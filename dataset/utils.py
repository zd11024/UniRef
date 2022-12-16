import re
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from tqdm import tqdm

from utils.hdfs_io import hexists, hcopy, hopen
# from vqaTools.vqaEval import VQAEval
from refTools.evaluation.refEvaluation import RefEvaluation
import copy

def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption


def vqa_eval(vqa, result_file, test_ques_path):
    vqaRes = vqa.loadRes(result_file, test_ques_path)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    # evaluate results
    vqaEval.evaluate()

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    return vqaEval


def write_json(result: list, wpath: str):
    if wpath.startswith('hdfs'):
        with hopen(wpath, 'w') as f:
            for res in result:
                to_write = json.dumps(res) + '\n'
                f.write(to_write.encode())
    else:
        with open(wpath, 'wt') as f:
            for res in result:
                f.write(json.dumps(res) + '\n')


def read_json(rpath: str):
    result = []
    if rpath.startswith('hdfs'):
        with hopen(rpath, 'r') as f:
            for line in f:
                result.append(json.loads(line.decode().strip()))
    else:
        with open(rpath, 'rt') as f:
            for line in f:
                result.append(json.loads(line.strip()))

    return result


def collect_result(result, filename, local_wdir, hdfs_wdir, write_to_hdfs=False, save_result=False):
    assert isinstance(result, list)
    write_json(result, os.path.join(hdfs_wdir if write_to_hdfs else local_wdir,
                                    '%s_rank%d.json' % (filename, utils.get_rank())))
    dist.barrier()

    result = []
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            result += read_json(os.path.join(hdfs_wdir if write_to_hdfs else local_wdir,
                                             '%s_rank%d.json' % (filename, rank)))

        if save_result:
            final_result_file = os.path.join(local_wdir, '%s.json' % filename)
            json.dump(result, open(final_result_file, 'w'))
            print('result file saved to %s' % final_result_file)
            if write_to_hdfs:
                hcopy(final_result_file, os.path.join(hdfs_wdir, '%s.json' % filename))
                print('result file saved to %s' % os.path.join(hdfs_wdir, '%s.json' % filename))

    dist.barrier()

    return result


def collect_tensor_result(result, filename, local_wdir, hdfs_wdir, write_to_hdfs=False):
    wpath = os.path.join(local_wdir, '%s_rank%d.pth' % (filename, utils.get_rank()))
    torch.save(result, wpath)
    if write_to_hdfs:
        hcopy(wpath, hdfs_wdir)

    dist.barrier()

    result = []
    if utils.is_main_process():
        # combine results from all processes
        for rank in range(utils.get_world_size()):
            rpath = os.path.join(local_wdir, '%s_rank%d.pth' % (filename, rank))
            if write_to_hdfs:
                hcopy(os.path.join(hdfs_wdir, '%s_rank%d.pth' % (filename, rank)), rpath)

            result += torch.load(rpath)

    dist.barrier()

    return result


def grounding_eval(results, dets, cocos, refer, alpha, mask_size=24):
    correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
    correct_A, correct_B, correct_val = 0, 0, 0
    num_A, num_B, num_val = 0, 0, 0

    for res in tqdm(results):

        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        mask = res['pred'].cuda().view(1, 1, mask_size, mask_size)
        mask = F.interpolate(mask, size=(image['height'], image['width']), mode='bicubic').squeeze()

        # rank detection boxes
        max_score = 0
        for det in dets[str(ref['image_id'])]:
            score = mask[int(det[1]):int(det[1] + det[3]), int(det[0]):int(det[0] + det[2])]
            area = det[2] * det[3]
            score = score.sum() / area ** alpha
            if score > max_score:
                pred_box = det[:4]
                max_score = score

        IoU_det = computeIoU(ref_box, pred_box)

        if ref['split'] == 'testA':
            num_A += 1
            if IoU_det >= 0.5:
                correct_A_d += 1
        elif ref['split'] == 'testB':
            num_B += 1
            if IoU_det >= 0.5:
                correct_B_d += 1
        elif ref['split'] == 'val':
            num_val += 1
            if IoU_det >= 0.5:
                correct_val_d += 1

    eval_result = {'val_d': correct_val_d / num_val, 'testA_d': correct_A_d / num_A, 'testB_d': correct_B_d / num_B}

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


def grounding_eval_bbox(results, refer):
    correct_A_d, correct_B_d, correct_val_d, correct_test_d = 0, 0, 0, 0
    num_A, num_B, num_val, num_test = 0, 0, 0, 0

    for res in tqdm(results):
        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        coord = res['pred'].cuda()
        coord[0::2] *= image['width']
        coord[1::2] *= image['height']

        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2

        IoU_det = computeIoU(ref_box, coord)

        if ref['split'] == 'testA':
            num_A += 1
            if IoU_det >= 0.5:
                correct_A_d += 1
        elif ref['split'] == 'testB':
            num_B += 1
            if IoU_det >= 0.5:
                correct_B_d += 1
        elif ref['split'] == 'val':
            num_val += 1
            if IoU_det >= 0.5:
                correct_val_d += 1
        elif ref['split'] == 'test':
            num_test += 1
            if IoU_det >= 0.5:
                correct_test_d += 1

    eval_result = {'val_d': correct_val_d / num_val}
    if num_A>0:
        eval_result['testA_d'] = correct_A_d / num_A
    if num_B>0:
        eval_result['testB_d'] = correct_B_d / num_B
    if num_test>0:
        eval_result['test_d'] = correct_test_d / num_test

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result

def grounding_eval_bbox(results, refer):
    correct_A_d, correct_B_d, correct_val_d, correct_test_d = 0, 0, 0, 0
    num_A, num_B, num_val, num_test = 0, 0, 0, 0
    pred_acc, pred_recall, pred_f1 = [], [], []
    for res in tqdm(results):
        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        coord = res['pred'].clone().cuda()
        coord[0::2] *= image['width']
        coord[1::2] *= image['height']

        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2

        IoU_det = computeIoU(ref_box, coord)

        if ref['split'] == 'testA':
            num_A += 1
            if IoU_det >= 0.5:
                correct_A_d += 1
        elif ref['split'] == 'testB':
            num_B += 1
            if IoU_det >= 0.5:
                correct_B_d += 1
        elif ref['split'] == 'val':
            num_val += 1
            if IoU_det >= 0.5:
                correct_val_d += 1
        elif ref['split'] == 'test':
            num_test += 1
            if IoU_det >= 0.5:
                correct_test_d += 1
        
        if 'acc' in res:
            pred_acc += [res['acc']]
            pred_recall += [res['recall']]
            pred_f1 += [res['f1']]

    eval_result = {'val_d': correct_val_d / num_val}
    if num_A>0:
        eval_result['testA_d'] = correct_A_d / num_A
    if num_B>0:
        eval_result['testB_d'] = correct_B_d / num_B
    if num_test>0:
        eval_result['test_d'] = correct_test_d / num_test
    
    if len(pred_acc)>0:
        eval_result['pred_acc'] = sum(pred_acc) / len(pred_acc)
        eval_result['pred_recall'] = sum(pred_recall) / len(pred_recall)
        eval_result['pred_f1'] = sum(pred_f1) / len(pred_f1)

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


def grounding_pred(results, refer):
    eval_result = []

    for res in tqdm(results):
        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        coord = res['pred'].clone().cuda()
        coord[0::2] *= image['width']
        coord[1::2] *= image['height']

        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2

        IoU_det = computeIoU(ref_box, coord)
        flg = 0
        if ref['split'] == 'testA':
            if IoU_det >= 0.5:
                flg = 1
        elif ref['split'] == 'testB':
            if IoU_det >= 0.5:
                flg = 1
        elif ref['split'] == 'val':
            if IoU_det >= 0.5:
                flg = 1
        elif ref['split'] == 'test':
            if IoU_det >= 0.5:
                flg = 1
        new_res = copy.deepcopy(res)
        new_res['pred'] = coord.tolist()
        new_res['correct'] = flg
        eval_result.append(new_res)
    return eval_result


def grounding_eval_bbox_by_position(results, refer):
    # correct_A_d, correct_B_d, correct_val_d, correct_test_d = 0, 0, 0, 0
    # num_A, num_B, num_val, num_test = 0, 0, 0, 0
    correct_pos, correct_non_pos = 0, 0
    num_pos, num_non_pos = 0, 0
    pos_word = ['left', 'right', 'front', 'behind', 'top', 'bottom']
    correct_per_pos = {'left':0, 'right':0, 'front':0, 'behind':0, 'top':0, 'bottom':0}
    num_per_pos = {'left':0, 'right':0, 'front':0, 'behind':0, 'top':0, 'bottom':0}

    for res in tqdm(results):
        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        coord = res['pred'].cuda()
        coord[0::2] *= image['width']
        coord[1::2] *= image['height']

        coord[0] -= coord[2] / 2
        coord[1] -= coord[3] / 2

        IoU_det = computeIoU(ref_box, coord)

        flg = False
        for p in pos_word:
            if p in res['text']:
                flg = True
                num_per_pos[p] += 1
                if IoU_det >= 0.5:
                    correct_per_pos[p] += 1

        if flg:
            num_pos += 1
            if IoU_det >= 0.5:
                correct_pos += 1
        else:
            num_non_pos += 1
            if IoU_det >= 0.5:
                correct_non_pos += 1
                
    eval_result = {
        'pos_d': correct_pos / (num_pos+1e-9), 
        'non_pos_d': correct_non_pos / (num_non_pos+1e-9),
        'num_pos': num_pos,
        'num_non_pos': num_non_pos
    }
    for p in pos_word:
        eval_result.update({
            f'per_pos_d_{p}': correct_per_pos[p] / (num_per_pos[p]+1e-9),
            f'num_per_pos_{p}': num_per_pos[p]
        })

    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')

    return eval_result


# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union


def prepare_inputs(texts, tokenizer, config, mask_generator=None):
    batch_size = len(texts)
    text_ids_batch = []
    text_atts_batch = []
    text_ids_masked_batch = []
    masked_pos_batch = []
    masked_ids_batch = []

    tokens_all = []
    max_len = 0

    for b in range(batch_size):
        text = texts[b]
        text = pre_caption(text, config['max_tokens'])
        tokens = tokenizer.tokenize(text)
        tokens = [tokenizer.cls_token] + tokens[:config['max_tokens']-2] + [tokenizer.sep_token]
        tokens_all += [tokens]
        max_len = max(max_len, len(tokens))
    
    for tokens in tokens_all:
        text_ids = tokenizer.convert_tokens_to_ids(tokens)

        # pad
        n_text = len(text_ids)
        n_pad = max_len - len(text_ids)
        text_ids += [tokenizer.pad_token_id] * n_pad
        text_atts = [1] * n_text + [0] * n_pad

        text_ids_batch.append(text_ids)
        text_atts_batch.append(text_atts)

        ### mask inputs
        if mask_generator:
            tokens_masked, masked_pos = mask_generator(copy.deepcopy(tokens))
            text_ids_masked = tokenizer.convert_tokens_to_ids(tokens_masked)
            masked_ids = [text_ids[p] for p in masked_pos]
            text_ids_masked += n_pad * [tokenizer.pad_token_id]        
     
            # pad
            n_pad_mask = config['max_masks'] - len(masked_pos)
            masked_pos += [tokenizer.pad_token_id] * n_pad_mask
            masked_ids += [-100] * n_pad_mask

            text_ids_masked_batch.append(text_ids_masked)
            masked_pos_batch.append(masked_pos)
            masked_ids_batch.append(masked_ids)

    text_ids_batch = torch.tensor(text_ids_batch)
    text_atts_batch = torch.tensor(text_atts_batch)
    uni_text_atts = get_uni_attetion_mask(text_atts_batch)
    if mask_generator:
        text_ids_masked_batch = torch.tensor(text_ids_masked_batch)
        masked_pos_batch = torch.tensor(masked_pos_batch)
        masked_ids_batch = torch.tensor(masked_ids_batch)

    # return dict
    ret = {
        'text_ids': text_ids_batch,
        'uni_text_atts': uni_text_atts,
        'bi_text_atts': text_atts_batch,
    }
    if mask_generator:
        ret.update({
            'text_ids_masked': text_ids_masked_batch,
            'masked_pos': masked_pos_batch,
            'masked_ids': masked_ids_batch
        })
    return ret
    


def get_uni_attetion_mask(text_atts):
    batch_size, seq_length = text_atts.size()
    seq_ids = torch.arange(seq_length, device=text_atts.device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    causal_mask = causal_mask.to(text_atts.dtype)

    if causal_mask.shape[1] < text_atts.shape[1]:
        prefix_seq_len = text_atts.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                ),
                causal_mask,
            ],
            axis=-1,
        )
    uni_text_atts = causal_mask * text_atts[:, None, :]
    return uni_text_atts