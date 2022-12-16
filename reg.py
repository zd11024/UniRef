import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from dataset import create_dataset, create_sampler, create_loader
from dataset.utils import collect_tensor_result, grounding_eval_bbox
from models import load_pretrained
from models.uniref import UniRef, UniRefDecoder
from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer
from optim import create_optimizer
from refTools.refer_python3 import REFER
from scheduler import create_scheduler
from utils.hdfs_io import hmkdir, hcopy, hexists
from utils.torch_io import save as hdfs_torch_save
from refTools.evaluation.refEvaluation import RefEvaluation
from dataset.pretrain_dataset import TextMaskingGenerator
from dataset.utils import prepare_inputs


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator, task_list=['reg']):

    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if 'rec' in task_list:
        metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        
        metric_logger.add_meter('loss_pred', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('acc_pred', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('recall_pred', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('f1_pred', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    if 'reg' in task_list:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100

    for image, texts, image_atts, target_bbox, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        batch_size = image.size(0)
        image = image.to(device)
        image_atts = image_atts.to(device)
        idx_to_group_img = torch.arange(batch_size).to(device)
        target_bbox = target_bbox.to(device)

        inputs = prepare_inputs(texts, tokenizer, config, mask_generator)
        text_atts, text_ids_masked, masked_pos, masked_ids = inputs['bi_text_atts'], inputs['text_ids_masked'], inputs['masked_pos'], inputs['masked_ids']
        text_atts, text_ids_masked, masked_pos, masked_ids = \
            text_atts.to(device), text_ids_masked.to(device), masked_pos.to(device), masked_ids.to(device)
        text_ids = inputs['text_ids'].to(device)

        if ('rec' in task_list) and ('reg' in task_list):
            loss_mlm, loss_bbox, loss_giou, loss_pred, acc_pred, recall_pred, f1_pred = model(text_atts=text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                image=image, image_atts=image_atts, idx_to_group_img=idx_to_group_img, 
                text_ids=text_ids, target_bbox=target_bbox,
                ret_bbox_loss=True, ret_mlm_loss=True, task_id=1, use_new_segment=config['use_new_segment'])
            loss = loss_mlm + (loss_bbox + loss_giou) * config['rec_weight'] + loss_pred * config['p_weight']
        
        elif 'reg' in task_list:
            loss_outputs = model(text_atts=text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                image=image, image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox,
                ret_bbox_loss=False, ret_mlm_loss=True, task_id=1, use_new_segment=config['use_new_segment'])
            loss_mlm = loss_outputs[0]
            loss = loss_mlm

        elif 'rec' in task_list:
            loss_bbox, loss_giou, loss_pred, acc_pred, recall_pred, f1_pred = model(text_atts=text_atts, text_ids=text_ids, image=image, image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox,
                ret_bbox_loss=True, ret_mlm_loss=False, use_new_segment=config['use_new_segment'])
            loss = loss_bbox + loss_giou + loss_pred * config['p_weight']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if 'rec' in task_list:
            metric_logger.update(loss_bbox=loss_bbox.item())
            metric_logger.update(loss_giou=loss_giou.item())
            metric_logger.update(loss_pred=loss_pred.item())
            metric_logger.update(acc_pred=acc_pred.item())
            metric_logger.update(recall_pred=recall_pred.item())
            metric_logger.update(f1_pred=f1_pred.item())
        if 'reg' in task_list:
            metric_logger.update(loss_mlm=loss_mlm.item())
        

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def val(model, data_loader, tokenizer, device, mode='beam search'):
    

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50

    
    decode_args = {
        'max_length': 20,
        'bos_token_id': tokenizer.cls_token_id,
        'eos_token_id': tokenizer.sep_token_id,
        'min_length': 2
    }

    # exists bugs in beam search
    if mode=='beam search':
        decode_args.update({
            'num_beams': 5,
            'early_stopping': True,
        })

    result = []
    for image, text, image_atts, target_boxes, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        batch_size = image.size(0)
        image = image.to(device)
        image_atts = image_atts.to(device)
        idx_to_group_img = torch.arange(batch_size).to(device)
        input_ids = torch.zeros([batch_size, 1], dtype=torch.int).to(device) + tokenizer.cls_token_id
        attention_mask = torch.ones([batch_size, 1], dtype=torch.int).to(device)

        with torch.no_grad():

            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                image_atts=image_atts,
                idx_to_group_img=idx_to_group_img,
                use_new_segment=config['use_new_segment'],
                **decode_args
            )

        for b in range(batch_size):
            sent = tokenizer.decode(output_sequences[b].tolist(), skip_special_tokens=True)
            ref_id = ref_ids[b].item()
            result.append({'ref_id': ref_id, 'sent': sent, 'gt_text': text[b], 'image': data_loader.dataset.refid2img[ref_id]})
        
    return result


def val_rec(model, data_loader, tokenizer, device):
    

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50

    result = []
    for image, texts, image_atts, target_bbox, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device, non_blocking=True)
        target_bbox = target_bbox.to(device)
        inputs = prepare_inputs(texts, tokenizer, config)

        with torch.no_grad():
            outputs_coord = model.predict(text_ids=inputs['text_ids'].to(device), text_atts=inputs['bi_text_atts'].to(device), image=image)

        assert len(ref_ids) == outputs_coord.shape[0]

        for r_id, coord, text in zip(ref_ids, outputs_coord, texts):
            result.append({'ref_id': r_id.item(), 'pred': coord, 'text': text})

    return result


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # refcoco evaluation tools
    split_map = {'refcoco': 'unc', 'refcoco+': 'unc', 'refcocog': 'umd'}
    split_eval = {'refcoco': ['val', 'testA', 'testB'], 'refcoco+': ['val', 'testA', 'testB'], 'refcocog': ['val', 'test']}
    refer = REFER(config['refcoco_data'], args.dataset, split_map[args.dataset])

    print("Creating dataset")
    grd_train_dataset, grd_test_dataset = create_dataset(args.dataset, config, refer=refer)
    datasets = [grd_train_dataset, grd_test_dataset]

    train_dataset_size = len(grd_train_dataset)
    train_batch_size = config['batch_size'] // world_size

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size'], config['batch_size']],
                                              num_workers=[4, 4], is_trains=[True, False], collate_fns=[None, None])


    print("Creating model")
    model = UniRef(config=config, load_vision_params=False)
    if args.checkpoint:
        model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module   

    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        gen_model = UniRefDecoder.from_pretrained(config, tokenizer, model_without_ddp)
        result = val(gen_model, test_loader, tokenizer, device, mode='beam search')
        results = collect_tensor_result(result, filename='grounding_bbox_eval', local_wdir=args.result_dir,
                                        hdfs_wdir=args.output_hdfs,
                                        write_to_hdfs=world_size > 8)

        if utils.is_main_process():
            eval_results = {}
            final_results = []
            for split in split_eval[args.dataset]:
                ref_ids = refer.getRefIds(split=split)
                split_results = []
                for d in results:
                    if d['ref_id'] in ref_ids:
                        split_results.append(d)
                
                refEval = RefEvaluation(refer, split_results)
                refEval.evaluate()
                print(f'Split: {split}')
                for metric, score in refEval.eval.items():
                    eval_results[f'{split}_{metric}'] = score
                
                for y in refEval.evalRefs:
                    y['sent']=refEval.refToRes[y['ref_id']][0]
                    final_results.append(y)  

            for d in final_results:
                ref_id = d['ref_id']
                ref = refer.Refs[ref_id]
                image = refer.Imgs[ref['image_id']]
                if 'train2014' in image['file_name']:
                    img_path = os.path.join('train2014', image['file_name'])
                else:
                    img_path = os.path.join('val2014', image['file_name'])
                d['image'] = img_path
                
            for metric, score in eval_results.items():
                print('%s: %.3f'%(metric, score))
             
            with open(os.path.join(args.output_file), 'w') as f:
                json.dump(final_results, f)


        dist.barrier()

    else:
        mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'],
                                                   config['max_masks'], config['skipgram_prb'],
                                                   config['skipgram_size'], config['mask_whole_word'])

        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        best = -1
        best_epoch = 0
        start_epoch = 0
        if args.checkpoint and args.resume:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']

            file_hdfs = os.path.join(args.output_dir, 'log.txt')
            if hexists(file_hdfs):
                print('### Read from hdfs log.txt')
                hcopy(file_hdfs, 'log.txt')


        print(f'### Start from {start_epoch}...')

        for epoch in range(start_epoch, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            training_loader = train_loader
            task_list = ['reg']
            if config['joint_training']:
                task_list = ['rec', 'reg']

            print('###Train', ','.join(task_list))

            train_stats = train(model, training_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config, mask_generator, task_list=task_list)

            if config['joint_training']:
                result_rec = val_rec(model_without_ddp, test_loader, tokenizer, device)
                results_rec = collect_tensor_result(result_rec, filename='rec_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                    write_to_hdfs=world_size > 8)

                if utils.is_main_process():
                    grounding_acc = grounding_eval_bbox(results_rec, refer)

            gen_model = UniRefDecoder.from_pretrained(config, tokenizer, model_without_ddp)
            result = val(gen_model, test_loader, tokenizer, device, mode='beam search')
            results = collect_tensor_result(result, filename='reg_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                     write_to_hdfs=world_size > 8)

            if utils.is_main_process():
                eval_results = {}
                for split in split_eval[args.dataset]:
                    ref_ids = refer.getRefIds(split=split)
                    split_results = []
                    for d in results:
                        if d['ref_id'] in ref_ids:
                            split_results.append(d)
                    
                    refEval = RefEvaluation(refer, split_results)
                    refEval.evaluate()
                    print(f'Split: {split}')
                    for metric, score in refEval.eval.items():
                        eval_results[f'{split}_{metric}'] = score
                
                log_stats = {'epoch': epoch,
                            **{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'{k}': v for k, v in eval_results.items()},
                }
                if config['joint_training']:
                    log_stats.update({
                            **{f'{k}': v for k, v in grounding_acc.items()}
                    })

                # save the best
                if eval_results['val_CIDEr'] >= best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                    }
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.th'))
                    hdfs_torch_save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.th'))
                    best = eval_results['val_CIDEr']
                    best_epoch = epoch
                
                ## save the current
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'best': best,
                    'best_epoch': best_epoch
                }
                hdfs_torch_save(save_obj, os.path.join(args.output_dir, 'training_state_latest.th'))

                with open(os.path.join("log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                hcopy('log.txt', os.path.join(args.output_dir, 'log.txt'))

            dist.barrier()

        if utils.is_main_process():
            with open(os.path.join("log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)
            hcopy('log.txt', os.path.join(args.output_dir, 'log.txt'))

            os.system(f"cat log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, default='configs/uniref_finetune.yaml')
    parser.add_argument('--output_dir', type=str, default='output/refcoco_bbox')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--load_bbox_pretrain', action='store_true')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--epochs', default=-1, type=int)
    parser.add_argument('--lr', default=-1., type=float)
    parser.add_argument('--mask_prob', default=-1., type=float)
    parser.add_argument('--dataset', type=str, default='refcoco+', choices=['refcoco', 'refcoco+', 'refcocog'])

    parser.add_argument('--joint_training', type=str, default=None, help='joint training with rec')
    parser.add_argument('--rec_weight', default=1, type=float)

    parser.add_argument('--output_file', type=str, default='gen.json')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    if args.bs > 0:
        config['batch_size'] = args.bs
    if args.epochs > 0:
        config['schedular']['epochs'] = args.epochs
    if args.lr > 0:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr
    if args.mask_prob >= 0:
        config['mask_prob'] = args.mask_prob
    
    config['joint_training'] = args.joint_training
    config['rec_weight'] = args.rec_weight

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    hmkdir(args.output_dir)

    main(args, config)