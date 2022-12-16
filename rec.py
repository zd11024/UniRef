import argparse
import datetime
import json
import math
import os
import random
import time
import copy
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from dataset import create_dataset, create_sampler, create_loader
from dataset.utils import grounding_pred, collect_tensor_result, grounding_eval_bbox, grounding_eval_bbox_by_position
from models import load_pretrained
from models.uniref import UniRef
from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer
from optim import create_optimizer
from refTools.refer_python3 import REFER
from scheduler import create_scheduler
from utils.hdfs_io import hmkdir, hcopy, hexists
from utils.torch_io import save as hdfs_torch_save
from refTools.evaluation.refEvaluation import RefEvaluation
from dataset.pretrain_dataset import TextMaskingGenerator
from dataset.utils import pre_caption
from dataset.utils import prepare_inputs


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_pred', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('acc_pred', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('recall_pred', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('f1_pred', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
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

        loss_bbox, loss_giou, loss_pred, acc_pred, recall_pred, f1_pred = model(text_atts=text_atts, text_ids=text_ids, image=image, image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox,
                ret_bbox_loss=True, ret_mlm_loss=False, use_new_segment=config['use_new_segment'])
        loss = loss_bbox + loss_giou + loss_pred * config['p_weight']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(loss_pred=loss_pred.item())
        metric_logger.update(acc_pred=acc_pred.item())
        metric_logger.update(recall_pred=recall_pred.item())
        metric_logger.update(f1_pred=f1_pred.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def val(model, data_loader, tokenizer, device, refer=None):
    

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    result = []
    for image, texts, image_atts, target_bbox, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device, non_blocking=True)
        target_bbox = target_bbox.to(device)
        # text_input = tokenizer(texts, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
        inputs = prepare_inputs(texts, tokenizer, config)
        text_ids = inputs['text_ids'].to(device)
        text_atts = inputs['bi_text_atts'].to(device)
        image_atts = image_atts.to(device)

        with torch.no_grad():
            image_embeds, _ = model.get_vision_embeds(image)
            text_embeds = model.get_text_embeds(text_ids, text_atts)
            outputs_coord, region_att = model.predict(text_ids=text_ids, text_atts=text_atts, image=image, ret_region_att=True, image_embeds=image_embeds, text_embeds=text_embeds)

            acc, recall = 0, 0
            target = image_atts[:, 1:].float()
            pred = region_att[:,-1,:]
            acc += (pred * target).sum(dim=1) / (pred.sum(dim=1) + 1e-9)
            recall += (pred * target).sum(dim=1) / target.sum(dim=1)
            f1 = 2 * acc * recall / (acc + recall + 1e-9)
        
        assert len(ref_ids) == outputs_coord.shape[0]

        # for r_id, coord, text, att in zip(ref_ids, outputs_coord, texts, region_att):
        #     result.append({'ref_id': r_id.item(), 'pred': coord, 'text': text, 'att': att.tolist()})
        
        batch_size = image.size(0)
        for bn in range(batch_size):
            result.append({
                'ref_id': ref_ids[bn].item(),
                'pred': outputs_coord[bn],
                'text': texts[bn], 
                'att': region_att[bn].tolist(),
                'acc': acc[bn].item(),
                'recall': recall[bn].item(),
                'f1': f1[bn].item()
            })


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

    mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'],
                                        config['max_masks'], config['skipgram_prb'],
                                        config['skipgram_size'], config['mask_whole_word'])


    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")

        if args.input_file:
            config['reg_file'] = args.input_file
            reg_dataset = create_dataset('reg', config, refer=refer)
            reg_sampler =  create_sampler([reg_dataset], [False], utils.get_world_size(), utils.get_rank())
            reg_loader = create_loader([reg_dataset], reg_sampler,
                                batch_size=[config['batch_size']],
                                num_workers=[4], is_trains=[False], collate_fns=[None])[0]
            result = val(model_without_ddp, reg_loader, tokenizer, device, refer=refer)
        else:
            result = val(model_without_ddp, test_loader, tokenizer, device, refer=refer)
        
        results = collect_tensor_result(result, filename='grounding_bbox_eval', local_wdir=args.result_dir,
                                        hdfs_wdir=args.output_hdfs,
                                        write_to_hdfs=world_size > 8)

        if utils.is_main_process():
            grounding_acc = grounding_eval_bbox(results, refer)
            log_stats = {**{f'{k}': v for k, v in grounding_acc.items()}}
            print(log_stats)


            pred_list = grounding_pred(results, refer)

            with open(args.output_file, 'w') as f:
                json.dump(pred_list, f)

        dist.barrier()

    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(train_batch_size*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        if args.epochs > 0:
            max_epoch = args.epochs
        best = 0
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
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config, mask_generator)

            result = val(model_without_ddp, test_loader, tokenizer, device)
            results = collect_tensor_result(result, filename='epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                     write_to_hdfs=world_size > 8)

            if utils.is_main_process():
                grounding_acc = grounding_eval_bbox(results, refer)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'{k}': v for k, v in grounding_acc.items()},
                             'epoch': epoch}

                if grounding_acc['val_d'] >= best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                    }
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.th'))
                    hdfs_torch_save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.th'))
                    best = grounding_acc['val_d']
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
    parser.add_argument('--dataset', type=str, default='refcoco+', choices=['refcoco', 'refcoco+', 'refcocog'])

    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='pred.json')


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

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)