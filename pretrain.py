'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from model.model_pretrain import ALBEF
from model.xvlm import XVLM, load_pretrained
from TCL.models.vit import interpolate_pos_embed
from TCL.models.tokenization_bert import BertTokenizer

import utils
from TCL.scheduler import create_scheduler
from TCL.optim import create_optimizer

from tqdm import tqdm
import wandb

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_p2w', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    if config['mid_mlm']:
        metric_logger.add_meter('loss_ctext', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
   # if config['cl_cls'] or config['cl_bw']:
    #    metric_logger.add_meter('loss_bw', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
   
    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()
        
        image = image.to(device,non_blocking=True) 

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)  
        
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        weight = config['weight']
        
        loss_all = model(image, text_input, alpha = alpha) 

        if config['mid_mlm']:
           # if config['cl_cls'] or config['cl_bw']:
            #    loss_mlm, loss_ita, loss_itm, loss_p2w, loss_ctext, loss_bw = loss_all
           # else:
            loss_mlm, loss_ita, loss_itm, loss_p2w, loss_ctext = loss_all
        else:
            loss_mlm, loss_ita, loss_itm, loss_p2w = loss_all 
            
        loss = loss_mlm + loss_ita + loss_itm + loss_p2w *weight 
        if config['mid_mlm']:
            loss += loss_ctext
        #if config['cl_cls'] or config['cl_bw']:
        #    loss += loss_bw
          
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_p2w=loss_p2w.item())
        if config['mid_mlm']:
            metric_logger.update(loss_ctext = loss_ctext.item())
        #if config['cl_cls'] or config['cl_bw'] :
        #    metric_logger.update(loss_bw=loss_bw.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])    

        #print(i,'  loss_mlm:', loss_mlm.item(), 'loss_ita:', loss_ita.item(), 'loss_itm:', loss_itm.item(), 'loss_mim:',loss_mim.item())     
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)  
        '''
        wandb.log({
            "loss_p2w": loss_p2w.item(),
            "loss_mlm": loss_ctext.item(),
            "loss_im": loss_cimg.item(),
            #"max_pre": max_pre.item()
        })
        '''
        #break
        
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}   
    

def evaluate(model, data_loader, tokenizer, device, config, alpha):
    # eval
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    header = 'Eval Epoch: '
    print_freq = len(data_loader) 

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
  
        image = image.to(device,non_blocking=True) 

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)  
        
        loss_all = model(image, text_input, alpha = alpha)  
        if config['mid_mlm']:
            #if config['cl_cls'] or config['cl_bw'] :
            #    loss_mlm, loss_ita, loss_itm, loss_p2w, loss_ctext, loss_bw = loss_all
            #else:
            loss_mlm, loss_ita, loss_itm, loss_p2w, loss_ctext = loss_all
        else:
            loss_mlm, loss_ita, loss_itm, loss_p2w = loss_all 
            
        loss = loss_mlm + loss_ita + loss_itm + loss_p2w 
        if config['mid_mlm']:
            loss += loss_ctext
        #if config['cl_cls'] or config['cl_bw'] :
        #    loss += loss_bw
        
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_p2w=loss_p2w.item())
        if config['mid_mlm']:
            metric_logger.update(loss_ctext = loss_ctext.item())
        #if config['cl_cls'] or config['cl_bw'] :
        #    metric_logger.update(loss_bw=loss_bw.item())
        #break         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 

def patch2token(model, dataset, tokenizer, device):
    # eval
    model.eval()
    data_loader = DataLoader(dataset,batch_size=4)  
    for i, (image_o, text) in enumerate(data_loader):
        image = image_o[:4]
        text = text[:4]
        image = image.to(device,non_blocking=True) 
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        logits, label = model.module.p2t(image, text_input)
        values, index = logits.topk(5, largest=True, sorted=True)
        v_l, i_l = label.topk(5, largest=True, sorted=True)
        tokens = []
        labels = []
        for j, tmpimg in enumerate(index):
            img = image_o[j] #3,256,256
            tmptext = []
            tmplabel = []
            for k, patch in enumerate(tmpimg):
                tmpt = tokenizer.decode(patch)
                tmptext.append(tmpt)
                tmpt = tokenizer.decode(i_l[j][k])
                tmplabel.append(tmpt)
                #break
            #break
            tokens.append(tmptext)
            labels.append(tmplabel)
            utils.imageSavePLT(image_o[j], './output/img/'+str(j)+'.jpg')
        dic = {'text': tokens, 'label': labels, 'text_pro:': values.cpu().detach().numpy().tolist(), 'labels_pro': v_l.cpu().detach().numpy().tolist()}
        with open('./output/result.json', 'w') as file_obj:
            json.dump(dic, file_obj)
        #break


def main(args, config):
    #wandb.init(project="emb_label", entity="multi-retrieval")

    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    #### Dataset ####  
    print("Creating dataset")
    datasets = [utils.create_dataset('pretrain', config)]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = utils.create_sampler(datasets, [True], num_tasks, global_rank)         
    else:
        samplers = [None]

    data_loader= utils.create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    config['text_encoder'] = args.text_encoder
    #### Model #### 
    print("Creating model")
    teacher_model = None
    if config['use_teacher']:
        ckpt_rpath = os.path.join(config['ckpt_rpath'], config['model_type']+'.pt')
        teacher_model = XVLM(config=config)
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=True, load_text=True)
        msg = teacher_model.load_state_dict(state_dict, strict=False) 
        teacher_model = teacher_model.to(device)
        print(msg) 

    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer,use_distill_loss=config['mid_mlm'], teacher_model=teacher_model)
    
    model = model.to(device) 

    
        
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1       
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)     
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped                    
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    #patch2token(model, datasets[0], tokenizer, device)

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
            
        train_stats= train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config) 
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
              
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
        dist.barrier()  
        #break
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)