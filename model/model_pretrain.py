'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
 
from functools import partial
from model.vit import VisionTransformer, Block, interpolate_pos_embed
from model.swin_transformer import interpolate_relative_pos_embed, SwinTransformer, Mlp
from TCL.models.xbert import BertConfig, BertForMaskedLM, BertOnlyMLMHead, BertLayer, BertModel
 
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torch.distributed as dist
 
import numpy as np
import random
import os
import math
from utils import read_json
from model.xvlm import XVLM, load_pretrained, build_mlp

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply
 
class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 use_contrastive_loss = True,
                 use_matching_loss = True,
                 use_mlm_loss = True,
                 use_p2w_loss = True,
                 use_distill_loss = True,
                 teacher_model = None,
                 init_model = True,
                 ):
        super().__init__()
        

        self.tokenizer = tokenizer  
        self.model_type = config['model_type']
        if config['model_type']=='tcl'or config['model_type']=='albef':
            self.visual_encoder = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
            vision_width = config['vision_width']
        elif config['model_type']=='xvlm':
            vision_config = read_json(config['vision_config'])
            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'], patch_size=4, in_chans=3, embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'], num_heads=vision_config['num_heads'], window_size=vision_config['window_size'],
                mlp_ratio=4., qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, ape=False, patch_norm=True, use_checkpoint=False)
            vision_width = vision_config['vision_width']
              
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        if use_mlm_loss:
            self.mlm_probability = config['mlm_probability']
            self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)    
        else:
            self.text_encoder = BertModel(config=bert_config, add_pooling_layer=False) 

        text_width = self.text_encoder.config.hidden_size
        if use_matching_loss:
            #self.itm_head = nn.Linear(text_width, 2)
            self.itm_head = build_mlp(input_dim=text_width, output_dim=2)

        if init_model:
            self.init_pretrained(config['ckpt_rpath'], config['model_type'], config)

        if use_contrastive_loss:
            embed_dim = config['embed_dim']
            self.vision_proj = nn.Linear(vision_width, embed_dim)
            self.text_proj = nn.Linear(text_width, embed_dim)    
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])   

        if use_p2w_loss:
            self.im_probability = config['im_probability']
            self.open_voc = config['open_voc']
            self.img_pooling = config['img_pooling']
            self.img_head = BertOnlyMLMHead(bert_config)
            #self.img_head = self.text_encoder.cls

        self.v2t_proj = nn.Linear(vision_width, text_width)
        #self.v2t_proj = Mlp(vision_width, vision_width*text_width//math.gcd(vision_width, text_width), text_width)
        self.add_img_head = config['img_head']
        if self.add_img_head:
            
            self.cimg_head = nn.ModuleList([BertLayer(bert_config, i) for i in range(2)])
            if init_model:
                self.cimg_head.apply(self.text_encoder._init_weights)
            '''
            dpr = [x.item() for x in torch.linspace(0, 0, 2)] 
            self.cimg_head =nn.ModuleList([
                Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                    drop=0, attn_drop=0, drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6))
                for i in range(2)])
            self.cimg_head.apply(self.visual_encoder._init_weights)
            '''

        self.cl_bw = config['cl_bw']
        self.cl_cls = config['cl_cls']

        self.mid_mlm = use_distill_loss
        if use_distill_loss:
            self.mlm_head = self.text_encoder.cls
            #self.mlm_head = self.img_head
        self.add_text_head = config['text_head']
        if self.add_text_head:
            self.ctext_head = nn.ModuleList([BertLayer(bert_config,i) for i in range(2)])
            if init_model:
                self.ctext_head.apply(self.text_encoder.bert._init_weights)
        
        self.teacher_model = teacher_model       
        if self.teacher_model is not None:
            for name, parameter in teacher_model.named_parameters():
                parameter.requires_grad = False
        if init_model:
            self._tie_or_clone_weights(self.img_head.predictions.decoder, self.text_encoder.cls.predictions.decoder, bert_config)
        
    def get_teacher_model(self, config):
        ckpt_rpath = os.path.join(config['ckpt_rpath'], config['model_type']+'.pt')
        xvlm_model = XVLM(config=config)
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=True, load_text=True)
        msg = xvlm_model.load_state_dict(state_dict, strict=False)
        for name, parameter in xvlm_model.named_parameters():
            parameter.requires_grad = False
        return xvlm_model
    
    def init_pretrained(self, ckpt_rpath, model_type, config):
        ckpt_rpath = os.path.join(ckpt_rpath,model_type+'.pt')
        checkpoint = torch.load(ckpt_rpath, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        if model_type=='tcl' or model_type=='albef':
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],self.visual_encoder)     
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            state_dict['visual_encoder'] = {}
            state_dict['text_encoder'] = {}
            for key in state_dict.keys():
                if key.startswith('visual_encoder.'):
                    new_key = key.split('visual_encoder.')[-1]
                    state_dict['visual_encoder'][new_key] = state_dict[key]
                if key.startswith('text_encoder.'):
                    new_key = key.split('text_encoder.')[-1]
                    state_dict['text_encoder'][new_key] = state_dict[key]

        elif model_type=='xvlm': #swin
            window_size = read_json(config['vision_config'])['window_size']
            state_dict['visual_encoder'] = {}
            state_dict['text_encoder'] = {}
            state_dict['itm_head'] = {}
            for k in list(state_dict.keys()):
                if k.startswith('itm_head.'):
                    new_key = k.split('itm_head.')[-1]
                    state_dict['itm_head'][new_key] = state_dict[k]
                '''
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * window_size - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]
                    continue
                '''
                if k.startswith('vision_encoder.'):
                    new_key = k.split('vision_encoder.')[-1]
                    state_dict['visual_encoder'][new_key] = state_dict[k]
                if k.startswith('text_encoder.'):
                    if 'crossattention' in k:
                        continue
                    new_key = k.split('text_encoder.')[-1]
                    state_dict['text_encoder'][new_key] = state_dict[k]
            msg = self.itm_head.load_state_dict(state_dict['itm_head'],strict=False)
            print(msg)

        msg = self.visual_encoder.load_state_dict(state_dict['visual_encoder'],strict=False)
        print(msg)
        msg = self.text_encoder.load_state_dict(state_dict['text_encoder'], strict=False)
        print(msg)
    
    def _tie_or_clone_weights(self, first_module, second_module, config):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
 
    def get_text_embeds(self, text):
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        text_output = encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', output_hidden_states=True)  
        hidden_states = text_output[0]
        return hidden_states
    
    def get_image_embeds(self, image, att=None):
        image_embeds = self.visual_encoder(image)
        #image_embeds = self.v2t_proj(image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        return image_embeds, image_atts
    
    def get_img_bw(self, image_embeds):
        token_embeddings = self.img_head(image_embeds)[:,1:,:]
        img_w = torch.softmax(torch.max(token_embeddings, dim=1).values, dim=-1)
        word_embeddings_matrix = self.text_encoder.bert.embeddings.word_embeddings.weight
        img_lexicon = torch.matmul(img_w, word_embeddings_matrix.detach()).unsqueeze(1)
        return img_lexicon

    def get_text_bw(self, text_embeds, mask_att):
        token_embeddings = self.mlm_head(text_embeds)[:,1:,:]
        mask_att = mask_att[:,1:]
        token_embeddings.masked_fill_((mask_att == 0).unsqueeze(-1), 0.)  # apply mask
        text_w = torch.softmax(torch.max(token_embeddings, dim=1).values, dim=-1)
        word_embeddings_matrix = self.text_encoder.bert.embeddings.word_embeddings.weight
        text_lexicon = torch.matmul(text_w, word_embeddings_matrix.detach()).unsqueeze(1)
        return text_lexicon

    def get_features(self, image_embeds=None, text_embeds=None, mask_att=None):
        if image_embeds is None:
            if self.cl_bw:
                text_lexicon = self.get_text_bw(text_embeds, mask_att)
                text_fbw = F.normalize(self.text_proj(text_lexicon.squeeze()),dim=-1)
                return text_fbw
            return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            if self.cl_bw:
                img_lexicon = self.get_img_bw(image_embeds)
                img_fbw = F.normalize(self.vision_proj(img_lexicon.squeeze()),dim=-1)
                return img_fbw
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        else:
            if self.cl_bw:
                img_lexicon = self.get_img_bw(image_embeds)
                img_fbw = F.normalize(self.vision_proj(img_lexicon.squeeze()),dim=-1)
                text_lexicon = self.get_text_bw(text_embeds, mask_att)
                text_fbw = F.normalize(self.text_proj(text_lexicon.squeeze()),dim=-1)
                return img_fbw, text_fbw
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1), \
                   F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        assert image_feat.size(-1) == text_feat.size(-1)

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2
    
    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        """
        Matching Loss with hard negatives
        """
        text_encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        bs = image_embeds.size(0)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_t2i = F.softmax(sim_t2i, dim=1)

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)
        
        cross_pos = text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = text_atts,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       ).last_hidden_state[:, 0, :]
        cross_neg = text_encoder(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       ).last_hidden_state[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        return F.cross_entropy(output, itm_labels)

    def get_mlm_loss(self, text, image, image_embeds, image_atts, alpha):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
 
        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        mlm_output = self.text_encoder(input_ids, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        labels = labels,   
                                        alpha = alpha
                                        )                           
        loss_mlm = mlm_output.loss  
        
        if self.mid_mlm:
            if self.teacher_model is not None:
                image_embeds_teacher, _ = self.teacher_model.get_vision_embeds(image)
                mlm_output = self.teacher_model.text_encoder(input_ids,
                                 attention_mask=text.attention_mask,
                                 encoder_hidden_states=image_embeds_teacher,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=labels,
                                 alpha = alpha)
            text_output = self.text_encoder.bert(input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', output_hidden_states=True) 
            hidden_states = text_output[0]
            if self.add_text_head:
                attention_mask = self.text_encoder.bert.get_extended_attention_mask(
                    text.attention_mask,
                    text.attention_mask.shape,
                    image.device,
                    False
                )
                for layer in self.ctext_head:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask,
                    )
                    hidden_states = layer_outputs[0]
            pre_scores_all = self.mlm_head(hidden_states)
            pre_scores = pre_scores_all.view(-1, self.text_encoder.config.vocab_size)
            label_scores = mlm_output.logits.view(-1, self.text_encoder.config.vocab_size)
            idx = torch.where(labels.view(-1)!=-100)
            pre_scores = pre_scores[idx[0],:]
            label_scores = label_scores[idx[0],:]
            kl_loss = F.kl_div(pre_scores.softmax(dim=-1).log(), label_scores.softmax(dim=-1), reduction='sum')/pre_scores.shape[0]
            return loss_mlm, kl_loss

        return loss_mlm

    def get_p2w_loss(self, image, text, image_atts, image_embeds, text_embeds, alpha):
        image_embeds_mask, target = self.visual_encoder(image, mask=True, mask_pro=self.im_probability) 
        indexs = torch.where(target!=-100)
    
        if self.teacher_model is None:
            img_temb = self.vision_proj(image_embeds[:,1:,:])
            if self.img_pooling:
                img_temb = self.patch_pooling(img_temb)
            text_temb = self.text_proj(text_embeds[:,1:,:])
        else:
            image_emb_label, _ = self.teacher_model.get_vision_embeds(image, image_atts)
            text_emb_label = self.teacher_model.get_text_embeds(text.input_ids, text.attention_mask)
            img_temb, text_temb = self.teacher_model.vision_proj(image_emb_label[:,1:,:]), self.teacher_model.text_proj(text_emb_label[:,1:,:])

        #print('img', img_temb[torch.isnan(img_temb)],'txt:', len(text_embeds[torch.isnan(text_embeds)]), torch.where(torch.isnan(text_embeds)))
        txt_pad = ~text.attention_mask.bool()
        img_pad = ~image_atts.bool()
        if self.img_pooling:
            img_pad = img_pad[:,0:img_temb.shape[1]+1]
        distance, T = optimal_transport_dist(text_temb, img_temb, txt_pad[:,1:], img_pad[:,1:])
        #print(distance, torch.sum(T, 2), T[0][0])
        '''
        tsum = 1/torch.sum(T, 2, keepdim=True)
        if torch.isinf(tsum).any().item():
            tsum = torch.where(torch.isinf(tsum), torch.full_like(tsum, 1), tsum)
        labels = T*(tsum)
        '''
        scale = 2e3
        #scale, _ = torch.max(T, dim=-1, keepdim=True)
        #scale = 1/scale
        #if torch.isinf(scale).any().item():
        #    scale = torch.where(torch.isinf(scale), torch.full_like(scale, 1), scale)
        max_T,_ = torch.max(T, -1)
        _, top_ids = torch.topk(max_T, math.floor(T.shape[1]*0.3),-1)
        mask_T = torch.zeros_like(max_T)
        mask_T = mask_T.scatter(-1, top_ids, 1)
        indexs = torch.where(mask_T!=0)
        #print(indexs, top_ids[0])

        values, indx = torch.topk(T, 3, -1)
        T = T *scale
        T[T==0] = float('-inf')
        values, indx = torch.topk(T, 3, -1)
        mask_arg = torch.zeros_like(T)
        mask_arg = mask_arg.scatter(-1, indx, 1)
        T = T.masked_fill(mask_arg==0, 0)
        T[T==0] = float('-inf')
        labels = T.softmax(-1) 
        #print(distance, labels[0].sum(), torch.sum(labels, 2), labels.max(), labels[0][0], T[0][0])
        #print('labels', labels.max())

        if self.open_voc:
            '''
            mlm_output = self.text_encoder(text.input_ids, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,  
                                        alpha = alpha
                                        ).logits
            '''
            hidden_states = text_embeds
            if self.add_text_head:
                attention_mask = self.text_encoder.bert.get_extended_attention_mask(
                    text.attention_mask,
                    text.attention_mask.shape,
                    image.device,
                    False
                )
                for layer in self.ctext_head:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask,
                    )
                    hidden_states = layer_outputs[0]
            mlm_output = self.text_encoder.cls(hidden_states)
            voc_max = F.softmax(mlm_output[:, 1:, :], dim=2)
            #voc_max = self.text_encoder.cls(text_embeds[:, 1:, :]).softmax(dim=-1)
            #print("voc", torch.topk(voc_max,10,dim=-1).indices[0,:10,:], torch.topk(voc_max,10,dim=-1).values[0,:10,:])
            labels_voc = torch.matmul(labels, voc_max)
            #print(torch.topk(labels_voc[indexs[0], indexs[1],:],10,dim=-1).values[:10,:], torch.topk(labels_voc[indexs[0], indexs[1],:],10,dim=-1).indices[:10,:])
        else:
            text_ids = text.input_ids[:,1:]
            text_ids = text_ids.repeat(1,T.shape[1]).reshape(text_ids.shape[0], T.shape[1], -1)
            ids = torch.where(labels!=0)
            new_ids = text_ids[ids[0],ids[1],ids[2]]
            labels_voc = torch.zeros(labels.shape[0], labels.shape[1], self.text_encoder.config.vocab_size).to(image.device)
            labels_voc[ids[0], ids[1], new_ids.long()] = labels[ids[0],ids[1],ids[2]]

        if not self.img_pooling and self.model_type!='xvlm':
            labels_voc = labels_voc[indexs[0], indexs[1],:]
        labels_voc = labels_voc.view(-1, labels_voc.shape[-1])

        if self.img_pooling:
            cls_tmp = image_embeds_mask[:,0:1,:]
            image_embeds_mask = self.patch_pooling(image_embeds_mask[:,1:,:])
            image_embeds_mask = torch.cat([cls_tmp,image_embeds_mask], 1)
        v2t_embed = self.v2t_proj(image_embeds_mask)

        if self.add_img_head:
            x = v2t_embed
            hidden_states = x
            attention_mask = self.text_encoder.bert.get_extended_attention_mask(
                image_atts,
                image_atts.shape,
                image.device,
                False
            )
            for layer in self.cimg_head:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                )
                hidden_states = layer_outputs[0]
            v2t_embed = hidden_states

        prediction_scores_all = self.img_head(v2t_embed)
        prediction_scores = prediction_scores_all[:,1:,:]
        if not self.img_pooling and self.model_type!='xvlm':
            prediction_scores = prediction_scores[indexs[0], indexs[1],:]
        prediction_scores = prediction_scores.reshape(-1, prediction_scores.shape[-1])
        #log_likelihood = -F.log_softmax(prediction_scores, dim=1)
        #loss_p2w = torch.sum(torch.mul(log_likelihood, labels_voc.detach())) / prediction_scores.shape[0]
        loss_p2w = F.kl_div(prediction_scores.softmax(dim=-1).log(), labels_voc.detach(), reduction='sum')/prediction_scores.shape[0]
        return loss_p2w




    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds, image_atts = self.get_image_embeds(image)
        text_embeds = self.get_text_embeds(text)
        
        image_feat, text_feat = self.get_features(image_embeds, text_embeds, text.attention_mask)

        loss_ita = self.get_contrastive_loss(image_feat, text_feat)


        #==========================p2w=======================================
        loss_p2w = self.get_p2w_loss(image, text, image_atts, image_embeds, text_embeds, alpha)
        

        ###=================================###
        # forward the positve image-text pair
        if self.cl_bw:
            img_lexicon = self.get_img_bw(image_embeds)
            text_lexicon = self.get_text_bw(text_embeds, text.attention_mask)
            image_embeds = torch.cat([img_lexicon, image_embeds[:,1:,:]],1)
            text_embeds = torch.cat([text_lexicon, text_embeds[:,1:,:]],1)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text.attention_mask, text_feat)
        
        ##================= MLM ========================##                                         
        loss_mlm = self.get_mlm_loss(text, image, image_embeds, image_atts, alpha)

        if self.mid_mlm:
            loss_mlm, kl_loss = loss_mlm

        if self.mid_mlm:
            if self.cl_cls or self.cl_bw:
                return loss_mlm, loss_ita, loss_itm, loss_p2w, kl_loss#, loss_bw
            else:
                return loss_mlm, loss_ita, loss_itm, loss_p2w, kl_loss

        return loss_mlm, loss_ita, loss_itm, loss_p2w
 
        
 
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    
 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            
 
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
 
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids



    # jinyu: patch pooling of image patches to reduce computation and enlarge receptive field
    def patch_pooling(self, x):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, c1*c1, dim)
        return x
        
    def p2t(self, image, text):
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_embeds_head = image_embeds
        if self.add_img_head:
            x = image_embeds_head
            for i,blk in enumerate(self.cimg_head):
                x = blk(x)
            x = self.visual_encoder.norm(x)
            image_embeds_head = x
        image_emb = self.v2t_proj(image_embeds_head[:,1:,:])
        if self.img_pooling:
            image_emb = self.patch_pooling(image_emb)
        prediction_scores = self.img_head(image_emb)
        log_likelihood = F.softmax(prediction_scores, dim=2)
        #print(log_likelihood)
        

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', output_hidden_states=True)            
        text_embeds = text_output.last_hidden_state
        txt_pad = ~text.attention_mask.bool()
        img_pad = ~image_atts.bool()
        img_temb = self.vision_proj(image_embeds[:,1:,:])
        if self.img_pooling:
            img_temb = self.patch_pooling(img_temb)
            img_pad = img_pad[:,0:img_temb.shape[1]+1]
        text_temb = self.text_proj(text_embeds[:,1:,:])
        distance, T = optimal_transport_dist(text_temb, img_temb, txt_pad[:,1:], img_pad[:,1:])
        '''
        tsum = 1/torch.sum(T, 2, keepdim=True)
        if torch.isinf(tsum).any().item():
            tsum = torch.where(torch.isinf(tsum), torch.full_like(tsum, 1), tsum)
        labels = T*(tsum)
        text_ids = text.input_ids[:,1:]
        text_ids = text_ids.repeat(1,T.shape[1]).reshape(text_ids.shape[0], T.shape[1], -1)
        ids = torch.where(labels!=0)
        new_ids = text_ids[ids[0],ids[1],ids[2]]
        labels_voc = torch.zeros(labels.shape[0], labels.shape[1], self.text_encoder.config.vocab_size).to(image.device)
        labels_voc[ids[0], ids[1], new_ids.long()] = labels[ids[0],ids[1],ids[2]]
        '''
        _, max_T= torch.topk(T, 3,-1)

        #patch_dis = torch.softmax(self.mlp(image_emb[:,1:,:]), 1)
        #patch_dis = torch.sum(T, dim=2, keepdim=True)
        #print(patch_dis)
        
        return log_likelihood, max_T
 
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
 
    output = torch.cat(tensors_gather, dim=0)
    return output

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist, cosine_sim


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace

@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost, sim = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    
    m = cost.matmul(T.detach())
    distance = trace(m)
    return distance, T*sim.permute(0,2,1)