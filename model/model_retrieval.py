'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
 
from functools import partial
from model.vit import VisionTransformer, interpolate_pos_embed
from TCL.models.xbert import BertConfig, BertForMaskedLM, BertOnlyMLMHead, BertLayer
from model.swin_transformer import interpolate_relative_pos_embed, SwinTransformer
from model.model_pretrain import ALBEF
 
import torch
import torch.nn.functional as F
from torch import nn
 
import numpy as np
import random
from utils import read_json
 
class ALBEF_Retrieval(ALBEF):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 ):
        config['mlm_probability'] = 0
        self.lexicon_loss = config['lexicon_loss']
        super().__init__(text_encoder, tokenizer, config, use_contrastive_loss = True, use_matching_loss = True, use_mlm_loss = self.lexicon_loss, use_p2w_loss = False, use_distill_loss = False, teacher_model = None, init_model = False)
        self.align_loss = config['align_loss']
        self.lexicon_loss = config['lexicon_loss']
        if self.lexicon_loss:
            self.img_head = self.text_encoder.cls
            self.mlm_head = self.text_encoder.cls
        
    def get_image_bw_(self, image_feat):
        token_embeddings = self.img_head(image_feat)[:,1:,:]
        image_embed = torch.log(1+torch.relu(torch.max(token_embeddings, dim=1).values))
        return image_embed

    def get_text_bw_(self, text_feat, text, device):
        hidden_states = text_feat
        '''
        if self.add_text_head:
            attention_mask = self.text_encoder.bert.get_extended_attention_mask(
                text.attention_mask,
                text.attention_mask.shape,
                device,
                False
            )
            for layer in self.ctext_head:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                )
                hidden_states = layer_outputs[0]
        '''
        pre_scores_all = self.mlm_head(hidden_states)
        token_embeddings = pre_scores_all[:,1:,:]
        mask_att = text.attention_mask[:,1:]
        token_embeddings.masked_fill_((mask_att == 0).unsqueeze(-1), 0.)  # apply mask
        text_bw = torch.log(1+torch.relu(torch.max(token_embeddings, dim=1).values))
        return text_bw

    def get_lexicon_loss(self, text, image, image_embeds, text_embeds, idx):
        out = {}
        prediction_scores_all = self.img_head(image_embeds)
        hidden_states = text_embeds
        pre_scores_all = self.mlm_head(hidden_states)

        loss_bw = 0
        token_embeddings = prediction_scores_all[:,1:,:]
        #mask_att = image_atts[:, 1:]
        #token_embeddings.masked_fill_((mask_att == 0).unsqueeze(-1), 0.)  # apply mask

        # fix 17
        torch.relu_(token_embeddings)
        with torch.no_grad():
            max_indices = torch.argmax(1. + token_embeddings, dim=1).unsqueeze(1).detach()  # [bs,1,V]
        sentence_embedding = torch.gather(token_embeddings, dim=1, index=max_indices).squeeze(1)
        img_bw = torch.log(1. + sentence_embedding)

        out['img'] = img_bw
        if self.cl_cls:
            text_bw = F.normalize(self.text_proj(hidden_states[:,0,:]),dim=-1)
            out['text'] = text_bw
            loss_bw += self.get_contrastive_loss(img_bw, text_bw, idx=idx)

        if self.cl_bw:
            token_embeddings = pre_scores_all[:,1:,:]
            mask_att = text.attention_mask[:,1:]
            token_embeddings.masked_fill_((mask_att == 0).unsqueeze(-1), 0.)  # apply mask
            #text_bw = torch.log(1+torch.relu(torch.max(token_embeddings, dim=1).values))
            torch.relu_(token_embeddings)
            with torch.no_grad():
                max_indices = torch.argmax(1. + token_embeddings, dim=1).unsqueeze(1).detach()  # [bs,1,V]
            sentence_embedding = torch.gather(token_embeddings, dim=1, index=max_indices).squeeze(1)
            text_bw = torch.log(1. + sentence_embedding)

            out['text'] = text_bw
            loss_bw += self.get_contrastive_loss(img_bw, text_bw, idx=idx)
        loss_lc = loss_bw
        return loss_lc, out

 
    def forward(self, image, text, alpha, idx):
        
        image_embeds, image_atts = self.get_image_embeds(image)
        text_embeds = self.get_text_embeds(text)
        
        image_feat, text_feat = self.get_features(image_embeds, text_embeds, text.attention_mask)

        if self.align_loss:
            loss_ita = self.get_contrastive_loss(image_feat, text_feat, idx=idx)

        #==========================lexicon contrastive=======================================
        if self.lexicon_loss:
           loss_lc, out = self.get_lexicon_loss(text, image, image_embeds, text_embeds, idx)
        
        if self.cl_bw:
            img_lexicon = self.get_img_bw(image_embeds)
            text_lexicon = self.get_text_bw(text_embeds, text.attention_mask)
            image_embeds = torch.cat([img_lexicon, image_embeds[:,1:,:]],1)
            text_embeds = torch.cat([text_lexicon, text_embeds[:,1:,:]],1)
        #==============================================#
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text.attention_mask, text_feat, idx=idx)

        if self.lexicon_loss:
            if self.align_loss:
                return loss_ita, loss_itm, loss_lc, out
            else:
                return loss_itm, loss_lc, out
        return loss_ita, loss_itm


    # jinyu: patch pooling of image patches to reduce computation and enlarge receptive field
    def patch_pooling(self, x):
        pooled_patch_length = 16
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = b1 // int(np.sqrt(pooled_patch_length))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, pooled_patch_length, dim)
        return x
        



