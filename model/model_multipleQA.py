from functools import partial
from TCL.models.vit import VisionTransformer
from TCL.models.xbert import BertConfig, BertModel, BertForMultipleChoice

import torch
from torch import nn
import torch.nn.functional as F

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        self.image = config['image']
        
        if self.image:
            self.visual_encoder = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=True)   
        self.classifier = nn.Sequential(nn.Dropout(self.text_encoder.config.hidden_dropout_prob), nn.Linear(self.text_encoder.config.hidden_size, 1))     

        if self.distill:
            if self.image:
                self.visual_encoder_m = VisionTransformer(
                    img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                    mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=True)
            self.classifier_m = nn.Sequential(nn.Dropout(self.text_encoder.config.hidden_dropout_prob), nn.Linear(self.text_encoder.config.hidden_size, 1))     
            if self.image:
                self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                    [self.text_encoder,self.text_encoder_m],
                                    [self.classifier,self.classifier_m],
                                    ]
            else:
                self.model_pairs = [[self.text_encoder,self.text_encoder_m],
                                    [self.classifier,self.classifier_m],
                                    ]
            self.copy_params()        
            self.momentum = 0.995
            
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        num_choices = text['input_ids'].shape[1]
        input_ids = text['input_ids']
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = text['attention_mask']
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        if self.image:
            image_embeds = self.visual_encoder(image) 
            image_embeds = torch.cat([torch.cat([image_embeds[i:i+1,:,:]]*num_choices) for i in range(image_embeds.shape[0])])
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        else:
            image_embeds = torch.ones((input_ids.shape[0],1,self.text_encoder.config.hidden_size)).to(text['input_ids'].device)
            image_atts = torch.zeros(image_embeds.size()[:-1],dtype=torch.long).to(text['input_ids'].device)    
           
        if train:
            output = self.text_encoder(input_ids, 
                                       attention_mask = attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )    
            logits = self.classifier(output.pooler_output)   
            reshaped_logits = logits.view(-1, num_choices) 
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    if self.image:
                        image_embeds_m = self.visual_encoder_m(image) 
                        image_embeds_m = torch.cat([torch.cat([image_embeds_m[i:i+1,:,:]]*num_choices) for i in range(image_embeds_m.shape[0])])
                    else:
                        image_embeds_m = torch.ones((input_ids.shape[0],1,self.text_encoder.config.hidden_size)).to(text['input_ids'].device)
                    output_m = self.text_encoder_m(input_ids, 
                                               attention_mask = attention_mask, 
                                               encoder_hidden_states = image_embeds_m,
                                               encoder_attention_mask = image_atts,        
                                               return_dict = True
                                              )  
                    logits_m = self.classifier_m(output_m.pooler_output)   
                    reshaped_logits_m = logits_m.view(-1, num_choices)          

                loss = (1-alpha)*F.cross_entropy(reshaped_logits, targets) - alpha*torch.sum(
                    F.log_softmax(reshaped_logits, dim=1)*F.softmax(reshaped_logits_m, dim=1),dim=1).mean()
            else:
                loss = F.cross_entropy(reshaped_logits, targets)             
            return loss 
            
        else:
            output = self.text_encoder(input_ids, 
                                       attention_mask = attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )  
            logits = self.classifier(output.pooler_output)   
            reshaped_logits = logits.view(-1, num_choices)                              
            return reshaped_logits
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

