from functools import partial
from TCL.models.vit import VisionTransformer
#from TCL.models.xbert import BertConfig, BertModel, BertForMultipleChoice
from transformers import BertTokenizer, BertForMultipleChoice

import torch
from torch import nn
import torch.nn.functional as F

class ALBEFBase(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer  

        self.text_encoder = BertForMultipleChoice.from_pretrained(config['model_name'])   
            
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        num_choices = text['input_ids'].shape[1]
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
           
        if train:
            output = self.text_encoder(input_ids, 
                                       attention_mask = attention_mask,         
                                       return_dict = True,
                                       labels = targets
                                      )     
            loss = output.loss            
            return loss 
            
        else:
            output = self.text_encoder(input_ids, 
                                       attention_mask = attention_mask,         
                                       return_dict = True
                                      )                                
            return output.logits

