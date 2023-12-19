import numpy as np
import pandas as pd
import time
import datetime
import gc
import random
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split

from transformers import TextClassificationPipeline
import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup


class pred:
    def __init__(self,text):
        self.text = text

    def pred_class(self):

        tokenizer = BertTokenizer.from_pretrained("E:/Assignment/bert/result/content/results/tokenizer", local_files_only=True)
        max_len=512
        device="cpu"
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path="E:/Assignment/bert/result/content/results/model", num_labels = 5,output_attentions = False, output_hidden_states = False,)
        model = model.to(device)
        encoded_dict = tokenizer.encode_plus(
                        self.text,                     
                        add_special_tokens = True, 
                        max_length = max_len,           
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                   )
        
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
        pred=pipe(self.text)

        idtolabel=max(pred[0], key=lambda x:x['score'])["label"]

        dict_decode={"LABEL_0":"digestive system diseases","LABEL_1":"cardiovascular diseases","LABEL_2":"neoplasms","LABEL_3":"nervous system diseases","LABEL_4":"general pathological conditions"}

        return dict_decode[idtolabel]


