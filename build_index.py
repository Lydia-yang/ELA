import torch
import clip
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.load import load_dataset
from dataset.dataset_base import baseDataset
from dataset.dataset_comQA import comQADataset
from dataset.dataset_winGrande import winGDataset
from dataset.dataset_test import testDataset
from dataset.utils import pre_question
import os
import faiss
import json
import utils
import argparse

from get_features import load_model

def build_index(feature_path, save_path, device):
    index = None
    for file in os.listdir(feature_path):
        path = os.path.join(feature_path, file)
        data = np.load(path)
        ids = data['id']
        features = data['features']
        ids = np.array(ids).astype('int64')
        features = np.array(features).astype('float32')
        #print(ids.shape, features.shape)
        if index is None:
            index = faiss.IndexFlatL2(features.shape[1])
            index = faiss.IndexIDMap(index)
            if device=='cuda':
                ngpus = faiss.get_num_gpus()
                print("number of GPUs:", ngpus)
                index = faiss.index_cpu_to_all_gpus(index)
        index.add_with_ids(features, ids)
    if device=='cuda':
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, save_path)

def search_img_example(index, query, k, dic, pair_dic=None):
    features = np.array(query).astype('float32')
    D, I = index.search(features, k)
    img_path = get_value(I, dic, pair_dic)
    return img_path

def search_img_glue(sentence, faiss_path, save_path, dataset, model, tokenizer, device, dic, pair_dic=None):
    sentence1_key, sentence2_key = sentence
    max_words = 30
    k = 50
    k = k//2 if sentence2_key is not None else k
    index = faiss.read_index(faiss_path)
    def preprocess_function(examples):
        with torch.no_grad():
            text = examples[sentence1_key]
            text = [pre_question(tmp, max_words) for tmp in text]
            query = tokenizer(text, truncate=True)
            features = model(query.to(device))
            features = features.cpu().numpy()
            img = search_img_example(index, features, k, dic, pair_dic)
            if sentence2_key is not None:
                text = examples[sentence2_key]
                text = [pre_question(tmp, max_words) for tmp in text]
                query = tokenizer(text, truncate=True)
                features = model(query.to(device))
                features = features.cpu().numpy()
                img2 = search_img_example(index, features, k, dic, pair_dic)
                for i1, i2 in zip(img, img2):
                    i1.extend(i2)
            #print(img[0])
            result = {'img': img}
        return result
    dataset = dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
            desc="Running img search on dataset",
        )
    #dataset.to_csv(save_path)
    dataset.to_json(save_path)

def search_img(faiss_path, save_path, dataset, model, tokenizer, device, dic, pair_dic=None):
    index = faiss.read_index(faiss_path)
    bs = 512
    k = 50
    querys = []
    imgs = []
    with torch.no_grad():
        for text,_,_ in tqdm(DataLoader(dataset, batch_size=bs)):
            querys.extend(text)
            query = tokenizer(text)
            features = model(query.to(device))
            features = features.cpu().numpy()
            features = np.array(features).astype('float32')
            D, I = index.search(features, k)
            img_path = get_value(I, dic, pair_dic)
            imgs.extend(img_path)
            #break
    result = {'querys':querys, 'images':imgs}
    with open(save_path, 'w') as file_obj:
        json.dump(result, file_obj)

def get_value(I, dic, pair_dic):
    ans = []
    for i in range(len(I)):
        tmp = []
        for j in range(len(I[0])):
            id = I[i][j]
            key = str(id)
            if pair_dic:
                 key = pair_dic[key][0]
                 key = str(key)
            value = dic[key]
            tmp.append(value)
        ans.append(tmp)
    return ans

def print_result(path, count):
    res = json.load(open(path, 'r'))
    text = res['querys']
    img = res['images']
    for i in range(count):
        line = 'text:{0}, img:{1}'.format(text[i], img[i])
        print(line)

def get_dic(path, name):
    dic = {}
    for file in os.listdir(path):
        m = file.split('.')[0].split('_')[-1]
        if m!=name:
            continue
        file_path = os.path.join(path, file)
        tmp_dic = json.load(open(file_path, 'r'))
        dic.update(tmp_dic)
    return dic

def build_all_index():
    #build index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modality = ['img', 'txt']
    save_path = 'data'
    for m in modality:
        feature_path = os.path.join(save_path, 'features', m)
        index_path = os.path.join(save_path, 'index')
        if not os.path.exists(index_path):
            os.mkdir(index_path)
        index_path = os.path.join(index_path, m+'.idx')
        build_index(feature_path, index_path, device)


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    clip_name = 'ViT-B/32'
    data_path = 'data/ids'
    img_dic = get_dic(data_path, 'img')
    pair_dic = get_dic(data_path, 'T2I')
    #dataset = baseDataset('data/ids/COCO_txt.json')
    model, preprocess = load_model(device, clip_name, False)
    name = 'glue'
    if name=='comQA':
        dataset = [comQADataset('data/comQA/dev_rand_split.jsonl')]
        save_txt = ['data/result/ans_comQA_txt_val.json']
        save_img = ['data/result/ans_comQA_img_val.json']
    elif name=='winG':
        dataset = [winGDataset('data/winogrande/dev.jsonl')]
        save_txt = ['data/result/ans_winG_txt_val.json']
        save_img = ['data/result/ans_winG_img_val.json']
    elif name=='glue':
        task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        }
        datas = ['rte', 'mrpc', 'stsb']
        dataset = []
        save_txt = []
        sentence = []
        for tmp in datas:
            raw_datasets = load_dataset("glue", tmp)
            dataset.append(raw_datasets["validation"])
            dataset.append(raw_datasets["train"])
            save_txt.append('data/glue/{0}/val_img.json'.format(tmp))
            save_txt.append('data/glue/{0}/train_img.json'.format(tmp))
            sentence.append(task_to_keys[tmp])
            sentence.append(task_to_keys[tmp])
            #break

    
    for i, (d, p) in enumerate(zip(dataset, save_txt)):
        if name=='glue':
            search_img_glue(sentence[i], 'data/index/txt.idx', p, d, model.encode_text, clip.tokenize,
               device, img_dic, pair_dic)
        else:
            search_img('data/index/txt.idx', p, d, model.encode_text, clip.tokenize,
               device, img_dic, pair_dic)
    
   # search_img('data/index/img.idx', save_img, dataset, model.encode_text, clip.tokenize,
    #           device, img_dic)
    #print_result('data/result/ans_comQA_img.json', 37)
    #build_all_index()
    