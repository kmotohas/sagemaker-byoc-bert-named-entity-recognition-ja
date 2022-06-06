import os
import sys
import json
import random
import unicodedata
from glob import glob
from logging import getLogger, DEBUG, StreamHandler

import datasets

from NER_tokenizer_BIO import NER_tokenizer_BIO

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler(sys.stdout))


def create_dataset(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力できる形に整形。
    """
    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels= []
    
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )
        input_ids.append(encoding['input_ids'])
        token_type_ids.append(encoding['token_type_ids'])
        attention_mask.append(encoding['attention_mask'])
        labels.append(encoding['labels'])
    
    d = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask, 
        "labels": labels
    }
        
    return d
    
    
        
        
def preprocess():
    raw_data_dir = '/opt/ml/processing/input/raw'
    
    input_files = glob(f'{raw_data_dir}/*')
    logger.debug(input_files)
    
    dataset = []
    for input_file in input_files:
        dataset += json.load(open(input_file, 'r'))
        
    # 固有表現のタイプとIDを対応付る辞書 
    type_id_dict = {
        "人名": 1,
        "法人名": 2,
        "政治的組織名": 3,
        "その他の組織名": 4,
        "地名": 5,
        "施設名": 6,
        "製品名": 7,
        "イベント名": 8,
    }
    
    # カテゴリーをラベルに変更、文字列の正規化する。
    for sample in dataset:
        sample['text'] = unicodedata.normalize('NFKC', sample['text'])
        for e in sample["entities"]:
            e['type_id'] = type_id_dict[e['type']]
            del e['type']
        
    # データセットの分割
    random.seed(42)
    random.shuffle(dataset)
    n = len(dataset)
    n_train = int(n*0.6)
    n_val = int(n*0.2)
    dataset_train = dataset[:n_train]
    dataset_val = dataset[n_train:n_train+n_val]
    dataset_test = dataset[n_train+n_val:]
    # バッチ変換の入力のためにJSONL形式に変換
    # https://huggingface.co/docs/sagemaker/reference#inference-toolkit-api
    dataset_jsonl = [{'inputs': d['text']} for d in dataset_test]
    test_dir = 'opt/ml/processing/output/data/test'
    os.makedirs(test_dir, exist_ok=True)
    with open(f'{test_dir}/input.jsonl', 'w') as jsonl:
        for data in dataset_jsonl:
            json.dump(data, jsonl, ensure_ascii=False)
            jsonl.write('\n')
    
    tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    # トークナイザのロード
    # 固有表現のカテゴリーの数`num_entity_type`を入力に入れる必要がある。
    tokenizer = NER_tokenizer_BIO.from_pretrained(
        tokenizer_name,
        num_entity_type=8
    )
    
    # データセットの作成
    max_length = 128
    
    dataset_train = create_dataset(
        tokenizer, 
        dataset_train, 
        max_length
    )
    
    dataset_val = create_dataset(
        tokenizer, 
        dataset_val, 
        max_length
    )
    
    #dataset_test = create_dataset(
    #    tokenizer, 
    #    dataset_test, 
    #    max_length
    #)
    

    dataset_train = datasets.Dataset.from_dict(dataset_train)
    dataset_val = datasets.Dataset.from_dict(dataset_val)
    #dataset_test = datasets.Dataset.from_dict(dataset_test)
    
    # set format for pytorch
    dataset_train.set_format(
        'torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    )
    dataset_val.set_format(
        'torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    )
    #dataset_test.set_format(
    #    'torch',
    #    columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    #)
    
    dataset_train.save_to_disk('/opt/ml/processing/output/data/train')
    dataset_val.save_to_disk('/opt/ml/processing/output/data/validation')
    #dataset_test.save_to_disk('/opt/ml/processing/output/data/test')
    
if __name__ == '__main__':
    preprocess()
    