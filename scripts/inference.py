import os
import torch
from transformers import AutoModelForTokenClassification

from NER_tokenizer_BIO import NER_tokenizer_BIO


def model_fn(model_dir):
    tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = NER_tokenizer_BIO.from_pretrained(tokenizer_name, num_entity_type=8)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    if int(os.environ['SM_NUM_GPUS']) > 0:
        model = model.cuda() # GPUで推論する場合
    return model, tokenizer
    
def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    sentences = data.pop('inputs', data)
    encoding, spans = tokenizer.encode_plus_untagged(
        sentences,
        return_tensors='pt'
    )
    if int(os.environ['SM_NUM_GPUS']) > 0:
        encoding = { k: v.cuda() for k, v in encoding.items() }  # GPUで推論する場合
    
    with torch.no_grad():
        output = model(**encoding)
        scores = output.logits
        scores = scores[0].cpu().numpy().tolist()
        
    # 分類スコアを固有表現に変換する
    entities_predicted = tokenizer.converdft_bert_output_to_entities(
        sentences, scores, spans
    )
    
    return entities_predicted
