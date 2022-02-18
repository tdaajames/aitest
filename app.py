# -*- coding: UTF-8 -*-
from flask import Flask, request, jsonify
import requests
import json
from transformers import AutoTokenizer
import yaml
import tensorflow as tf
import annoy
import pickle

app = Flask(__name__)


def embed(input):
    payload = {
        "inputs": {"input_ids": input['input_ids'].tolist(), "attention_mask": input['attention_mask'].tolist()}
    }

    url = "http://" + tf_serving_url + "/v1/models/paraphrase-multilingual-mpnet-base-v2:predict"
    r = requests.post(url, json=payload)
    pred = json.loads(r.content.decode('utf-8'))
    return pred['outputs']


def mean_pooling(model_output, attention_mask):
    result_tmp = model_output['output_0']

    mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
    masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
            tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
    input_mask = tf.cast(attention_mask, tf.float32)
    pooled = masked_reduce_mean(result_tmp, input_mask)
    return pooled


def get_embedding(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    model_output = embed(encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding_list = sentence_embeddings.numpy().tolist()

    return embedding_list


def find_similar_items(embedding, num_matches=5):
    '''Finds similar items to a given embedding in the ANN index'''
    ids, distance = index.get_nns_by_vector(
        embedding, num_matches, search_k=-1, include_distances=True)
    items = [mapping[i] for i in ids]
    return items, distance


# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/qa/predict', methods=['POST'])
def qa_predict():
    # Decoding and pre-processing base64 image
    sentences = request.json['sentences']
    embedding_list = get_embedding(sentences)
    reply_list = []
    for i in range(len(embedding_list)):
        embedding = embedding_list[i]
        items, distance = find_similar_items(embedding, 1)
        sim = json.loads(items[0])
        r = {'sentence': sentences[i], 'question': sim["question"], 'label': sim["label"], 'answer': sim["answer"],
             'distance': distance}
        reply_list.append(r)

    # Returning JSON response to the frontend
    return jsonify(reply_list)


# 获取配置信息
yamlConfig = 'config.yaml'

with open(yamlConfig, 'rb') as f:
    # yaml文件通过---分节，多个节组合成一个列表
    config = yaml.load(f)
    print(type(config))
    index_path = config['index_path']
    print(index_path)
    tf_serving_url = config['tf_serving_url']
    print(tf_serving_url)
    tokenizer_path = config['tokenizer_path']
    print(tokenizer_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
embed_fun_handler = get_embedding

# 加载索引数据
projected_dim = 768
embedding_dimension = projected_dim
index_filename = index_path

index = annoy.AnnoyIndex(embedding_dimension)
index.load(index_filename, prefault=True)
print('Annoy index is loaded.')
with open(index_filename + '.mapping', 'rb') as handle:
    mapping = pickle.load(handle)
print('Mapping file is loaded.')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
