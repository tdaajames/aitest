import os
import json
import pickle
from collections import namedtuple
from datetime import datetime
import apache_beam as beam
from apache_beam.transforms import util
import tensorflow as tf
import tensorflow_hub as hub
import annoy
import tensorflow_text as text  # A dependency of the preprocessing model
from transformers import AutoTokenizer
import yaml


def embed(input):
    return model(**input)


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


def generate_embeddings(text_list, random_projection_matrix=None):
    # Beam will run this function in different processes that need to
    label_list = []
    question_list = []
    answer_list = []
    for i in range(len(text_list)):
        text = text_list[i]
        data_list = text.split(',', 2)
        label = data_list[0]
        question = data_list[1]
        answer = data_list[2]
        label_list.append(label)
        question_list.append(question)
        answer_list.append(answer)
    embedding_list = embed_fun_handler(question_list)
    if random_projection_matrix is not None:
        embedding_list = embedding_list.dot(random_projection_matrix)
    return label_list, question_list, answer_list, embedding_list


def to_tf_example(entries):
    examples = []

    label_list, question_list, answer_list, embedding_list = entries
    for i in range(len(label_list)):
        label = label_list[i]
        question = question_list[i]
        answer = answer_list[i]
        embedding = embedding_list[i]

        features = {
            'label': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
            'question': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[question.encode('utf-8')])),
            'answer': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[answer.encode('utf-8')])),
            'embedding': tf.train.Feature(
                float_list=tf.train.FloatList(value=embedding))
        }

        example = tf.train.Example(
            features=tf.train.Features(
                feature=features)).SerializeToString(deterministic=True)

        examples.append(example)

    return examples


def run_hub2emb(args):
    '''Runs the embedding generation pipeline'''

    options = beam.options.pipeline_options.PipelineOptions(**args)
    args = namedtuple("options", args.keys())(*args.values())

    with beam.Pipeline(args.runner, options=options) as pipeline:
        (
                pipeline
                | 'Read sentences from files' >> beam.io.ReadFromText(file_pattern=args.data_dir)
                | 'Batch elements' >> util.BatchElements(min_batch_size=args.batch_size, max_batch_size=args.batch_size)
                | 'Generate embeddings' >> beam.Map(generate_embeddings, args.random_projection_matrix)
                | 'Encode to tf example' >> beam.FlatMap(to_tf_example)
                | 'Write to TFRecords files' >> beam.io.WriteToTFRecord(
            file_path_prefix='{}/emb'.format(args.output_dir), file_name_suffix='.tfrecords', num_shards=1)
        )


def _parse_example(example):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example, feature_description)


def build_index(embedding_files_pattern, index_filename, vector_length,
                metric='angular', num_trees=100):
    '''Builds an ANNOY index'''

    annoy_index = annoy.AnnoyIndex(vector_length, metric=metric)
    # Mapping between the item and its identifier in the index
    mapping = {}

    embed_files = tf.io.gfile.glob(embedding_files_pattern)
    num_files = len(embed_files)
    print('Found {} embedding file(s).'.format(num_files))

    item_counter = 0
    for i, embed_file in enumerate(embed_files):
        print('Loading embeddings in file {} of {}...'.format(i + 1, num_files))
        dataset = tf.data.TFRecordDataset(embed_file)
        for record in dataset.map(_parse_example):
            label = record['label'].numpy().decode("utf-8")
            question = record['question'].numpy().decode("utf-8")
            answer = record['answer'].numpy().decode("utf-8")
            embedding = record['embedding'].numpy()
            data = {'label': label, 'question': question, 'answer': answer}
            json_data = json.dumps(data)

            mapping[item_counter] = json_data
            annoy_index.add_item(item_counter, embedding)
            item_counter += 1
            if item_counter % 100000 == 0:
                print('{} items loaded to the index'.format(item_counter))

    print('A total of {} items added to the index'.format(item_counter))

    print('Building the index with {} trees...'.format(num_trees))
    annoy_index.build(n_trees=num_trees)
    print('Index is successfully built.')

    print('Saving index to disk...')
    annoy_index.save(index_filename)
    print('Index is saved to disk.')
    print("Index file size: {} GB".format(
        round(os.path.getsize(index_filename) / float(1024 ** 3), 2)))
    annoy_index.unload()

    print('Saving mapping to disk...')
    with open(index_filename + '.mapping', 'wb') as handle:
        pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Mapping is saved to disk.')
    print("Mapping file size: {} MB".format(
        round(os.path.getsize(index_filename + '.mapping') / float(1024 ** 2), 2)))


# 获取配置信息
yamlConfig = 'config.yaml'

with open(yamlConfig, 'rb') as f:
    # yaml文件通过---分节，多个节组合成一个列表
    config = yaml.load(f)
    print(type(config))
    pb_path = config['pb_path']
    print(pb_path)
    data_path = config['data_path']
    print(data_path)
    base_data_path = config['base_data_path']
    print(base_data_path)
    index_path = config['index_path']
    print(index_path)
    tokenizer_path = config['tokenizer_path']
    print(tokenizer_path)

module_url = pb_path
model = hub.load(module_url)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
embed_fun_handler = get_embedding

projected_dim = 768  # @param {type:"number"}

output_dir = data_path
random_projection_matrix = None

args = {
    'job_name': 'hub2emb-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S')),
    'runner': 'DirectRunner',
    'batch_size': 1024,
    'data_dir': base_data_path,
    'output_dir': output_dir,
    'random_projection_matrix': None,
}

print("Pipeline args are set.")

embed_file = os.path.join(output_dir, 'emb-00000-of-00001.tfrecords')
if os.path.exists(embed_file):
    os.remove(embed_file)

print("Running pipeline...")
run_hub2emb(args)
print("Pipeline is done.")

sample = 5

# Create a description of the features.
feature_description = {
    'label': tf.io.FixedLenFeature([], tf.string),
    'question': tf.io.FixedLenFeature([], tf.string),
    'answer': tf.io.FixedLenFeature([], tf.string),
    'embedding': tf.io.FixedLenFeature([projected_dim], tf.float32)
}

dataset = tf.data.TFRecordDataset(embed_file)
for record in dataset.take(sample).map(_parse_example):
    print("{}: {}".format(record['question'].numpy().decode('utf-8'), record['embedding'].numpy()[:10]))

embedding_files = "{}/emb-*.tfrecords".format(output_dir)
embedding_dimension = projected_dim
index_filename = index_path

build_index(embedding_files, index_filename, embedding_dimension)
