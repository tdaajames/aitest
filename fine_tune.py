import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.tokenization
from collections import namedtuple
from datetime import datetime
import apache_beam as beam
from apache_beam.transforms import util
import tensorflow_text as text  # A dependency of the preprocessing model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 指定用CPU
hub_url_bert = "D:\\workspace\\20211029_AI\\bert_zh_L-12_H-768_A-12_4"

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(hub_url_bert, "assets\\vocab.txt"),
    do_lower_case=True)

def create_dataset(text_list):
    for i in range(len(text_list)):
        text = text_list[i]
        data_list = text.split('\t', 2)
        label = int(data_list[0])
        sentence1 = data_list[1]
        sentence2 = data_list[2]
        label_list.append(label)
        sentence1_list.append(sentence1)
        sentence2_list.append(sentence2)

def get_dataset(args):
    '''Runs the embedding generation pipeline'''

    options = beam.options.pipeline_options.PipelineOptions(**args)
    args = namedtuple("options", args.keys())(*args.values())

    with beam.Pipeline(args.runner, options=options) as pipeline:
        (
                pipeline
                | 'Read sentences from files' >> beam.io.ReadFromText(file_pattern=args.data_dir)
                | 'Batch elements' >> util.BatchElements(min_batch_size=args.batch_size, max_batch_size=args.batch_size)
                | 'Generate embeddings' >> beam.Map(create_dataset)
        )


args = {
    'job_name': 'hub2emb-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S')),
    'runner': 'DirectRunner',
    'batch_size': 2048,
    'data_dir': 'D:\\workspace\\20211029_AI\\test\\corpus\\test.csv',
    'random_projection_matrix': None,
}

global label_list
global sentence1_list
global sentence2_list
label_list = []
sentence1_list = []
sentence2_list = []
print("Running pipeline...")
get_dataset(args)
print("Pipeline is done.")

glue = {'train': {}, 'test': {}}
glue['train']['label'] = tf.constant(label_list)
glue['train']['sentence1'] = tf.constant(sentence1_list)
glue['train']['sentence2'] = tf.constant(sentence2_list)

def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(glue_dict, tokenizer):
    num_examples = len(glue_dict["sentence1"])

    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence1"])])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence2"])])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs

glue_train = bert_encode(glue['train'], tokenizer)
glue_train_labels = glue['train']['label']

# Set up epochs and steps
epochs = 3
batch_size = 32

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def build_classifier_model():
    class Classifier(tf.keras.Model):
        def __init__(self, encoder_inputs, net, encoder):
            super(Classifier, self).__init__(inputs=encoder_inputs, outputs=net, name="prediction")
            self.encoder = encoder

        def call(self, preprocessed_text, training=False):
            x = super(Classifier, self).call(preprocessed_text)
            return x
    input_word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')
    encoder_inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }
    encoder = hub.KerasLayer(hub_url_bert, trainable=True)
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(2, activation=None, name='classifier')(net)
    tf.keras.Model()
    return Classifier(encoder_inputs, net, encoder)

classifier_model = build_classifier_model()

classifier_model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

classifier_model.fit(
    glue_train, glue_train_labels,
    batch_size=32,
    epochs=epochs)

export_dir = './exported_model'
tf.saved_model.save(classifier_model, export_dir=export_dir)
