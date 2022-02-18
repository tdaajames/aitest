from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import apache_beam as beam
from apache_beam.transforms import util
from collections import namedtuple
import yaml


def create_dataset(text_list):
    for i in range(len(text_list)):
        text = text_list[i]
        data_list = text.split('\t', 2)
        label = int(data_list[0])
        sentence1 = data_list[1]
        sentence2 = data_list[2]
        inp_example = InputExample(texts=[sentence1, sentence2], label=label)
        train_samples.append(inp_example)


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


# 获取配置信息
yamlConfig = 'config.yaml'

with open(yamlConfig, 'rb') as f:
    # yaml文件通过---分节，多个节组合成一个列表
    config = yaml.load(f)
    print(type(config))
    pre_model_path = config['pre_model_path']
    print(pre_model_path)
    tune_model_path = config['tune_model_path']
    print(tune_model_path)

# 使用CPU训练
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logging.info("Start yoka fine tune")

train_batch_size = 16
num_epochs = 4

# Load a pre-trained sentence transformer model
model = SentenceTransformer(pre_model_path)

# Read the dataset
train_batch_size = 16
num_epochs = 4

train_samples = []
dev_samples = []
test_samples = []

args = {
    'job_name': 'hub2emb-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S')),
    'runner': 'DirectRunner',
    'batch_size': 2048,
    'data_dir': 'D:\\wyf\\workspace\\20211029_AI\\test\\corpus\\sgs-train-reduce.csv',
}
print("Running pipeline...")
get_dataset(args)
print("Pipeline is done.")

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.CosineSimilarityLoss(model=model)
train_loss = losses.ContrastiveLoss(model=model)

# Development set: Measure correlation between cosine score and gold labels
# logging.info("Read STSbenchmark dev dataset")
# evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          # evaluator=evaluator,
          epochs=num_epochs,
          # evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=tune_model_path)
