import argparse
import os
import zipfile
import shutil
import json
import time
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset as hf_Dataset

from data.data_config import continuous_test_dataset_labels, discrete_single_test_dataset_labels, \
    discrete_multi_test_dataset_labels, datasets_name, yes_no_to_bool_mapper
from utils.helper import compute_regression_metrics, compute_classification_metrics
from utils.pre_processing import pre_process_eval_dataset
from models.encoder_models import load_bert_model

warnings.filterwarnings("ignore")

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SaveMetricsCallbackRegression(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.metrics_per_epoch = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            clean_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        clean_metrics[key] = float(value)
                    else:
                        clean_metrics[key] = value.tolist()
                else:
                    clean_metrics[key] = value

            self.metrics_per_epoch.append(clean_metrics)
            with open(self.save_path, "w") as f:
                json.dump(self.metrics_per_epoch, f, indent=4)


class SaveMetricsCallbackClassification(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.metrics_per_epoch = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.metrics_per_epoch.append(metrics)
            with open(self.save_path, "w") as f:
                json.dump(self.metrics_per_epoch, f, indent=4)


def evaluate_encoding_model(args):
    device = 'cuda'
    os.makedirs(args.eval_output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log_dir, exist_ok=True)

    encoder_path = os.path.join(args.encoder_dir, 'Training_Output', 'best_encoder.pt')

    if os.path.isdir(args.raw_data_extract_dir_name):
        shutil.rmtree(args.raw_data_extract_dir_name)
    os.makedirs(args.raw_data_extract_dir_name)

    with zipfile.ZipFile('emotion_datasets_raw.zip', 'r') as zip_ref:
        zip_ref.extractall(args.raw_data_extract_dir_name)

    original_data_dir_path = os.path.join(args.raw_data_extract_dir_name, 'data')

    if args.dataset_category == datasets_name[0]:
        dataset_meta_info = continuous_test_dataset_labels.copy()
        compute_metrics_function = compute_regression_metrics
        metrics_callback_class = SaveMetricsCallbackRegression
    elif args.dataset_category == datasets_name[1]:
        dataset_meta_info = discrete_single_test_dataset_labels.copy()
        compute_metrics_function = compute_classification_metrics
        metrics_callback_class = SaveMetricsCallbackClassification
    else:
        dataset_meta_info = discrete_multi_test_dataset_labels.copy()
        compute_metrics_function = compute_classification_metrics
        metrics_callback_class = SaveMetricsCallbackClassification

    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder_model)

    for dataset_name, dataset_info in dataset_meta_info.items():
        if os.path.exists(os.path.join(args.eval_output_dir, dataset_name)):
            shutil.rmtree(os.path.join(args.eval_output_dir, dataset_name))
        os.makedirs(os.path.join(args.eval_output_dir, dataset_name), exist_ok=True)

        if os.path.exists(os.path.join(args.tensorboard_log_dir, dataset_name)):
            shutil.rmtree(os.path.join(args.tensorboard_log_dir, dataset_name))
        os.makedirs(os.path.join(args.tensorboard_log_dir, dataset_name), exist_ok=True)

        dataset = hf_Dataset.from_file(os.path.join(original_data_dir_path, dataset_name, 'data-00000-of-00001.arrow'))
        label_column_names = dataset_meta_info[dataset_name]['labels']
        text_column = 'text'

        model = load_bert_model(encoder_path, label_column_names, dataset_type=args.dataset_category,
                                freeze_encoder=yes_no_to_bool_mapper[args.freeze_encoder],
                                pre_trained_encoder=yes_no_to_bool_mapper[args.pre_trained_encoder])
        pre_processed_dataset = pre_process_eval_dataset(dataset, tokenizer, text_column, label_column_names,
                                                         dataset_type=args.dataset_category, split=0.2)

        metrics_callback = metrics_callback_class(os.path.join(args.eval_output_dir, dataset_name, 'epoch_metrics.json'))

        training_args = TrainingArguments(
            output_dir=os.path.join(args.eval_output_dir, dataset_name),
            logging_dir=os.path.join(args.tensorboard_log_dir, dataset_name),
            report_to="tensorboard",
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_strategy='epoch',
            logging_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=pre_processed_dataset['train'],
            eval_dataset=pre_processed_dataset["validation"],
            compute_metrics=compute_metrics_function,
            callbacks=[metrics_callback]
        )

        trainer.train()

        test_metrics = trainer.evaluate(pre_processed_dataset["test"])
        print("Test metrics:", test_metrics)
        with open(os.path.join(args.eval_output_dir, dataset_name, 'test_metrics.json'), "w") as f:
            json.dump(test_metrics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_category', default=datasets_name[0], type=str,
                        help='Name of the Dataset Category to Test',
                        choices=[datasets_name[0], datasets_name[1], datasets_name[2]])
    parser.add_argument('--freeze_encoder', default='yes', type=str,
                        help='Flag to freeze the encoder training',
                        choices=['yes', 'no'])
    parser.add_argument('--pre_trained_encoder', default='yes', type=str,
                        help='Flag to specify using pretrained encoder',
                        choices=['yes', 'no'])
    parser.add_argument('--eval_output_dir', default='Eval_Results', type=str,
                        help='Name of the Output Directory')
    parser.add_argument('--encoder_dir', default='Results', type=str,
                        help='Name of the Preprocessing Data Directory')
    parser.add_argument('--raw_data_extract_dir_name', default='raw_data', type=str,
                        help='Name of the Raw Data Extract Directory')
    parser.add_argument('--tensorboard_log_dir', default='eval_tensorboard_logs', type=str,
                        help='Path for the Tensorboard log directory')

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of Epochs to Train the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size for Dataloader')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Starting value of Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Learning rate weight decay')

    parser.add_argument('--base_encoder_model', default='FacebookAI/roberta-base', type=str,
                        help='Base Encoder Model to use', choices=['FacebookAI/roberta-base'])

    parser_args = parser.parse_args()
    start_time = time.time()
    evaluate_encoding_model(parser_args)
    end_time = time.time()
    print(f'Time taken for evaluation: {(end_time - start_time) / 60} minutes')
