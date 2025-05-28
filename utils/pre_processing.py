import os
from glob import glob
import random
import numpy as np
import warnings
from functools import partial
import torch
from datasets import Dataset as hf_Dataset, DatasetDict as hf_DatasetDict, concatenate_datasets, Value

from utils.helper import create_missing_labels_column, replace_nan, get_combined_label_column_cont, \
    get_new_kmeans_cluster_lables, get_plut_mapped_label_column_multi, create_eval_labels_column_vector_reg, \
    create_eval_labels_column_vector_class, tokenize_text
from utils.dataset import ConvertToTorchDataset
from data.data_config import continuous_train_dataset_labels, discrete_single_train_dataset_labels, \
    discrete_multi_train_dataset_labels, rename_dict, datasets_name

warnings.filterwarnings("ignore")

seed = 1234

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def preprocess_and_join_continuous_datasets(original_data_dir_path, save_path, n_label_clusters=8):
    continuous_labels_renamed = {
        key: {label: rename_dict[label] if rename_dict.get(label) else label for label in value['labels']} for
        key, value in continuous_train_dataset_labels.items()}
    continuous_new_labels = {key: list(continuous_labels_renamed[key].values()) for key, value in
                             continuous_labels_renamed.items()}
    continuous_labels_renamed['SentimentalLIAR']['statement'] = 'text'

    continuous_dataset_names = continuous_new_labels.keys()
    continuous_datasets = dict()
    for data_folder in continuous_dataset_names:
        if os.path.isdir(os.path.join(original_data_dir_path, data_folder)):
            for file_path in glob(os.path.join(original_data_dir_path, data_folder, '*.arrow')):
                dataset = hf_Dataset.from_file(file_path)
                for old_label_name, new_label_name in continuous_labels_renamed[data_folder].items():
                    if old_label_name != new_label_name:
                        dataset = dataset.rename_column(old_label_name, new_label_name)
                continuous_datasets[data_folder] = dataset

    final_labels_continuous = list()
    _temp = [final_labels_continuous.extend(x) for x in continuous_new_labels.values()]
    final_labels_continuous = list(sorted(list(set(sorted(final_labels_continuous))), key=lambda x: x, reverse=False))

    continuous_new_label_datasets_dict = continuous_datasets.copy()
    continuous_new_label_datasets_dict = {
        key: value.map(partial(replace_nan, label_column_names=continuous_new_labels[key], fill_with_value=0.0),
                       num_proc=4) for key, value in continuous_new_label_datasets_dict.items()}
    continuous_new_label_datasets_dict = {key: value.map(partial(create_missing_labels_column,
                                                                 missing_label_column_names=list(
                                                                     set(final_labels_continuous).difference(
                                                                         set(continuous_new_labels[key])))), num_proc=4)
                                          for key, value in continuous_new_label_datasets_dict.items()}
    continuous_new_label_datasets_dict = {
        key: value.map(partial(get_combined_label_column_cont, label_column_names=final_labels_continuous), num_proc=4)
        for key, value in continuous_new_label_datasets_dict.items()}

    processed_continuous_datasets = dict()
    for dataset_name, dataset in continuous_new_label_datasets_dict.items():
        columns_to_keep = ['text', 'combined_label', 'original_value']
        dataset = dataset.remove_columns(
            [column_name for column_name in dataset.column_names if column_name not in columns_to_keep])

        dataset = dataset.remove_columns(
            [column_name for column_name in dataset.column_names if column_name not in columns_to_keep])
        processed_continuous_datasets[dataset_name] = dataset

    combined_continuous_dataset = concatenate_datasets(list(processed_continuous_datasets.values()))
    combined_continuous_dataset = combined_continuous_dataset.filter(
        lambda x: ((x['text'] is not None) and (str(x['text']) not in ['', 'nan'])))

    combined_continuous_dataset = get_new_kmeans_cluster_lables(combined_continuous_dataset, n_label_clusters)
    train_val_test = combined_continuous_dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=seed)

    combined_continuous_dataset_train = train_val_test['train']
    combined_continuous_dataset_val = val_test['train']
    combined_continuous_dataset_test = val_test['test']

    combined_continuous_dataset_train = combined_continuous_dataset_train.remove_columns(['new_label'])
    combined_continuous_dataset_val = combined_continuous_dataset_val.remove_columns(['new_label'])
    combined_continuous_dataset_test = combined_continuous_dataset_test.remove_columns(['new_label'])

    combined_continuous_dataset_dict = hf_DatasetDict(
        {'train': combined_continuous_dataset_train, 'validation': combined_continuous_dataset_val,
         'test': combined_continuous_dataset_test})

    combined_continuous_dataset_dict.save_to_disk(os.path.join(save_path, 'combined_continuous_dataset'))


def preprocess_and_join_discrete_single_datasets(original_data_dir_path, save_path):
    discrete_single_labels_renamed = {
        key: {label: rename_dict[label] if rename_dict.get(label) else label for label in value['labels']} for
        key, value in discrete_single_train_dataset_labels.items()}
    discrete_single_new_labels = {key: list(discrete_single_labels_renamed[key].values()) for key, value in
                                  discrete_single_labels_renamed.items()}

    discrete_single_dataset_names = discrete_single_new_labels.keys()
    discrete_single_label_datasets = dict()
    for data_folder in discrete_single_dataset_names:
        if os.path.isdir(os.path.join(original_data_dir_path, data_folder)):
            for file_path in glob(os.path.join(original_data_dir_path, data_folder, '*.arrow')):
                dataset = hf_Dataset.from_file(file_path)
                for old_label_name, new_label_name in discrete_single_labels_renamed[data_folder].items():
                    if old_label_name != new_label_name:
                        dataset = dataset.rename_column(old_label_name, new_label_name)
                discrete_single_label_datasets[data_folder] = dataset

    final_labels_discrete_single = list()
    _temp = [final_labels_discrete_single.extend(x) for x in discrete_single_new_labels.values()]
    final_labels_discrete_single = list(sorted(list(set(sorted(final_labels_discrete_single))), key=lambda x: x,
                                               reverse=False))

    processed_discrete_single_label_datasets_train = dict()
    processed_discrete_single_label_datasets_val = dict()
    processed_discrete_single_label_datasets_test = dict()
    for dataset_name, dataset in discrete_single_label_datasets.items():
        existing_labels = discrete_single_new_labels[dataset_name]
        missing_labels = list(set(final_labels_discrete_single).difference(set(existing_labels)))
        if discrete_single_train_dataset_labels[dataset_name]['convert_to_bool']:
            missing_val = 0.0
        else:
            missing_val = False

        dataset = dataset.map(partial(create_missing_labels_column, missing_label_column_names=missing_labels,
                                      fill_with_value=missing_val), num_proc=4)

        if discrete_single_train_dataset_labels[dataset_name]['convert_to_bool']:
            dataset = dataset.map(lambda x: {y: True if x[y] > 0 else False for y in final_labels_discrete_single})
            for label in final_labels_discrete_single:
                dataset = dataset.cast_column(label, Value("bool"))

        dataset = dataset.map(
            partial(replace_nan, label_column_names=final_labels_discrete_single, fill_with_value=False), num_proc=4)

        columns_to_keep = ['text']
        columns_to_keep.extend(final_labels_discrete_single)

        dataset = dataset.remove_columns(
            [column_name for column_name in dataset.column_names if column_name not in columns_to_keep])
        train_val_test = dataset.train_test_split(test_size=0.2, seed=seed)
        val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=seed)

        processed_discrete_single_label_datasets_train[dataset_name] = train_val_test['train']
        processed_discrete_single_label_datasets_val[dataset_name] = val_test['train']
        processed_discrete_single_label_datasets_test[dataset_name] = val_test['test']

    combined_discrete_single_label_dataset_train = concatenate_datasets(
        list(processed_discrete_single_label_datasets_train.values()))
    combined_discrete_single_label_dataset_val = concatenate_datasets(
        list(processed_discrete_single_label_datasets_val.values()))
    combined_discrete_single_label_dataset_test = concatenate_datasets(
        list(processed_discrete_single_label_datasets_test.values()))

    combined_discrete_single_label_dataset_train = combined_discrete_single_label_dataset_train.filter(
        lambda x: len([column_name for column_name in final_labels_discrete_single if x[column_name]]) == 1)
    combined_discrete_single_label_dataset_val = combined_discrete_single_label_dataset_val.filter(
        lambda x: len([column_name for column_name in final_labels_discrete_single if x[column_name]]) == 1)
    combined_discrete_single_label_dataset_test = combined_discrete_single_label_dataset_test.filter(
        lambda x: len([column_name for column_name in final_labels_discrete_single if x[column_name]]) == 1)

    combined_discrete_single_label_dataset_dict = hf_DatasetDict({
        'train': combined_discrete_single_label_dataset_train,
        'validation': combined_discrete_single_label_dataset_val,
        'test': combined_discrete_single_label_dataset_test})

    combined_discrete_single_label_dataset_dict.save_to_disk(os.path.join(save_path,
                                                                          'combined_discrete_single_dataset'))


def preprocess_and_join_discrete_multi_datasets(original_data_dir_path, save_path, n_label_clusters=10):
    discrete_multi_labels_renamed = {
        key: {label: rename_dict[label] if rename_dict.get(label) else label for label in value['labels']} for
        key, value in discrete_multi_train_dataset_labels.items()}
    discrete_multi_new_labels = {key: list(discrete_multi_labels_renamed[key].values()) for key, value in
                                 discrete_multi_labels_renamed.items()}

    discrete_multi_dataset_names = discrete_multi_new_labels.keys()
    discrete_multi_label_datasets = dict()
    for data_folder in discrete_multi_dataset_names:
        if os.path.isdir(os.path.join(original_data_dir_path, data_folder)):
            for file_path in glob(os.path.join(original_data_dir_path, data_folder, '*.arrow')):
                dataset = hf_Dataset.from_file(file_path)
                for old_label_name, new_label_name in discrete_multi_labels_renamed[data_folder].items():
                    if old_label_name != new_label_name:
                        dataset = dataset.rename_column(old_label_name, new_label_name)
                discrete_multi_label_datasets[data_folder] = dataset

    final_labels_discrete_multi = list()
    _temp = [final_labels_discrete_multi.extend(x) for x in discrete_multi_new_labels.values()]
    final_labels_discrete_multi = list(sorted(list(set(sorted(final_labels_discrete_multi))), key=lambda x: x,
                                              reverse=False))

    discrete_multi_new_label_datasets_dict = discrete_multi_label_datasets.copy()
    discrete_multi_new_label_datasets_dict = {
        key: value.map(partial(replace_nan, label_column_names=discrete_multi_new_labels[key], fill_with_value=0),
                       num_proc=4) for key, value in discrete_multi_new_label_datasets_dict.items()}
    discrete_multi_new_label_datasets_dict = {
        key: value.map(partial(get_plut_mapped_label_column_multi, label_column_names=discrete_multi_new_labels[key]),
                       num_proc=4) for key, value in discrete_multi_new_label_datasets_dict.items()}

    processed_discrete_multi_datasets = dict()
    for dataset_name, dataset in discrete_multi_new_label_datasets_dict.items():
        columns_to_keep = ['text', 'combined_label', 'original_value']
        dataset = dataset.remove_columns(
            [column_name for column_name in dataset.column_names if column_name not in columns_to_keep])
        processed_discrete_multi_datasets[dataset_name] = dataset

    combined_discrete_multi_dataset = concatenate_datasets(list(processed_discrete_multi_datasets.values()))
    combined_discrete_multi_dataset = combined_discrete_multi_dataset.filter(
        lambda x: ((x['text'] is not None) and (str(x['text']) not in ['', 'nan'])))

    combined_discrete_multi_dataset = combined_discrete_multi_dataset.filter(lambda x: x['original_value'] > 0.0)

    combined_discrete_multi_dataset = get_new_kmeans_cluster_lables(combined_discrete_multi_dataset, n_label_clusters)
    train_val_test = combined_discrete_multi_dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=seed)

    combined_discrete_multi_dataset_train = train_val_test['train']
    combined_discrete_multi_dataset_val = val_test['train']
    combined_discrete_multi_dataset_test = val_test['test']

    combined_discrete_multi_dataset_train = combined_discrete_multi_dataset_train.remove_columns(['new_label'])
    combined_discrete_multi_dataset_val = combined_discrete_multi_dataset_val.remove_columns(['new_label'])
    combined_discrete_multi_dataset_test = combined_discrete_multi_dataset_test.remove_columns(['new_label'])

    combined_discrete_multi_dataset_dict = hf_DatasetDict(
        {'train': combined_discrete_multi_dataset_train, 'validation': combined_discrete_multi_dataset_val,
         'test': combined_discrete_multi_dataset_test})

    combined_discrete_multi_dataset_dict.save_to_disk(os.path.join(save_path, 'combined_discrete_multi_dataset'))


def pre_process_eval_dataset(dataset, tokenizer, text_column, label_columns, dataset_type=datasets_name[0], split=0.2,
                             shuffle=True):
    dataset = dataset.map(partial(replace_nan, label_column_names=label_columns), num_proc=4)
    if dataset_type == datasets_name[0]:
        dataset = dataset.map(partial(create_eval_labels_column_vector_reg, label_column_names=label_columns),
                              num_proc=4)
    else:
        dataset = dataset.map(partial(create_eval_labels_column_vector_class, label_column_names=label_columns),
                              num_proc=4)
    dataset = dataset.filter(lambda x: ((x[text_column] is not None) and (str(x[text_column]) not in ['', 'nan'])))
    dataset = dataset.map(partial(tokenize_text, text_column=text_column, tokenizer=tokenizer), num_proc=4)

    dataset = dataset.remove_columns([column_name for column_name in dataset.column_names if column_name not in
                                      ['input_ids', 'attention_mask', 'labels']])

    if shuffle:
        dataset = dataset.shuffle()

    train_test = dataset.train_test_split(test_size=split, seed=seed)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=seed)
    dataset_dict = hf_DatasetDict({
        'train': ConvertToTorchDataset(train_test['train']),
        'validation': ConvertToTorchDataset(val_test['train']),
        'test': ConvertToTorchDataset(val_test['test'])
    })
    return dataset_dict
