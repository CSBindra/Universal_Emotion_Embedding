import os
import json
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr
import torch
from datasets import Dataset as hf_Dataset

from data.data_config import emotion_grouping_mapping, new_embedding_label_columns


def create_missing_labels_column(row_data, missing_label_column_names, fill_with_value=0.0):
    for missing_label_column in missing_label_column_names:
        row_data[missing_label_column] = fill_with_value
    return row_data


def replace_nan(row_data, label_column_names, fill_with_value=0.0):
    for key in label_column_names:
        if type(row_data[key]) == bool:
            row_data[key] = False if (str(row_data[key]) == 'nan' or not row_data[key]) else row_data[key]
        else:
            row_data[key] = fill_with_value if (str(row_data[key]) == 'nan' or not row_data[key]) else row_data[key]
    return row_data


def get_combined_label_column_cont(row_data, label_column_names):
    combined_list = np.array([row_data[label_column_name] for label_column_name in label_column_names],
                             dtype=np.float32)
    min_vals = np.min(combined_list)
    max_vals = np.max(combined_list)
    combined_list_normalized = (combined_list - min_vals) / (max_vals - min_vals + 1e-8)
    row_data['combined_label'] = combined_list_normalized
    row_data['original_value'] = np.sum(combined_list_normalized)
    return row_data


def get_new_label_column(row_data, combined_to_new_label_map):
    row_data['new_label'] = combined_to_new_label_map[tuple(row_data['combined_label'])]
    return row_data


def create_eval_labels_column_vector_reg(row_data, label_column_names):
    row_data["labels"] = [row_data[column_name] for column_name in label_column_names]
    return row_data


def create_eval_labels_column_vector_class(row_data, label_column_names):
    row_data["labels"] = torch.zeros((len(label_column_names)))
    for i, label in enumerate(label_column_names):
        row_data["labels"][i] = min((int(row_data[label]) * i), 1)
    row_data["labels"] = (torch.argmax(row_data["labels"])).unsqueeze(0).squeeze(-1)
    return row_data


def tokenize_text(row_data, text_column, tokenizer):
    return tokenizer(row_data[text_column], padding='max_length', truncation=True)


def expand_new_label_column(row_data, n_label_clusters):
    for i in range(n_label_clusters):
        if row_data['new_label'] == i:
            row_data[f'Emotion_Label_{i+1}'] = True
        else:
            row_data[f'Emotion_Label_{i+1}'] = False
    return row_data


def get_new_kmeans_cluster_lables(combined_dataset, n_label_clusters, seed=1234):
    kmeans = KMeans(n_clusters=n_label_clusters, random_state=seed)
    kmeans.fit(combined_dataset['combined_label'])

    new_labels = kmeans.labels_

    combined_to_new_label_map = dict()
    for row_index, row_data in enumerate(combined_dataset):
        combined_to_new_label_map[tuple(row_data['combined_label'])] = new_labels[row_index]

    combined_dataset = combined_dataset.map(partial(get_new_label_column,
                                                    combined_to_new_label_map=combined_to_new_label_map),
                                            num_proc=4)
    combined_dataset = combined_dataset.remove_columns(['combined_label'])

    combined_dataset = combined_dataset.map(partial(expand_new_label_column, n_label_clusters=n_label_clusters),
                                            num_proc=4)
    return combined_dataset


def get_plut_mapped_label_column_multi(row_data, label_column_names):
    new_label_val_dict = {new_label: 0.0 for new_label in new_embedding_label_columns}
    for label_column_name in label_column_names:
        if (type(row_data[label_column_name]) == bool) and row_data[label_column_name]:
            new_label_val_dict[emotion_grouping_mapping[label_column_name]] = 1.0
        elif int(row_data[label_column_name]) > 0:
            new_label_val_dict[emotion_grouping_mapping[label_column_name]] = 1.0
        else:
            continue
    combined_list = list(new_label_val_dict.values())
    min_vals = np.min(combined_list)
    max_vals = np.max(combined_list)
    combined_list_normalized = (combined_list - min_vals) / (max_vals - min_vals + 1e-8)
    row_data['combined_label'] = combined_list_normalized
    row_data['original_value'] = np.sum(combined_list_normalized)
    return row_data


def convert_labels_to_index(row_data, labels):
    row_data['label'] = torch.zeros(1, dtype=torch.long)
    for i, label in enumerate(labels):
        if row_data[label]:
            row_data['label'] += i
    return row_data


def load_dataset(base_dataset_path, exclude_feature_columns, shuffle=True, seed=1234):
    with open(os.path.join(base_dataset_path, 'train', 'dataset_info.json'), 'r') as file:
        dataset_labels = json.load(file)

    dataset_labels = [x for x in dataset_labels['features'] if x not in list(exclude_feature_columns)]
    dataset_labels_mapping = {i: x for i, x in enumerate(dataset_labels)}

    dataset_train = hf_Dataset.from_file(os.path.join(base_dataset_path, 'train', 'data-00000-of-00001.arrow'))
    dataset_val = hf_Dataset.from_file(os.path.join(base_dataset_path, 'validation', 'data-00000-of-00001.arrow'))
    dataset_test = hf_Dataset.from_file(os.path.join(base_dataset_path, 'test', 'data-00000-of-00001.arrow'))

    dataset_train = dataset_train.map(partial(convert_labels_to_index, labels=dataset_labels), num_proc=4, batched=False)
    dataset_val = dataset_val.map(partial(convert_labels_to_index, labels=dataset_labels), num_proc=4, batched=False)
    dataset_test = dataset_test.map(partial(convert_labels_to_index, labels=dataset_labels), num_proc=4, batched=False)

    if shuffle:
        dataset_train = dataset_train.shuffle(seed=seed)
        dataset_val = dataset_val.shuffle(seed=seed)
        dataset_test = dataset_test.shuffle(seed=seed)
    return dataset_train, dataset_val, dataset_test, dataset_labels_mapping


def smooth_curve(values, beta=0.8):
    smoothed = []
    avg = 0
    for i, val in enumerate(values):
        avg = beta * avg + (1 - beta) * val
        corrected = avg / (1 - beta ** (i + 1))
        smoothed.append(corrected)
    return smoothed


def save_loss_curve(loss_list, val_check_interval, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.title(title)
    x_labels = [(i + 1) * val_check_interval for i in np.arange(len(loss_list))]
    plt.plot(x_labels, smooth_curve(loss_list), label="Smoothed")
    plt.plot(x_labels, loss_list, alpha=0.3, label="Raw")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.xticks(x_labels)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_regression_metrics(eval_pred):
    preds, labels = eval_pred
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    mape = mean_absolute_percentage_error(labels, preds)
    r2 = r2_score(labels, preds)

    pearson_corrs = []
    spearman_corrs = []
    for i in range(preds.shape[1]):
        p_corr, _ = pearsonr(labels[:, i], preds[:, i])
        pearson_corrs.append(p_corr)
        s_corr, _ = spearmanr(labels[:, i], preds[:, i])
        spearman_corrs.append(s_corr)

    pearson_corr = np.mean(pearson_corrs)
    spearman_corr = np.mean(spearman_corrs)

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "pearson_corr": float(pearson_corr),
        "spearman_corr": float(spearman_corr)
    }


def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
