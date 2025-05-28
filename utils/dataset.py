import random
import torch
from torch.utils.data import Dataset, Sampler

from data.data_config import datasets_name


def collate_fn(batch, contrastive_dataset, tokenizer, max_length=128):
    all_texts = []
    all_labels = []
    all_original_values = []

    source = ''
    for group in batch:
        dataset_idx, anchor_label, anchor_idx = group
        if dataset_idx == 0:
            source = datasets_name[0]
        elif dataset_idx == 1:
            source = datasets_name[1]
        else:
            source = datasets_name[2]
        data = contrastive_dataset.datasets[dataset_idx]
        anchor = data[anchor_idx]
        anchor_text = anchor['text']
        if source in [datasets_name[0], datasets_name[2]]:
            anchor_original_value = anchor['original_value']
        else:
            anchor_original_value = 0.0

        pos_indices = [i for i in contrastive_dataset.label_to_indices[dataset_idx][anchor_label] if i != anchor_idx]
        pos_sample_count = min(2, len(pos_indices))
        positive_indices = random.sample(pos_indices, pos_sample_count) if pos_sample_count > 0 else []
        positive_texts = [data[pos_idx]['text'] for pos_idx in positive_indices]
        if source in [datasets_name[0], datasets_name[2]]:
            positive_original_values = [data[pos_idx]['original_value'] for pos_idx in positive_indices]
        else:
            positive_original_values = [anchor_original_value] * len(positive_texts)

        neg_label_candidatess = [l for l in contrastive_dataset.label_to_indices[dataset_idx] if l != anchor_label]
        negative_texts = []
        negative_labels = []
        negative_original_values = []
        while len(negative_texts) < 2:
            neg_label = random.choice(neg_label_candidatess)
            neg_idx = random.choice(contrastive_dataset.label_to_indices[dataset_idx][neg_label])
            negative_texts.append(data[neg_idx]['text'])
            negative_labels.append(neg_label)
            if source in [datasets_name[0], datasets_name[2]]:
                negative_original_values.append(data[neg_idx]['original_value'])
            else:
                negative_original_values.append(anchor_original_value)

        group_texts = [anchor_text] + positive_texts + negative_texts
        all_texts.extend(group_texts)
        group_labels = [anchor_label] * (1 + len(positive_texts)) + [neg_label for neg_label in negative_labels]
        all_labels.extend(group_labels)
        group_original_values = [anchor_original_value] + positive_original_values + negative_original_values
        all_original_values.extend(group_original_values)

    tokens = tokenizer(all_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return tokens, all_labels, all_original_values, source


def eval_collate_fn(batch, tokenizer):
    texts = [item['text'] for item in batch]
    labels = [item['label'][0] for item in batch]
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    return tokens, labels


class ContrastiveTextDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.label_to_indices = [self._build_label_index(data) for data in datasets]
        self.total_len = sum(len(data) for data in datasets)

    def _build_label_index(self, data):
        label_map = {}
        for i, example in enumerate(data):
            label = example['label'][0]
            label_map.setdefault(label, []).append(i)
        return label_map

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return idx


class LabelBatchSampler(Sampler):
    def __init__(self, contrastive_dataset, group_size=5, batch_size=32):
        super().__init__()
        self.contrastive_dataset = contrastive_dataset
        self.group_size = group_size
        self.batch_size = batch_size
        self.labels_per_batch = batch_size // group_size
        self.available_indices = {}
        self.reset()

    def reset(self):
        self.available_indices = {}
        for dataset_idx, label_map in enumerate(self.contrastive_dataset.label_to_indices):
            for label, indices in label_map.items():
                self.available_indices[(dataset_idx, label)] = set(indices)
        self._calculate_total_anchors()

    def _calculate_total_anchors(self):
        self.total_anchors = sum(len(indices) for indices in self.available_indices.values())

    def __iter__(self):
        while True:
            valid_dataset_indices = []
            for dataset_idx in range(len(self.contrastive_dataset.label_to_indices)):
                labels_with_enough = [
                    label for label, indices in self.contrastive_dataset.label_to_indices[dataset_idx].items()
                    if len(self.available_indices[(dataset_idx, label)]) >= self.group_size
                ]
                if len(labels_with_enough) >= self.labels_per_batch:
                    valid_dataset_indices.append((dataset_idx, labels_with_enough))

            if not valid_dataset_indices:
                break

            dataset_idx, label_pool = random.choice(valid_dataset_indices)
            chosen_labels = random.sample(label_pool, self.labels_per_batch)

            batch = []
            for label in chosen_labels:
                sampled_indices = random.sample(list(self.available_indices[(dataset_idx, label)]), self.group_size)
                self.available_indices[(dataset_idx, label)].difference_update(sampled_indices)
                batch.extend([(dataset_idx, label, anchor_idx) for anchor_idx in sampled_indices])

            self.total_anchors -= len(batch)
            yield batch

    def __len__(self):
        return self.total_anchors // self.batch_size


class ConvertToTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.input_ids = torch.tensor(hf_dataset['input_ids'])
        self.attention_mask = torch.tensor(hf_dataset['attention_mask'])
        self.labels = torch.tensor(hf_dataset['labels'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
