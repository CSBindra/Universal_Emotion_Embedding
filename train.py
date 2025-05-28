import argparse
import os
import zipfile
import shutil
import pickle
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap.umap_ as umap
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utils.helper import load_dataset, save_loss_curve
from models.encoder_models import TextEncoder
from models.supcon_loss import supervised_contrastive_loss
from data.data_config import datasets_name
from utils.dataset import ContrastiveTextDataset, LabelBatchSampler, collate_fn, eval_collate_fn
from utils.pre_processing import preprocess_and_join_continuous_datasets, \
    preprocess_and_join_discrete_single_datasets, preprocess_and_join_discrete_multi_datasets

warnings.filterwarnings("ignore")

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate(model, tokenizer, datasets, device, dataset_labels_mapping, base_seaborn_plots_path, desc='Validation',
             global_step=0):
    model.eval()
    dataset_eval_loss = [0, 0, 0]
    dataset_samples = [0, 0, 0]
    avg_dataset_eval_loss = [0, 0, 0]

    with torch.no_grad():
        pbar = tqdm(total=len(datasets), desc=desc, unit='dataset', leave=False)
        for dataset_idx, dataset in enumerate(datasets):
            dataset_embeddings = []
            dataset_labels = []
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                                    collate_fn=lambda batch_data: eval_collate_fn(batch_data, tokenizer), num_workers=4,
                                    pin_memory=True, persistent_workers=True)
            for batch in dataloader:
                input_ids = batch[0]['input_ids'].to(device)
                attention_mask = batch[0]['attention_mask'].to(device)
                labels = torch.tensor(batch[1]).to(device)

                with autocast(device):
                    embeddings = model(input_ids, attention_mask)
                    loss = supervised_contrastive_loss(embeddings, labels, temperature=0.2)

                dataset_embeddings.append(embeddings.cpu())
                dataset_labels.append(labels.cpu())

                batch_size = input_ids.size(0)

                dataset_eval_loss[dataset_idx] += loss.item() * batch_size
                dataset_samples[dataset_idx] += batch_size

            if desc == 'Validation':
                dataset_embeddings = torch.cat(dataset_embeddings, dim=0).numpy()
                dataset_labels = torch.cat(dataset_labels, dim=0).numpy()
                dataset_labels_string = [dataset_labels_mapping[dataset_idx][label_idx] for label_idx in dataset_labels]

                pca = PCA(n_components=50)
                pca_result = pca.fit_transform(np.array(dataset_embeddings))

                reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed)
                embeddings_2d = reducer.fit_transform(pca_result)

                plt.figure(figsize=(10, 8))
                palette = sns.color_palette("hls", len(set(dataset_labels_string)))
                sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=dataset_labels_string, legend="full",
                                palette=palette, s=3)
                plt.title("UMAP of Sentence Embeddings")
                plt.savefig(os.path.join(base_seaborn_plots_path, datasets_name[dataset_idx],
                                         f'embedding_umap_plot_step_{global_step}.png'), bbox_inches='tight', dpi=300)
                plt.close()

            avg_dataset_eval_loss[dataset_idx] = dataset_eval_loss[dataset_idx] / dataset_samples[dataset_idx]

            pbar.update(1)
        pbar.close()
    return avg_dataset_eval_loss


def pre_process_dataset_and_train_supcon_encoder(args):
    device = 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.raw_data_extract_dir_name):
        shutil.rmtree(args.raw_data_extract_dir_name)
    os.makedirs(args.raw_data_extract_dir_name)

    with zipfile.ZipFile('emotion_datasets_raw.zip', 'r') as zip_ref:
        zip_ref.extractall(args.raw_data_extract_dir_name)

    original_data_dir_path = os.path.join(args.raw_data_extract_dir_name, 'data')

    preprocessing_data_dir_path = os.path.join(args.output_dir, args.preprocessing_data_dir)
    os.makedirs(preprocessing_data_dir_path, exist_ok=True)

    model_output_dir_path = os.path.join(args.output_dir, 'Training_Output')
    if os.path.isdir(model_output_dir_path):
        shutil.rmtree(model_output_dir_path)
    os.makedirs(model_output_dir_path)

    base_seaborn_plots_path = os.path.join(model_output_dir_path, 'seaborn_results')
    os.makedirs(base_seaborn_plots_path, exist_ok=True)
    os.makedirs(os.path.join(base_seaborn_plots_path, datasets_name[0]), exist_ok=True)
    os.makedirs(os.path.join(base_seaborn_plots_path, datasets_name[1]), exist_ok=True)
    os.makedirs(os.path.join(base_seaborn_plots_path, datasets_name[2]), exist_ok=True)

    if not os.path.isdir(args.tensorboard_log_dir):
        os.makedirs(args.tensorboard_log_dir)

    if not os.path.isdir(os.path.join(preprocessing_data_dir_path, 'combined_continuous_dataset')):
        _ = preprocess_and_join_continuous_datasets(original_data_dir_path, preprocessing_data_dir_path)
        _ = preprocess_and_join_discrete_single_datasets(original_data_dir_path,
                                                         preprocessing_data_dir_path)
        _ = preprocess_and_join_discrete_multi_datasets(original_data_dir_path,
                                                        preprocessing_data_dir_path)

    continuous_dataset_loaded = load_dataset(os.path.join(preprocessing_data_dir_path, 'combined_continuous_dataset'),
                                             ['text', 'original_value'])
    continuous_dataset_train = continuous_dataset_loaded[0]
    continuous_dataset_val = continuous_dataset_loaded[1]
    continuous_dataset_test = continuous_dataset_loaded[2]
    continuous_dataset_labels = continuous_dataset_loaded[3]

    discrete_single_dataset_loaded = load_dataset(os.path.join(preprocessing_data_dir_path,
                                                               'combined_discrete_single_dataset'), ['text'])
    discrete_single_dataset_train = discrete_single_dataset_loaded[0]
    discrete_single_dataset_val = discrete_single_dataset_loaded[1]
    discrete_single_dataset_test = discrete_single_dataset_loaded[2]
    discrete_single_dataset_labels = discrete_single_dataset_loaded[3]

    discrete_multi_dataset_loaded = load_dataset(os.path.join(preprocessing_data_dir_path,
                                                              'combined_discrete_multi_dataset'),
                                                 ['text', 'original_value'])
    discrete_multi_dataset_train = discrete_multi_dataset_loaded[0]
    discrete_multi_dataset_val = discrete_multi_dataset_loaded[1]
    discrete_multi_dataset_test = discrete_multi_dataset_loaded[2]
    discrete_multi_dataset_labels = discrete_multi_dataset_loaded[3]

    dataset_dict_train = {
        datasets_name[0]: continuous_dataset_train,
        datasets_name[1]: discrete_single_dataset_train,
        datasets_name[2]: discrete_multi_dataset_train
    }

    dataset_dict_val = {
        datasets_name[0]: continuous_dataset_val,
        datasets_name[1]: discrete_single_dataset_val,
        datasets_name[2]: discrete_multi_dataset_val
    }

    dataset_dict_test = {
        datasets_name[0]: continuous_dataset_test,
        datasets_name[1]: discrete_single_dataset_test,
        datasets_name[2]: discrete_multi_dataset_test
    }

    dataset_labels_mapping = [continuous_dataset_labels, discrete_single_dataset_labels, discrete_multi_dataset_labels]

    best_val_loss = float('inf')
    best_val_loss_per_dataset = [float('inf')] * 3
    global_val_loss_impact_weights = [1.0, 1.0, 1.0]
    epochs_without_improvement = 0
    global_step = 0

    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder_model)

    encoder = TextEncoder(model_name=args.base_encoder_model).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    contrastive_dataset_train = ContrastiveTextDataset(
        [dataset_dict_train[datasets_name[0]], dataset_dict_train[datasets_name[1]],
         dataset_dict_train[datasets_name[2]]])

    batch_sampler = LabelBatchSampler(contrastive_dataset_train, group_size=args.group_size, batch_size=args.batch_size)
    dataloader_train = DataLoader(contrastive_dataset_train, batch_sampler=batch_sampler,
                                  collate_fn=lambda batch_data: collate_fn(batch_data, contrastive_dataset_train,
                                                                           tokenizer),
                                  num_workers=4, pin_memory=True, persistent_workers=True)

    contrastive_val_datasets = [val_data for val_data in list(dataset_dict_val.values())]

    train_loss_list = []

    val_loss_list_continuous = []
    val_loss_list_discrete_single = []
    val_loss_list_discrete_multi = []
    val_loss_list = []

    step_loss = 0.0

    num_training_steps = len(dataloader_train) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=num_training_steps)

    scaler = GradScaler(device)
    for epoch in range(args.num_epochs):
        batch_sampler.reset()
        encoder.train()
        for batch in tqdm(dataloader_train, desc=f'Epoch {epoch + 1}', leave=False):
            tokens, labels, original_values, source = batch
            input_ids = tokens["input_ids"].view(-1, tokens["input_ids"].size(-1)).to(device)
            attention_mask = tokens["attention_mask"].view(-1, tokens["attention_mask"].size(-1)).to(device)
            labels = torch.tensor(labels).to(device)
            original_values = torch.tensor(original_values).to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                embeddings = encoder(input_ids, attention_mask)
                if source in [datasets_name[0], datasets_name[2]]:
                    train_loss = supervised_contrastive_loss(embeddings, labels, original_values=original_values,
                                                             step=global_step, temperature=0.2)
                else:
                    train_loss = supervised_contrastive_loss(embeddings, labels, temperature=0.2)
            scaler.scale(train_loss).backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if source == datasets_name[0]:
                step_loss += (global_val_loss_impact_weights[0] * train_loss.item())
            elif source == datasets_name[2]:
                step_loss += (global_val_loss_impact_weights[1] * train_loss.item())
            else:
                step_loss += (global_val_loss_impact_weights[2] * train_loss.item())

            global_step += 1
            if global_step % args.val_check_interval == 0:
                with torch.no_grad():
                    emb_norms = embeddings.norm(dim=1)
                    print(
                        f"[DEBUG] Embedding norm mean: {emb_norms.mean().item():.4f}, std: {emb_norms.std().item():.4f}")

                avg_train_loss = step_loss / args.val_check_interval
                train_loss_list.append(avg_train_loss)
                print(f"Step {global_step}: Avg Train Loss = {avg_train_loss:.4f}")
                step_loss = 0.0

                val_loss_datasets = evaluate(encoder, tokenizer, contrastive_val_datasets, device,
                                             dataset_labels_mapping, base_seaborn_plots_path, global_step=global_step)
                val_loss_continuous, val_loss_discrete_single, val_loss_discrete_multi = val_loss_datasets
                val_loss_list_continuous.append(val_loss_continuous)
                val_loss_list_discrete_single.append(val_loss_discrete_single)
                val_loss_list_discrete_multi.append(val_loss_discrete_multi)
                print(f"Step {global_step}: Continuous Label Validation Loss = {val_loss_continuous:.4f}")
                print(f"Step {global_step}: Discrete Single Label Validation Loss = {val_loss_discrete_single:.4f}")
                print(f"Step {global_step}: Discrete Multi Label Validation Loss = {val_loss_discrete_multi:.4f}")

                if best_val_loss_per_dataset[0] - val_loss_continuous > args.min_delta:
                    change_in_cont_val_loss = best_val_loss_per_dataset[0] - val_loss_continuous
                    best_val_loss_per_dataset[0] = val_loss_continuous
                else:
                    change_in_cont_val_loss = best_val_loss_per_dataset[0] - val_loss_continuous

                if best_val_loss_per_dataset[1] - val_loss_discrete_single > args.min_delta:
                    change_in_discrete_single_val_loss = best_val_loss_per_dataset[1] - val_loss_discrete_single
                    best_val_loss_per_dataset[1] = val_loss_discrete_single
                else:
                    change_in_discrete_single_val_loss = best_val_loss_per_dataset[1] - val_loss_discrete_single

                if best_val_loss_per_dataset[2] - val_loss_discrete_multi > args.min_delta:
                    change_in_discrete_multi_val_loss = best_val_loss_per_dataset[2] - val_loss_discrete_multi
                    best_val_loss_per_dataset[2] = val_loss_discrete_multi
                else:
                    change_in_discrete_multi_val_loss = best_val_loss_per_dataset[2] - val_loss_discrete_multi

                val_loss = global_val_loss_impact_weights[0] * val_loss_continuous + global_val_loss_impact_weights[
                    1] * val_loss_discrete_single + global_val_loss_impact_weights[2] * val_loss_discrete_multi
                val_loss_list.append(val_loss)
                print(f"Step {global_step}: Average Validation Loss = {val_loss:.4f}")

                if best_val_loss - val_loss > args.min_delta:
                    if (change_in_cont_val_loss >= 0.01) and (change_in_discrete_single_val_loss >= 0.01) and (
                            change_in_discrete_multi_val_loss >= 0.01):
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        torch.save(encoder.state_dict(), os.path.join(model_output_dir_path, 'best_encoder.pt'))
                        with open(os.path.join(model_output_dir_path, 'global_val_loss_weights.pickle'), 'wb') as file:
                            pickle.dump(global_val_loss_impact_weights, file)
                        print(f"Best Model Saved after step {global_step}.")
                    else:
                        epochs_without_improvement += 1
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping triggered after step {global_step}.")
                    break

                if change_in_cont_val_loss < 0.01:
                    global_val_loss_impact_weights[0] *= 1.2
                else:
                    global_val_loss_impact_weights[0] *= 0.95

                if change_in_discrete_single_val_loss < 0.01:
                    global_val_loss_impact_weights[1] *= 1.2
                else:
                    global_val_loss_impact_weights[1] *= 0.95

                if change_in_discrete_multi_val_loss < 0.01:
                    global_val_loss_impact_weights[2] *= 1.2
                else:
                    global_val_loss_impact_weights[2] *= 0.95

                global_val_loss_impact_weights = [min(max(w, 0.5), 3.0) for w in global_val_loss_impact_weights]
                print(f'Global Validation Impact Loss Weights: {global_val_loss_impact_weights}')
        if epochs_without_improvement >= args.patience:
            break

    _ = save_loss_curve(train_loss_list, args.val_check_interval, 'Train Loss', 'Train Step', 'Loss',
                        os.path.join(model_output_dir_path, 'train_loss.png'))
    _ = save_loss_curve(val_loss_list_continuous, args.val_check_interval, 'Continuous Label Validation Loss',
                        'Validation Step', 'Loss', os.path.join(model_output_dir_path, 'continuous_label_val_loss.png'))
    _ = save_loss_curve(val_loss_list_discrete_single, args.val_check_interval,
                        'Discrete Single Label Validation Loss', 'Validation Step', 'Loss',
                        os.path.join(model_output_dir_path, 'discrete_single_label_val_loss.png'))
    _ = save_loss_curve(val_loss_list_discrete_multi, args.val_check_interval, 'Discrete Multi Label Validation Loss',
                        'Validation Step', 'Loss',
                        os.path.join(model_output_dir_path, 'discrete_multi_label_val_loss.png'))
    _ = save_loss_curve(val_loss_list, args.val_check_interval, 'Average Validation Loss', 'Validation Step', 'Loss',
                        os.path.join(model_output_dir_path, 'avg_val_loss.png'))

    contrastive_test_datasets = [test_data for test_data in list(dataset_dict_test.values())]
    test_loss_datasets = evaluate(encoder, tokenizer, contrastive_test_datasets, device, dataset_labels_mapping,
                                  base_seaborn_plots_path, desc='Test')
    print(f"Step {global_step}: Continuous Test Loss = {test_loss_datasets[0]:.4f}")
    print(f"Step {global_step}: Discrete Single Label Test Loss = {test_loss_datasets[1]:.4f}")
    print(f"Step {global_step}: Discrete Multi Label Test Loss = {test_loss_datasets[2]:.4f}")
    test_loss = np.mean(test_loss_datasets)
    print(f"Average Test Loss = {test_loss:.4f}")

    with open(os.path.join(model_output_dir_path, 'train_loss_list.pickle'), 'wb') as file:
        pickle.dump(train_loss_list, file)

    with open(os.path.join(model_output_dir_path, 'val_loss_list_continuous.pickle'), 'wb') as file:
        pickle.dump(val_loss_list_continuous, file)

    with open(os.path.join(model_output_dir_path, 'val_loss_list_discrete_single.pickle'), 'wb') as file:
        pickle.dump(val_loss_list_discrete_single, file)

    with open(os.path.join(model_output_dir_path, 'val_loss_list_discrete_multi.pickle'), 'wb') as file:
        pickle.dump(val_loss_list_discrete_multi, file)

    with open(os.path.join(model_output_dir_path, 'val_loss_list.pickle'), 'wb') as file:
        pickle.dump(val_loss_list, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='Results', type=str,
                        help='Name of the Output Directory')
    parser.add_argument('--raw_data_extract_dir_name', default='raw_data', type=str,
                        help='Name of the Raw Data Extract Directory')
    parser.add_argument('--preprocessing_data_dir', default='Concatenated_Datasets', type=str,
                        help='Name of the Preprocessing Data Directory')
    parser.add_argument('--tensorboard_log_dir', default='tensorboard_logs', type=str,
                        help='Path for the Tensorboard log directory')

    parser.add_argument('--val_check_interval', type=int, default=1000, help='Validation Interval Steps')
    parser.add_argument('--patience', type=int, default=2, help='Interval steps to tolerate for no improvement')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum deviation in loss to tolerate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of Epochs to Train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for Dataloader')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Starting value of Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Learning rate weight decay')
    parser.add_argument('--group_size', type=int, default=5, help='Group size for positive and negative examples in '
                                                                  'contrastive loss')
    parser.add_argument('--num_warmup_steps', type=int, default=500, help='Warmup steps for lr scheduler')
    parser.add_argument('--base_encoder_model', default='FacebookAI/roberta-base', type=str,
                        help='Base Encoder Model to use', choices=['FacebookAI/roberta-base'])

    parser_args = parser.parse_args()
    start_time = time.time()
    pre_process_dataset_and_train_supcon_encoder(parser_args)
    end_time = time.time()
    print(f'Time taken for training: {(end_time - start_time) / 60} minutes')
