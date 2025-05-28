import torch


def supervised_contrastive_loss(embeddings, labels, original_values=None, temperature=0.07, alpha=0.5, step=0,
                                annealing_steps=2000):
    device = embeddings.device
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    self_mask = torch.eye(labels.size(0), device=device)
    mask = mask * (1 - self_mask)

    if original_values is not None:
        # Compute soft weights based on original labels
        original_values = original_values.view(-1, 1)
        label_diff = torch.abs(original_values - original_values.T)
        label_similarity = torch.exp(-label_diff/alpha)
        label_similarity = torch.clamp(label_similarity, min=0.05, max=1.0)

        # Compute annealed soft weight
        soft_weight = min(1.0, step / annealing_steps)

        # Interpolate between hard and soft mask
        weighted_mask = (1 - soft_weight) * mask + soft_weight * (mask * label_similarity)
    else:
        weighted_mask = mask

    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * (1 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (weighted_mask * log_prob).sum(1) / (weighted_mask.sum(1) + 1e-12)

    loss = -mean_log_prob_pos.mean()
    return loss
