import time
import torch
from sklearn.decomposition import PCA, SparsePCA
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Accuracy, F1Score
from torch.nn import functional as F
from typing import Union


def exec_training_loop(train_loader: DataLoader, model: nn.Module,
                       criterion: nn.Module, aux_criterion: nn.Module,
                       optimizer: AdamW, device: str,
                       epoch: int, epochs: int,
                       num_classes: int, ignore_label: int,
                       multi_modal: bool, multi_backbone: bool,
                       pca: Union[PCA, SparsePCA]) -> dict:
    start_time = time.time()

    epoch_loss = 0.
    iou_evaluator_weighted = JaccardIndex(task='multiclass',
                                          num_classes=num_classes,
                                          ignore_index=ignore_label,
                                          average='weighted')
    accuracy_evaluator_weighted = Accuracy(task='multiclass',
                                           num_classes=num_classes,
                                           ignore_index=ignore_label,
                                           average='weighted')
    iou_evaluator_macro = JaccardIndex(task='multiclass',
                                       num_classes=num_classes,
                                       ignore_index=ignore_label,
                                       average='macro')
    accuracy_evaluator_macro = Accuracy(task='multiclass',
                                        num_classes=num_classes,
                                        ignore_index=ignore_label,
                                        average='macro')

    mean_iou_weighted = 0.
    mean_accuracy_weighted = 0.
    mean_iou_macro = 0.
    mean_accuracy_macro = 0.

    model.train()

    for iteration, (sample, true_mask) in enumerate(train_loader):
        if multi_modal:
            sample = [s.to(device) for s in sample]
            true_mask = true_mask.to(device)
            pred_mask = model(xs=sample, pca=pca)
        else:
            sample = sample[0]
            sample = sample.to(device)
            true_mask = true_mask.to(device)
            pred_mask = model(sample)['out']

        loss = criterion(pred_mask, true_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        pred_mask = pred_mask.detach().cpu().argmax(dim=1)
        true_mask = true_mask.detach().cpu()

        mean_iou_weighted += iou_evaluator_weighted(pred_mask, true_mask)
        mean_accuracy_weighted += accuracy_evaluator_weighted(pred_mask, true_mask)
        mean_iou_macro += iou_evaluator_macro(pred_mask, true_mask)
        mean_accuracy_macro += accuracy_evaluator_macro(pred_mask, true_mask)

        if (iteration + 1) % 50 == 0 or iteration == len(train_loader) - 1 or iteration == 0:
            debug_str = f"Epoch: [{epoch}/{epochs}] |\t " \
                        f"Iteration: [{iteration + 1}/{len(train_loader)}] |\t " \
                        f"Elapsed time: {time.time() - start_time:.2f}s |\t " \
                        f"Train loss: {epoch_loss / (iteration + 1):.3f} |\t " \
                        f"Train mean accuracy (macro): {mean_accuracy_macro / (iteration + 1):.3f} |\t " \
                        f"Train mean accuracy (weighted): {mean_accuracy_weighted / (iteration + 1):.3f} |\t " \
                        f"Train mean IoU (macro): {mean_iou_macro / (iteration + 1):.3f} |\t" \
                        f"Train mean IoU (weighted): {mean_iou_weighted / (iteration + 1):.3f}"
            print(debug_str, flush=True)

    print(flush=True)

    return {
        'mean_iou_weighted': mean_iou_weighted / len(train_loader),
        'mean_accuracy_weighted': mean_accuracy_weighted / len(train_loader),
        'mean_iou_macro': mean_iou_macro / len(train_loader),
        'mean_accuracy_macro': mean_accuracy_macro / len(train_loader),
        'loss': epoch_loss / len(train_loader)
    }
