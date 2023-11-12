import time
import yaml
import torch
import torch.nn as nn
from model.utils import fix_seeds
from dataset.deliver_dataset import Deliver
from dataset.augmentation import get_val_augmentations
from model.model_retriever import get_model
from model.optimizer_retriever import get_optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.validation_loop import exec_validation_loop
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


def evaluate():
    start_time = time.time()

    with open('configs/deliver_dataset.yml', 'r') as f:
        dataset_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    with open('configs/model.yml', 'r') as f:
        model_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds()
    # device = model_cfg['DEVICE']
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    modals = ''.join([modal[0].upper() for modal in dataset_cfg['DATASET']['MODALS']])

    val_transform = get_val_augmentations(cropped_size=model_cfg['EVAL']['IMAGE_SIZE'],
                                          normalization_mean=dataset_cfg['DATASET']['MEAN'],
                                          normalization_std=dataset_cfg['DATASET']['STD'])

    val_set = Deliver(root=dataset_cfg['DATASET']['ROOT'],
                      split='test',
                      transform=val_transform,
                      modals=dataset_cfg['DATASET']['MODALS'],
                      cases=dataset_cfg['DATASET']['CASES'],
                      classes=dataset_cfg['DATASET']['CLASSES'],
                      palette=dataset_cfg['DATASET']['PALETTE'])

    val_loader = DataLoader(dataset=val_set,
                            batch_size=model_cfg['EVAL']['BATCH_SIZE'],
                            num_workers=dataset_cfg['DATASET']['WORKERS'],
                            shuffle=False)

    save_dir_prefix = f"{model_cfg['SAVE_DIR']}/{model_cfg['MODEL']['NAME']}_{model_cfg['MODEL']['BACKBONE']}_{modals}"
    classes = dataset_cfg['DATASET']['CLASSES']
    n_classes = val_set.n_classes()
    multi_modal = len(dataset_cfg['DATASET']['MODALS']) > 1
    multi_backbone = (len(dataset_cfg['DATASET']['MODALS']) > 1) and model_cfg['MODEL']['MULTI_BACKBONE']

    model = get_model(model_name=model_cfg['MODEL']['NAME'],
                      backbone=model_cfg['MODEL']['BACKBONE'],
                      multi_modal=multi_modal,
                      multi_backbone=multi_backbone,
                      classes=n_classes,
                      old_error=False)
    model.to(device)

    '''
    criterion = get_criterion(loss_name=model_cfg['LOSS']['NAME'],
                              n_classes=n_classes,
                              ignore_label=dataset_cfg['DATASET']['IGNORE_LABEL'])

    aux_criterion = get_criterion(loss_name='CrossEntropy',
                                  n_classes=n_classes,
                                  ignore_label=dataset_cfg['DATASET']['IGNORE_LABEL'])
    '''

    criterion = nn.CrossEntropyLoss(ignore_index=n_classes - 1)

    optimizer = get_optimizer(optimizer_name=model_cfg['OPTIMIZER']['NAME'],
                              model_params=model.parameters(),
                              lr=model_cfg['OPTIMIZER']['LR'],
                              weight_decay=model_cfg['OPTIMIZER']['WEIGHT_DECAY'])

    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                     mode='max',
                                     factor=0.2,
                                     patience=2,
                                     min_lr=1e-7,
                                     verbose=True)

    decomposition = PCA(n_components=1, random_state=3407) if model_cfg['MODEL']['DECOMPOSITION'] == 'PCA' \
        else Isomap(n_components=1, n_neighbors=10, n_jobs=-1) if model_cfg['MODEL']['DECOMPOSITION'] == 'Isomap' \
        else None

    debug_str = f"""
{'*' * 80}
Evaluation setup completed.
Model: {model_cfg['MODEL']['NAME']}.
Backbone: {model_cfg['MODEL']['BACKBONE']}.
Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}.
Loss: {model_cfg['LOSS']['NAME']}.
Optimizer: {model_cfg['OPTIMIZER']['NAME']}.
Learning rate: {model_cfg['OPTIMIZER']['LR']}.
Multimodal: {multi_modal}.
Decomposition algorithm: {model_cfg['MODEL']['DECOMPOSITION']}.
Multi Backbone: {multi_backbone}.
Weights path: '{model_cfg["EVAL"]["MODEL_PATH"]}'.
Evaluating on: {'GPU' if torch.cuda.is_available() else 'CPU'}.
Elapsed time for setup: {(time.time() - start_time) / 60 :.2f} minutes.
{'*' * 80}\n    
"""
    print(debug_str, flush=True)

    global_report = torch.load(model_cfg["EVAL"]["MODEL_PATH"], map_location='cpu')

    model.load_state_dict(global_report['model'])
    optimizer.load_state_dict(global_report['optimizer'])

    report = exec_validation_loop(val_loader=val_loader,
                                  model=model,
                                  criterion=criterion,
                                  aux_criterion=criterion,
                                  device=device,
                                  ignore_label=dataset_cfg['DATASET']['IGNORE_LABEL'],
                                  num_classes=n_classes,
                                  multi_modal=multi_modal,
                                  multi_backbone=multi_backbone,
                                  pca=decomposition,
                                  use_aux=model_cfg['TRAIN']['AUX'],
                                  save=False)

    print(f'\n{"*" * 80}\nValidation results:\n'
          f'Validation loss: {report["loss"]}\n'
          f'Validation accuracy (macro): {report["mean_accuracy_macro"]}\n'
          f'Validation accuracy (weighted): {report["mean_accuracy_weighted"]}\n'
          f'Validation IoU (macro): {report["mean_iou_macro"]}\n'
          f'Validation IoU (weighted): {report["mean_iou_weighted"]}\n'
          f'Validation accuracy (single): {report["mean_accuracy_single"]}\n'
          f'Validation IoU (single): {report["mean_iou_single"]}\n'
          f'{"*" * 80}')

    global_report['valid']['mean_iou_weighted'].append(report['mean_iou_weighted'])
    global_report['valid']['mean_accuracy_weighted'].append(report['mean_accuracy_weighted'])
    global_report['valid']['mean_iou_macro'].append(report['mean_iou_macro'])
    global_report['valid']['mean_accuracy_macro'].append(report['mean_accuracy_macro'])
    global_report['valid']['loss'].append(report['loss'])

    print(f"{'*' * 80}\nEvaluation completely ended. Total elapsed time: "
          f"{(time.time() - start_time) / 60:.2f} minutes.\n{'*' * 80}", flush=True)


if __name__ == '__main__':
    evaluate()
