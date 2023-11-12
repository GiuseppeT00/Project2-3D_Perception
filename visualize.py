import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataset.deliver_dataset import Deliver
from dataset.augmentation import get_val_augmentations


def visualize_both(pred_mask: torch.Tensor, true_mask: torch.Tensor, palette: list):
    pred_mask = np.array([palette[label] for label in torch.flatten(pred_mask)]).reshape((1024, 1024, 3))
    true_mask = np.array([palette[label] for label in torch.flatten(true_mask)]).reshape((1024, 1024, 3))
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(pred_mask)
    ax[1].imshow(true_mask)
    plt.show()

def save_mask(pred_mask: torch.Tensor, palette: list):
    img0 = np.array([palette[label] for label in torch.flatten(pred_mask[0])]).reshape((1024, 1024, 3))
    plt.imshow(img0)
    plt.savefig(f'predictions/DeepLabV3_ResNet101_masks/mask{idx}_0.png')
    plt.clf()

    img1 = np.array([palette[label] for label in torch.flatten(pred_mask[1])]).reshape((1024, 1024, 3))
    plt.imshow(img1)
    plt.savefig(f'predictions/DeepLabV3_ResNet101_masks/mask{idx}_1.png')
    plt.clf()

if __name__ == '__main__':
    with open('configs/deliver_dataset.yml', 'r') as f:
        dataset_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    '''
    val_transform = get_val_augmentations(cropped_size=[1024, 1024],
                                          normalization_mean=dataset_cfg['DATASET']['MEAN'],
                                          normalization_std=dataset_cfg['DATASET']['STD'])
    test_set = Deliver(transform=val_transform, split='test', cases=dataset_cfg['DATASET']['CASES'],
                       classes=dataset_cfg['DATASET']['CLASSES'],
                       modals=['img'], palette=dataset_cfg['DATASET']['PALETTE'], root='data')
    test_loader = iter(DataLoader(test_set, batch_size=2, num_workers=2, drop_last=False, pin_memory=False, shuffle=False))
    '''

    for idx in range(1, 949):
        pred_mask = torch.load(f'predictions/DeepLabV3_ResNet101/pred{idx}.pth', map_location='cpu')
        save_mask(pred_mask, dataset_cfg['DATASET']['PALETTE'])
        print(f'Mask {idx} correctly saved.')
        '''
        _, true_mask = next(test_loader)
        visualize_both(pred_mask[0], true_mask[0], dataset_cfg['DATASET']['PALETTE'])
        visualize_both(pred_mask[1], true_mask[1], dataset_cfg['DATASET']['PALETTE'])
        '''



