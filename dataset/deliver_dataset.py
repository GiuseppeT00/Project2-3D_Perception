import os
import torch
from torch.utils.data import Dataset
from torchvision import io
import torchvision.transforms.functional as TF
import glob
from typing import List, Tuple


class Deliver(Dataset):
    def __init__(self, root: str, split: str, transform, modals: List[str],
                 cases: List[str], classes: List[str], palette: List[List[int]]):
        super().__init__()

        assert len(classes) > 0 and len(palette) > 0 \
               and len(modals) > 0 and len(cases) > 0
        assert split in ['train', 'test'], f'Invalid split selected: {split}'
        for modal in modals:
            assert modal in ['img', 'lidar', 'depth', 'event'], \
                f'Invalid modal selected: {modal}'
        for case in cases:
            assert case in ['cloud', 'fog', 'night', 'sun', 'rain'], \
                f'Invalid case selected: {case}'

        self._split = split
        self._transform = transform
        self._modals = modals
        self._cases = cases
        self._classes = classes
        self._n_classes = len(classes) + 1
        self._palette = palette

        self._files = list()
        for case in cases:
            self._files.append(
                sorted(
                    glob.glob(
                        os.path.join(
                            *[root, 'img', case, split, '*', '*']
                        )
                    )
                )
            )
            # self._files.append(sorted(glob.glob(os.path.join(*['..', root, 'img', case, split, '*', '*.png']))))
        self._files = [file for subfiles in self._files for file in subfiles]

        debug_str = f"""
{'*' * 80}
Deliver dataset correctly initialized.
Selected split: {'Train' if self._split == 'train' else 'Val'}.
Selected modals: {self._modals}.
Selected cases: {self._cases}.
Number of classes: {self._n_classes}.
Number of images found (for a single modal): {len(self._files)}.
{'*' * 80}\n
        """
        print(debug_str, flush=True)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> Tuple[list, torch.Tensor]:
        base_path = str(self._files[index])
        sample_paths = {
            key: base_path.replace(
                '/img', f'/{"hha" if key == "depth" else key}'
            ).replace('_rgb', f'_{"rgb" if key == "img" else key}')
            #key: base_path.replace('\\img', f'\\{key}').replace('_rgb', f'_{"rgb" if key == "img" else key}')
            for key in self._modals
        }
        # label_path = base_path.replace('/img', '/semantic').replace('_rgb', '_semantic')
        label_path = base_path.replace('\\img', '\\semantic').replace('_rgb', '_semantic')
        sample = dict()
        H, W = -1, -1
        for key, path in sample_paths.items():
            sample[key] = self._open_img(path)
            if H == -1:
                H, W = sample[key].shape[1:]
            if key == 'event':
                sample[key] = TF.resize(sample[key], [H, W], TF.InterpolationMode.NEAREST)
        sample['mask'] = io.read_image(label_path)[0, ...].unsqueeze(0)
        # Testato da me: i valori unici nelle mask vanno da 0 a 25 compresi.
        # sample['mask'][sample['mask'] == 255] = 0
        sample['mask'] -= 1
        sample['mask'][sample['mask'] == 255] = 25

        if self._transform:
            sample = self._transform(sample)
        label = sample['mask'].squeeze().numpy()
        # print(f'In dataset: {sample["mask"].squeeze().unique()}')
        del sample['mask']
        return list(sample.values()), torch.from_numpy(label).long()

    def _open_img(self, file):
        img = io.read_image(file)
        C, _, _ = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def n_classes(self):
        return self._n_classes


if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    CLASSES = [
        "Building",
        "Fence",
        "Other",
        "Pedestrian",
        "Pole",
        "RoadLine",
        "Road",
        "SideWalk",
        "Vegetation",
        "Cars",
        "Wall",
        "TrafficSign",
        "Sky",
        "Ground",
        "Bridge",
        "RailTrack",
        "GroundRail",
        "TrafficLight",
        "Static",
        "Dynamic",
        "Water",
        "Terrain",
        "TwoWheeler",
        "Bus",
        "Truck"
    ]
    PALETTE = [
        [70, 70, 70],
        [100, 40, 40],
        [55, 90, 80],
        [220, 20, 60],
        [153, 153, 153],
        [157, 234, 50],
        [128, 64, 128],
        [244, 35, 232],
        [107, 142, 35],
        [0, 0, 142],
        [102, 102, 156],
        [220, 220, 0],
        [70, 130, 180],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
        [250, 170, 30],
        [110, 190, 160],
        [170, 120, 50],
        [45, 60, 150],
        [145, 170, 100],
        [0, 0, 230],
        [0, 60, 100],
        [0, 0, 70],
        [0, 0, 0]
    ]
    trainset = Deliver(transform=None, split='train', cases=cases, classes=CLASSES,
                       modals=['img'], palette=PALETTE, root='data')
    from torch.utils.data import DataLoader
    import numpy as np
    import torch.nn.functional as F
    trainloader = DataLoader(trainset, batch_size=1, num_workers=1, drop_last=False, pin_memory=False)
    import matplotlib.pyplot as plt
    for i, (sample, lbl) in enumerate(trainloader):
        if 25 in lbl[0].unique():
            # print(lbl[0].unique(return_counts=True))
            mask = lbl[0].cpu().numpy()
            img = [[255, 255, 255] for _ in range(1042*1042)]
            img = np.array(img).reshape((1042, 1042, 3))
            for row in range(len(mask[0])):
                for col in range(len(mask[1])):
                    if mask[row][col] == 2:
                        img[row][col] = [0, 0, 0]

            mask = np.array([PALETTE[label] for label in mask.reshape(-1)]).reshape((1042, 1042, 3))

            f, ax = plt.subplots(1, 2)
            ax[0].imshow(np.transpose(sample[0][0].cpu().numpy(), (1, 2, 0)))
            ax[1].imshow(mask)
            plt.show()
            exit(0)
