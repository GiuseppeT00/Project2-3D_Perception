import torchvision.transforms.functional as TF
from torch import Tensor
import random
import math
from typing import Union, Tuple, List


class Compose:
    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, sample: dict) -> dict:
        img, mask = sample['img'], sample['mask']
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self._transforms:
            sample = transform(sample)

        return sample


class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self._mean = mean
        self._std = std

    def __call__(self, sample: dict) -> dict:
        for k in list(sample.keys()):
            if k == 'mask':
                continue
            elif k == 'img':
                sample[k] = sample[k].float()
                sample[k] /= 255
                sample[k] = TF.normalize(sample[k], self._mean, self._std)
            else:
                sample[k] = sample[k].float()
                sample[k] /= 255

        return sample


class RandomColorJitter:
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self._p:
            brightness = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_brightness(sample['img'], brightness)
            contrast = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_contrast(sample['img'], contrast)
            saturation = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_saturation(sample['img'], saturation)
        return sample


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1.):
        self._gamma = gamma
        self._gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self._gamma, self._gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5):
        self._sharpness = sharpness_factor
        self._p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self._p:
            sample['img'] = TF.adjust_sharpness(sample['img'], self._sharpness)
        return sample


class RandomAutoContrast:
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self._p:
            sample['img'] = TF.autocontrast(sample['img'])
        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size: Union[int, List[int]] = 3, p: float = 0.5):
        self._kernel_size = kernel_size
        self._p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self._p:
            sample['img'] = TF.gaussian_blur(sample['img'], self._kernel_size)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self._p:
            for k in list(sample.keys()):
                sample[k] = TF.hflip(sample[k])
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self._p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self._p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits: int = 2):
        self._bits = bits  # 0-8

    def __call__(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.posterize(image, self._bits), label


class Affine:
    def __init__(self, angle=0, translate: List[int] = (0, 0), scale=1.0,
                 shear: List[float] = (0., 0.), seg_fill: float = 0.):
        self._angle = angle
        self._translate = translate
        self._scale = scale
        self._shear = shear
        self._seg_fill = seg_fill

    def __call__(self, img: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.affine(img, self._angle, self._translate, self._scale,
                         self._shear, TF.InterpolationMode.BILINEAR, 0.), \
            TF.affine(label, self._angle, self._translate, self._scale, self._shear,
                      TF.InterpolationMode.NEAREST, self._seg_fill)


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2,
                 seg_fill: float = 0., expand: bool = False):
        self._p = p
        self._angle = degrees
        self._expand = expand
        self._seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        random_angle = random.random() * 2 * self._angle - self._angle
        if random.random() < self.p:
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.NEAREST,
                                          self._expand, fill=self._seg_fill)
                else:
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.BILINEAR,
                                          self._expand, fill=0.)
        return sample


class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]):

        self._size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self._size), TF.center_crop(mask, self._size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5):
        self._size = (size, size) if isinstance(size, int) else size
        self._p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self._size
        if random.random() < self._p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0):
        self._size = size
        self._seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self._size[1] - img.shape[2], self._size[0] - img.shape[1])
        return TF.pad(img, padding), TF.pad(mask, padding, self._seg_fill)


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
        self._size = size
        self._seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self._size

        # scale the image
        scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, [nH, nW], TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [nH, nW], TF.InterpolationMode.NEAREST)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self._seg_fill)
        return img, mask


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]):
        self._size = size

    def __call__(self, sample: dict) -> dict:
        H, W = sample['img'].shape[1:]

        # scale the image
        scale_factor = self._size[0] / min(H, W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        for k, v in sample.items():
            if k == 'mask':
                sample[k] = TF.resize(v, [nH, nW], TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, [nH, nW], TF.InterpolationMode.BILINEAR)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32

        for k, v in sample.items():
            if k == 'mask':
                sample[k] = TF.resize(v, [alignH, alignW], TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, [alignH, alignW], TF.InterpolationMode.BILINEAR)
        return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]],
                 scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0):
        self._size = size
        self._scale = scale
        self._seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        H, W = sample['img'].shape[1:]
        tH, tW = self._size

        # get the scale
        ratio = random.random() * (self._scale[1] - self._scale[0]) + self._scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        for k, v in sample.items():
            if k == 'mask':
                sample[k] = TF.resize(v, [nH, nW], TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, [nH, nW], TF.InterpolationMode.BILINEAR)

        # random crop
        margin_h = max(sample['img'].shape[1] - tH, 0)
        margin_w = max(sample['img'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        # pad the image
        if sample['img'].shape[1:] != self._size:
            padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = TF.pad(v, padding, fill=self._seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample


def get_train_augmentations(cropped_size: Union[int, Tuple[int], List[int]], seg_fill: int,
                            normalization_mean: List[float], normalization_std: List[float]):
    return Compose([
        RandomColorJitter(p=0.2),
        RandomHorizontalFlip(p=0.5),
        RandomGaussianBlur([3, 3], p=0.2),
        RandomResizedCrop(cropped_size, scale=(0.5, 2.0), seg_fill=seg_fill),
        Normalize(normalization_mean, normalization_std)
    ])


def get_val_augmentations(cropped_size: Union[int, Tuple[int], List[int]],
                          normalization_mean: List[float], normalization_std: List[float]):
    return Compose([
        Resize(cropped_size),
        Normalize(normalization_mean, normalization_std)
    ])
