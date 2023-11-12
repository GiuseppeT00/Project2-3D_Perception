import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from typing import Union
from sklearn.decomposition import PCA, SparsePCA


def fuse_latent_features(self, pca: Union[PCA, SparsePCA], latent_features: list) -> torch.Tensor:
    required_shape = latent_features[0].shape
    latent_features = [torch.flatten(feature).numpy() for feature in latent_features]
    latent_features = np.array(latent_features)
    latent_features = latent_features.T
    fused_features = torch.Tensor(pca.fit_transform(latent_features))
    fused_features = torch.flatten(fused_features)
    fused_features = fused_features.view(required_shape)
    return fused_features


def overloaded_forward_single_backbone(self, xs: list, pca: Union[PCA, SparsePCA]):
    input_shape = xs[0].shape[-2:]

    x = [self.backbone(x)['out'].detach().cpu() for x in xs]
    x = self.fuse_latent_features(pca=pca, latent_features=x)
    x = x.to(torch.device('cuda')).requires_grad_()
    x = self.classifier(x)
    x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
    return x


def overloaded_forward_multi_backbone(self, xs: list, pca: Union[PCA, SparsePCA]):
    pass


def get_model(model_name: str, backbone: str, multi_modal: bool, multi_backbone: bool, classes: int) -> nn.Module:
    models = {
        'FCN_ResNet50': fcn_resnet50,
        'FCN_ResNet101': fcn_resnet101,
        'DeepLabV3_ResNet50': deeplabv3_resnet50,
        'DeepLabV3_ResNet101': deeplabv3_resnet101
    }
    weights = {
        'FCN_ResNet50': FCN_ResNet50_Weights,
        'FCN_ResNet101': FCN_ResNet101_Weights,
        'DeepLabV3_ResNet50': DeepLabV3_ResNet50_Weights,
        'DeepLabV3_ResNet101': DeepLabV3_ResNet101_Weights
    }

    selected_model = f'{model_name}_{backbone}'

    assert selected_model in list(models.keys()), f'Invalid selected model ({selected_model}).'

    model = models[selected_model](weights=weights[selected_model], weights_backbone=None)

    if model_name == 'FCN':
        model.classifier[4] = nn.Conv2d(in_channels=512, out_channels=classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=classes, kernel_size=(1, 1), stride=(1, 1))
    # model.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=classes, kernel_size=(1, 1), stride=(1, 1))

    if multi_modal:
        bound_method = fuse_latent_features.__get__(model, model.__class__)
        setattr(model, 'fuse_latent_features', bound_method)
        if multi_backbone:
            bound_method = overloaded_forward_multi_backbone.__get__(model, model.__class__)
        else:
            bound_method = overloaded_forward_single_backbone.__get__(model, model.__class__)
        setattr(model, 'forward', bound_method)

    debug_str = f"""
{'*' * 80}
Model creation completed. Model:
{model}
{'*' * 80}\n    
"""
    print(debug_str, flush=True)

    return model

"""
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from random import randint


    pca = PCA(n_components=1)
    imgs_orig = [torch.rand((2, 3, 56, 56), dtype=torch.float32) for i in range(4)]
    print(f'1 STEP: len(imgs)={len(imgs_orig)} ; shape(imgs[i])={imgs_orig[0].shape}')
    '''
    #print(imgs_orig)
    imgs = [torch.flatten(img) for img in imgs_orig]
    #print(imgs)
    print(f'2 STEP: len(imgs)={len(imgs)} ; shape(imgs[i])={imgs[0].shape}')
    df = pd.DataFrame({str(idx): np.array(img) for idx, img in enumerate(imgs)})
    # print(df)
    m1 = torch.flatten(torch.Tensor(pca.fit_transform(df))).view((2, 3, 5, 5))
    m2 = fuse_latent_features(0, pca, imgs_orig)
    print("ARE EQUALS" if torch.equal(m1, m2) else "VAFAMMOKK")
    print(f'3 STEP MINE: shape(m2)={m2.shape}')
    #print(m2)'''
    model = get_model('FCN', 'ResNet50', multi_modal=True, multi_backbone=False, classes=26)
    out = model(xs=imgs_orig, pca=pca)
    print(f'MODEL OUT SHAPE: {out.shape}')
    exit(0)
    pca = PCA(n_components=1)
    truth = pd.DataFrame({
        'f1': [randint(1, 10) for _ in range(25)],
        'f2': [randint(1, 10) for _ in range(25)],
        'f3': [randint(1, 10) for _ in range(25)],
        'f4': [randint(1, 10) for _ in range(25)],
    })
    print(f'TRUTH:\n{pca.fit_transform(truth)}')
    exit(0)

    imgs = [torch.full((1, 3, 5, 5), fill_value=i+1, dtype=torch.float32) for i in range(4)]
    print(imgs)
    imgs = np.array([torch.flatten(img).numpy() for img in imgs]).T

    imgs = torch.Tensor(pca.fit_transform(imgs)).view((1, 3, 5, 5))
    print(imgs)
    print(imgs.shape)

    exit(0)


    '''
    imgs = torch.cat([torch.unsqueeze(torch.full((1,3,512,512), fill_value=i+1, dtype=torch.float32), 0) for i in range(4)], dim=0)
    print(imgs.shape)
    imgs = torch.mean(imgs, dim=0)
    print(imgs)
    print(imgs.shape)
    exit(0)'''

    model = get_model('FCN', 'ResNet50', multi_modal=True, multi_backbone=False, classes=26)
    pca = PCA(n_components=1)
    imgs = [torch.full((1,3,512,512), fill_value=i+1, dtype=torch.float32) for i in range(4)]

    out = model(xs=imgs, pca=pca)
    print(out.shape)
    exit(0)

    bb = model.backbone
    print("partial")
    out = model._modules['backbone'](img)['out']
    print(out.shape)
    #print(out)
    pred = model._modules['classifier'](out)
    print(pred.shape)
    print("full")
    pred = model(img)['out']
    print(pred.shape)"""

