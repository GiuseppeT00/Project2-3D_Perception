from torch.optim import AdamW


def get_optimizer(optimizer_name: str, model_params, lr: float, weight_decay: float):
    assert optimizer_name in ['AdamW'], f'Invalid optimizer selected ({optimizer_name})'
    if optimizer_name == 'AdamW':
        return AdamW(params=model_params,
                     lr=lr,
                     weight_decay=weight_decay)
