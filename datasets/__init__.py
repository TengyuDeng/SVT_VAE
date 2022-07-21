from .datasets import RWCDataset, DALIDataset, MyLibriSpeechDataset
from .collate_fns import get_collate_fn
from .transforms import get_transform
import os, torch

num_workers = 8

dataset_classes = {
    "RWC": RWCDataset,
    "DALI": DALIDataset,
    "LibriSpeech": MyLibriSpeechDataset, 
}

def get_dataloaders(configs):
    
    transform = get_transform(configs['feature_type'], configs['transform_params'])
    if 'target_type' in configs:
        target_transform = get_transform(configs['target_type'], configs['target_transform_params'])
    else:
        target_transform = vanila_transform
        
    train_dataloaders = {}
    val_dataloaders = {}
    test_dataloaders = {}
    for dataloader_name in configs['train']:
        dataloader = get_dataloader(configs['train'][dataloader_name], transform, target_transform, train=True)
        train_dataloaders[dataloader_name] = dataloader
    for dataloader_name in configs['val']:
        dataloader = get_dataloader(configs['val'][dataloader_name], transform, target_transform, train=False)
        val_dataloaders[dataloader_name] = dataloader
    for dataloader_name in configs['test']:
        dataloader = get_dataloader(configs['test'][dataloader_name], transform, target_transform, train=False)
        test_dataloaders[dataloader_name] = dataloader

    return train_dataloaders, val_dataloaders, test_dataloaders

def get_dataloader(configs, transform, target_transform, train):
    configs = configs.copy()
    dataset_type = configs.pop("type")
    collate_fn = get_collate_fn(configs.pop("collate_fn"))
    batch_size = configs.pop("batch_size")

    dataset = get_dataset(dataset_type, configs, transform, target_transform, train)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        )

    return dataloader

def get_dataset(dataset_type, configs, transform, target_transform, train):

    dataset = dataset_classes[dataset_type](
        transform=transform,
        target_transform=target_transform,
        **configs
        )
    return dataset