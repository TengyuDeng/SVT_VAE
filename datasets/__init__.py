from .datasets import RWCDataset, DALIDataset, MyLibriSpeechDataset
from .collate_fns import get_collate_fn
import os, torch

num_workers = 8

dataset_classes = {
    "RWC": RWCDataset,
    "DALI": DALIDataset,
    "LibriSpeech": MyLibriSpeechDataset, 
}

def get_dataloaders(configs):
    
    transforms = configs['transforms']
        
    train_dataloaders = {}
    val_dataloaders = {}
    test_dataloaders = {}
    for dataloader_name in configs['train']:
        dataloader_configs = configs['train'][dataloader_name].copy()
        dataloader_configs["transforms"] = transforms.copy()
        dataloader = get_dataloader(dataloader_configs)
        train_dataloaders[dataloader_name] = dataloader
    for dataloader_name in configs['val']:
        dataloader_configs = configs['val'][dataloader_name].copy()
        dataloader_configs["transforms"] = transforms.copy()
        dataloader = get_dataloader(dataloader_configs)
        val_dataloaders[dataloader_name] = dataloader
    for dataloader_name in configs['test']:
        dataloader_configs = configs['test'][dataloader_name].copy()
        dataloader_configs["transforms"] = transforms.copy()
        dataloader = get_dataloader(dataloader_configs)
        test_dataloaders[dataloader_name] = dataloader

    return train_dataloaders, val_dataloaders, test_dataloaders

def get_dataloader(configs):
    
    dataset_type = configs.pop("type")
    collate_fn = get_collate_fn(configs.pop("collate_fn"))
    batch_size = configs.pop("batch_size")

    dataset = get_dataset(dataset_type, configs)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        )

    return dataloader

def get_dataset(dataset_type, configs):

    dataset = dataset_classes[dataset_type](**configs)
    return dataset