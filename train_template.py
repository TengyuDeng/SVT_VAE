import argparse
import torch
import os, pickle

from datasets import get_dataloaders
from models import get_model
from utils import *
from train_utils import *
from jiwer import wer, cer
from decoders import get_CTC_decoder, Mycodec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}!")

@train_loop_decorator
def train_loop(data, model, loss_function, optimizer, scheduler):
    down_sample = model.down_sample
    optimizer.zero_grad()
    # torch.cuda.empty_cache()
    audio_features, pitches, onsets, targets, input_lengths, target_lengths = data
    audio_features = audio_features.to(DEVICE)
    # Forward
    outputs = model(audio_features)
    # outputs: (length, batch_size, num_classes)
    outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
    output_lengths = downsample_length(input_lengths, down_sample)

    # outputs: (max_length, batch_size, num_classes)
    # targets: (sum(target_lengths),)
    # output_lengths: (batch_size,)
    # target_lengths: (batch_size,)
    
    loss = loss_function(outputs, targets, output_lengths, target_lengths)

    # Backprop
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    return loss.item()


@test_loop_decorator
def test_loop(data, model, dataloader_name, decoder, codec):
    down_sample = model.down_sample
    audio_features, pitches, onsets, targets, input_lengths, target_lengths = data  
    audio_features = audio_features.to(DEVICE) 
    output_lengths = downsample_length(input_lengths, down_sample).numpy()
    with torch.no_grad():
        # outputs: (length, batch_size, num_classes)
        # softmax_outputs: (batch_size, length, num_classes)
        outputs = model(audio_features)
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=-1).transpose(0, 1).cpu().numpy()

    texts = []
    targets = targets.numpy()
    target_lengths = target_lengths.numpy()
    start = 0
    batch_size = len(target_lengths)
    for i in range(batch_size):
        text = codec.decode(targets[start: start + target_lengths[i]])
        if not text:
            text += '.'
        texts.append(text)
        start += target_lengths[i]

    predicts = decoder.decode(softmax_outputs, output_lengths)

    try:
        error_wer = wer(texts, predicts)
        error_cer = cer(texts, predicts)
    except ValueError:
        texts = [text + "." for text in texts]
        error_wer = wer(texts, predicts)
        error_cer = cer(texts, predicts)
    return error_wer, error_cer

def main(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    # Read config.yaml
    configs = read_yaml(config_yaml)
    
    # Get directories
    checkpoints_dir, statistics_dir = get_dirs(workspace, configs['task'], config_yaml)

    # Get codec
    codec = Mycodec(target_type="word")

    # Construct dataloaders
    train_dataloaders, val_dataloaders, test_dataloaders = get_dataloaders(configs["dataloaders"])

    # Get model
    model = get_model(num_classes_lyrics=len(codec.characs), **configs["model"])

    # if torch.cuda.DEVICE_count() > 1:
    #     print(f"Using {torch.cuda.DEVICE_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)
    
    model = model.to(DEVICE)

    # Get decoder
    decoder = get_CTC_decoder(blank_id=0, **configs["decoder"])

    # Loss, optimizer, and scheduler
    training_configs = configs["training"]
    error_names = training_configs['error_names']
    if 'resume_checkpoint' in training_configs:
        model.load_state_dict(torch.load(training_configs['resume_checkpoint']), strict=False)
    max_epoch = training_configs['max_epoch']
    learning_rate = float(training_configs['learning_rate'])
    warm_up_steps = training_configs['warm_up_steps']
    es_monitor = training_configs['early_stop_monitor']
    es_patience = training_configs['early_stop_patience']
    reduce_lr_steps = training_configs['reduce_lr_steps'] if 'reduce_lr_steps' in training_configs else None
    epoch_0 = training_configs['continue_epoch'] if 'continue_epoch' in training_configs else 0
    
    loss_function = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda param : param.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
        )
    
    lr_lambda = lambda step : get_lr_lambda(step, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    early_stop = EarlyStopping(mode='min', patience=es_patience)
    
    stop_flag = False
    for epoch in range(epoch_0, epoch_0 + max_epoch + 1):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for dataloader_name in val_dataloaders:
            statistics = test_loop(
                dataloader=val_dataloaders[dataloader_name],
                model=model,
                dataloader_name=dataloader_name,
                error_names=error_names,
                statistics_dir=statistics_dir,
                epoch=epoch,
                decoder=decoder,
                codec=codec,
                val=True,
                )
            if dataloader_name == es_monitor:
                if early_stop(statistics['errors'][0]):
                    best_epoch = epoch - early_stop.patience
                    stop_flag = True
        if stop_flag:
            break
        if epoch == epoch_0 + max_epoch:
            best_epoch = epoch
            break
        for dataloader_name in train_dataloaders:
            train_loop(
                dataloader=train_dataloaders[dataloader_name], 
                model=model,
                statistics_dir=statistics_dir,
                epoch=epoch + 1,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        
        save_checkpoints(model, checkpoints_dir, epoch + 1)

        # scheduler.step(error_wer)
    print("Testing with model on best validation error.")
    load_checkpoints(model, checkpoints_dir, best_epoch)
    test_statistics = {}
    for dataloader_name in test_dataloaders:
        statistics_dict = test_loop(
            dataloader=test_dataloaders[dataloader_name],
            model=model,
            dataloader_name=dataloader_name,
            error_names=error_names,
            statistics_dir=None,
            epoch=None,
            decoder=decoder,
            codec=codec,
            val=False,
            )
        test_statistics[dataloader_name] = statistics_dict
    print(test_statistics)
    test_path = os.path.join(statistics_dir, "test_statistics.pkl")
    pickle.dump(test_statistics, open(test_path, 'wb'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--workspace", type=str, default="/n/work1/deng/workspaces/", help="Directory of workspace.")
    parser.add_argument("--config_yaml", type=str, required=True, help="User defined config file.")
    args = parser.parse_args()

    main(args)
