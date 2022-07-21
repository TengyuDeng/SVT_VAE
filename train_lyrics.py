import argparse
import torch
import os, pickle, re

from datasets import get_dataloaders
from models import get_model, get_language_model
from utils import *
from train_utils import *

from decoders import get_CTC_decoder, Mycodec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}.")

def mse_loss(reconst, target_features, input_lengths):
    for i in range(len(input_lengths)):

        if i == 0:
            loss = torch.sum((reconst[i, :, :input_lengths[i]] - target_features[i, :, :input_lengths[i]]) ** 2)
        else:
            loss += torch.sum((reconst[i, :, :input_lengths[i]] - target_features[i, :, :input_lengths[i]]) ** 2)

    loss /= torch.sum(input_lengths) * target_features.shape[1]
    return loss

def mae_loss(reconst, target_features, input_lengths):
    for i in range(len(input_lengths)):

        if i == 0:
            loss = torch.sum(torch.abs(reconst[i, :, :input_lengths[i]] - target_features[i, :, :input_lengths[i]]))
        else:
            loss += torch.sum(torch.abs(reconst[i, :, :input_lengths[i]] - target_features[i, :, :input_lengths[i]]))

    loss /= torch.sum(input_lengths) * target_features.shape[1]
    return loss

@train_loop_decorator
def train_loop(data, model, optimizers, schedulers, loss_functions, loss_weights):
    for optimizer in optimizers:
        optimizer.zero_grad()
    # torch.cuda.empty_cache()
    input_features, target_features, pitches, onsets, input_lengths, tatum_frames, texts, text_lengths = data
    input_features = input_features.to(DEVICE)
    target_features = target_features.to(DEVICE)
    
    # Forward lyrics
    lyrics_logits = model['lyrics'](input_features)
    lyrics_logprobs = torch.nn.functional.log_softmax(lyrics_logits, dim=-1)
    loss_lyrics = loss_functions['lyrics'](lyrics_logprobs, texts, downsample_length(input_lengths, model['lyrics'].down_sample), text_lengths)
    

    # # The reconstruction loss

    # reconst = adjust_shape(model['reconst'](lyrics_logprobs).transpose(-1, -2), target_features)
    
    # loss_reconst = mse_loss(reconst, target_features, input_lengths)
    
    # loss = loss_weights['lyrics'] * loss_lyrics + loss_weights['reconst'] * loss_reconst
    
    loss = loss_lyrics

    # Backprop
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    for scheduler in schedulers:
        scheduler.step()
    
    return loss.item()

@test_loop_decorator
def test_loop(data, model, dataloader_name, decoder, codec):
    input_features, target_features, pitches, onsets, input_lengths, tatum_frames, texts, text_lengths = data
    input_features = input_features.to(DEVICE)
    target_features = target_features.to(DEVICE)
    
    with torch.no_grad():
        lyrics_logits = model['lyrics'](input_features)
        lyrics_probs = torch.nn.functional.softmax(lyrics_logits, dim=-1)
        lyrics_logprobs = torch.nn.functional.log_softmax(lyrics_logits, dim=-1)
        
        # The reconstruction loss

        # reconst = adjust_shape(model['reconst'](lyrics_logprobs).transpose(-1, -2), target_features)
        # loss_reconst = mse_loss(reconst, target_features, input_lengths).detach().cpu().item()
    loss_reconst = 0.
    
    output_lengths = downsample_length(input_lengths, model['lyrics'].down_sample)
    start = 0
    batch_size = len(text_lengths)
    # predicts = decoder.decode(lyrics_probs.transpose(0, 1))
    error_per = 0
    for i in range(batch_size):
        text = texts[start: start + text_lengths[i]]
        # text = codec.decode(texts[start: start + text_lengths[i]])
        predict = decoder.decode_one(lyrics_probs[:output_lengths[i], i, :])
        error_per += mycer(text, predict)
        start += text_lengths[i]
    error_per /= batch_size

    # predicts = decoder.decode(lyrics_probs, input_lengths)
    
    # try:
    #     error_wer = wer(texts, predicts)
    #     error_cer = cer(texts, predicts)
    # except ValueError:
    #     texts = [text + "." for text in texts]
    #     error_wer = wer(texts, predicts)
    #     error_cer = cer(texts, predicts)
    # return error_wer, error_cer

    return error_per, loss_reconst

def main(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    # Read config.yaml
    configs = read_yaml(config_yaml)

    # Get directories
    checkpoints_dir, statistics_dir = get_dirs(workspace, configs['task'], config_yaml)

    # Construct dataloaders
    train_dataloaders, val_dataloaders, test_dataloaders = get_dataloaders(configs["dataloaders"])
    # Get model
    model = {}
    for model_name in configs["model"]:
        if model_name == "language_model":
            model[model_name] = get_language_model(**configs["model"][model_name])
        else:
            model[model_name] = get_model(model_name, **configs["model"][model_name])
        model[model_name] = model[model_name].to(DEVICE)
        os.makedirs(os.path.join(checkpoints_dir, model_name), exist_ok=True)

    decoder = get_CTC_decoder(blank_id=0, **configs["decoder"])
    codec = Mycodec(target_type=configs["decoder"]['target_type'])
    # if torch.cuda.DEVICE_count() > 1:
    #     print(f"Using {torch.cuda.DEVICE_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)

    # Loss, optimizer, and scheduler
    training_configs = configs["training"]
    error_names = training_configs['error_names']
    if 'resume_checkpoint' in training_configs:
        for model_name in training_configs['resume_checkpoint']:
            model[model_name].load_state_dict(torch.load(training_configs['resume_checkpoint'][model_name]), strict=False)
    max_epoch = training_configs['max_epoch']
    learning_rate = float(training_configs['learning_rate'])
    warm_up_steps = training_configs['warm_up_steps']
    es_monitor = training_configs['early_stop_monitor']
    es_patience = training_configs['early_stop_patience']
    es_mode = training_configs['early_stop_mode']
    es_index = training_configs['early_stop_index']
    loss_weights = training_configs['loss_weights']
    reduce_lr_steps = training_configs['reduce_lr_steps'] if 'reduce_lr_steps' in training_configs else None
    epoch_0 = training_configs['continue_epoch'] if 'continue_epoch' in training_configs else 0
    
    loss_functions = {
    'pitch': CrossEntropyLossWithProb().to(DEVICE),
    'onset': torch.nn.BCEWithLogitsLoss().to(DEVICE),
    'lyrics': torch.nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE),
    }
    # loss_functions = set_weights(loss_functions, training_configs, device=DEVICE)
    
    optimizers = []
    for model_name in model:
        if model_name != "rendering":
            optimizers.append(torch.optim.Adam(
        filter(lambda param : param.requires_grad, model[model_name].parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
        ))
    # optimizers = [
    # torch.optim.Adam(
    #     filter(lambda param : param.requires_grad, model[model_name].parameters()),
    #     lr=learning_rate,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=0.0,
    #     amsgrad=True,
    #     ) for model_name in model
    # ]
    
    lr_lambda = lambda step : get_lr_lambda(step, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps)
    schedulers = [
    torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for optimizer in optimizers
    ]
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    early_stop = EarlyStopping(mode=es_mode, patience=es_patience)
    
    stop_flag = False
    for epoch in range(epoch_0, epoch_0 + max_epoch + 1):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for dataloader_name in val_dataloaders:
            statistics = test_loop(
                dataloader=val_dataloaders[dataloader_name],
                model=model,
                dataloader_name=dataloader_name,
                decoder=decoder,
                codec=codec,
                error_names=error_names,
                statistics_dir=statistics_dir,
                epoch=epoch,
                val=True,
                )
            if dataloader_name == es_monitor:
                if early_stop(statistics['errors'][es_index]):
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
                optimizers=optimizers,
                schedulers=schedulers,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
            )
        
        for model_name in model:
            save_checkpoints(model[model_name], os.path.join(checkpoints_dir, model_name), epoch + 1)

        # scheduler.step(error_wer)
    print("Testing with model on best validation error.")
    for model_name in model:
        load_checkpoints(model[model_name], os.path.join(checkpoints_dir, model_name), best_epoch)
    test_statistics = {}
    for dataloader_name in test_dataloaders:
        statistics_dict = test_loop(
            dataloader=test_dataloaders[dataloader_name],
            model=model,
            dataloader_name=dataloader_name,
            decoder=decoder,
            codec=codec,
            error_names=error_names,
            statistics_dir=None,
            epoch=None,
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
