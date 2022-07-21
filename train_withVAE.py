import argparse
import torch
import os, pickle, re

from datasets import get_dataloaders
from models import get_model, get_language_model
from utils import *
from train_utils import *
from vae import VAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}.")

@train_loop_decorator
def train_loop(data, model, optimizers, schedulers, loss_functions, loss_weights, supervised=False):
    for optimizer in optimizers:
        optimizer.zero_grad()
    # torch.cuda.empty_cache()
    input_features, target_features, pitches, onsets, input_lengths, tatum_frames, _, _ = data
    input_features = input_features.to(DEVICE)
    target_features = target_features.to(DEVICE)
    pitches = pitches.transpose(-1, -2).to(DEVICE)
    onsets = onsets.to(DEVICE)
    tatum_frames = tatum_frames.to(DEVICE)
    # Forward melody
    pitches_logits, onsets_logits = model['melody'](input_features, tatum_frames)
    loss_pitch = loss_functions['pitch'](pitches_logits, pitches)
    loss_onset = loss_functions['onset'](onsets_logits, onsets)
    
    if loss_weights['reconst'] != 0.:
    # The reconstruction loss
        if supervised:
            pitches_pre = pitches
            onsets_pre = onsets
        else:
            pitches_pre = hardmax(pitches_logits, dim=-1)
            onsets_pre = hardmax_bernoulli(onsets_logits, threshold=0.5)
        pitches_rendered = model['rendering'](pitches_pre[..., :-1])
        # loss_reconst = torch.mean((target_features != 0) * (target_features - reconst) ** 2)
        # loss_reconst = torch.mean(((target_features - reconst)[target_features != 0]) ** 2)    
        loss_reconst = loss_functions['reconst'](
            pitches_rendered, onsets_pre,
            target_features,
            input_lengths,
            tatum_frames,
            supervised=supervised,
        )

        if supervised:
            loss = loss_reconst
        else:
            loss = loss_weights['pitch'] * loss_pitch + loss_weights['onset'] * loss_onset + loss_weights['reconst'] * loss_reconst
            
    else:
        loss = loss_weights['pitch'] * loss_pitch + loss_weights['onset'] * loss_onset
    # Backprop
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    for scheduler in schedulers:
        scheduler.step()
    
    return loss.item()

@test_loop_decorator
def test_loop(data, model, dataloader_name, loss_functions, supervised=False):
    input_features, target_features, pitches, onsets, input_lengths, tatum_frames, _, _ = data
    input_features = input_features.to(DEVICE)
    target_features = target_features.to(DEVICE)
    tatum_frames = tatum_frames.to(DEVICE)
    with torch.no_grad():
        # pitches_logits: (batch_size, num_tatums, num_pitches)
        # onsets_logits: (batch_size, num_tatums)
        pitches_logits, onsets_logits = model['melody'](input_features, tatum_frames)
        if supervised:
            pitches_pre = pitches.transpose(-1, -2).to(DEVICE)
            onsets_pre = onsets.to(DEVICE)
        else:
            pitches_pre = hardmax(pitches_logits, dim=-1)
            onsets_pre = hardmax_bernoulli(onsets_logits, threshold=0.5)
        pitches_rendered = model['rendering'](pitches_pre[..., :-1])
        loss_reconst = loss_functions['reconst'](
            pitches_rendered, onsets_pre,
            target_features,
            input_lengths,
            tatum_frames,
            supervised=supervised,
        ).detach().cpu().item()

    # Melody evaluation
    pitches = np.argmax(pitches.numpy(), axis=1)
#     onsets = peakpicking(onsets.numpy(), window_size=1, threshold=0.3)
    onsets = onsets.numpy()
    pitches_pre = np.argmax(pitches_pre.cpu().numpy(), axis=-1)
    onsets_pre = onsets_pre.cpu().numpy()
    # onsets_pre = (onsets_prob.cpu().numpy() > 0.5).astype(int)
    note_results = evaluate_notes(pitches, onsets, pitches_pre, onsets_pre, input_lengths, sr=22050, hop_length=256)
    # frame_err = evaluate_frames_batch(pitches, pitches_pre, input_lengths)
    return note_results[2], note_results[5], note_results[8], loss_reconst

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
    supervised = training_configs['supervised']
    reduce_lr_steps = training_configs['reduce_lr_steps'] if 'reduce_lr_steps' in training_configs else None
    epoch_0 = training_configs['continue_epoch'] if 'continue_epoch' in training_configs else 0
    
    loss_functions = {
    'pitch': CrossEntropyLossWithProb().to(DEVICE),
    'onset': torch.nn.BCEWithLogitsLoss().to(DEVICE),
    'reconst': VAE(
        model['z_pre'], model['reconst'], model['language_model'], 
        loss_weights=loss_weights,
        )
    }
    loss_functions = set_weights(loss_functions, training_configs, device=DEVICE) 

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
                loss_functions=loss_functions,
                error_names=error_names,
                statistics_dir=statistics_dir,
                epoch=epoch,
                val=True,
                supervised = [True, False][epoch % 2] if supervised == "mix" else supervised,
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
                supervised = [True, False][epoch % 2] if supervised == "mix" else supervised,
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
            loss_functions=loss_functions,
            error_names=error_names,
            statistics_dir=None,
            epoch=None,
            val=False,
            supervised = [True, False][epoch % 2] if supervised == "mix" else supervised,
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
