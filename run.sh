nohup gpujob -d 7 python3 train_baseline.py --config_yaml="./configs/train_baseline/RWC_pitch_cqt_1_1_2ch_tatum_fold0.yaml" > outputs/20220607/RWC_pitch_cqt_1_1_2ch_tatum_fold0.out &
nohup gpujob -d 6 python3 train_baseline.py --config_yaml="./configs/train_baseline/RWC_pitch_cqt_1_1_2ch_tatum_fold1.yaml" > outputs/20220607/RWC_pitch_cqt_1_1_2ch_tatum_fold1.out &
nohup gpujob -d 5 python3 train_baseline.py --config_yaml="./configs/train_baseline/RWC_pitch_cqt_1_1_2ch_tatum_fold2.yaml" > outputs/20220607/RWC_pitch_cqt_1_1_2ch_tatum_fold2.out &
nohup gpujob -d 4 python3 train_baseline.py --config_yaml="./configs/train_baseline/RWC_pitch_cqt_1_1_2ch_tatum_fold3.yaml" > outputs/20220607/RWC_pitch_cqt_1_1_2ch_tatum_fold3.out &
nohup gpujob -d 3 python3 train_baseline.py --config_yaml="./configs/train_baseline/RWC_pitch_cqt_1_1_2ch_tatum_fold4.yaml" > outputs/20220607/RWC_pitch_cqt_1_1_2ch_tatum_fold4.out &
# nohup gpujob -d 2 python3 train_baseline.py --config_yaml="./configs/train_baseline/RWC_pitch_cqt_1_1_2ch_tatum_mix_fold0.yaml" > outputs/20220607/RWC_pitch_cqt_1_1_2ch_tatum_mix_fold0.out &
# nohup gpujob -d 1 python3 train_baseline.py --config_yaml="./configs/train_baseline/RWC_pitch_cqt_1_1_2ch_tatum_separated_fold0.yaml" > outputs/20220607/RWC_pitch_cqt_1_1_2ch_tatum_separated_fold0.out &
