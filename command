# train
python3 main.py --model RFSR --scale 3 --save RFSR_x3 --n_resblocks 16 --n_feats 64 --n_columns 4 --epochs 1100 --patch_size 144 --chop --batch_size 16 --dir_data ../../ --data_train DIV2K --data_range 1-800/801-810 --data_test DIV2K --loss "1*L1" --save_results --lr 1e-6 --decay 20000 --gamma 0.6 --load RFSR_x3 --pre_train ../experiment/RFSR_x3/model/model_best.pt

# test
python3 main.py --model RFSR --scale 3 --save RFSR_x3 --n_resblocks 16 --n_feats 64 --n_columns 4 --epochs 1100 --patch_size 144 --chop --batch_size 16 --dir_data ../../ --data_train DIV2K --data_range 1-800/801-810 --data_test DIV2K --loss "1*L1" --save_results --lr 1e-6 --decay 20000 --gamma 0.6 --load RFSR_x3 --pre_train ../experiment/RFSR_x3/model/model_best.pt

# MSRN
python3 main.py --model MSRN --scale 2 --save MSRN_8blocks_x2 --n_resblocks 8 --n_feats 64 --n_columns 4 --epochs 1000 --patch_size 96 --chop --batch_size 16 --dir_data ../../ --data_train DIV2K --data_range 1-800/801-810 --data_test DIV2K --loss "1*L1" --save_results --lr 1e-4 --decay 200-400-600-800 --load MSRN_8blocks_x2 --pre_train ../experiment/MSRN_8blocks_x2/model/model_latest.pt

# OpCounter
python3 opcounter.py --model MSRN --n_resblocks 10 --n_feats 64 --n_columns 4 --save temp --epochs 1000 --scale 4 --patch_size 192 --batch_size 16 --dir_data ../../ --data_train DIV2K --data_range 1-800/801-810 --data_test DIV2K --loss 1*L1 --save_results --lr 1e-4 --decay 200-400-600-800 --cpu