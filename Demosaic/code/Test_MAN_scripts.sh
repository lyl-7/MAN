#!/bin/bash
# pytorch 0.4.0
# test scripts
# use different testsets (Mcm18, Kodak24, CBSD68, Urban100) to reproduce the results in the paper.
CUDA_VISIBLE_DEVICES=0 python main.py --model man --data_test McM      --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 
CUDA_VISIBLE_DEVICES=0 python main.py --model man --data_test Kodak24  --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 
CUDA_VISIBLE_DEVICES=0 python main.py --model man --data_test BSD68    --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 
CUDA_VISIBLE_DEVICES=0 python main.py --model man --data_test Urban100 --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 