#!/bin/bash/
# pytorch0.4.0

# X2
CUDA_VISIBLE_DEVICES=3 python main.py --model MAN --data_test Set5 --scale 2 --patch_size 96  --n_resblocks 32 --n_feats 256 --save MAN_SR_F256B32P48BIX2  --chop  --save_results
# X3
CUDA_VISIBLE_DEVICES=2 python main.py --model MAN --data_test Set5 --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --save MAN_SR_F256B32P48BIX3  --chop  --save_results
# X4
CUDA_VISIBLE_DEVICES=1 python main.py --model MAN --data_test Set5 --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --save MAN_SR_F256B32P48BIX4  --chop  --save_results

