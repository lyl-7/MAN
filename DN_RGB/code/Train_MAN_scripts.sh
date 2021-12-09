#!/bin/bash/
# pytorch 0.4.0
# train scripts
# MAN_DN_F64G10P48L2N10
python main.py --model MAN --noise_level 10 --save MAN_DN_F256B32P48N10 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE
# MAN_DN_F64G10P48L2N30
python main.py --model MAN --noise_level 30 --save MAN_DN_F256B32P48N30 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE 
# MAN_DN_F64G10P48L2N50
python main.py --model MAN --noise_level 50 --save MAN_DN_F256B32P48N30 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE
# MAN_DN_F64G10P48L2N70
python main.py --model MAN --noise_level 70 --save MAN_DN_F256B32P48N30 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE


