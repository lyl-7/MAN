#!/bin/bash/
# pytorch 0.3.1
# train scripts
python main.py --model man  --save MAN_Demosaic_F256B32P48 --n_resblocks 32 --n_feats 256 --patch_size 48 --save_results --chop --loss 1*L1  
