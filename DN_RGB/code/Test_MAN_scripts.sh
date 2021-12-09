#!/bin/bash/
# pytorch0.4.0
# test scripts
# use different testsets (Kodak24, CBSD68, Urban100) to reproduce the results in the paper.
# N=10
CUDA_VISIBLE_DEVICES=3 python main.py --model MAN --data_test Kodak24 --noise_level 10  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N10.pt --test_only  --save_results
# N=30
CUDA_VISIBLE_DEVICES=2 python main.py --model MAN --data_test Kodak24 --noise_level 30  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N30.pt --test_only 
# N=50
CUDA_VISIBLE_DEVICES=1 python main.py --model MAN --data_test Kodak24 --noise_level 50  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N50.pt --test_only 
# N=70
CUDA_VISIBLE_DEVICES=1 python main.py --model MAN --data_test Kodak24 --noise_level 70  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N70.pt --test_only 

