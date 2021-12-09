#!/bin/bash/
# pytorch0.4.0
# test scripts
# No self-ensemble, use different testsets (Set5, Set14, B100, Urban100, Manga109) to reproduce the results in the paper.
# X2
CUDA_VISIBLE_DEVICES=3 python main.py --model MAN --data_test Set5 --scale 2 --patch_size 96  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX2.pt --test_only --save_results
# X3
CUDA_VISIBLE_DEVICES=2 python main.py --model MAN --data_test Set5 --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX3.pt --test_only --save_results
# X4
CUDA_VISIBLE_DEVICES=1 python main.py --model MAN --data_test Set5 --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX4.pt --test_only --save_results

# use self-ensemble
# X2
CUDA_VISIBLE_DEVICES=3 python main.py --model MAN --data_test Set5 --scale 2 --patch_size 96  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX2.pt --test_only --save_results --self_ensemble
# X3
CUDA_VISIBLE_DEVICES=2 python main.py --model MAN --data_test Set5 --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX3.pt --test_only --save_results --self_ensemble
# X4
CUDA_VISIBLE_DEVICES=1 python main.py --model MAN --data_test Set5 --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX4.pt --test_only --save_results --self_ensemble




