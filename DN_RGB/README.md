# [DN_RGB] Mix-order Attention Networks for Image Restoration

["Mix-order Attention Networks for Image Restoration"](https://dl.acm.org/doi/10.1145/3474085.3475205), is published on ACM MM 2021. 


[DN_RGB]This code is built on [RCAN(pytorch)](https://github.com/yulunzhang/RCAN) and [RNAN(pytorch)](https://github.com/yulunzhang/RNAN) and tested on Ubuntu 16.04 (Pytorch 0.4.0).

## 1.Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' as the HR and LR image path.

### Train the model
You can retrain the model:

1. cd to code.

2. run the following scripts to train the models.

    **You can use scripts in file 'Train_MNAN_scripts' to retrain models for our paper.**

    ```bash
    # MAN_DN_F64G10P48L2N10
    python main.py --model MAN --noise_level 10 --save MAN_DN_F256B32P48N10 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE
    # MAN_DN_F64G10P48L2N30
    python main.py --model MAN --noise_level 30 --save MAN_DN_F256B32P48N30 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE 
    # MAN_DN_F64G10P48L2N50
    python main.py --model MAN --noise_level 50 --save MAN_DN_F256B32P48N30 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE
    # MAN_DN_F64G10P48L2N70
    python main.py --model MAN --noise_level 70 --save MAN_DN_F256B32P48N30 --n_resblocks 32 --n_feats 256 --patch_size 48  --save_results --chop --loss 1*MSE
    ```
## Test
### Quick start
1. Download models for our paper and place them in '/MAN/DN_RGB/experiment/model'.

    All the models can be downloaded from [modelzoo](https://drive.google.com/drive/folders/1wj4iHHphBwY42kFQu_QNEZtYx5vk5wxk?usp=sharing).

2. Cd to 'MAN/DN_RGB/code', run the following scripts.

    **You can use scripts in file 'Test_MAN_scripts' to produce results for our paper.**

    ```bash
    # use different testsets (Kodak24, CBSD68, Urban100) to reproduce the results in the paper.
    # N=10
    python main.py --model MAN --data_test Kodak24 --noise_level 10  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N10.pt --test_only  --save_results
    # N=30
    python main.py --model MAN --data_test Kodak24 --noise_level 30  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N30.pt --test_only 
    # N=50
    python main.py --model MAN --data_test Kodak24 --noise_level 50  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N50.pt --test_only 
    # N=70
    python main.py --model MAN --data_test Kodak24 --noise_level 70  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_DN_F256B32P48N70.pt --test_only 
    ```



## Results
Some of the test results can be [downloaded]().

## Citation
If the the work or the code is helpful, please cite the following papers
```
@inproceedings{dai2021mix,
  title={Mix-order Attention Networks for Image Restoration},
  author={Dai, Tao and Lv, Yalei and Chen, Bin and Wang, Zhi and Zhu, Zexuan and Xia, Shu-Tao},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={2880--2888},
  year={2021}
}

@inproceedings{lv2021hoca,
  title={HOCA: Higher-Order Channel Attention for Single Image Super-Resolution},
  author={Lv, Yalei and Dai, Tao and Chen, Bin and Lu, Jian and Xia, Shu-Tao and Cao, Jingchao},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1605--1609},
  year={2021},
  organization={IEEE}
}
```
## Acknowledgements
This code is built on [RCAN(pytorch)](https://github.com/yulunzhang/RCAN) and [RNAN(pytorch)](https://github.com/yulunzhang/RNAN). We thank the authors for sharing their codes.
