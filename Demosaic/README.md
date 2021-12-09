# [Demosaic] Mix-order Attention Networks for Image Restoration

["Mix-order Attention Networks for Image Restoration"](https://dl.acm.org/doi/10.1145/3474085.3475205), is published on ACM MM 2021. 


[Demosaic]This code is built on [RCAN(pytorch)](https://github.com/yulunzhang/RCAN) and [RNAN(pytorch)](https://github.com/yulunzhang/RNAN) and tested on Ubuntu 16.04 (Pytorch 0.4.0).

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
   python main.py --model man  --save MAN_Demosaic_F256B32P48 --n_resblocks 32 --n_feats 256 --patch_size 48 --save_results --chop --loss 1*L1  
    ```
## Test
### Quick start
1. Download models for our paper and place them in '/MAN/Demosaic/experiment/model'.

    All the models can be downloaded from [modelzoo](https://drive.google.com/drive/folders/1wj4iHHphBwY42kFQu_QNEZtYx5vk5wxk?usp=sharing).

2. Cd to 'MAN/Demosaic/code', run the following scripts.

    **You can use scripts in file 'Test_MAN_scripts' to produce results for our paper.**

    ```bash
    # different testsets (Mcm18, Kodak24, CBSD68, Urban100) to reproduce the results in the paper.
    python main.py --model man --data_test McM      --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 
    python main.py --model man --data_test Kodak24  --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 
    python main.py --model man --data_test BSD68    --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 
    python main.py --model man --data_test Urban100 --save Test_MAN  --n_resblocks 32 --n_feats 256 --test_only  --chop --pre_train ../experiment/model/MAN_Demosaic_F256B32P48.pt 
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
