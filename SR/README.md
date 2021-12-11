# [SR] Mix-order Attention Networks for Image Restoration

["Mix-order Attention Networks for Image Restoration"](https://dl.acm.org/doi/10.1145/3474085.3475205), is published on ACM MM 2021. 


[SR]This code is built on [RCAN(pytorch)](https://github.com/yulunzhang/RCAN) and [RNAN(pytorch)](https://github.com/yulunzhang/RNAN) and tested on Ubuntu 16.04 (Pytorch 0.4.0).

## 1.Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Place the dataset as the following directory structure:
   ```
   -dataset
    -DIV2K
      -DIV2K_train_HR
      -DIV2K_train_LR_bicubic
      -DIV2K_train_DNRGB
      -DIV2K_train_Mosaic
    -benchmark
      -Urban100
        -HR
        -LR_bicubic
        -DN-RGB
        -Mosaic
      -Set5
        ...
   ```
3. Specify '--dir_data' as the HR and LR image path.

### Train the model
You can retrain the model:

1. cd to code.

2. run the following scripts to train the models.

    **You can use scripts in file 'Train_MNAN_scripts' to retrain models for our paper.**

    ```bash
    # X2
    CUDA_VISIBLE_DEVICES=3 python main.py --model MAN --data_test Set5 --scale 2 --patch_size 96  --n_resblocks 32 --n_feats 256 --save MAN_SR_F256B32P48BIX2  --chop  --save_results
    # X3
    CUDA_VISIBLE_DEVICES=2 python main.py --model MAN --data_test Set5 --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --save MAN_SR_F256B32P48BIX3  --chop  --save_results
    # X4
    CUDA_VISIBLE_DEVICES=1 python main.py --model MAN --data_test Set5 --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --save MAN_SR_F256B32P48BIX4  --chop  --save_results
    ```
## Test
### Quick start
1. Download models for our paper and place them in '/MAN/SR/experiment/model'.

    All the models can be downloaded from [modelzoo](https://drive.google.com/drive/folders/1wj4iHHphBwY42kFQu_QNEZtYx5vk5wxk?usp=sharing).

2. Cd to 'MAN/SR/code', run the following scripts.

    **You can use scripts in file 'Test_MAN_scripts' to produce results for our paper.**

    ```bash
    # use different testsets (Set5, Set14, B100, Urban100, Manga109) to reproduce the results in the paper.
    # X2
    python main.py --model MAN --data_test Set5 --scale 2 --patch_size 96  --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX2.pt --test_only --save_results
    # X3
    python main.py --model MAN --data_test Set5 --scale 3 --patch_size 144 --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX3.pt --test_only --save_results
    # X4
    python main.py --model MAN --data_test Set5 --scale 4 --patch_size 192 --n_resblocks 32 --n_feats 256 --save Test_MAN  --test_only --chop --pre_train ../experiment/model/MAN_SR_F256B32P48BIX4.pt --test_only --save_results
    ```



## Results
Some of the test results can be [downloaded](https://drive.google.com/drive/folders/13dirpQ6a68FsVLSBj9L3tMWvIlkVAK7Z?usp=sharing).

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
