# Mix-order Attention Networks for Image Restoration

["Mix-order Attention Networks for Image Restoration"](https://dl.acm.org/doi/10.1145/3474085.3475205), is published on ACM MM 2021. 


This code is built on [RCAN(pytorch)](https://github.com/yulunzhang/RCAN) and [RNAN(pytorch)](https://github.com/yulunzhang/RNAN) and tested on Ubuntu 16.04 (Pytorch 0.4.0).

## Introdcution
Convolutional neural networks (CNNs) have obtained great success
in image restoration tasks, like single image denoising, demosaicing,
and super-resolution. However, most existing CNN-based methods
neglect the diversity of image contents and degradations in the
corrupted images and treat channel-wise features equally, thus
hindering the representation ability of CNNs. To address this issue,
we propose a deep mix-order attention networks (MAN) to extract
features that capture rich feature statistics within networks. Our
MAN is mainly built on simple residual blocks and our mix-order
channel attention (MOCA) module, which further consists of feature
gating and feature pooling blocks to capture different types of
semantic information. With our MOCA, our MAN can be flexible to
handle various types of image contents and degradations. Besides,
our MAN can be generalized to different image restoration tasks,
like image denoising, super-resolution, and demosaicing. Extensive
experiments demonstrate that our method obtains favorably against
state-of-the-art methods in terms of quantitative and qualitative
metrics.

![framework](/Figs/MAN_framework.PNG)


## Tasks
### Super-resolution
![SR_PSNR](/Figs/SR_PSNR.PNG)
![SR_Visual](/Figs/SR_visual.PNG)
More details at [SR](https://github.com/lyl-7/MAN/tree/main/SR/).

### Color Image Denoising
![DN_PSNR](/Figs/DN_PSNR.PNG)
![DN_Visual](/Figs/DN_visual.PNG)
More details at [DN_RGB](https://github.com/lyl-7/MAN/tree/main/DN_RGB/).

### Image Demosaicing
![Demosaic_PSNR](/Figs/Demosaic_PSNR.PNG)
![Demosaic_Visual](/Figs/Demosaic_visual.PNG)
More details at [Demosaic](https://github.com/lyl-7/MAN/tree/main/Demosaic/).



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
