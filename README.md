# WeakFixation

## Overview
This repository is the implementation of **'Weakly Supervised Visual Saliency Prediction (TIP2022)'**.

Qiuxia Lai, Tianfei Zhou, Salman Khan, Hanqiu Sun, Jianbing Shen, Ling Shao.

## Requirements
Create  an anaconda environment:

```commandline
$ conda env create -f environment.yml
```

Activate the environment:

```commandline
$ source activate torch36
$ <run_python_command> # see the examples below
```

## Results Download
Prediction results on **MIT300**, **MIT1003**, **PASCAL-S**, **SALITON-Test**, **TORONTO**, and **DUT-OMRON** can be downloaded from:

Google Drive: <https://drive.google.com/file/d/1CWWv79RYwh1tRY82VtsjZ1N4KxyccTa_/view?usp=sharing>

Baidu Disk: <https://pan.baidu.com/s/1HfZNfNAsKqzJRAbX4WU7eA>  (password:`s216`)

Our evaluation code is adapted from [this matlab tool](https://github.com/cvzoya/saliency/tree/master/code_forMetrics).
A python version can be found at [this repository](https://github.com/tarunsharma1/saliency_metrics).

## Datasets Preparation

The bounding boxes of each dataset are generated using [EdgeBox](https://github.com/pdollar/edges).

**Training**: The images are downloaded from the [official website](https://cocodataset.org). The bounding boxes of `MS-COCO` can be downloaded from [Baidu Disk Link (train)](https://pan.baidu.com/s/11Jb4P0h_tECbmkypVri2oA) (password:`5ecm`) and [Baidu Disk Link (eval)](https://pan.baidu.com/s/11WzSLZ_BxbpuMoxqD96lcg) (password:`qdrg`).

**Testing**: Taking `MIT1003` as an example. You may download the dataset along with the bounding boxes from this [Baidu Disk Link](https://pan.baidu.com/s/1amWZZeNbGVHgSmD9vlsPCg) (password:`ten8`) OR [Google Drive](https://drive.google.com/file/d/1PCuxks2f5Aem8u0GwcbdLP2e11zySLyQ/view?usp=sharing) for a fast try.

The datasets are arranged as:

        DataSets
        |----MS_COCO
             |---train2014
             |---val2014
             |---train2014_eb500
             |---val2014_eb500
        |----MIT1003
             |---ALLSTIMULI
                 |--xxx.jpeg
                 |--xxx.jpeg
                 |--...
             |---ALLFIXATIONMAPS
                 |--...
             |---ALLFIXATIONS
                 |--...
             |---eb500
                 |--xxx.mat
                 |--...
        |----PASCAL-S
        |--...
        



## Testing

### Download weights from:

Google Drive: <https://drive.google.com/file/d/1KxyXNWo_mxPkRo1sf2jHFMB_Jxzc6msY/view?usp=sharing>

Baidu Disk: <https://pan.baidu.com/s/1Mn7U3UTKOVUW7w6WC5w65w> password:`bgft`

### Configure the directories

### Run the following command:

```commandline

```

## Training

Coming soon.

## Citation
If you find this repository useful, please cite the following reference.
```

```

## Contact

Qiuxia Lai: ashleylqx`at`gmail.com
