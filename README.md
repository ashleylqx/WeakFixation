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
<!--
A python version can be found at [this repository](https://github.com/tarunsharma1/saliency_metrics).
-->

## Datasets Preparation

### For training
We use `MS-COCO 2014 - train` for training, and a subset of `MS-COCO - val` (i.e., the `val` of `SALICON`) for evaluation.

**Images** available at the [official website](https://cocodataset.org). 

**Bounding boxes**: [Baidu Disk Link (train)](https://pan.baidu.com/s/11Jb4P0h_tECbmkypVri2oA) (password:`5ecm`) and [Baidu Disk Link (eval)](https://pan.baidu.com/s/11WzSLZ_BxbpuMoxqD96lcg) (password:`qdrg`). 
OR generated using [EdgeBox](https://github.com/pdollar/edges).

**Saliency prior maps**: [Baidu Disk Link (train)](https://pan.baidu.com/s/1WrKHsTyMjygf9MUtxk4cqQ) (password:`u8h2`) and [Baidu Disk Link (val)](https://pan.baidu.com/s/1os1ryvubS_p0ueRr3tPUDA) (password:`qxxe`). 
OR generated using the [official matlab code](http://www.houxiaodi.com/publication.html) of *Dynamic visual attention: Searching for coding length increments, NeurIPS 2008*.




### For testing
Taking `MIT1003` as an example. 
You may download the dataset along with the bounding boxes from this [Baidu Disk Link](https://pan.baidu.com/s/1amWZZeNbGVHgSmD9vlsPCg) (password:`ten8`) OR [Google Drive](https://drive.google.com/file/d/1PCuxks2f5Aem8u0GwcbdLP2e11zySLyQ/view?usp=sharing) for a fast try.


### Dataset arrangement
The datasets are arranged as:

        DataSets
        |----MS_COCO
             |---train2014
             |---val2014
             |---train2014_eb500
             |---val2014_eb500
             |---train2014_nips08
             |---val2014_nips08
        |----SALICON
             |---images
                |---train
                |---val
                |---test
             |---fixations
                |---train
                |---val
                |---test
             |---maps
                |---train
                |---val
             |---eb500
                |---train
                |---val
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
        



## Testing

### Download weight `model_best.pt` from:

Google Drive: <https://drive.google.com/file/d/1KxyXNWo_mxPkRo1sf2jHFMB_Jxzc6msY/view?usp=sharing>

Baidu Disk: <https://pan.baidu.com/s/1Mn7U3UTKOVUW7w6WC5w65w> password:`bgft`

### Configurations
1. Set the `base_path` in `config.py` be the parent folder of `DataSets`.
2. Put the downloaded `model_best.pt` in `<code_path>/WF/Models/best/`.
3. Download weight `resnet50.pth` from [pytorch github](https://download.pytorch.org/models/resnet50-0676ba61.pth) and put it in `<code_path>/Weights/`.

### Prediction
Run 
```commandline
python main.py --phase test --model_name best --bestname model_best.pt --batch-size 2
```

The saliency prediction results will be saved in `<code_path>/WF/Preds/MIT1003/<model_name>_multiscale/`.

Please evaluate the prediction results using the above mentioned [matlab tool](https://github.com/cvzoya/saliency/tree/master/code_forMetrics).


## Training

Coming soon.

## Citation
If you find this repository useful, please consider citing the following reference.
```
@ARTICLE{lai2022weakly,
    title={Weakly supervised visual saliency prediction},
    author={Qiuxia Lai and Tianfei Zhou and Salman Khan and Hanqiu Sun and Jianbing Shen and Ling Shao},
    journal={IEEE Trans. on Image Processing},
    year={2022}
}
```

## Contact

Qiuxia Lai: ashleylqx`at`gmail.com | qxlai`at`cuc.edu.cn
