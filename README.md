# Recommended Environment

Anaconda Environment: Download and install from [Anaconda](https://www.anaconda.com/).

```
conda create --name deeplab python=3.10 -y
```

Activate the environment using the command:

```
conda activate deeplab`
```

Install PyTorch from [PyTorch](https://pytorch.org/get-started/locally/)

```
pip install numpy scikit-learn opencv-python pillow 

git clone git@gitlab.lrz.de:av2.0/aerial_imagery/lane-marking-detection.git

cd lane-marking-detection/DeepLabV3Plus/pytorch-deeplab-xception/
``` 


# Dataset preparation

Training Dataset used: [skyscapes](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes)

Download the datsaset and copy it to the folder `skyscapes`

### Skyscapes dataset folder structure
```
├── DeepLabV3Plus
├── skyscapes
│   ├── test
│   │   ├──images
│   ├── train
│   │   ├──images
│   │   ├──labels
│   ├── val
│   │   ├──images
│   │   ├──labels
```
Trained model was then tested on the aerial images obtained from [Bavarian open data](https://geodaten.bayern.de/opengeodata/)

Next, to obtain the train dataset augmentation

```
cd DeepLabV3Plus/pytorch-deeplab-xception-master/dataloaders/datasets
```

Update the skyscapes dataset base directory path and run this to get the augmented dataset

```
python skyscapes_data_aug.py
```


# Training

```
cd DeepLabV3Plus/pytorch-deeplab-xception-master/
```

change the hyperparameters as needed in train.py and run

```
python train.py
```


## Changing pretrained weights of the backbone
Xception and ResNet101 are two backbones used in this model.

Download the pretrained weights from the following Model Zoo.
- [PyTorch Model Zoo](https://pytorch.org/vision/main/models.html)
- [Cadene Model Zoo](https://data.lip6.fr/cadene/pretrainedmodels/)

Copy them to `/DeepLabV3Plus/pytorch-deeplab-xception-master/modeling/backbone/`

For Xception backbone

`cd modeling/backbone/`

Open xcpetion.py, change the weights path as required.

Same process for ResNet101 backbone.


## Bavarian photo resize

Update the parameters in DeepLabV3Plus/pytorch-deeplab-xception/bav_photo_resize.py

run 
`python bav_photo_resize.py`


## Inference

`cd DeepLabV3Plus/pytorch-deeplab-xception-master/`

update the directory paths in predict.py

run `python predict.py`


### Calculation of number of pixels per class

*Used only for multi-class dataset

`cd DeepLabV3Plus/pytorch-deeplab-xception-master/`

run `python calc_num_pixels.py`

# Results
### Training Results on Skyscapes dataset
<div>
	<img src="/home/upendra/lane-marking-detection/DeepLabV3Plus/assets/ValActual.png" width="33%" />
	<p><em>Sample Scene</em></p>
    <img src="/home/upendra/lane-marking-detection/DeepLabV3Plus/assets/ValGT.png" width="33%" />
	<p><em>Ground Truth</em></p>
</div>
<div>
	<img src="/home/upendra/lane-marking-detection/DeepLabV3Plus/assets/ValXception.png" width="33%" />
	<p><em>Inference 1 (Xception Backbone)</em></p>
    <img src="/home/upendra/lane-marking-detection/DeepLabV3Plus/assets/ValResNet101.png" width="33%" />
	<p><em>Inference 2 (ResNet101 Backbone)</em></p>
</div>


### Test Results on Bavarian Open Data
<div>
	<img src="/home/upendra/lane-marking-detection/DeepLabV3Plus/assets/TestImageMunich.png" width="33%" />
	<p><em>Sample Scene</em></p>
    <img src="/home/upendra/lane-marking-detection/DeepLabV3Plus/assets/TestResultMultiClass.png" width="33%" />
	<p><em>Inference 1 - Multi-Class</em></p>
    <img src="/home/upendra/lane-marking-detection/DeepLabV3Plus/assets/TestResultBinaryClass.png" width="33%" />
	<p><em>Inference 2 - Binary-Class</em></p>
</div>

# References and links
- **DeepLabv3+ Pytorch implementation**: [DeepLabV3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
- **Inference**: [predict](https://github.com/alpemek/aerial-segmentation)
- **TUM Course - Introduction to Deep Learning**: [IN2346](https://dvl.in.tum.de/teaching/i2dl-ss22/)