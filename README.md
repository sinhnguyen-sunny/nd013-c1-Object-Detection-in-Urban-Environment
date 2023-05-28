# nd013-c1-Object-Detection-in-Urban-Environment

### Project overview

The objective of this project is to utilize the Tensorflow object detection apis to conduct object detection on camera data extracted from the [Waymo](waimo.com/open) Urban driving dataset. we aim to analyze the camera data and identify various objects present in the driving scene, thereby enhancing the perception abilities of autonomous systems.

### Set up

The experiments were performed on the Udacity GPU workspace in which data was downloaded from the Waymo's Google Cloud bucket and split into "train", "test", "val" in the directory "/home/workspace/data/"

### Dataset

#### Dataset analysis

Initial Dataset analysis was performed in the `Exploratory Data Analysis.ipynb` jupyer notebook. The dataset, which contains groundtruth bounding boxes for pedestrians, cyclists and cars, is somewhat challenging. The Driving scenarios are quite diverse: different places, weather conditions as well ifferent times of the day.

![Dataset1](media/dataset.png 'Samples from Waymo Open Dataset')

Moreover, additional data analysis show that the dataset in not well balanced as well: the classes are not well distributed with the majority of of objects in the image being vehicles, with very few cyclists!

![Dataset2](media/eda1.png 'Classes distribution in Waymo Open Dataset')

Object quantity in each image is also not homogeneous at all: with images from highway driving containing only up to 20 objects while crowded city scenaris can contain as well up to 70 objects.

![Dataset2](media/eda2.png 'Objects in Waymo Open Dataset')

### Download pretrained model & Edit the config file

As explaining during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

Follow the steps below:

```
cd /home/workspace/experiments/pretrained_model/

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

Next, editing the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:

```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 4 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```

A new config file has been created, `pipeline_new.config`.

Finally, moving the new config file into the directory "/home/workspace/experiments/reference/"

```
experiments
├── augmentations_1
├── augmentations_2
├── augmentations_3
├── pretrained_model
│  └── ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
└── reference
```

### Training & Evaluation

#### Reference experiment

Launch the training process:

- a training process:

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```

- monitor the training, you can launch a tensorboard instance by running

```
python -m tensorboard.main --logdir experiments/reference/
```

Once the training is finished, launch the evaluation process. Launching evaluation process in parallel with training process will lead to OOM error in the workspace

- an evaluation process:

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

- evaluation results:
  Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.019
  Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.047
  Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.011
  Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
  Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.085
  Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.094
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.007
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.028
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.056
  Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
  Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.202
  Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.250

```

```

### Improve on the reference

#### Run experiments

I run 3 experiments
Each experiment reported in this project can be replicated using the preconfigured `pipeline.config` file in the experimet's subdirectory.

- Model training (while <EXPERIMENT_DIR> is: "augmentations_1", "augmentations_2", "augmentations_3")

```
python experiments/model_main_tf2.py --model_dir=experiments/<EXPERIMENT_DIR>/ --pipeline_config_path=experiments/<EXPERIMENT_DIR>/pipeline_new.config
```

Model evaluation:

```
python experiments/model_main_tf2.py --model_dir=/app/project/experiments/<EXPERIMET_DIR>/ --pipeline_config_path=/app/project/experiments/<EXPERIMENT_DIR>/pipeline_new.config --checkpoint_dir=/app/project/experiments/<EXPERIMENT_DIR>/
```

Tensorboard plots:

```
python -m tensorboard.main --logdir experiments/<EXPERIMENT_DIR>/
```

#### Evaluating results of experiments

1. Experiment 1:

- Training configuration:

  - Augmented training data: Randomly adjust the brightness and contrast
  - Optimizer: momentum
  - Learning Rate: warmup + decaying 0.04

- Evaluating results:
  Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.023
  Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.051
  Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.017
  Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
  Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.106
  Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.101
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.008
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.030
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.055
  Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
  Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.210
  Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.203

2. Experiment 2:

- Training configuration:

  - Augmented training data: Different lighting conditions
  - Optimizer: momentum
  - Learning Rate: warmup + decaying 0.04

- Evaluating results:

  Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.046
  Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.098
  Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.037
  Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.014
  Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.175
  Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.223
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.012
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.051
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.081
  Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
  Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.282
  Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.290

3. Experiment 3:

- Training configuration:

  - Augmented training data: Randomly scale and distort the images
  - Optimizer: adam
  - Learning Rate: decaying 0.01

- Evaluating results:
  Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.000
  Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.001
  Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.000
  Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
  Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
  Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.003
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.000
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.001
  Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.006
  Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
  Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.033
  Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.034

### Test Results

Finally, the best trained model - Experiment 2 was exported, here is a gif animation of the model on one of the unseen tfrecords:

![Test](animation.gif)
