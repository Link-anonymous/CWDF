# Robust RGB-D Salient Object Detection via Channel-wise Deep Fusion

This repository contains code for paper "Robust RGB-D Salient Object Detection via Channel-wise Deep Fusion".
* The "Ours" folder contains the source code of our network, which utilizes original RGB and depth maps to calculate the saliency maps.
* The "Ours+" folder contains the source code of our improved network, which replaces the original depth maps with the saliency predictions generated above.

## Prerequisites
| [Caffe](https://github.com/BVLC/caffe) | [CUDA10](https://developer.nvidia.com/cuda-downloads) | [CUDNN7.5](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) | [Matlab2016b](https://www.mathworks.com/) |

## Testing
1. Download Testing sets from the [anonymous link](abc) and extract it to `./CWDF/Dataset/Test/`

2. Download our pretrained model from the [anonymous link](abc)
* "Ours.caffemodel" should put into the folder "Ours"
* "Ours+.caffemodel" should put into the folder "Ours+"

3. Run the test demo
* Firstly, run "test.m" in "Ours" folder to generate the saliency maps. There are one output folders as "fuse" for the final results.
* Secondly, run "test+.m" in "Ours+" folder to obtain the improved results. Similarly, there are one output folders as "fuse+" for the final improved results.

## Training
1. Download training data from the [anonymous link](abc), and extract it to `./CWDF/Dataset/Train/`
2. Download [initial VGG16 model](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) and put it into `./CWDF/Model/`
3. Start to train our network with `sh ./CWDF/ours/finetune.sh`.
4. Calculate the saliency predictions refer to "Testing 3.1".
5. Start to train our improved network with `sh ./CWDF/ours+/finetune.sh`.
