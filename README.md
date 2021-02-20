# Domain Adaptive Faster R-CNN in Detectron 

This is a Caffe2 implementation of 'Domain Adaptive Faster R-CNN for Object Detection in the Wild', implemented by Haoran Wang(whrzxzero@gmail.com). The original paper can be found [here](https://arxiv.org/pdf/1803.03243.pdf). This implementation is built on [Detectron](https://github.com/facebookresearch/Detectron) @ [5ed75f9](https://github.com/facebookresearch/Detectron/tree/5ed75f9d672b3c78b7da92d9b2321d04f33a7ccc).

If you find this repository useful, please cite the oringinal paper:

```
@inproceedings{chen2018domain,
  title={Domain Adaptive Faster R-CNN for Object Detection in the Wild},
      author =     {Chen, Yuhua and Li, Wen and Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
      booktitle =  {Computer Vision and Pattern Recognition (CVPR)},
      year =       {2018}
  }
```

and Detectron:

```
@misc{Detectron2018,
  author =       {Ross Girshick and Ilija Radosavovic and Georgia Gkioxari and
                  Piotr Doll\'{a}r and Kaiming He},
  title =        {Detectron},
  howpublished = {\url{https://github.com/facebookresearch/detectron}},
  year =         {2018}
}
```
## Installation

Please follow the instruction in [Detectron](https://github.com/facebookresearch/Detectron) to install and use Detectron-DomainAdaptive-Faster-RCNN.

## Usage Example

An example of adapting from **Sim10k** dataset to **Cityscapes** dataset is provided:
1. Download the Cityscapes datasets from [here](https://www.cityscapes-dataset.com/downloads/) and Sim10k datasets from [here](https://fcav.engin.umich.edu/sim-dataset).

2. Convert the labels of Cityscapes datasets and Sim10k datasets to coco format using the scripts 'tools/convert_cityscapes_to_caronly_coco.py' and 'tools/convert_sim10k_to_coco.py'.

3. Convert ImageNet-pretrained VGG16 Caffe model to Detectron format with 'tools/pickle_caffe_blobs.py' or use my converted VGG16 model in [here](https://drive.google.com/file/d/1nlo6TJt0AwlPIkG8e3aXjdVNdmaLOytg/view?usp=sharing) 

4. Train the Domain Adaptive Faster R-CNN:
    ```Shell
    cd $DETECTRON
    python2 tools/train_net.py --cfg configs/da_faster_rcnn_baselines/e2e_da_faster_rcnn_vgg16-sim10k.yaml
    
5. Test the trained model:
    ```Shell
    cd $DETECTRON
    python2 tools/test_net.py --cfg configs/da_faster_rcnn_baselines/e2e_da_faster_rcnn_vgg16-sim10k.yaml TEST.WEIGHTS /<path_to_trained_model>/model_final.pkl NUM_GPUS 1

### Pretrained Model & Results

The best results for different adaptation are reported. Due to the instable nature of adversarial training, the best models are obtained through a model selection on a randomly picked mini validation set.

|                  | image                | instsnace            | consistency          | car AP| Pretrained models|
|------------------|:--------------------:|:--------------------:|:--------------------:|:-----:|:---:|
| Faster R-CNN     |                      |                      |                      | 32.58 ||
| DA Faster R-CNN  |✓                     |                      |                      | 38.60 |[model](https://polybox.ethz.ch/index.php/s/sSasYjKd2mZOiGL)| 
| DA Faster R-CNN  |                      |✓                     |                      | 35.55 |[model](https://polybox.ethz.ch/index.php/s/rIsd4rup5u35Ym1)|
| DA Faster R-CNN  |✓                     |✓                     |                      | 39.23 |[model](https://polybox.ethz.ch/index.php/s/He0YvLrAhWB1Amc)| 
| DA Faster R-CNN  |✓                     |✓                     |✓                     | 40.01 |[model](https://polybox.ethz.ch/index.php/s/apeC2oZ1iPD5dgw)|

## Other Implementation
[da-faster-rcnn](https://github.com/yuhuayc/da-faster-rcnn) based on Caffe. (original code by paper authors)

[Domain-Adaptive-Faster-RCNN-PyTorch](https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch) based on PyTorch and maskrcnn-benchmark.
