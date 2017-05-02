# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## MCWPrune

MCWPrune/Caffe is developed by MulticoreWare for squeezing Convolution and InnerProduct layers by pruning the network. Though the repository is designed to work for any model, it is exclusively tested with [VGG-16's Imagenet Classifier](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel), available at [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). This repository is an enhancement of [Dynamic Network Surgery](https://github.com/yiwenguo/Dynamic-Network-Surgery) with additional features and techniques to improve accuracy. Switch to  Prune branch to use the repository.

The code changes are made upon Caffe-rc5 tag. This fork of Caffe has two new additional layers namely SqueezeConvolution and SqueezeInnerProduct for pruning and splicing the Convolution and InnerProduct layers respectively. To enable these layers, use layer type as  SqueezeConvolution and SqueezeInnerProduct in the prototxt and for the layer params - squeeze_convolution_param and squeeze_inner_product_param refer the example below: 
  ~~~~
  squeeze_convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    gamma: 0.00001
    power: 1
    c_rate: 0.3
    iter_stop: 40000
    weight_mask_filler {
      type: "constant"
      value: 1
    }
    bias_mask_filler {
      type: "constant"
      value: 1
    }
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
~~~~
Other tools for processing the caffemodel - applying the masks on weights, printing the compression factor, truncating masks from the caffemodel can be found in miscellaneous_scripts folder.

### Hints for Pruning and Retaining Accuracy


  C_rate controls the rate for pruning for every layer. Higher the c_rate, more the pruning happens. After many trial and error experiments, one might end up finding the optimal c_rate values which yields maximum compression factor with minimal loss in accuracy. Proper analysis of the intermediate results gives the understanding of the layers which are more sensitive to the accuracy. Set lower c_rates for such layers. [Layers which are pretty close to the input data and Softmax fall into this category.]

 To improve the performance of the pruned caffemodel, techniques like Retraining and Dynamic Splicing are implemented in the code. For enabling Retraining and Dynamic Splicing, turn on the corresponding flags in squeeze_conv_layer.hpp and squeeze_inner_product_layer.hpp. Splicing rates can be adjusted in the same files if required.

 Best results are obtained by finetuning other learning hyper parameters like Base Learning rate, Learning rate multipliers, Dropout ratio and Training batch size. 
 

 
 ### Results
 
 
 
| Compression Factor 	| Top-5 Accuracy on Validation set 	|
|:------------------:	|:--------------------------------:	|
| 1.00X (Default) 	| 88.44% 	|
| 10.38X  	| 89.32% 	|
| 12.30X  	| 89.00% 	|
| 13.05X 	| 88.9% 	|
| 14.4X 	| 88.62% 	|
| 15.00X 	| 88.39% 	|

As seen from the above table, the accuracy of the compressed models(For eg. 13.3 million parameters - 10X reduced) are better than the original model(138.3 million parameters)
 

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
