#!/usr/bin/env python
#************Purpose: To check the compression factor of a caffemodel*******************************************
#************Usage:   python printXFactor.py <path of deploy.prototxt> <path of target caffemodel>**************
#************Authors: Rohini Priya <rohini@multicorewareinc.com>************************************************
#*******************  Zibiah Esme <zibiah@multicorewareinc.com> ************************************************
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys
import numpy as np

prototxt   = sys.argv[1] #Relative or Absolute path to the deploy file
caffemodel = sys.argv[2] #Relative or Absolute path to the caffemodel

net_inputdata = {}

print_config = True

def get_inputs(net):
  net_inputdata['num'] = net.blobs['data'].data.shape[0]
  net_inputdata['channels'] = net.blobs['data'].data.shape[1]
  net_inputdata['height'] = net.blobs['data'].data.shape[2]
  net_inputdata['width'] = net.blobs['data'].data.shape[3]

if __name__ == '__main__':

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    cnet= caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(), cnet)

    get_inputs(net)
    net.blobs['data'].reshape(net_inputdata['num'],
                            net_inputdata['channels'],
                            net_inputdata['height'],
                            net_inputdata['width'])
    vision_layers = ["Convolution", "InnerProduct", "SqueezeConvolution", "SqueezeInnerProduct"]
    grand_total_weights = 0
    non_pruned = 0
    print "\n"
    for i, layer in enumerate(cnet.layer):
      layer_det = cnet.layer[i]
      layer_name = layer.name
      if layer.type in vision_layers:
        if layer.type == "SqueezeConvolution":
          if layer_det.squeeze_convolution_param.bias_term:
            filters_mask= net.params[layer_name][2].data
          else:
            filters_mask= net.params[layer_name][1].data
        elif layer.type == "SqueezeInnerProduct":
          if layer.squeeze_inner_product_param.bias_term:
            filters_mask= net.params[layer_name][2].data
          else:
            filters_mask= net.params[layer_name][1].data
        one = 0
        total = 0
        total = len(filters_mask.flatten())
        one = np.count_nonzero(filters_mask > 0)
        dec = one / float(total)
        print "Compression Rate for ", layer_name, " is ", one, 1 - dec
        grand_total_weights += total
        non_pruned += one
print "******************************************************************"
compression_factor = float(float(grand_total_weights)/float(non_pruned))
print "Total Compression Factor ", compression_factor
print "******************************************************************"
