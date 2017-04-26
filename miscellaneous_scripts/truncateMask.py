#!/usr/bin/env python
#************Purpose: To truncate masks and generate a new model with default blobs****************************************************************************************
#************Usage:   python truncateMask.py <path of src_deploy.prototxt> <path of src caffemodel> <path of target_deploy.prototxt> <path of dummy caffemodel>***********
#************Authors: Rohini Priya <rohini@multicorewareinc.com>***********************************************************************************************************
#*******************  Zibiah Esme <zibiah@multicorewareinc.com> ***********************************************************************************************************

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys

prototxt_4   = sys.argv[1] # Path of prototxt with Squeezed Layers
caffemodel_4 = sys.argv[2] # Path of source caffemodel with 4 blobs - generated by MCWPrune/caffe
prototxt_2   = sys.argv[3] # Path of deploy.prototxt with normal layers - without Squeezed Layers
caffemodel_2 = sys.argv[4] # Path of dummy caffemodel with 2 blobs - generated by default Caffe

net_inputdata = {}
print_config = True


def get_inputs(net):
  net_inputdata['num'] = net.blobs['data'].data.shape[0]
  net_inputdata['channels'] = net.blobs['data'].data.shape[1]
  net_inputdata['height'] = net.blobs['data'].data.shape[2]
  net_inputdata['width'] = net.blobs['data'].data.shape[3]

if __name__ == '__main__':

    caffe.set_mode_cpu()
    net_4 = caffe.Net(prototxt_4, caffemodel_4, caffe.TEST)
    net_2 = caffe.Net(prototxt_2, caffemodel_2, caffe.TEST)
    cnet= caffe_pb2.NetParameter()
    cnet2= caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_4).read(), cnet)
    text_format.Merge(open(prototxt_2).read(), cnet2)

    get_inputs(net_4)
    get_inputs(net_2)

    net_4.blobs['data'].reshape(net_inputdata['num'],
                            net_inputdata['channels'],
                            net_inputdata['height'],
                            net_inputdata['width'])
    net_2.blobs['data'].reshape(net_inputdata['num'],
                            net_inputdata['channels'],
                            net_inputdata['height'],
                            net_inputdata['width'])

    vision_layers = ["Convolution", "InnerProduct", "SqueezeConvolution", "SqueezeInnerProduct"]
    for i, layer in enumerate(cnet.layer):
      layer_name = layer.name
      if layer.type in vision_layers:
            name = layer.name
            filters = weights = net_4.params[layer_name][0].data
            filters_mask= net_4.params[layer_name][2].data

            if layer.type == "InnerProduct" or layer.type == 'SqueezeInnerProduct':
                for ii in range(0, len(weights)):
                    for j in range(0, len(weights[ii])):
                        weights[ii][j] = filters_mask[ii][j] * weights[ii][j]
            else:
                for ii in range(0, len(weights)):
                    for j in range(0, len(weights[ii])):
                        for k in range(0, len(weights[ii][j])):
                            for l in range(0, len(weights[ii][j][k])):
                                weights[ii][j][k][l] = filters_mask[ii][j][k][l]* weights[ii][j][k][l]

            print name, layer.type
            net_2.params[layer_name][0].data.flat = weights # Update the new weights in the dummy caffemodel

net_2.save("MaskTruncated_" + caffemodel_4.split('/')[-1]);
print "> Done!"