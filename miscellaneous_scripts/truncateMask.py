#/usr/bin/python
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys
import numpy as np
import argparse

net_inputdata = {}

def get_inputs(net):
  net_inputdata['num'] = net.blobs['data'].data.shape[0]
  net_inputdata['channels'] = net.blobs['data'].data.shape[1]
  net_inputdata['height'] = net.blobs['data'].data.shape[2]
  net_inputdata['width'] = net.blobs['data'].data.shape[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting pruned model to default model')
    parser.add_argument('--pruned_prototxt', type=str, default=None, help='Prototxt used for pruning')
    parser.add_argument('--pruned_weights', type=str, default=None, help='Pruned caffemodel')
    parser.add_argument('--original_prototxt', type=str, default=None, help='Original Prototxt')
    parser.add_argument('--original_weights', type=str, default=None, help='Original caffemodel')
    args = parser.parse_args()

    if (args.pruned_prototxt == None) or not (args.pruned_prototxt.endswith(".prototxt")):
        print "Invalid pruned model file"
        exit(1)
    elif (args.pruned_weights == None) or not (args.pruned_weights.endswith(".caffemodel")):
        print "Invalid pruned weight file"
        exit(1)
    elif (args.original_prototxt == None) or not (args.original_prototxt.endswith(".prototxt")):
        print "Invalid original model file"
        exit(1)
    elif (args.original_weights == None) or not (args.original_weights.endswith(".caffemodel")):
        print "Invalid original weight file"
        exit(1)

    caffe.set_mode_cpu()
    net_4 = caffe.Net(args.pruned_prototxt, args.pruned_weights, caffe.TEST)
    net_2 = caffe.Net(args.original_prototxt, args.original_weights, caffe.TEST)
    cnet= caffe_pb2.NetParameter()
    text_format.Merge(open(args.pruned_prototxt).read(), cnet)

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
      layer_det = cnet.layer[i]
      layer_name = layer.name
      if layer_name in net_4.params.keys():
          if layer.type in vision_layers:
            name = layer.name
            #For vision layers with bias term
            if (layer.type == "SqueezeConvolution" and layer_det.squeeze_convolution_param.bias_term) or (layer.type == "SqueezeInnerProduct" and layer_det.squeeze_inner_product_param.bias_term):
                filters = net_4.params[layer_name][0].data
                biases = net_4.params[layer_name][1].data
                filters_mask = net_4.params[layer_name][2].data
                biases_mask = net_4.params[layer_name][3].data
                bias_flatten = biases.flatten()
                bias_mask_flatten = biases_mask.flatten()
                #Multiply biases and its mask values
                bias_flatten = bias_flatten * bias_mask_flatten
                net_2.params[layer_name][1].data.flat = bias_flatten.reshape(biases.shape)     # Update the new biases in the default caffemodel
            else:
                filters = net_4.params[layer_name][0].data
                filters_mask = net_4.params[layer_name][1].data
            filters_flatten = filters.flatten()
            filters_mask_flatten = filters_mask.flatten()
            #Multiply weights and its mask values
            filters_flatten = filters_flatten * filters_mask_flatten
            net_2.params[layer_name][0].data.flat = filters_flatten.reshape(filters.shape)     # Update the new weights in the default caffemodel
          #For other layers with blob parameters like Batchnormalization - copy the weights from src to destination
          else:
            for blob_no in range(len(net_4.params[layer_name])):
                net_2.params[layer_name][blob_no].data.flat = net_4.params[layer_name][blob_no].data
net_2.save("MaskTruncated_" + args.pruned_weights.split('/')[-1]);
print "Conversion Done!"
