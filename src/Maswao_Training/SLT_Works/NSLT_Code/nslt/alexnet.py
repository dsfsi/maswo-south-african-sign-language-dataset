"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, source, keep_prob, device_str):
        """Create the graph of the AlexNet model.

        Args:
            source: Tensor for the input.
            keep_prob: Dropout keep probability (1.0 = no dropout).
            device_str: Device string for placement.
        """
        # Parse input arguments into class variables
        self.device_str = device_str
        self.input = source
        self.keep_prob = keep_prob
        self.weights_path = '../BaseModel/bvlc_alexnet.npy'
        
        # Store layers for variable tracking
        self.layers = {}
        self.trainable_variables = []

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        with tf.device(self.device_str):
            # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
            conv1 = conv(self.input, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            self.layers['conv1'] = conv1
            norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
            pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

            # 2nd Layer: Conv (w ReLu) -> Lrn -> Pool with 2 groups
            conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
            self.layers['conv2'] = conv2
            norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
            pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

            # 3rd Layer: Conv (w ReLu)
            conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
            self.layers['conv3'] = conv3

            # 4th Layer: Conv (w ReLu) splitted into two groups
            conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
            self.layers['conv4'] = conv4

            # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
            conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
            self.layers['conv5'] = conv5
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            flattened = tf.reshape(pool5, [-1, 6*6*256])
            fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
            self.layers['fc6'] = fc6
            dropout6 = dropout(fc6, self.keep_prob, name='dropout6')

            # 7th Layer: FC (w ReLu) -> Dropout
            fc7 = fc(dropout6, 4096, 4096, name='fc7')
            self.layers['fc7'] = fc7
            dropout7 = dropout(fc7, self.keep_prob, name='dropout7')

            self.output = tf.expand_dims(dropout7, 1, name='cnn_output')
            
            # Collect all trainable variables from the layers
            self._collect_trainable_variables()

    def _collect_trainable_variables(self):
        """Collect all trainable variables from the created layers."""
        # In TF2, we need to track variables explicitly
        # Variables are created within variable_scope contexts in helper functions
        for layer_name in self.layers:
            # Get variables from this layer's scope
            layer_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope=layer_name
            )
            self.trainable_variables.extend(layer_vars)

    def load_initial_weights(self):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        
        Note: In TF2, this no longer requires a session parameter.
        """
        # Load the weights into memory
        weights_dict = np.load(self.weights_path, encoding='bytes', allow_pickle=True).item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            if op_name not in ['fc8']:
                # Get variables for this operation
                try:
                    # Try to get variables using TF2 approach
                    scope_vars = [v for v in tf.compat.v1.global_variables() 
                                 if op_name in v.name]
                    
                    for data in weights_dict[op_name]:
                        # Biases
                        if len(data.shape) == 1:
                            bias_var = [v for v in scope_vars if 'biases' in v.name]
                            if bias_var:
                                bias_var[0].assign(data)
                        # Weights
                        else:
                            weight_var = [v for v in scope_vars if 'weights' in v.name]
                            if weight_var:
                                weight_var[0].assign(data)
                                
                except Exception as e:
                    print(f"Warning: Could not load weights for {op_name}: {e}")


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, 
         name, padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(
        i, k,
        strides=[1, stride_y, stride_x, 1],
        padding=padding
    )

    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases of the conv layer
        # TF2: Use compat.v1.get_variable for backward compatibility
        weights = tf.compat.v1.get_variable(
            'weights', 
            shape=[filter_height, filter_width, 
                   input_channels // groups, num_filters],
            dtype=tf.float32
        )
        biases = tf.compat.v1.get_variable(
            'biases', 
            shape=[num_filters],
            dtype=tf.float32
        )

        if groups == 1:
            conv_result = convolve(x, weights)
        else:
            # In the cases of multiple groups, split inputs & weights
            # Split input and weights and convolve them separately
            input_groups = tf.split(
                axis=3, 
                num_or_size_splits=groups, 
                value=x
            )
            weight_groups = tf.split(
                axis=3, 
                num_or_size_splits=groups,
                value=weights
            )
            output_groups = [
                convolve(i, k) 
                for i, k in zip(input_groups, weight_groups)
            ]

            # Concat the convolved output together again
            conv_result = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(
            tf.nn.bias_add(conv_result, biases), 
            tf.shape(conv_result)
        )

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases
        weights = tf.compat.v1.get_variable(
            'weights', 
            shape=[num_in, num_out],
            trainable=True,
            dtype=tf.float32
        )
        biases = tf.compat.v1.get_variable(
            'biases', 
            shape=[num_out], 
            trainable=True,
            dtype=tf.float32
        )

        # Matrix multiply weights and inputs and add bias
        # TF2: tf.nn.xw_plus_b removed, use tf.matmul + tf.nn.bias_add
        act = tf.nn.bias_add(tf.matmul(x, weights), biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu_result = tf.nn.relu(act)
            return relu_result
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, 
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(
        x, 
        ksize=[1, filter_height, filter_width, 1], 
        strides=[1, stride_y, stride_x, 1], 
        padding=padding, 
        name=name
    )


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer.
    
    Note: LRN is deprecated in TF2 but still available via tf.nn.local_response_normalization
    Consider replacing with batch normalization for better performance.
    """
    return tf.nn.local_response_normalization(
        x, 
        depth_radius=radius, 
        alpha=alpha, 
        beta=beta, 
        bias=bias, 
        name=name
    )


def dropout(x, keep_prob, name):
    """Create a dropout layer.
    
    TF2: tf.nn.dropout signature changed - now uses rate (1 - keep_prob) instead of keep_prob
    """
    # TF2: Convert keep_prob to dropout rate
    # If keep_prob is a float/int, calculate rate
    # If it's 1.0 (no dropout), rate should be 0.0
    if isinstance(keep_prob, (int, float)):
        rate = 1.0 - keep_prob
    else:
        # If it's a tensor/variable
        rate = 1.0 - keep_prob
    
    return tf.nn.dropout(x, rate=rate, name=name)
