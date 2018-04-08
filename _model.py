#!/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

import coco_input

def _var(name, shape, wd=0.001,initializer=None):
    #sqrt(3. / (in + out))
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
        
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

_COLLECTION_BATCHNORM_VARIABLES = "batchnorms"

def get_restore_variables():
    """
    return variables to be restored for eval
    """
    variables = tf.get_collection(_COLLECTION_BATCHNORM_VARIABLES)

    return variables + tf.trainable_variables()

def get_pretrain_variables():
    """
    return variables to be restored from pretrained model    
    """
    return tf.trainable_variables()



class Layer(object):
    """
    Layer with convolution and pooling
    """
    def __init__(self, name, output_ch, num_conv=1, retain_ratio=0.5, is_train=False):
        self.name = name
        self.output_ch = output_ch
        self.num_conv = num_conv
        self.retain_ratio = retain_ratio
        self.is_train = is_train

    def inference(self, in_feat):
        self.input_shape = in_feat.get_shape()
        N, H, W, C = self.input_shape
        feat = in_feat
        now_ch = C

        with tf.variable_scope(self.name):
            for idx_conv in range(3):
                with tf.variable_scope("conv{}".format(idx_conv)) as scope:
                    self.w = _var("W", [3,3,now_ch,self.output_ch])
                    
                    feat = tf.nn.conv2d(feat, self.w, strides=[1,1,1,1],padding="VALID")

                    feat = tf.contrib.layers.batch_norm(feat, center=True, scale=False, scope=scope, is_training=self.is_train, variables_collections=[_COLLECTION_BATCHNORM_VARIABLES])
                    
                    feat = tf.nn.relu(feat)
                    now_ch = self.output_ch
                    
            feat = tf.nn.max_pool(feat, [1,2,2,1], strides=[1,2,2,1],padding="SAME")

            """
            if self.is_train:
                feat = tf.nn.dropout(feat, keep_prob=self.retain_ratio)
            """
                    
        return feat


class Network(object):
    def __init__(self, is_train):
        self.is_train = is_train

    
    def inference(self, x):
        """
        forward network
        """
        layers = []
        feats = []
        
        D = x.get_shape()[3]
        output_ch = 32
        dataset = coco_input.get_dataset()
        num_classes = dataset.num_classes
        feat = x

        # 28x28
        for idx_layer in range(5):
            name = "layer{}".format(idx_layer)

            if idx_layer == 0:
                layer = Layer(name, output_ch, retain_ratio=0.8, is_train=self.is_train)
            else:
                layer = Layer(name, output_ch, is_train=self.is_train)

            feat = layer.inference(feat)
            output_ch = int(output_ch * 2)

            feats.append(feat)
            layers.append(layer)


        self.conv_outputs = feats
        self.conv_layers = layers


        # Global Average Pooling
        with tf.variable_scope("GAP"):
            N, H, W, C = feat.get_shape()

            # avg pool
            feat = tf.nn.avg_pool(feat, [1,H,W,1], strides=[1,1,1,1],padding="VALID") 

            # [batch_size, 1, 1, C] -> [batch_size, C]
            feat = tf.reshape(feat, [int(N), int(C)])
            
            # fc layer
            W = _var("W", [C,num_classes], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = _var("b", [num_classes],initializer=tf.constant_initializer())
                    
            logits = tf.matmul(feat, W) + b
    
        return logits


def get_loss(labels, logits):
    dataset = coco_input.get_dataset()
    tf.histogram_summary("logits", logits)    
    
    vector_labels = tf.one_hot(labels, dataset.num_classes, dtype=tf.float32)
    tf.histogram_summary("labels", labels)    
    
    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, vector_labels), name="entropy")
    decays = tf.add_n(tf.get_collection('losses'), name="weight_loss")
    total_loss = tf.add(entropy, decays, "total_loss")

    tf.scalar_summary("entropy", entropy)
    tf.scalar_summary("total_loss",total_loss)
    
    return entropy, total_loss