#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
validate Ladder Network
"""

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import coco_input
import model
import config

FLAGS = tf.app.flags.FLAGS


def restore_model(saver, sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.dir_parameter)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return None
    
    return global_step

def eval_once(summary_writer, top_k_op, entropy):
    saver = tf.train.Saver(model.get_restore_variables())
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    gpu_options = tf.GPUOptions(allow_growth=True) 
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True))
    sess.run(init)
    global_step = restore_model(saver, sess)

    if global_step is None:
        return

    # Start the queue runners.
    coord = tf.train.Coordinator()   
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    dataset = coco_input.get_dataset()
    true_count = 0    # Counts the number of correct predictions.
    total_sample_count = dataset.get_validation_size()

    num_iter = total_sample_count / FLAGS.batch_size
    entropies = []
    use_sample_count = 0
        
    #for i in range(num_iter):
    # limit for performance (samples are chosen randomly)
    for i in range(min(num_iter, 1000)):
        predictions, value_entropy = sess.run([top_k_op, entropy])
        true_count += np.sum(predictions)
        use_sample_count += FLAGS.batch_size
        entropies.append(value_entropy)

    # Compute precision @ 1.
    precision = true_count / float(use_sample_count)
    mean_entropy = float(np.mean(entropies))

    print ("use data {} / {}".format(use_sample_count, total_sample_count))
    print('step %d precision @ 1 = %.2f entropy = %.2f' % (int(global_step), precision, mean_entropy))

    summary = tf.Summary()
    summary.value.add(tag='Precision @ 1', simple_value=precision)
    summary.value.add(tag='entropy', simple_value=mean_entropy)    
    summary_writer.add_summary(summary, global_step)                        


    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
        

def evaluate():
    with tf.Graph().as_default() as g, tf.device("/gpu:0"):
        dataset = coco_input.get_dataset()
        labels, images = dataset.validate_input()

        network = model.Network(is_train=False)
        logits = network.inference(images)
        entropy, _ = model.get_loss(labels, logits)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        summary_writer = tf.train.SummaryWriter(FLAGS.dir_log_val, g)

        while True:
            eval_once(summary_writer, top_k_op, entropy)
            time.sleep(FLAGS.eval_interval_secs)

                
def main(argv=None):  
    if tf.gfile.Exists(FLAGS.dir_log_val):
        tf.gfile.DeleteRecursively(FLAGS.dir_log_val)
    tf.gfile.MakeDirs(FLAGS.dir_log_val)
    
    evaluate()

if __name__ == '__main__':
    tf.app.run()
