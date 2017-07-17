import tensorflow as tf
from random import shuffle
import numpy as np
import time
import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import scipy.misc
import provider
import pc_util
import importlib
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import pointnet_cls as MODEL


parser = argparse.ArgumentParser()
parser.add_argument('--ply_path', default='', help='the 0 centered normalized to unit box ply file to classify')
FLAGS = parser.parse_args()



BATCH_SIZE = 2
NUM_POINT = 2048
MODEL_PATH = 'log/model.ckpt'
testFile=FLAGS.ply_path
reverseDict=dict({0:"bird",1:"bond",2:"can",3:"cracker",4:"house",5:"shoe",6:"teapot"})
NUM_CLASSES = 7
def evaluate(num_votes):
    is_training = False
     
   
    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
    is_training_pl = tf.placeholder(tf.bool, shape=())

    # simple model
    pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
    #loss = MODEL.get_loss(pred, labels_pl, end_points)
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
        
    # Create a session

    sess = tf.Session()

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    #log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           }

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    #fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    #for fn in range(len(TEST_FILES)):
    #log_string('----'+str(fn)+'----')
    current_data = provider.load_ply_data(testFile)
    #current_label = np.squeeze(current_label)
    current_data=np.asarray([current_data,np.zeros_like(current_data)])
    print(current_data.shape)
    
    file_size = current_data.shape[0]
    num_batches = 1
    print(file_size)
      
    
    batch_pred_sum = np.zeros((2, NUM_CLASSES)) # score for classes
    batch_pred_classes = np.zeros((2, NUM_CLASSES)) # 0/1 for classes
    feed_dict = {ops['pointclouds_pl']: current_data,
                 
                 ops['is_training_pl']: is_training}
    pred_val = sess.run( ops['pred'],feed_dict=feed_dict)
    print(reverseDict[np.argmax(pred_val[0])])


if __name__=='__main__':
    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            evaluate(num_votes=1)
    #LOG_FOUT.close()
