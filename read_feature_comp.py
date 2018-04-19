import numpy as np
import os
import sys
import argparse
import glob
import time
#import _init_paths
# from units import SClassifier, AverageMeter, convert_secs2time
# import caffe
import scipy.io as sio


def comp_feature(feature_1,feature_2):
  feature_1=feature_1.reshape(-1)
  feature_2=feature_2.reshape(-1)
  feature_1_mult = feature_1*feature_1
  feature_2_mult = feature_2*feature_2
  sum1=np.sqrt(sum(feature_1_mult))
  feature_1=feature_1/sum1
  sum2=np.sqrt(sum(feature_2_mult))
  feature_2=feature_2/sum2
  mult=feature_1*feature_2
  feature_1_mult = feature_1*feature_1
  feature_2_mult = feature_2*feature_2
  # print feature_1.shape
  # print feature_1_mult
  # print sum1
  # print feature_1
  ret = sum(feature_1_mult)+sum(feature_2_mult)-2*sum(mult)
  return ret

def main(argv):
  dir_name="/storage2/liushuai/gs6_env/result/lafang1/"
  mat_1 = sio.loadmat(dir_name+"nelson00047_112.feature.mat")
  mat_1 = sio.loadmat(dir_name+"nelson00361_77.feature.mat")
  feature_1= mat_1["feature"]
  #mat_2 = sio.loadmat(dir_name+"nelson00048_112.feature.mat")
  mat_2 = sio.loadmat(dir_name+"nelson00346_80.feature.mat")
  
  feature_2= mat_2["feature"]
  ret = comp_feature(feature_1,feature_2)
  print ret

if __name__ == '__main__':
  main(sys.argv)