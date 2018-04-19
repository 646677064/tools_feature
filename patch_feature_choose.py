import numpy as np
import os
import sys
import argparse
import glob
import time
#import _init_paths
from units import SClassifier, AverageMeter, convert_secs2time
import caffe
import scipy.io as sio
import sys,os,subprocess,commands
from subprocess import Popen,PIPE
import random
import math

def load_txt(xfile):
  img_files = []
  labels = []
  for line in open(xfile):
    line = line.strip('\n').split(' ')
    assert(len(line) == 2)
    img_files.append(line[0])
    labels.append(int(float(line[1])))
  return img_files, labels

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

def Popen_do(pp_string,b_pip_stdout=True):
  #print pp_string
  if b_pip_stdout==True:
      p = Popen(pp_string, shell=True, stdout=PIPE, stderr=PIPE)#,close_fds=True)
  else:
      p = Popen(pp_string, shell=True, stderr=PIPE)#,close_fds=True)
  out, err = p.communicate()
  #p.wait()
  print pp_string
  if p.returncode != 0:
      print err
      #return 0
  return 1

def main(argv):

  parser = argparse.ArgumentParser()
  # Required arguments: input and output files.
  parser.add_argument(
    "input_file",
    help="Input image, directory"
  )
  parser.add_argument(
    "feature_file",
    help="Feature mat filename."
  )
  parser.add_argument(
    "score_file",
    help="Score Output mat filename."
  )
  # Optional arguments.
  parser.add_argument(
    "--model_def",
    default=os.path.join(
            "./models/market1501/caffenet/feature.proto"),
    help="Model definition file."
  )
  parser.add_argument(
    "--pretrained_model",
    default=os.path.join(
            "./models/market1501/caffenet/caffenet_iter_17000.caffemodel"),
    help="Trained model weights file."
  )
  parser.add_argument(
    "--gpu",
    type=int,
    default=-1,
    help="Switch for gpu computation."
  )
  parser.add_argument(
    "--center_only",
    action='store_true',
    help="Switch for prediction from center crop alone instead of " +
         "averaging predictions across crops (default)."
  )
  parser.add_argument(
    "--images_dim",
    default='256,256',
    help="Canonical 'height,width' dimensions of input images."
  )
  parser.add_argument(
    "--mean_value",
    default=os.path.join(
                         'examples/market1501/market1501_mean.binaryproto'),
    help="Data set image mean of [Channels x Height x Width] dimensions " +
         "(numpy array). Set to '' for no mean subtraction."
  )
  parser.add_argument(
    "--input_scale",
    type=float,
    help="Multiply input features by this scale to finish preprocessing."
  )
  parser.add_argument(
    "--raw_scale",
    type=float,
    default=255.0,
    help="Multiply raw input by this scale before preprocessing."
  )
  parser.add_argument(
    "--channel_swap",
    default='2,1,0',
    help="Order to permute input channels. The default converts " +
         "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
  )
  parser.add_argument(
    "--ext",
    default='jpg',
    help="Image file extension to take as input when a directory " +
         "is given as the input file."
  )
  parser.add_argument(
    "--feature_name",
    default="fc7",
    help="feature blob name."
  )
  parser.add_argument(
    "--score_name",
    default="prediction",
    help="prediction score blob name."
  )
  args = parser.parse_args()

  image_dims = [int(s) for s in args.images_dim.split(',')]

  channel_swap = None
  if args.channel_swap:
    channel_swap = [int(s) for s in args.channel_swap.split(',')]

  mean_value = None
  if args.mean_value:
    mean_value = [float(s) for s in args.mean_value.split(',')]
    mean_value = np.array(mean_value)

  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    print("GPU mode, device : {}".format(args.gpu))
  else:
    caffe.set_mode_cpu()
    print("CPU mode")

  # Make classifier.
  classifier = SClassifier(args.model_def, args.pretrained_model,
        image_dims=image_dims, mean_value=mean_value,
        input_scale=args.input_scale, raw_scale=args.raw_scale,
        channel_swap=channel_swap)

  args.input_file = os.path.expanduser(args.input_file)
  if os.path.isdir(args.input_file):
    list_dir = os.listdir(args.input_file)
  for idx_dir in list_dir:
    print idx_dir
    start_time = time.time()
    epoch_time = AverageMeter()
    if  os.path.isdir(args.input_file +"/"+idx_dir):
      #print idx_dir
      file_list=glob.glob(args.input_file +"/"+idx_dir+ '/*.' + args.ext)
      labels = [-1 for _ in xrange(len(file_list))]
      if not os.path.exists(args.feature_file+"/"+idx_dir+'/'):
        os.mkdir(args.feature_file+"/"+idx_dir+'/')
      with open(args.feature_file+"/"+idx_dir+"/list_file.txt","w") as z_f:
        tmp_file_list = [line+"\n" for line in file_list]
        z_f.writelines(tmp_file_list)

      save_feature = None
      size = len(file_list)
      for idx_f, _file_i in enumerate(file_list):
        _input=caffe.io.load_image(_file_i)
        _ = classifier.predict([_input], not args.center_only)
        feature = classifier.get_blob_data(args.feature_name)
        assert (feature.shape[0] == 1 )
        #assert (feature.shape[0] == 1 and score.shape[0] == 1)
        feature_shape = feature.shape
        #score   = classifier.get_blob_data(args.score_name)
       # score_shape = score.shape
        if save_feature is None:
            print('feature : {} : {}'.format(args.feature_name, feature_shape))
            save_feature = np.zeros((len(file_list), feature.size),dtype=np.float32)
        save_feature[idx_f, :] = feature.reshape(1, feature.size)
        tmp_file_name=os.path.basename(file_list[idx_f])
        #sio.savemat(args.feature_file+"/"+idx_dir+'/'+os.path.splitext(tmp_file_name)[0]+".feature", {'feature':feature})
      
      same_file_list=[]
      if len(same_file_list) == 0:
        tmp_list=[0]
        same_file_list.append(tmp_list)
      print size
      for aa_fea in range(1,size):
        #print len(same_file_list)
        b_same_class=False
        for bb_fea in range(0,len(same_file_list)):
          #print aa_fea," ",bb_fea," ",same_file_list[bb_fea][0]
          ret = comp_feature(save_feature[aa_fea],save_feature[same_file_list[bb_fea][0]])
          if ret <0.2:
            b_same_class=True
            same_file_list[bb_fea].append(aa_fea)
            break
        if b_same_class==False:
          tmp_list_in=[aa_fea]
          same_file_list.append(tmp_list_in)
      one_file_list=[ file_list[same_file_list[ss][0]] for ss in range(0,len(same_file_list))]
      with open(args.feature_file+"/"+idx_dir+"/everyclass_one_list_file.txt","w") as one_f:
        tmp_file_one = [line+"\n" for line in one_file_list]
        one_f.writelines(tmp_file_one)
      for cp_file in one_file_list:
        ppsring= "cp "+cp_file+" "+args.feature_file+"/"+idx_dir+"/"
        assert Popen_do(ppsring),ppsring+" error!"
      print idx_dir," different pic :",len(one_file_list)
      epoch_time.update(time.time() - start_time)
      start_time = time.time()
      need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (len(file_list)-1))
      need_time = '{:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)
      print need_time


if __name__ == '__main__':
  main(sys.argv)
