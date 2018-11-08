#!/bin/bash

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
cd /home/liushuai/caffe-ssd
./build/tools/caffe train \
--solver="/home/liushuai/caffe-ssd/laji/squeezeNet_512_512/git-squeeze-ssd-v2//solver.prototxt" \
--weights="/home/liushuai/caffe-ssd/zPrice/squeezeNet/squeezenet_v1.1.caffemodel" \
--gpu 3 2>&1 | tee /home/liushuai/caffe-ssd/laji/squeezeNet_512_512/git-squeeze-ssd-v2//squeeze_ssd_v2.log
