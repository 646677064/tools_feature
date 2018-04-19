#!/bin/bash
#mean=103.94,116.78,123.68 #densenet201
mean=110.676,115.771,123.191 #siamese
base_model=resnet
model_file=/storage2/liushuai/gs6_env/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_95000.caffemodel
prototxt_file=/storage2/liushuai/gs6_env/caffe-reid/models/market1501/res50_near5/dev.proto
feature_name=pool5
gpu=6
num=1
LOG="/storage2/liushuai/gs6_env/result//1.txt"
exec &> >(tee -a "$LOG")
python /storage2/liushuai/gs6_env/market1501_extract_freature/patch_feature_choose.py \
	/storage2/tiannuodata/patch/ \
	/storage2/liushuai/gs6_env/result/ \
	/storage2/liushuai/gs6_env/result/ \
	--gpu $gpu \
	--model_def $prototxt_file \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value $mean 
#2>&1|tee /storage2/liushuai/gs6_env/result//1.txt