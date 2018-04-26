#!/bin/bash
#mean=103.94,116.78,123.68 #densenet201
mean=110.676,115.771,123.191 #siamese
#model_file=/storage2/liushuai/gs6_env/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_95000.caffemodel
#prototxt_file=/storage2/liushuai/gs6_env/caffe-reid/models/market1501/res50_near5/dev.proto
# model_file=/storage2/for_gs4/compare/classify_siamese.2.0.caffemodel
# prototxt_file=/storage2/for_gs4/compare/classify_siamese.2.0.prototxt
# model_file=/storage2/for_gs4/compare/res101_no_dropout_calssfy.caffemodel
# prototxt_file=/storage2/for_gs4/compare/res101_no_dropout_calssfy.prototxt
model_file=/storage2/for_gs4/compare/res101_dropout_calssfy.caffemodel
prototxt_file=/storage2/for_gs4/compare/res101_dropout_calssfy.prototxt
feature_name=pool5
gpu=4
LOG="/storage2/liushuai/gs6_env/baiwei_patch_result//1.txt"
exec &> >(tee -a "$LOG")
python /storage2/liushuai/gs6_env/market1501_extract_freature/patch_feature_choose.py \
	/storage2/tiannuodata/work/projdata/baiwei/testdata//patch329/ \
	/storage2/liushuai/gs6_env/baiwei329_patch_result/ \
	/storage2/liushuai/gs6_env/baiwei329_patch_result/ \
	--gpu $gpu \
	--model_def $prototxt_file \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value $mean 
#2>&1|tee /storage2/liushuai/gs6_env/result//1.txt