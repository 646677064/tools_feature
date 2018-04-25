#!/bin/bash
#mean=103.94,116.78,123.68 #densenet201
mean=110.676,115.771,123.191 #siamese
#base_model=resnet
# model_file=/storage2/liushuai/gs6_env/caffe-reid/models/market1501/res50_near5/snapshot/res50.full_iter_95000.caffemodel
# prototxt_file=/storage2/liushuai/gs6_env/caffe-reid/models/market1501/res50_near5/dev.proto
model_file=/storage2/for_gs4/compare/classify_siamese.2.0.caffemodel
prototxt_file=/storage2/for_gs4/compare/classify_siamese.2.0.prototxt
feature_name=pool5
gpu=4
patch_dir_root=/storage2/liushuai/gs6_env/baiwei_patch_result/
jpg_dir=/storage2/tiannuodata/work/projdata/baiwei/baiweiproj2/JPEGImages/
shape2_xml_dir=/storage2/liushuai/gs6_env/shape_baiwei_result_xml/
dir_out=/storage2/liushuai/gs6_env/feature_baiwei_result_xml
LOG="/storage2/liushuai/gs6_env//feature_baiwei_result_xml.txt"
#exec &> >(tee -a "$LOG")
python /storage2/liushuai/gs6_env/market1501_extract_freature/models_make_xml_from_feature.py \
	$patch_dir_root \
	$jpg_dir \
	$shape2_xml_dir \
	$dir_out \
	--gpu $gpu \
	--model_def $prototxt_file \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value $mean 
#2>&1|tee /storage2/liushuai/gs6_env/result//1.txt
