set -e
if [ ! -n "$1" ] ;then
    echo "\$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

if [ ! -n "$2" ] ;then
    echo "\$2 is empty, default is vgg_reduce"
    base_model='vgg16'
else
    echo "use $2 as base model"
    base_model=$2
fi

if [ ! -n "$3" ] ;then
    echo "\$3 is empty, default is feature_name"
    feature_name='pool5'
else
    echo "use $3 as base model"
    feature_name=$3
fi
#base_model=caffenet
#base_model=vgg16
#base_model=googlenet
#base_model=res50
#feature_name=fc7
#feature_name=pool5/7x7_s1
#feature_name=pool5
#model_file=./models/market1501/$base_model/snapshot/${base_model}.full_iter_50000.caffemodel
model_file=./models/market1501/$base_model/snapshot/res50.full_iter_200000.caffemodel
#res50.full_iter_200000.caffemodel
#model_file=/home/liushuai/RFCN/py-R-FCN-master/caffe/models/50_droupout_siamese/mnist_siamese_iter_175000.caffemodel
#model_file=/home/liushuai/RFCN/py-R-FCN-master/data//ResNet-50-model.caffemodel
#model_file=/home/liushuai/RFCN/py-R-FCN-master/data//DenseNet_201.caffemodel
#mean=103.94,116.78,123.68 #densenet201
mean=110.676,115.771,123.191 #siamese

# num='0'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean
# # res50 103.062623801 115.902882574 123.151630838          densenet mean 103.94,116.78,123.68
# #110.676,115.771,123.191
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# num='1'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean
# num='11'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

num='12'
python examples/market1501/extract/extract_feature.py \
	/storage/liushuai/work/test/gallery/test"$num".txt \
	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/$base_model/dev.proto \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value $mean 2>&1|tee /storage2/liushuai/gs3/work/caffe-reid/1.txt

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# num='13'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# num='15'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean


# num='5'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean
	

# num='6'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# num='81'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean


# num='82'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean


# num='83'
# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/gallery/test"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/test"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean

# python examples/market1501/extract/extract_feature.py \
# 	/storage/liushuai/work/test/query/query"$num".txt \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# 	--gpu $gpu \
# 	--model_def ./models/market1501/$base_model/dev.proto \
# 	--feature_name $feature_name \
# 	--pretrained_model $model_file \
# 	--mean_value $mean
# # python examples/market1501/extract/extract_feature.py \
# # 	/storage/liushuai/work/test/query/query"$num".txt \
# # 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.feature.mat \
# # 	/storage/liushuai/work/test/result"$2"/query"$num".lst.${base_model}.score.mat \
# # 	--gpu $gpu \
# # 	--model_def ./models/market1501/$base_model/dev.proto \
# # 	--feature_name $feature_name \
# # 	--pretrained_model $model_file \
# # 	--mean_value 0.0,0.0,0.0

# # python examples/market1501/extract/extract_feature.py \
# # 	examples/market1501/lists/test.lst \
# # 	examples/market1501/datamat/test.lst.${base_model}.feature.mat \
# # 	examples/market1501/datamat/test.lst.${base_model}.score.mat \
# # 	--gpu $gpu \
# # 	--model_def ./models/market1501/$base_model/dev.proto \
# # 	--feature_name $feature_name \
# # 	--pretrained_model $model_file \
# # 	--mean_value 0.0,0.0,0.0

# # python examples/market1501/extract/extract_feature.py \
# # 	examples/market1501/lists/query.lst \
# # 	examples/market1501/datamat/query.lst.${base_model}.feature.mat \
# # 	examples/market1501/datamat/query.lst.${base_model}.score.mat \
# # 	--gpu $gpu \
# # 	--model_def ./models/market1501/$base_model/dev.proto \
# # 	--feature_name $feature_name \
# # 	--pretrained_model $model_file \
# # 	--mean_value 0.0,0.0,0.0
