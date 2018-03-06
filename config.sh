#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/zhc1124/.conda/envs/tf/lib/python3.6/site-packages/tensorflow/models/research:/home/zhc1124/.conda/envs/tf/lib/python3.6/site-packages/tensorflow/models/research/slim
python genplate_scence.py
python ./VOCdevkit/VOC2012/image_sets.py
python create_pascal_tf_record.py \
    --label_map_path=tf_datasets/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=tf_datasets/pascal_train.record
python ./create_pascal_tf_record.py \
    --label_map_path=tf_datasets/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=./tf_datasets/pascal_val.record