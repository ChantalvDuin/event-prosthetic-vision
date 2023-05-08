"""
This script is an adapted script of vision-kit from giacomobartoli to transform csv data into a .record file.
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import random

import pandas as pd
import tensorflow as tf
from PIL import Image

import re
from object_detection.utils import dataset_util
from collections import namedtuple

# set tensorflow api 1.0 flags for hyperparameters
flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('dir_path', '', 'Path to directory of images')
FLAGS = flags.FLAGS

# some regex used for finding session, obj
re_find_session = '(?<=.{5}).\d'
re_find_object = '(?<=.{8}).\d'

# find CORe50 object or session
def find_obj(s, regex):
    obj = re.search(regex, s)
    return obj.group()

# split csv data
def split(df, group):
    data = namedtuple('data', ['Filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

# create tensorflow example
def create_tf_example(group, path):
    name = group.Filename
    temp = find_obj(name, re_find_session)
    if temp.startswith('0'):
        temp = temp[1:]
    session = 's' + temp

    temp = find_obj(name, re_find_object)
    if temp.startswith('0'):
        temp = temp[1:]
    object = 'o' + temp

    file_path = path + session + '/' + object + '/'


    with tf.io.gfile.GFile(os.path.join(file_path, '{}'.format(group.Filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.Filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes.append((row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/Filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        #'image/object/class/text': dataset_util.int64_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# transform csv data to tensorflow record
def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = FLAGS.dir_path
    examples = pd.read_csv(FLAGS.csv_input)
    
    # shuffle csv entries for variance reduction to avoid object detection model overfitting
    grouped = split(examples, 'Filename')
    random.shuffle(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

# run main file
if __name__ == '__main__':
    tf.compat.v1.app.run()
