"""
This adapted script of vision-kit from giacomobartoli scans CORe50 dataset and creates a .csv file
"""
import csv
import re
import os
from fnmatch import fnmatch
import math
import tensorflow as tf
import numpy as np

# set tensorflow api 1.0 flags for hyperparameters
flags = tf.compat.v1.flags
flags.DEFINE_string('root', '', 'Path to the root folder for directory of images for CSV')
flags.DEFINE_string('output_path', '', 'Path to output CSV file')
FLAGS = flags.FLAGS

# set header for the .csv file:
column_name = ['Filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

# CORe50 images width and height
C50_W = 350
C50_H = 350

# CORe50 root: select if training dir or test
# root = '/home/chadui/data/dvs_phosphenes/core50_test'

# set CORe50 image data tye
pattern = "*.jpeg"

# Bounding boxes root
bbox = '/home/chadui/data/dvs_phosphenes/core50_bbox'
pattern_bbox = "*.txt"

# minimal time point per frame when using n_event_bins as frame accumulation
frame_t= np.load('/home/chadui/data/dvs_phosphenes/object_detection/core50_ev500_frame_t.npy')

# This is an empty list that will be filled with all the data: filename, width, height, session..etc
filenames = []

# some regex used for finding session, obj, frame
re_find_session = '(?<=.{5}).\d'
re_find_object = '(?<=.{8}).\d'
re_find_frame = '(?<=.{11})..\d'

# find CORe50 object number
def find_obj(s, regex):
    obj = re.search(regex, s)
    # print("obj : " + obj.group() )
    return obj.group()

# find bounding box corresponding to CORe50 frame
def find_bbox(session, obj, frame):
    bb_path = '/home/chadui/data/dvs_phosphenes/core50_bbox/'+session+'/'+'CropC_'+obj+'.txt'
    f1 = open(bb_path, 'r').readlines()
    for line in f1:
        regex_temp = 'Color%0.3d:' % int(frame)
        if line.startswith(regex_temp):
            #print(line[10:])
            return line[10:]

# c[0] = xmin, c[1] = ymin, c[2] = xmax, c[3] = ymax

# get bounding box parameters
def add_bbox_to_list(bbox, list):
    c = bbox.split()
    list.append(c[0])
    list.append(c[1])
    list.append(c[2])
    list.append(c[3])
    return list

# given an object, it returns the label
def add_class_to_list(object):
    index = int(object[1:])
    f = open('/home/chadui/data/dvs_phosphenes/core50_class_names.txt', 'r').readlines()
    return f[index-1].strip()

# scanning the file system, creating a list with all the data
for path, subdirs, files in os.walk(FLAGS.root):
    for name in sorted(files):
        if fnmatch(name, pattern):
            # append core50 image frame and image size
            listToAppend = []
            listToAppend.append(name)
            listToAppend.append(C50_W)
            listToAppend.append(C50_H)

            # find session
            temp = find_obj(name, re_find_session)
            if temp.startswith('0'):
                temp=temp[1:]
            session = 's' + temp
            ses_temp = temp

            temp = find_obj(name, re_find_object)
            if temp.startswith('0'):
                temp=temp[1:]
            object = 'o' + temp
            ob_temp = temp
            frame = find_obj(name, re_find_frame)

            # add class label
            # class = 50
            # listToAppend.append(int(object.strip('o')))

            # class = 10
            # get_class_ind =  math.floor(int(object.strip('o'))/6)
            listToAppend.append(1+math.floor((int(object.strip('o'))-1)/5))

            # adjust bounding box allocation for different frame rates than CORe40 frame rate of 20 fps
            # bbox_fr =  2*int(frame) # 10 fps to 20 fps
            # bbox_fr = int(0.5*int(frame))  # 10 fps to 40 fps
            # bbox_fr = int(2/3*int(frame))  # 10 fps to 30 fps
            # bbox_frame = f'{frame:03d}'

            # adjust bound box allocation for event frames by taking the nearest time points of event bins to the 20 fps bounding boxes
            # fs_frame = np.arange(0, 15, 1 / 20) # create array of time points of frames of camera using 20 fs for core50 dataset
            # min_t_frame = frame_t[int(ses_temp)-1, int(ob_temp)-1, int(frame)]
            # bbox_frame = (np.abs(fs_frame - min_t_frame)).argmin() # return index of fs_frame of nearest value from current minimum time point of frame_t

            # bounding_box = find_bbox(session, object, bbox_frame) # adjusted bounding box for alternative bounding boxes
            bounding_box = find_bbox(session, object, frame)
            add_bbox_to_list(bounding_box, listToAppend)
            # print(listToAppend)
            filenames.append(listToAppend)

# writing data to the .csv file
with open(FLAGS.output_path, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(column_name),
    for i in sorted(filenames):
        filewriter.writerow(i)

print ('Done! Your .csv file is ready!')