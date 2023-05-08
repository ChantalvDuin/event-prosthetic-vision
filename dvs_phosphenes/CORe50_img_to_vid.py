"""
This script takes CORe50 image frames and converts them into a single video
"""
import cv2

# set input image data path and output video path
data_path =  '/data/datasets/core50_128x128/'
video_path = '/home/chadui/data/dvs_phosphenes/core50_128x128_video/'

# set image frame parameters
height = 128
width =128

# set video parameters
frame_rate = 20
nr_frames= 300

# set CORe50 parameters
n_session = 11
n_object = 50

# converts image frames of each CORe50 object into a video
for ses in range(1, n_session+1) :
    for obj in range(1,n_object+1):
        # set images input and video output for each object and
        data_input = data_path + "s%d/o%d/" %(ses, obj)
        video_file = video_path + "s%d/s%d_o%d_vid.avi" %(ses,ses,obj)

        # set video writer configuration
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            video_file,
            fourcc,
            frame_rate,
            (width, height))

        # take each image frame to add to video
        for img in range(nr_frames):
            image_file = data_input + "C_%0.2d_%0.2d_%0.3d.png" % (ses,obj,img)
            frame = cv2.imread(image_file)
            out.write(frame)

        # close video writer
        out.release()