import cv2
from v2ecore.v2e_utils import video_writer

data_input =  '/data/chadui/core50_128x128/s1/o1'
video_path = '/home/chadui/data/dvs_phosphenes/core50_128x128/s1_01_video.avi'
height = 128
width =128
frame_rate = 20
nr_frames= 299

video_writer = video_writer(video_path,height, width, frame_rate)

for img in range(nr_frames):
    image_file= data_input + "/C_01_01_%0.3d.png" % img
    frame = cv2.imread(image_file)
    video_writer.write(frame)

video_writer.release()
