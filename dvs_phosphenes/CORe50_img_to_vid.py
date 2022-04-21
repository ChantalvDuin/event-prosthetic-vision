import cv2
data_path =  '/data/datasets/core50_128x128/'
video_path = '/home/chadui/data/dvs_phosphenes/core50_128x128_video/'

height = 128
width =128
frame_rate = 20
nr_frames= 300
n_session = 11
n_object = 50

for ses in range(1, n_session+1) :
    for obj in range(1,n_object+1):

        data_input = data_path + "s%d/o%d/" %(ses, obj)
        video_file = video_path + "s%d/s%d_o%d_vid.avi" %(ses,ses,obj)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            video_file,
            fourcc,
            frame_rate,
            (width, height))

        for img in range(nr_frames):
            image_file = data_input + "C_%0.2d_%0.2d_%0.3d.png" % (ses,obj,img)
            frame = cv2.imread(image_file)
            out.write(frame)

        out.release()