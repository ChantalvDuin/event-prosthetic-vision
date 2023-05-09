"""
This script is a pipeline to transform CORe50 images into Canny Edge detected phosphenes.

This script generates Canny Edge Detection representations for CORe50 images using either
 fixed hysteresis thresholds or median value auto-thresholds. These Canny Edge detected
 CORe50 images are transformed into their simulated phosphene representations using
 a phosphene simulator.
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from vr_phosphenes.defaults import Config
from vr_phosphenes.phosphene_simulation import (get_phosphene_simulator, GaussianSimulator, PhospheneSimulatorBasic)
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '4' # specify GPU to use

# set CORe50 data parameters
nr_ses = 11
nr_obj = 50
nr_fr = 300

# set fixed Canny hysteresis thresholds
t_edge_low = 225
t_edge_high = 250

# ---- Transform core50 data to Phosphenes  ----------------------------------------------------
for ses in range(1, nr_ses+1) :
    for obj in range(1,nr_obj+1):
        for fr in range(0,nr_fr):
            # read CORe50 image file
            img_filename= "C_%0.2d_%0.2d_%0.3d.png" %(ses, obj,fr)
            path =  '/home/chadui/data/dvs_phosphenes/core50_350x350/core50_350x350/s%d/o%d/' %(ses,obj) + img_filename
            img = cv.imread(path,0)

            # covert coloured CORe50 image to grey scale
            bw_img = Image.open(path)
            bw_img.convert('L')
            bw_img = np.array(bw_img)

            # determine median value Canny Edge auto-thresholds, comment out when using fixed thresholds
            sigma = 1/3
            v = np.median(bw_img)
            lower_1 = int(max(0, (1.0 - sigma) * v))
            upper_1 = int(min(255, (1.0 + sigma) * v))

            # transform CORe50 image to Canny Edge detection image
            edges = cv.Canny(img,lower_1,upper_1) # auto-generated thresholds
            # edges = cv.Canny(img, t_edge_low, t_edge_high)  # fixed thresholds

            # specify configure options for phosphene simulator, phosphene simulator used is from https://github.com/neuralcodinglab/dynaphos
            config = Config(display_size=(1600, 2880),
                            image_size=(350, 350),  # (260, 346),
                            zoom=1.,
                            phosphene_resolution=(32, 32),
                            # phosphene_intensity=8,
                            phosphene_intensity_decay=0.4,
                            receptive_field_size=4,
                            ipd=1240,
                            sigma=1.2,
                            phosphene_mode=1
                            )

            # set phosphene simulator
            phosphene_simulator = get_phosphene_simulator(config)

            # transform canny edge CORe50 images to phosphene representations
            norm_phosphenes_fr = phosphene_simulator(edges)
            norm_phosphenes_fr_rgb =  np.stack((norm_phosphenes_fr,)*3, axis=-1) # simulate RGB frames by duplicating phosphene frames three times

            # save phosphene representation
            phosphene_filename = "phos_%0.2d_%0.2d_%0.3d.jpeg" % (ses, obj, fr)
            if (ses == 3) | (ses == 7) | (ses == 10):
                phosphene_file = '/home/chadui/data/dvs_phosphenes/core50_test/s%d/o%d/' % (ses, obj) + phosphene_filename
            else:
                phosphene_file = '/home/chadui/data/dvs_phosphenes/core50_train/s%d/o%d/' % (ses, obj) + phosphene_filename
            plt.imsave(phosphene_file, norm_phosphenes_fr_rgb)
