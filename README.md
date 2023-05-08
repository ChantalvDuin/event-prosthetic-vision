# Event-prosthetic-vision
Pipeline for event-based simulated prosthetic vision and Canny Edge detected prosthetic vision for the task of object detection for the CORe50 dataset. 

## Description
This project was part of my Master Thesis "Optimisation of prosthetic vision for the detection of domestic objects using a silicon retina", in which the potential of a silicon retina known as Dynamic Vision Sensor (DVS) as a front-end sensor for prosthetic vision was explored. For this purpose, a pipeline for event-based prosthetic vision was created and tested using a Tensorflow object detection model (EfficientDet-D0) using the CORe50 dataset (https://vlomonaco.github.io/core50/). For exact details of the research, see the Master Thesis Event-based Prosthetic Vision.pdf file. 

See image below for an illustration of the event-based prosthetic vision pipeline: 
![alt text](https://github.com/ChantalvDuin/event-prosthetic-vision/blob/6e47f919587daac58c6b7bb7b87fb43e08255e38/event-based%20pipeline.png)
The pipeline includes the generation of event streams from a simulated dvs pixel using the v2e framework (Y. Hu, S-C. Liu, and T. Delbruck. v2e: From Video Frames to Realistic DVS Events. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), URL: https://arxiv.org/abs/2006.07722, 2021) as well as the tonic framework to generate the event representation. 

See image below for an illustration of the canny edge detection prosthetic vision pipeline: 
![alt text](https://github.com/ChantalvDuin/event-prosthetic-vision/blob/6e47f919587daac58c6b7bb7b87fb43e08255e38/canny%20pipeline.png)

## Requirements 
- v2e (https://github.com/SensorsINI/v2e)
- tonic (https://github.com/neuromorphs/tonic)
- numpy 
- cv2
- h5py
- PIL
- tensorflow
- pandas




