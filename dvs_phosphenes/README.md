## Description 
This folder contains all the python files and classes used for running and creating the event-based prosthetic vision and 
canny edge detection prosthetic vision pipeline. 

It also contains the classes and scripts for converting CORe50 dataset and its corresponding phosphene representations created with
the prosthetic vision pipelines into tf_records, .pbtxt labels ready to be used with Tensorflow Object Detection API.

Classes for Event-based prosthetic vision:
- **EventRepresentation.py:** it computes event representations from an event stream as well as ways of transforming the 
 event representations using the tonic framework.
- **EventStream.py:** it represens an event stream from either reading in a h5-file or converting a traditional video to
 an eventstream using the v2e framework.
- **event_vision_pipeline.py:** it transforms CORe50 images into CORe50 phosphene representations using the EventStream.py and 
 EventRepresentation.py. 

Class for Canny Edge detection prosthetic vision:
- **canny_edge_pipeline.py:** it generates canny edge detection CORe50 phosphene representations. 

Classes for Tensorflow Object Detection API of CORe50 dataset:
- **convert_labels.py:** it converts CORe50 classes into a readable format for tensorflow (.pbtxt)
- **create_csv.py:** it scans CORe50 dataset and it creates a .csv file
- **generate_tfrecord.py:** it writes csv data into a .record file.

Class and additional data to enable category-level prosthetic CORe50 object detection:
- **CORe50_img_to_vid.py:** it takes CORe50 image frames and converts them into a single video that can be used as input for the v2e input to generate the eventstream for the CORe50 object.
- **core50_category_class_names.txt:** text file for CORe50 labels for category level object detection
- **core50_category_labels.pbtxt:** corresponding .pbtxt file for CORe50 labels for category level object detection

## How to use
**1. Download CORe50 dataset **
