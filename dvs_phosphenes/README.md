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
**1. Download CORe50 dataset**

Download the full-size_350x350_images.zip file from the CORe50 repository at https://vlomonaco.github.io/core50/ or https://github.com/giacomobartoli/vision-kit/tree/master/core50_utils. 

**2. Convert CORe50 images to video**

Transform CORe50 image frames to video using the CORe50_img_to_vid.py.

**3. Generate synthetic eventstream from CORe50 videos**

Generate and save eventstream generated using EventStream.py and v2e framework with the EventStream function  'convert_video_to_events_v2e_command'. Make sure to have the v2e framework and needed packages installed. 

**4. Transform eventstreams of CORe50 to chosen Event representation and corresponding event-based prosthetic vision**

Generate the phosphene representation corresponding event-based CORe50 images using the 'event_vision_pipeline.py' with the specified different event represenentations and transformations of 'EventRepresentation.py'. Make sure to have the tonic framework and needed packages installed. 

**5. Generate Canny Edge detection CORe50 prosthetic vision**

Generate the canny edge detection CORe50 phosphene representations.

**6. Download CORe50 bounding boxes**

Download the CORe50 bounding boxes from the CORe50 repository at https://vlomonaco.github.io/core50/ or https://github.com/giacomobartoli/vision-kit/tree/master/core50_utils. 

**7. Generate CSV files**

Generate the csv files for the training and test CORe50 dataset (either original images or prosthetic vision CORe50 images) using 'python create_csv.py --root=PATH_TO_DATASET --output_path=PATH_TO_CSV_FILE.csv'.

**8. Generate .tf record**
Create corresponding .tf record using 'python generate_tfrecord.py  --csv_input=.PATH_TO_CSV_FILE.csv
--output_path=PATH_TO_TF_FILE.record --dir_path=PATH_TO_DATASET'.

**9. Configure neural network for Object Detection**
Configure chosen object detection model (in this case, EfficientDet-D0 was used) by adapting the configure file of the object detection model to the right paths of the .tf records and .pbtxt file as well as adjust number of classes. Then train and evaluate the object detection model on the origingal or prosthetic CORe50 dataset. Make sure to TensorFlow API object detection installed.

