import h5py
import numpy as np
import numpy.lib.recfunctions as rfn

import tonic.transforms as transforms

input_file = '/home/duinch/PycharmProjects/dvs_phosphenes/demo_output/events.h5'

# data file keys of h5 file : events, frame, frame_idx, frame_ts
data_file = h5py.File(input_file, 'r')
print(list(data_file.keys()))

events = data_file['events']
# events are in the form [time t, y coordinate, x coordinate, sign of events] with time in microseconds

frames = data_file['frame']
# frames are in the form [frame f, width, height]

frame_ind = data_file['frame_idx'] # frame index
frame_ts = data_file['frame_ts'] # delta frame time

# transform events into desired tonic events structure
dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
tonic_events = np.zeros((len(events),4))
tonic_events[:,[2,1,0,3]] = events[:,[0,1,2,3]] # tonic events have the shape (x, y, t, p)
tonic_events =  rfn.unstructured_to_structured(tonic_events, dtype)

# get frame information from v2e framework
width = frames.shape[1]
height = frames.shape[2]
sensor_size = (width, height, 2)

# print(tonic_events.shape)
# print(tonic_events[0])

n_time_bins = 20


frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)
tonic_frames = frame_transform(tonic_events)