import h5py
import numpy as np
import tonic.transforms as transforms

# input_file = '/home/duinch/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/v2e-box-output/events.h5'

data_file = h5py.File(input_file, 'r')
print(list(data_file.keys()))

events = data_file['events']
# events are in the form [time t, y coordinate, x coordinate, sign of events] with time in microseconds

frames = data_file['frame']
# frames are in the form [frame f, width, height]

frame_ind = data_file['frame_idx'] # frame index
frame_ts = data_file['frame_ts'] # delta frame time

tonic_events  = np.load('/home/duinch/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/v2e-output/demo_tonic_events.npy')
# dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
tonic_events = tonic_events.astype(np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]))

width = frames.shape[1]
height = frames.shape[2]
sensor_size = (width, height, 2)

nr_frames = 20
#
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=nr_frames)
tonic_frames = frame_transform(tonic_events)
#
