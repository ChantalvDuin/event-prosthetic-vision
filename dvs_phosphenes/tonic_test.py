import h5py
import numpy as np
import tonic.transforms as transforms

input_file = '/home/duinch/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/v2e-output/events.h5'
# input_file = '/home/duinch/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/v2e-box-output/events.h5'

data_file = h5py.File(input_file, 'r')
print(list(data_file.keys()))

events = data_file['events']
# events are in the form [time t, y coordinate, x coordinate, sign of events] with time in microseconds

frames = data_file['frame']
# frames are in the form [frame f, width, height]

frame_ind = data_file['frame_idx'] # frame index
frame_ts = data_file['frame_ts'] # delta frame time

N = len(events)
print(N)
tonic_events = np.zeros((N,4)) # tonic events have the shape (x, y, t, p)

for i in range(0, N):
    tonic_events[i,0] = events[i, 2] # x
    tonic_events[i,1] = events[i, 1] # y
    tonic_events[i,2] = events[i, 0] # t
    tonic_events[i,3] = events[i, 3] # p

width = frames.shape[1]
height = frames.shape[2]
sensor_size = (width, height, 2)

nr_frames = 20

frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=nr_frames)
tonic_frames = frame_transform(tonic_events)