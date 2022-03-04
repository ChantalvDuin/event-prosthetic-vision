import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
import tonic.transforms as transforms
import matplotlib.pyplot as plt

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

n_time_bins = 3
n_event_bins = 200

# frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)
# # frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_event_bins=n_event_bins)
# tonic_frames = frame_transform(tonic_events)
# #
# def plot_frames(frames):
#     fig, axes = plt.subplots(1, len(frames))
#     for axis, frame in zip(axes, frames):
#         axis.imshow(frame[1] - frame[0])
#         axis.axis("off")
#     plt.show(block=True)
#     plt.tight_layout()
# #
# #
# plot_frames(tonic_frames[0:5])

# volume = transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=n_time_bins)(tonic_events)
#
# def plot_voxel_grid(volume):
#     fig, axes = plt.subplots(1, len(volume))
#     for axis, slice in zip(axes, volume):
#         axis.imshow(slice)
#         axis.axis("off")
#     plt.show(block=True)
#     plt.tight_layout()
#
# plot_voxel_grid(volume[0:5])

denoise_transform = transforms.Denoise(filter_time=10000)

events_denoised = denoise_transform(tonic_events)

surfaces = transforms.ToTimesurface(sensor_size=sensor_size, surface_dimensions=None, tau=10000, decay='exp')(events_denoised)

n_events = events_denoised.shape[0]
n_events_per_slice = n_events // 3
fig, axes = plt.subplots(1, 3)
for i, axis in enumerate(axes):
    surf = surfaces[(i+1)*n_events_per_slice - 1]
    axis.imshow(surf[0] - surf[1])
    axis.axis("off")
plt.tight_layout()

