import numpy as np
import tonic.transforms as transforms
import matplotlib.pyplot as plt

class EventRepresentation(object):
    def __init__(self, num_events=100000, width=304, height=240):
        self.sensor_size = None
        self.events = np.rec.array(None, dtype= [("x", int), ("y", int), ("t", int), ("p", int)],
                                   shape=num_events)
        self.width = width
        self.height = height
        self.sensor_size = None
        self.transform = None
        self.representation = None

    def get_sensor_size(self):
        self.sensor_size = (self.width, self.height, 2)

    def get_eventstream(self, eventstream):
        self.events = eventstream.events
        self.height = eventstream.height
        self.width = eventstream.width
        self.get_sensor_size()

    def convert_to_frame(self, **kwargs):
        self.transform = transforms.ToFrame(sensor_size=self.sensor_size, **kwargs)
        self.representation = self.transform(self.events)

    def denoise(self, filter_time=10000):
        denoise_transform = transforms.Denoise(filter_time)
        self.events = denoise_transform(self.events)

    def plot_frames(self):
        frames = self.representation
        fig, axes = plt.subplots(1, len(frames))
        for axis, frame in zip(axes, frames):
            axis.imshow(frame[1] - frame[0])
            axis.axis("off")
        plt.show(block=True)
        plt.tight_layout()

    def convert_to_voxels(self, n_time_bins):
        self.transform = transforms.ToVoxelGrid(sensor_size=self.sensor_size, n_time_bins=n_time_bins)
        self.representation = self.transform(self.events)

    def plot_voxels(self):
        volume = self.representation
        fig, axes = plt.subplots(1, len(volume))
        for axis, slice in zip(axes, volume):
            axis.imshow(slice)
            axis.axis("off")
        plt.show(block=True)
        plt.tight_layout()

    def convert_to_time_surface(self, surface_dimensions, tau, decay):
        self.transform = transforms.ToTimesurface(sensor_size=self.sensor_size, surface_dimensions=surface_dimensions, tau=tau, decay=decay)
        self.representation = self.transform(self.events)

    def plot_time_surface(self):
        surfaces = self.representation
        n_events = self.events.shape[0]
        n_events_per_slice = n_events // 3
        fig, axes = plt.subplots(1, 3)
        for i, axis in enumerate(axes):
            surf = surfaces[(i + 1) * n_events_per_slice - 1]
            axis.imshow(surf[0] - surf[1])
            axis.axis("off")
        plt.tight_layout()