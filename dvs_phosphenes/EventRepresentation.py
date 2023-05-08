"""
    This module contains the class and functions for different ways of representing events from
    an eventstream using the tonic framework.
"""
import numpy as np
import tonic.transforms as transforms
import matplotlib.pyplot as plt

class EventRepresentation(object):
    """
     Computes event representation from an event stream.
     ---
     Parameters:
         sensor_size : a 3-tuple of x,y,p for sensor_size
         events (array) : ndarray, [x, y, t, p] where,
                 - x is the x-location of event in pixels
                 - y is the y-location of event in pixels
                 - t is the time-stamp of event
                 - p is the polarity of the event
         width : frame width in pixels
         height : frame height in pixels
         transform : transformation from events to specified event representation
         representation : event representation
     """
    def __init__(self, num_events=100000, width=304, height=240):
        self.sensor_size = None
        self.events = np.rec.array(None, dtype= [("x", int), ("y", int), ("t", int), ("p", int)],
                                   shape=num_events)
        self.width = width
        self.height = height
        self.transform = None
        self.representation = None
        self.pol_events = None

    def get_sensor_size(self):
        """
            Sets the sensor_size
        """
        self.sensor_size = (self.width, self.height, 2)

    def get_eventstream(self, eventstream):
        """
            Defines the events, height, width and sensor_size from a given eventstream
            ---
            Parameters :
                eventstream : eventstream (incl. events, height and width) from event vision object
        """
        self.events = eventstream.events
        self.height = int(eventstream.height)
        self.width = int(eventstream.width)
        self.get_sensor_size()

    def convert_to_frame(self, **kwargs):
        """
            Defines the transformation of the accumulation of events to frames from eventstream
            to frame event representation using the tonic ToFrame transformation. The exact
            representation is specified with **kwargs, where the accumulation of events are
            to frames are done by slicing :
             - along constant time (time_window),
             - constant number of events (event_count)
             - constant number of frames, along time axis (n_time_bins)
             - constant number of frames, along number of events (n_event_bins).
             And stores the transformation and resulting event representation accordingly.
        """
        self.transform = transforms.ToFrame(sensor_size=self.sensor_size, **kwargs)
        self.representation = self.transform(self.events)

    def denoise(self, filter_time=10000):
        """
            Drops events that are 'not sufficiently connected to other events in the recording.'
            ---
            Parameters :
                filter_time :  maximum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less events.
        """
        denoise_transform = transforms.Denoise(filter_time)
        self.events = denoise_transform(self.events)

    def plot_frames(self, plot_file=None, n_frames=None):
        """
            Plots the frames event representation of events and optionally store resulting plot
            ---
            Parameters
                plot_file : file name to store resulting plot
        """
        frames = self.representation
        frames = frames[0:n_frames]
        # fig, axes = plt.subplots(int(len(frames)/5), 5)
        fig, axes = plt.subplots(len(frames))
        for axis, frame in zip(axes.flatten(), frames):
            #axis.imshow(frame[1] - frame[0],'gray')
            axis.imshow(frame[0],'gray')
            axis.axis("off")
        plt.show(block=True)
        plt.tight_layout()
        if plot_file is not None :
            plt.savefig(plot_file)

    def convert_to_voxels(self, n_time_bins):
        """
            Defines the transformation to build a voxel grid of events from eventstream.
            The voxel grid is a voxel grid with bilinear interpolation in the time domain from a
            set of events. Implements the event volume from Zhu et al. 2019, Unsupervised
            event-based learning of optical flow, depth, and egomotion
            ---
            Parameters :
                n_time_bins :  fixed number of time bins to slice the event sample into.
            """
        self.transform = transforms.ToVoxelGrid(sensor_size=self.sensor_size, n_time_bins=n_time_bins)
        self.representation = self.transform(self.events)

    def plot_voxels(self, n_frames=None):
        """
            Plots the voxel grid event representation
        """
        volume = self.representation
        volume = volume[0:n_frames]
        fig, axes = plt.subplots(1, len(volume))
        for axis, slice in zip(axes, volume):
            axis.imshow(slice, 'gray')
            axis.axis("off")
        plt.show(block=True)
        plt.tight_layout()

    def convert_to_time_surface(self, surface_dimensions, tau, decay):
        """
            Defines the transformation of a representation that creates timesurfaces for each event
            in the recording from an eventstram . Modeled after the paper Lagorce et al. 2016,
            Hots: a hierarchy of event-based time-surfaces for pattern recognition
            https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
            ---
            Parameters :
             surface_dimensions: width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
             tau: time constant to decay events around occuring event with.
             decay: can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        """
        self.transform = transforms.ToTimesurface(sensor_size=self.sensor_size, surface_dimensions=surface_dimensions, tau=tau, decay=decay)
        self.representation = self.transform(self.events)

    def plot_time_surface(self):
        """
            Plots the time surface event representation
        """
        surfaces = self.representation
        n_events = self.events.shape[0]
        n_events_per_slice = n_events // 3
        fig, axes = plt.subplots(1, 3)
        for i, axis in enumerate(axes):
            surf = surfaces[(i + 1) * n_events_per_slice - 1]
            axis.imshow(surf[0] - surf[1], cmap="gray")
            axis.axis("off")
        plt.tight_layout()

    def get_event_repres(self):
        """
            Returns event representations
        :return: representation
        """
        return self.representation


    def normalise_frame_repres(self):
        """
        Normalise frame event representation to pixel values between 0 and 1
        :return: normalised frame event representation
        """
        n_frames = self.representation.shape[0]
        norm_represent = np.zeros((n_frames, self.width, self.height))
        for fr in range(n_frames):
            norm_fr = self.representation[fr][0][:][:]
            norm_represent[fr] = (norm_fr-np.min(norm_fr)) / (np.max(norm_fr)-np.min(norm_fr))
        self.representation = norm_represent
        return norm_represent

    def v2e_normalise(self):
        """
         Normalise frame event representation to pixel values between 0 and 1, the manner that
         v2e framework did
        :return:
        """
        dvs_vid_full_scale = 3
        n_frames = self.representation.shape[0]
        norm_represent = np.zeros((n_frames, self.width, self.height))
        for fr in range(n_frames):
            norm_fr = self.representation[fr][0][:][:]
            norm_represent[fr] = (norm_fr + dvs_vid_full_scale)/float(dvs_vid_full_scale*2)
        return norm_represent

    def remove_polar(self):
        """
        Merge polarities of events so that ON and OFF events have a singular polarity (a value of 1). T
        This is so that OFF events, registered as 0 values, are not factored out when normalising.
        :return: single polarity events
        """
        repre_merg_polar = transforms.MergePolarities()
        self.events = repre_merg_polar(self.events)

    def remove_pol_apply_refract_period(self, delta_time):
        """
        Merge polarities and apply a specified refractory period in which all
        events are dropped for that time period. This is for the purpose of evening out
        dense temporal event density.
        :param delta_time: refractory period in which events should be dropped
        :return: single polarity, refractory period dropped events
        """
        remove_pol_apply_refract_period =transforms.Compose([transforms.MergePolarities(), transforms.RefractoryPeriod(delta_time),])
        self.events = remove_pol_apply_refract_period(self.events)

    def apply_binary(self):
        """
        Binarise events by setting all enormalised non-zero event values accumualge within
        an event image frame to a valaue of 1.
        :return: binarised event image frame
        """
        apply_bin = self.representation
        apply_bin[apply_bin>0]=1
        self.representation = apply_bin
        return apply_bin






