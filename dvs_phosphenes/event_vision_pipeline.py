"""
This script is a pipeline to transform CORe50 images into phosphene representations using
a DVS encoder.

This script generates event-based phosphene representations for CORe50 images using a
simulated DVS camera. An eventstream is generated for each CORe50 data object using
the v2e framework and transformed into an event representation using tonic.
These resulting event representations are then transformed into their simulated
phosphene representations using a phosphene simulator.
"""
from EventRepresentation import EventRepresentation
from EventStream import EventStream, convert_video_to_events_v2e_command

from vr_phosphenes.defaults import Config
from vr_phosphenes.phosphene_simulation import (get_phosphene_simulator, GaussianSimulator, PhospheneSimulatorBasic)
import numpy as np
import matplotlib.pyplot as plt
import os

# set GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# specify CORe50 parameters
nr_ses = 11
nr_obj = 50

# Transform each CORe50 object over all sessions to a phosphene representations
for ses in range(1, nr_ses+1) : # for each CORe50 session
    for obj in range(1,nr_obj+1): # for each CORe50 object
        # -------- Get Event Stream --------
        # get pre-saved event stream file
        event_filename = "events_s%d_o%d.h5" %(ses, obj)
        event_file = '/home/chadui/data/dvs_phosphenes/core50_350x350_dvs/s%d/o%d/' %(ses,obj) + event_filename

        # read event stream file
        eventstream_h5 = EventStream()
        eventstream_h5.read_h5_file_different(event_file)

        # -------- Get Event Representation --------
        # create event representation
        frames = EventRepresentation()
        frames.get_eventstream(eventstream_h5)

        # frames.denoise() # denoise events
        frames.remove_polar() # merge ON and OFF events into singular polarity events

        # delta_time = 1000  # set a refractory period in us
        # frames.remove_pol_apply_refract_period(delta_time) # apply a refractory period

        # accumulate events in a frame representation by slicing along constant time window
        # timewindow = 3300 # set a timewindow in us
        # frames.convert_to_frame(time_window = timewindow)

        # accumulate events in a frame representation by slicing constant number of frames, along number of time bins
        n_time_bins = 300 # set number of time bins
        frames.convert_to_frame(n_time_bins = n_time_bins)

        # accumulate events in a frame representation by slicing constant number of frames, along number of event bins
        # n_event_bins = 300 # set number of events bins
        # frames.convert_to_frame(n_event_bins = n_event_bins)

        # normalise frames
        norm_frames =frames.normalise_frame_repres()

        # norm_frames = frames.apply_binary() # binarise events

        # Save the minimum time point in each event frame for the bounding box allocation
        # for object detection
        # n_events = len(frames.events) # get number of events
        # events_t = frames.events['t'] # get time point of each event
        # spike_count = int(n_events / (n_event_bins)) # compute spike count
        # ind_start = np.arange(n_event_bins) * spike_count; # get spikes indices for start of event bin
        # ind_end = ind_start + spike_count  # get spikes indices for end of event bin
        # # save each minimum time point of each event bin representation
        # for fr in range(0, n_event_bins):
        #     frame_t[ses-1, obj-1, fr] = min(events_t[ind_start[fr]:ind_end[fr]]) / 1000000
        # save resulting time points in a numpy array
        # np.save('/home/chadui/data/dvs_phosphenes/object_detection/core50_ev500_frame_t.npy', frame_t);

        # -------- Get Phosphene Representation --------
        # specify configure options for phosphene simulator
        config = Config(display_size=(1600, 2880),
                        image_size=(350, 350),  # (260, 346),
                        zoom=1.,
                        # phosphene_resolution=(64, 64),
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

        # transform event-based CORe50 frames to phosphene representations
        norm_phosphenes = np.zeros(norm_frames.shape)
        for fr in range(norm_phosphenes.shape[0]):
            norm_phosphenes_fr = phosphene_simulator(norm_frames[fr]) # covert to phosphenes
            norm_phosphenes[fr]= norm_phosphenes_fr
            norm_phosphenes_fr_rgb =  np.stack((norm_phosphenes_fr,)*3, axis=-1) # simulate RGB frames by duplicating phosphene frames three times

            # save phosphene representation
            phosphene_filename = "phos_%0.2d_%0.2d_%0.3d.jpeg" % (ses, obj, fr)
            if (ses == 3) | (ses == 7) | (ses == 10):
                phosphene_file = '/home/chadui/data/dvs_phosphenes/core50_test/s%d/o%d/' % (ses, obj) + phosphene_filename
            else:
                phosphene_file = '/home/chadui/data/dvs_phosphenes/core50_train/s%d/o%d/' % (ses, obj) + phosphene_filename
            plt.imsave(phosphene_file, norm_phosphenes_fr_rgb)
