"""
    This module contains the class and functions for representing an event stream from
    either reading in a h5-file or converting a traditional video to an eventstream using the v2e
    framework.
"""
import numpy as np
import os
from v2ecore.emulator import EventEmulator
import torch
import cv2
import numpy.lib.recfunctions as rfn
import h5py

class EventStream(object):
    """
    Computes event data from DVS camera or from traditional video using v2e framework
    ---
    Parameters:
        events (array) : ndarray, [x, y, t, p] where,
                - x is the x-location of event in pixels
                - y is the y-location of event in pixels
                - t is the time-stamp of event
                - p is the polarity of the event
        width : frame width in pixels
        height : frame height in pixels
    """
    def __init__(self, num_events=100000, width=360, height=240):
        self.events = np.rec.array(None, dtype= [("t", int), ("y", int), ("x", int), ("p", int)],
                                   shape=num_events)
        self.width = width
        self.height = height

    def read_h5_file(self, input_file):
        """
        Reads a given hdf5 event file and stores the events in desired event representation ordering
        ---
        Parameters :
            input_file : given hdf5 event file
        """
        data_file = h5py.File(input_file, 'r')
        h5_events = data_file['events']
        self.events = convert_h5_events_to_tonic_events(h5_events)
        frames = data_file['frame']
        self.width = frames.shape[1]
        self.height = frames.shape[2]

    def read_h5_file_different(self, input_file):
        """
        Reads a given hdf5 event file and stores the events in desired event representation ordering
        ---
        Parameters :
            input_file : given hdf5 event file
        """
        data_file = h5py.File(input_file, 'r')
        h5_events = data_file['events']
        self.events = convert_h5_events_to_tonic_events_different(h5_events)
        frames = data_file['frame']
        self.width = frames.shape[1]
        self.height = frames.shape[2]

    def convert_video_to_events_v2e_emulator(self, video_path, output_folder):
        """
        Synthesize fake DVS events from traditional video frames directly using the Emulator from the
        v2e framework
        ---
        Parameters :
            video_path: path of video to be converted
            output_folder: output folder of v2e command
        """
        torch.set_grad_enabled(False)
        cap = cv2.VideoCapture(video_path)

        # num of frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        delta_t = 1 / fps
        current_time = 0.

        # get pixel width and height
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define Emulator to convert video to events
        emulator = EventEmulator(
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.03,
            cutoff_hz=200,
            leak_rate_hz=1,
            shot_noise_rate_hz=10,
            output_folder=output_folder,
            device=torch_device
        )

        new_events = None
        idx = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret is True :
                # convert it to Luma frame
                luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # emulate events
                new_events = emulator.generate_events(luma_frame, current_time)
                ### The output events is in a numpy array that has a data type of np.int32
                ### THe shape is (N, 4), each row is one event in the format of (t, x, y, p)
                ### The unit of timestamp here is in microseconds

                # update time
                current_time += delta_t
                idx += 1
            else:
                break

        cap.release()
        # convert event ordering from the v2e desired ordering of (t, x, y, p) to the tonic preferred ordering of (x, y , t, p)
        self.events = convert_v2e_emulator_events_to_tonic_events(new_events)

def convert_h5_events_to_tonic_events(h5_events):
        """
        Returns the events after converting the event ordering of h5 event files (t, x, y ,p) to the event ordering
        preferred by tonic : (x, y, t, p)
        ---
        Parameters :
            h5_events: events as ndarray, [t, x, y, p] where,
                - t is the time-stamp of event
                - x is the x-location of event in pixels
                - y is the y-location of event in pixels
                - p is the polarity of the event
        """
        dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
        tonic_events = np.zeros((len(h5_events), 4))
        # print(h5_events[0])
        tonic_events[:, [2, 1, 0, 3]] = h5_events # tonic events have the shape (x, y, t, p) while h5 events : (t, y, x, p)
        # print(tonic_events[0])
        tonic_events = rfn.unstructured_to_structured(tonic_events, dtype)
        # print(tonic_events[0])
        return tonic_events


def convert_h5_events_to_tonic_events_different(h5_events):
    """
    Returns the events after converting the event ordering of h5 event files (t, x, y ,p) to the event ordering
    preferred by tonic : (x, y, t, p)
    ---
    Parameters :
        h5_events: events as ndarray, [t, x, y, p] where,
            - t is the time-stamp of event
            - x is the x-location of event in pixels
            - y is the y-location of event in pixels
            - p is the polarity of the event
    """
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    tonic_events = np.zeros((len(h5_events), 4))
    # print(h5_events[0])
    tonic_events[:, [2, 0, 1, 3]] = h5_events  # tonic events have the shape (x, y, t, p) while h5 events : (t, x, y, p)
    # print(tonic_events[0])
    tonic_events = rfn.unstructured_to_structured(tonic_events, dtype)
    # print(tonic_events[0])
    return tonic_events



def convert_v2e_emulator_events_to_tonic_events(v2e_events):
    """
    Returns the events after converting the event ordering of v2e event files (t, x, y ,p) from the emulator to the event ordering
    preferred by tonic : (x, y, t, p)
    ---
    Parameters :
        v2e_events: events as ndarray, [t, x, y, p] where,
            - t is the time-stamp of event
            - x is the x-location of event in pixels
            - y is the y-location of event in pixels
            - p is the polarity of the event
    """
    # dtype = np.dtype([("x", '<i8'), ("y", '<i8'), ("t", '<i8'), ("p", '<i8')])
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    # uint32
    tonic_events = np.zeros((len(v2e_events), 4))
    # print(v2e_events[0])
    tonic_events[:, [1, 2, 0, 3]] = v2e_events  # tonic events have the shape (x, y, t, p) while v2e events : (t, x , y, p)
    # print(tonic_events[0])
    tonic_events = rfn.unstructured_to_structured(tonic_events, dtype)
    # print(tonic_events[0])
    return tonic_events

def convert_video_to_events_v2e_command(video_path, output_folder, out_video, out_filename_h5, out_filename_aedat = "None", overwrite=True, skip_video_output=False, davis_output = True):
    """
    Extracts frames from the provided video, synthesizes fake DVS events from the video
    frames from the original video frames and generates different possible v2e outputs according to the
    specified v2e command. The v2e command is build in this function executed from the terminal.
    ---
    Parameters :
        video_path: path of video to be converted
        output_folder: folder to store v2e output
        out_video : name of video output
        out_filename_h5: output DVS events as hdf5 event database
        out_filename_aedat: output DVS events as aedat event database
        overwrite: overwrites files in existing output folder
        skip_video_output: skips producing video output
        davis_output: saves frames within hdf5 event database
    """
    dvs_exposure = "duration .033"
    input_frame_rate = 20
    input_slowmotion_factor = 1
    disable_slomo = True
    slomo_model = "/content/v2e/input/SuperSloMo39.ckpt"
    unique_output_folder = False
    timestamp_resolution = 0.001
    auto_timestamp_resolution = True

    condition = "Clean"
    thres = 0.2
    sigma = 0.03
    cutoff_hz = 200
    leak_rate_hz = 5.18
    shot_noise_rate_hz = 2.716

    if condition == "Clean":
        thres = 0.2
        sigma = 0.02
        cutoff_hz = 0
        leak_rate_hz = 0
        shot_noise_rate_hz = 0
    elif condition == "Noisy":
        thres = 0.2
        sigma_thres = 0.05
        cutoff_hz = 30
        leak_rate_hz = 0.1
        shot_noise_rate_hz = 5

    v2e_command = [""]
    v2e_command += ["-i", video_path]
    v2e_command += ["-o", output_folder]

    if overwrite:
        v2e_command.append("--overwrite")


    v2e_command += ["--unique_output_folder", "{}".format(unique_output_folder).lower()]

    v2e_command += ["--dvs_vid", out_video]

    if davis_output:
        v2e_command += ["--davis_output"]
    v2e_command += ["--dvs_h5", out_filename_h5]
    v2e_command += ["--dvs_aedat2", out_filename_aedat]

    v2e_command += ["--dvs_text", "None"]
    v2e_command += ["--no_preview"]

    # if skip video output
    if skip_video_output:
        v2e_command += ["--skip_video_output"]
    else:
        # set DVS video rendering params
        v2e_command += ["--dvs_exposure", dvs_exposure]

    # set slomo related options
    v2e_command += ["--input_frame_rate", "{}".format(input_frame_rate)]
    v2e_command += ["--input_slowmotion_factor", "{}".format(input_slowmotion_factor)]

    # set slomo data
    if disable_slomo:
        v2e_command += ["--disable_slomo"]
        v2e_command += ["--auto_timestamp_resolution", "false"]
    else:
        v2e_command += ["--slomo_model", slomo_model]
        if auto_timestamp_resolution:
            v2e_command += ["--auto_timestamp_resolution", "{}".format(auto_timestamp_resolution).lower()]
        else:
            v2e_command += ["--timestamp_resolution", "{}".format(timestamp_resolution)]

    v2e_command += ["--pos_thres", "{}".format(thres)]
    v2e_command += ["--neg_thres", "{}".format(thres)]
    v2e_command += ["--sigma_thres", "{}".format(sigma)]
    v2e_command += ["--cutoff_hz", "{}".format(cutoff_hz)]
    v2e_command += ["--leak_rate_hz", "{}".format(leak_rate_hz)]
    v2e_command += ["--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz)]

    # Final v2e command
    final_v2e_command = " ".join(v2e_command)

    input_terminal = '~/.conda/envs/dvs/bin/python ~/PycharmProjects/v2e/v2e.py%s' % final_v2e_command
    os.system(input_terminal)

