import numpy as np
import os
from v2ecore.emulator import EventEmulator
import torch
import cv2
import numpy.lib.recfunctions as rfn
import h5py

class EventStream(object):
    def __init__(self, num_events=100000, width=304, height=240):
        # num_spikes: number of events this instance will initially contain
        self.events = np.rec.array(None, dtype= [("x", int), ("y", int), ("t", int), ("p", int)],
                                   shape=num_events)
        self.width = width
        self.height = height

    def read_h5_file(self, input_file):
        data_file = h5py.File(input_file, 'r')
        h5_events = data_file['events']
        self.events = convert_v2e_events_to_tonic_events(h5_events)
        frames = data_file['frame']
        self.width = frames.shape[1]
        self.height = frames.shape[2]


    def convert_video_to_events_v2e_emulator(self, video_path, output_folder):

        torch.set_grad_enabled(False)
        cap = cv2.VideoCapture(video_path)

        # num of frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = num_of_frames / fps
        delta_t = 1 / fps
        current_time = 0.

        # get pixel width and height
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define Emulator to convert video to events
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
        # Only Emulate the first 10 frame
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret is True and idx < 10:
                # convert it to Luma frame
                luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # print("=" * 50)
                # print("Current Frame {} Time {}s".format(idx, current_time))
                # print("-" * 50)

                # emulate events
                # **IMPORTANT** The unit of timestamp here is in second, a floating number
                new_events = emulator.generate_events(luma_frame, current_time)

                ### The output events is in a numpy array that has a data type of np.int32
                ### THe shape is (N, 4), each row is one event in the format of (t, x, y, p)
                ### The unit of timestamp here is in microseconds

                # update time
                current_time += delta_t

                # print event stats
                if new_events is not None:
                    num_events = new_events.shape[0]
                    start_t = new_events[0, 0]
                    end_t = new_events[-1, 0]
                    event_time = (new_events[-1, 0] - new_events[0, 0])
                    event_rate_kevs = (num_events / delta_t) / 1e3

                    # print("Number of Events: {}\n"
                    #       "Duration: {}s\n"
                    #       "Start T: {:.5f}s\n"
                    #       "End T: {:.5f}s\n"
                    #       "Event Rate: {:.2f}KEV/s".format(
                    #     num_events, event_time, start_t, end_t,
                    #     event_rate_kevs))
                idx += 1
                # print("=" * 50)
            else:
                break

        cap.release()

        self.events = convert_v2e_events_to_tonic_events(new_events)

def convert_v2e_events_to_tonic_events(h5_events):
        dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
        tonic_events = np.zeros((len(h5_events), 4))
        tonic_events[:, [2, 1, 0, 3]] = h5_events[:, [0, 1, 2, 3]]  # tonic events have the shape (x, y, t, p)
        tonic_events = rfn.unstructured_to_structured(tonic_events, dtype)
        return tonic_events

def convert_video_to_events_v2e_command(video_path, output_folder, out_filename_h5, out_filename_aedat = "None", overwrite=True, skip_video_output=False, davis_output = True):

    dvs_exposure = "duration .033"
    input_frame_rate = 30
    input_slowmotion_factor = 1
    disable_slomo = True
    slomo_model = "/content/v2e/input/SuperSloMo39.ckpt"
    unique_output_folder = False
    timestamp_resolution = 0.001
    auto_timestamp_resolution = True

    condition = "Noisy"
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

    input_terminal = '~/miniconda3/envs/dvs/bin/python ~/PycharmProjects/v2e/v2e.py%s' % final_v2e_command
    os.system(input_terminal)

