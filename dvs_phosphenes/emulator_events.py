from v2ecore.emulator import EventEmulator
import torch
import cv2
import numpy as np

video_path = '/home/duinch/PycharmProjects/dvs_phosphenes/data/v2e_tutorial_video.avi'
output_folder = '/home/duinch/PycharmProjects/dvs_phosphenes/demo_output'
output_file = 'events_emulator.h5'

# **IMPORTANT** make torch static, likely get faster emulation
# might also cause memory issue
torch.set_grad_enabled(False)
# read a video from opencv
cap = cv2.VideoCapture(video_path)

# num of frames
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: {}".format(fps))
num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Num of frames: {}".format(num_of_frames))

duration = num_of_frames/fps
delta_t = 1/fps
current_time = 0.

print("Clip Duration: {}s".format(duration))
print("Delta Frame Time: {}s".format(delta_t))
print("="*50)

#get pixel width and height
pixel_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("Pixel width: {}".format(pixel_width))
pixel_height =cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Pixel height: {}".format(pixel_height))

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define Emulator to convert video to events
emulator = EventEmulator(
    pos_thres=0.2,
    neg_thres=0.2,
    sigma_thres=0.03,
    cutoff_hz=200,
    leak_rate_hz=1,
    shot_noise_rate_hz=10,
    # dvs_h5=output_file,
    output_folder=output_folder,
    # dvs_aedat2=output_file,
    # output_width= pixel_width,
    # output_height=pixel_height,
    device=torch_device
)

new_events = None

idx = 0
# Only Emulate the first 10 frame
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret is True and idx < 10:
        # convert it to Luma frame
        luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("="*50)
        print("Current Frame {} Time {}s".format(idx, current_time))
        print("-"*50)

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
            event_time = (new_events[-1, 0]-new_events[0, 0])
            event_rate_kevs = (num_events/delta_t)/1e3

            print("Number of Events: {}\n"
                  "Duration: {}s\n"
                  "Start T: {:.5f}s\n"
                  "End T: {:.5f}s\n"
                  "Event Rate: {:.2f}KEV/s".format(
                      num_events, event_time, start_t, end_t,
                      event_rate_kevs))
        idx += 1
        print("="*50)
    else:
        break

cap.release()

# output_string = output_folder + "/" + output_file
# np.save(output_string, new_events)