

# video_path = '/Users/chantal/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/Hallway_Demo_chantal.avi'
video_path = '/home/duinch/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/v2e_tutorial_video.avi'

# output_folder = '/Users/chantal/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/v2e-output' #@param {type:"string"}
output_folder = '/home/duinch/PycharmProjects/RL-mobility/Python/Experiments/CHANTAL/v2e-output'
overwrite = True #@param {type:"boolean"}

# overwrites files in existing folder (checks existence of non-empty output_folder). (default: True)
unique_output_folder = False #@param {type:"boolean"}
# makes unique output folder based on output_folder, e.g. output1 (if non-empty output_folder already exists) (default: False)
out_filename = "events.h5" #@param {type:"string"}
# Output DVS events as hdf5 event database.
davis_output = True #@param {type:"boolean"}
# If also save frames in HDF5. (default: False)

# Output DVS Video Options
skip_video_output = False #@param {type:"boolean"}
# Skip producing video outputs, including the original video, SloMo video, and DVS video (default: False)
dvs_exposure = "duration .033" #@param {type:"string"}
# Mode to finish DVS frame event integration: duration time: Use fixed accumulation time in seconds, e.g. --dvs_exposure duration .005; count n: Count n events per frame, -dvs_exposure count 5000; area_event N M: frame ends when any area of M x M pixels fills with N events, -dvs_exposure area_count 500 64 (default: duration 0.01)

# Input Options
input_frame_rate = 30 #@param {type:"number"}
# Manually define the video frame rate when the video is presented as a list of image files. When the input video is a video file, this option will be ignored.
input_slowmotion_factor =  1#@param {type:"number"}
# Sets the known slow-motion factor of the input video, i.e. how much the video is slowed down, i.e., the ratio of shooting frame rate to playback frame rate. input_slowmotion_factor<1 for sped-up video and input_slowmotion_factor>1 for slowmotion video.If an input video is shot at 120fps yet is presented as a 30fps video (has specified playback frame rate of 30Hz, according to file's FPS setting), then set --input_slowdown_factor=4.It means that each input frame represents (1/30)/4 s=(1/120)s.If input is video with intended frame intervals of 1ms that is in AVI file with default 30 FPS playback spec, then use ((1/30)s)*(1000Hz)=33.33333. (default: 1.0)

# DVS Time Resolution Options
disable_slomo = True #@param {type:"boolean"}
# Disables slomo interpolation; the output DVS events will have exactly the timestamp resolution of the source video (which is perhaps modified by --input_slowmotion_factor). (default: False)
timestamp_resolution = 0.001 #@param {type:"number"}
# Ignored by --disable_slomo.) Desired DVS timestamp resolution in seconds; determines slow motion upsampling factor; the video will be upsampled from source fps to achieve the at least this timestamp resolution.I.e. slowdown_factor = (1/fps)/timestamp_resolution; using a high resolution e.g. of 1ms will result in slow rendering since it will force high upsampling ratio. Can be combind with --auto_timestamp_resolution to limit upsampling to a maximum limit value. (default: None)
auto_timestamp_resolution = True #@param {type:"boolean"}
# # (Ignored by --disable_slomo.) If True (default), upsampling_factor is automatically determined to limit maximum movement between frames to 1 pixel. If False, --timestamp_resolution sets the upsampling factor for input video. Can be combined with --timestamp_resolution to ensure DVS events have at most some resolution. (default: False)

# This is the SloMo model path
slomo_model = "/content/v2e/input/SuperSloMo39.ckpt"

# DVS Model Options
condition = "Noisy" #@param ["Custom", "Clean", "Noisy"]
# Custom: Use following slidebar to adjust your DVS model.
# Clean: a preset DVS model, generate clean events, without non-idealities.
# Noisy: a preset DVS model, generate noisy events.

thres = 0.2 #@param {type:"slider", min:0.05, max:1, step:0.01}
# threshold in log_e intensity change to trigger a positive/negative event. (default: 0.2)
sigma = 0.03 #@param {type:"slider", min:0.01, max:0.25, step:0.001}
# 1-std deviation threshold variation in log_e intensity change. (default: 0.03)
cutoff_hz = 200 #@param {type:"slider", min:0, max:300, step:1}
# photoreceptor first-order IIR lowpass filter cutoff-off 3dB frequency in Hz - see https://ieeexplore.ieee.org/document/4444573 (default: 300)
leak_rate_hz = 5.18 #@param {type:"slider", min:0, max:100, step:0.01}
#  leak event rate per pixel in Hz - see https://ieeexplore.ieee.org/abstract/document/7962235 (default: 0.01)
shot_noise_rate_hz = 2.716 #@param {type:"slider", min:0, max:100, step:0.001}
#  Temporal noise rate of ON+OFF events in darkest parts of scene; reduced in brightest parts. (default: 0.001)

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


v2e_command = ["v2e"]

# set the input folder
# the video_path can be a video file or a folder of images
v2e_command += ["-i", video_path]

# set the output folder
v2e_command += ["-o", output_folder]

# if the output will rewrite the previous output
if overwrite:
    v2e_command.append("--overwrite")

# if there the output folder is unique
v2e_command += ["--unique_output_folder", "{}".format(unique_output_folder).lower()]

# set output configs, for the sake of this tutorial, let's just output HDF5 record
if davis_output:
    v2e_command += ["--davis_output"]
v2e_command += ["--dvs_h5", out_filename]
v2e_command += ["--dvs_aedat2", "None"]
v2e_command += ["--dvs_text", "None"]

# in Colab, let's say no preview
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

# threshold
v2e_command += ["--pos_thres", "{}".format(thres)]
v2e_command += ["--neg_thres", "{}".format(thres)]

# sigma
v2e_command += ["--sigma_thres", "{}".format(sigma)]

# DVS non-idealities
v2e_command += ["--cutoff_hz", "{}".format(cutoff_hz)]
v2e_command += ["--leak_rate_hz", "{}".format(leak_rate_hz)]
v2e_command += ["--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz)]

# Final v2e command

final_v2e_command = " ".join(v2e_command)

print("The Final v2e command:")
print(final_v2e_command)
