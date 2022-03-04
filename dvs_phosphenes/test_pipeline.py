from dvs_phosphenes.EventRepresentation import EventRepresentation
from dvs_phosphenes.EventStream import EventStream, convert_video_to_events_v2e_command

video_path = '/home/duinch/PycharmProjects/dvs_phosphenes/data/v2e_tutorial_video.avi'
output_folder = '/home/duinch/PycharmProjects/dvs_phosphenes/demo_output'
out_filename = "events.h5"
input_file = '/home/duinch/PycharmProjects/dvs_phosphenes/demo_output/events.h5'

eventstream_h5 = EventStream()
eventstream_h5.read_h5_file(input_file)
# eventstream_h5.convert_video_to_events_v2e_emulator(video_path, output_folder)
# convert_video_to_events_v2e_command(video_path, output_folder, "events_v2e.h5")

out_plot_tb_3 = '/home/duinch/PycharmProjects/dvs_phosphenes/data/n_time_bins_3.png'
frames = EventRepresentation()
frames.get_eventstream(eventstream_h5)
frames.denoise()
frames.convert_to_time_surface(surface_dimensions=None, tau=10000, decay='exp')
# frames.convert_to_frame(n_time_bins = 1000)
# frames.plot_frames(out_plot_tb_3)
