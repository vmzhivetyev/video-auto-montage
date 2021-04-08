import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path

# from video_montage.ffmpeg_processor import FFmpegProcessor
# from video_montage.segments_builder import SegmentsBuilder
# from video_montage.video_montage_config import MontageConfig


# def plot_audio(filename, start, end, config: MontageConfig):
#     segments_builder = SegmentsBuilder(config)
#     ffmpeg_processor = FFmpegProcessor()
#
#     start, end = audio_range(config.sample_rate, start, end)
#     audio = ffmpeg_processor.extract_audio(filename, config.sample_rate)[start:end]
#
#     plt.figure(1)
#
#     def show_data_and_peaks(plot, data, peaks, ranges):
#         plot.plot(data)
#         plot.plot(peaks, data[peaks], "x")
#         starts = [x[0] for x in ranges]
#         ends = [x[1] for x in ranges]
#         plot.plot(starts, [max(data) * 1.1] * len(starts), "g+")
#         plot.plot(ends, [max(data) * 1.1] * len(ends), "r+")
#
#     plot_a = plt.subplot(411)
#     show_data_and_peaks(plot_a, audio, *segments_builder.peak_ranges(audio))
#     plot_a.set_xlabel('sample rate * time')
#     plot_a.set_ylabel('energy')
#
#     plot_b = plt.subplot(412)
#     plot_b.specgram(audio, NFFT=segments_builder.nfft, Fs=config.sample_rate, noverlap=segments_builder.noverlap)
#     plot_b.set_xlabel('Time')
#     plot_b.set_ylabel('Frequency')
#
#     plot_c = plt.subplot(413)
#     speq, lows = segments_builder.fft_of_lows(audio)
#     plot_c.imshow(np.log(speq), cmap='viridis', aspect='auto')
#
#     plot_d = plt.subplot(414)
#     show_data_and_peaks(plot_d, lows, *segments_builder.peak_ranges(lows))
#
#     plt.show()


def time_to_sec(time):
    """ time is tuple of (minutes:seconds) """
    return time[0] * 60 + time[1]


def audio_range(sample_rate, start_time=(0, 0), end_time=(0, 10)):
    """ time is tuple of (minutes:seconds) """
    return int(sample_rate * time_to_sec(start_time)), int(sample_rate * time_to_sec(end_time))


def sec_to_time(sec):
    return int(sec / 60), sec % 60


def file_list_from_dir(dir_path, recursive=False):
    if recursive:
        return [str(x) for x in list(Path(dir_path).rglob('*')) if x.is_file()]

    return [x for x in [os.path.join(dir_path, x) for x in os.listdir(dir_path)]
            if os.path.isfile(x)]
