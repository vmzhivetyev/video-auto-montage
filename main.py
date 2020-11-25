import ffmpeg
from ffmpeg import Error
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib.pyplot import figure
import math

ffmpeg_cmd = 'bin/ffmpeg.exe'

# in1 = ffmpeg.input('vids/Desktop 08.28.2017 - 16.41.29.05.DVR.mp4')
# in2 = ffmpeg.input('in2.mp4')
# v1 = in1.video.hflip()
# a1 = in1.audio
# v2 = in2.video.filter('reverse').filter('hue', s=0)
# a2 = in2.audio.filter('areverse').filter('aphaser')
# joined = ffmpeg.concat(v1, a1, v2, a2, v=1, a=1).node
# v3 = joined[0]
# a3 = joined[1].filter('volume', 0.8)
# out = ffmpeg.output(v3, a3, 'out.mp4')
# out.run()


class FFmpegProcessor:
    def __init__(self):
        self.cmd = ffmpeg_cmd

    def extract_audio(self, filename, sample_rate):
        try:
            out, err = (
                ffmpeg
                    .input(filename)
                    .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=str(sample_rate))
                    .run(cmd=ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
            )
        except Error as err:
            print(err.stderr)
            raise

        return np.frombuffer(out, np.float32)


ap = FFmpegProcessor()

sample_rate = 44100


def peak_ranges(audio, max_distance_sec, min_count, sample_rate):
    # todo: widen ranges by lets say 1 second
    peaks, _ = find_peaks(audio, height=0.9, threshold=0.2)

    max_distance = max_distance_sec * sample_rate

    ranges = []
    start = -1
    last = -1
    count = 0
    for x in peaks:
        if start == -1:
            start = x
            last = x
            count = 1
        else:
            dist = x - last
            if dist < max_distance:
                last = x
                count = count + 1
            else:
                if count >= min_count:
                    ranges.append((start, last))
                start = x
                last = x
                count = 1

    if start != -1 and last != -1:
        ranges.append((start, last))

    return ranges


def plot_audio(filename, start, end, sample_rate):
    start, end = audio_range(sample_rate, start, end)
    audio = ap.extract_audio(filename, sample_rate)[start:end]

    plt.figure(1)

    ranges = peak_ranges(audio, 2, 3, sample_rate)

    plot_a = plt.subplot(211)
    plot_a.plot(audio)
    # plot_a.plot(peaks, audio[peaks], "x")
    # plot_a.plot(np.convolve(audio, np.ones(50) / 50, mode='full'))

    starts = [x[0] for x in ranges]
    ends = [x[1] for x in ranges]
    plot_a.plot(starts, audio[starts], "g+")
    plot_a.plot(ends, audio[ends], "r+")

    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('energy')

    plot_b = plt.subplot(212)
    plot_b.specgram(audio, NFFT=1024, Fs=sample_rate, noverlap=100)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

    plt.show()


def time_to_sec(time):
    """ time is tuple of (minutes:seconds) """
    return time[0] * 60 + time[1]


def audio_range(sample_rate, start_time=(0, 0), end_time=(0, 10)):
    """ time is tuple of (minutes:seconds) """
    return int(sample_rate * time_to_sec(start_time)), int(sample_rate * time_to_sec(end_time))

def filter_ranges(ranges, min_length_sec):
    min_length = min_length_sec * sample_rate

    def good(r):
        return (r[1] - r[0]) > min_length

    return [x for x in ranges if good(x)]


def sec_to_time(sec):
    return int(sec / 60), sec % 60


def cut_ranges(filename, ranges):
    """ ranges are in seconds """
    input_vid = ffmpeg.input(filename)

    import os
    dir = f'{filename[:-4]}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    count = 0
    for r in ranges:
        start = int(r[0])
        end = math.ceil(r[1])
        out_filename = f'{dir}/out_{count}.mp4'

        print(f'{filename}: Trimming {out_filename} (of {len(ranges)}) from {start} to {end}')

        vid = (
            input_vid
                .trim(start=start, end=end)
                .setpts('PTS-STARTPTS')
        )
        aud = (
            input_vid
                .filter_('atrim', start=start, end=end)
                .filter_('asetpts', 'PTS-STARTPTS')
        )

        joined = ffmpeg.concat(vid, aud, v=1, a=1)
        output = ffmpeg.output(joined, out_filename)
        output = ffmpeg.overwrite_output(output)
        output.run(cmd=ffmpeg_cmd)

        count = count + 1


def cut_video(filename):
    audio = ap.extract_audio(filename, sample_rate)

    ranges = peak_ranges(audio, 4, 3, sample_rate)
    ranges = filter_ranges(ranges, 1)

    sec_ranges = [(x[0] / sample_rate, x[1] / sample_rate) for x in ranges]
    time_ranges = [(sec_to_time(x[0]), sec_to_time(x[1])) for x in sec_ranges]

    print(sec_ranges)
    print(time_ranges)

    cut_ranges(filename, sec_ranges)


video_file_1 = "vids/Desktop 08.28.2017 - 16.41.29.05.DVR.mp4"  # 1:05 - 1:20 silenced scar
video_file_2 = "vids/Desktop 08.31.2017 - 22.50.22.05.DVR.mp4"  # 4:25 - 4:40 akm

cut_video(video_file_1)
cut_video(video_file_2)

# plot_audio(video_file_1, (4, 20), (4, 40), sample_rate)
# plot_audio(video_file_2, (0, 50), (0, 60), sample_rate)





