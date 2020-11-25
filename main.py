import ffmpeg
from ffmpeg import Error
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib.pyplot import figure
import math
import os

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


def concat_ranges(filename, ranges):
    """ ranges are in seconds """
    input_vid = ffmpeg.input(filename)

    dir = 'vids/done'
    if not os.path.exists(dir):
        os.makedirs(dir)

    out_filename = f'{dir}/{filename.split("/")[-1]}'

    if os.path.isfile(out_filename):
        print_log(f"Already exists '{out_filename}'")
        return

    print(f'Processing {out_filename} ({len(ranges)} ranges)')

    streams = []

    for r in ranges:
        start = int(r[0])
        end = math.ceil(r[1])

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

        streams.append(vid)
        streams.append(aud)

    joined = ffmpeg.concat(*streams, v=1, a=1)
    output = ffmpeg.output(joined, out_filename)
    output = ffmpeg.overwrite_output(output)
    output.run(cmd=ffmpeg_cmd)


def make_sec_ranges(filename):
    audio = ap.extract_audio(filename, sample_rate)

    ranges = peak_ranges(audio, 4, 2, sample_rate)
    ranges = filter_ranges(ranges, 1)

    sec_ranges = [(x[0] / sample_rate, x[1] / sample_rate) for x in ranges]
    time_ranges = [(sec_to_time(x[0]), sec_to_time(x[1])) for x in sec_ranges]

    print(sec_ranges)
    print(time_ranges)

    return sec_ranges


def cut_video_into_parts(filename):
    sec_ranges = make_sec_ranges(filename)
    cut_ranges(filename, sec_ranges)


def print_log(msg):
    with open("vids/skipped.txt", "a") as myfile:
        myfile.write(msg)

def log_video_ranges(filename, log):
    ranges = make_sec_ranges(filename)
    log.write(filename + '\n')
    for r in ranges:
        log.write(str(r) + '\n')

def cut_video_into_single(filename):
    sec_ranges = make_sec_ranges(filename)

    if len(sec_ranges) == 0:
        print_log(f"!!!!! NO RANGES FOR FILE '{filename}' !!!!! ")
    else:
        print_log(f"{len(sec_ranges)} for '{filename}'")
        concat_ranges(filename, sec_ranges)


video_file_1 = "vids/Desktop 08.28.2017 - 16.41.29.05.DVR.mp4"  # 1:05 - 1:20 silenced scar
video_file_2 = "vids/Desktop 08.31.2017 - 22.50.22.05.DVR.mp4"  # 4:25 - 4:40 akm

# cut_video_into_parts(video_file_1)
# cut_video_into_parts(video_file_2)

# cut_video_into_single(video_file_1)

# plot_audio(video_file_1, (4, 20), (4, 40), sample_rate)
# plot_audio(video_file_2, (0, 50), (0, 60), sample_rate)

files_list = """Desktop 09.22.2017 - 21.36.33.05.DVR.mp4
Desktop 10.02.2017 - 01.11.35.05.DVR.mp4
Desktop 10.04.2017 - 03.38.48.05.DVR.mp4
Desktop 10.06.2017 - 00.48.20.05.DVR.mp4
Desktop 10.08.2017 - 01.19.50.05.DVR.mp4
Desktop 10.15.2017 - 20.22.50.62.DVR.mp4
Desktop 10.16.2017 - 02.56.35.05.DVR.mp4
Desktop 10.17.2017 - 01.15.51.05.DVR.mp4
Desktop 10.19.2017 - 23.56.49.05.DVR.mp4
Desktop 10.22.2017 - 19.30.07.05.DVR.mp4
Desktop 10.24.2017 - 01.49.25.05.DVR.mp4
Desktop 10.24.2017 - 14.59.20.05.DVR.mp4
Desktop 10.25.2017 - 02.44.51.11.DVR.mp4
Desktop 10.28.2017 - 19.41.23.05.DVR.mp4
Desktop 10.30.2017 - 00.22.44.05.DVR.mp4
Desktop 10.31.2017 - 06.36.26.05.DVR.mp4
Desktop 11.01.2017 - 13.02.25.05.DVR.mp4
Desktop 11.01.2017 - 13.39.12.05.DVR.mp4
Desktop 11.09.2017 - 16.26.18.05.DVR.mp4
Desktop 11.12.2017 - 17.32.47.05.DVR.mp4
Desktop 11.16.2017 - 03.15.31.05.DVR.mp4
Desktop 11.19.2017 - 17.34.27.05.DVR.mp4
Desktop 12.10.2017 - 04.26.58.05.DVR.mp4
Desktop 12.17.2017 - 02.02.45.05.DVR.mp4
Desktop 12.20.2017 - 15.37.52.05.DVR.mp4
Desktop 12.23.2017 - 03.25.42.05.DVR.mp4
Desktop 12.26.2017 - 16.30.51.05.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 03.02.2018 - 05.21.29.05.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 03.11.2018 - 02.45.43.02.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 03.11.2018 - 02.55.52.03.DVR.mp4
PlayerUnknown's Battlegrounds 12.25.2017 - 05.01.10.14.DVR.mp4
PlayerUnknown's Battlegrounds 12.25.2017 - 05.20.16.17.DVR.mp4
PlayerUnknown's Battlegrounds 12.25.2017 - 05.21.59.20.DVR.mp4
PlayerUnknown's Battlegrounds 12.29.2017 - 23.06.41.13.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2018.07.08 - 22.50.06.02.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2018.07.09 - 02.07.32.03.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2018.07.09 - 02.16.21.04.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2018.12.19 - 22.37.27.10.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.02.09 - 06.27.39.22.DVR.1549683908413.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.02.14 - 20.27.26.113.DVR.1550183206257.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.02.25 - 21.19.58.26.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.02.25 - 23.39.46.41.DVR.1551127230291.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.02.28 - 03.57.48.21.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.02.28 - 04.02.10.24.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 04.40.50.88.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 21.23.57.95.DVR.1573239060421.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 21.27.20.96.DVR.1573239036546.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 21.40.42.97.DVR.1573238998647.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 21.47.53.98.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 21.48.25.99.DVR.1573238960847.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 22.27.01.101.DVR.1573241313219.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.08 - 22.27.45.102.DVR.1573241292709.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2019.11.09 - 13.16.45.122.DVR.1573294648954.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2020.04.05 - 22.52.55.18.DVR.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2020.04.24 - 09.11.48.63.DVR.1587708863505.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2020.04.29 - 19.35.37.107.DVR.1588178380155.mp4
PLAYERUNKNOWN'S BATTLEGROUNDS 2020.05.03 - 06.20.29.38.DVR.mp4
pubg 09.20.2017 - 02.24.06.08.DVR.mp4
pubg 10.11.2017 - 01.33.41.29.DVR.mp4
pubg 10.14.2017 - 13.48.09.40.DVR.mp4
pubg 10.21.2017 - 19.33.52.14.DVR.mp4
pubg 10.24.2017 - 20.46.50.23.DVR.mp4""".split('\n')

with open("vids/ranges.txt", "w") as ranges_log:
    for f in files_list:
        file = "E:/ShadowPlay-old/NvidiaReplays/PLAYERUNKNOWN'S BATTLEGROUNDS/" + f
        # cut_video_into_single(file)
        log_video_ranges(file, ranges_log)