import ffmpeg
from ffmpeg import Error
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib.pyplot import figure
import math
import os

ffmpeg_cmd = 'bin/ffmpeg.exe'


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

    goods = [x for x in ranges if good(x)]
    return [(x[0] - 2, x[1] + 2) for x in goods]


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


def concat_ranges(filename, out_filename, ranges):
    """ ranges are in seconds """
    input_vid = ffmpeg.input(filename)

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
            input_vid['a:0']
                .filter_('atrim', start=start, end=end)
                .filter_('asetpts', 'PTS-STARTPTS')
        )
        mic = (
            input_vid['a:1']
                .filter_('atrim', start=start, end=end)
                .filter_('asetpts', 'PTS-STARTPTS')
        )

        full_aud = ffmpeg.filter([aud, mic], 'amix', duration='shortest')

        streams.append(vid)
        streams.append(full_aud)

    joined = ffmpeg.concat(*streams, v=1, a=1)
    output = ffmpeg.output(joined, out_filename)
    output = output.global_args('-loglevel', 'error')
    output = ffmpeg.overwrite_output(output)

    print(' '.join([f'"{x}"' for x in ffmpeg.compile(output, cmd=ffmpeg_cmd)]).replace('/', '\\'))
    output.run(cmd=ffmpeg_cmd)


def make_sec_ranges(filename):
    audio = ap.extract_audio(filename, sample_rate)

    ranges = peak_ranges(audio, 4, 2, sample_rate)
    ranges = filter_ranges(ranges, 1)

    sec_ranges = [(x[0] / sample_rate, x[1] / sample_rate) for x in ranges]

    print(sec_ranges)

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

def cut_video_into_single(filename, out_dir):
    out_filename = f'{out_dir}/{filename.split("/")[-1]}'

    if os.path.isfile(out_filename):
        print_log(f"Already exists '{out_filename}'")
        return

    sec_ranges = make_sec_ranges(filename)

    if len(sec_ranges) == 0:
        print_log(f"!!!!! NO RANGES FOR FILE '{filename}' !!!!! ")
    else:
        print_log(f"{len(sec_ranges)} for '{filename}'")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        concat_ranges(filename, out_filename, sec_ranges)


# video_file_1 = "vids/Desktop 08.28.2017 - 16.41.29.05.DVR.mp4"  # 1:05 - 1:20 silenced scar
# video_file_2 = "vids/Desktop 08.31.2017 - 22.50.22.05.DVR.mp4"  # 4:25 - 4:40 akm

# cut_video_into_parts(video_file_1)
# cut_video_into_parts(video_file_2)

# plot_audio(video_file_1, (4, 20), (4, 40), sample_rate)
# plot_audio(video_file_2, (0, 50), (0, 60), sample_rate)

def file_list_from_dir(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


files = file_list_from_dir("E:/shadow play/replays/Apex Legends/")

for file in files:
    cut_video_into_single(file, 'vids/apex')
# with open("vids/ranges.txt", "w") as ranges_log:
    #log_video_ranges(file, ranges_log)