import ffmpeg
from ffmpeg import Error
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib.pyplot import figure
import math
import os
import time

ffmpeg_cmd = 'bin/ffmpeg.exe'


class VideoMontageConfig:
    def __init__(self,
                 input_dir,
                 output_dir,
                 bitrate_megabits=50,
                 mic_volume_multiplier=3,
                 peak_height=0.9,
                 peak_threshold=0.15,
                 max_seconds_between_peaks=4,
                 min_count_of_peaks=2,
                 min_duration_of_valid_range=1,
                 extend_range_bounds_by_seconds=2):
        '''
        :param input_dir:
        :param output_dir:
        :param bitrate_megabits:
        :param mic_volume_multiplier:
        :param peak_height:
        :param peak_threshold:
        :param max_seconds_between_peaks: distance between peaks to unite them to single time range
        :param min_count_of_peaks: if count of peaks in range is less than this value than the range is ignored
        :param min_duration_of_valid_range
        :param extend_range_bounds_by_seconds
        '''
        self.input_dir = input_dir
        self.peak_threshold = peak_threshold
        self.peak_height = peak_height
        self.mic_volume_multiplier = mic_volume_multiplier
        self.bitrate_megabits = bitrate_megabits
        self.output_dir = output_dir
        self.max_seconds_between_peaks = max_seconds_between_peaks
        self.min_count_of_peaks = min_count_of_peaks
        self.min_duration_of_valid_range = min_duration_of_valid_range
        self.extend_range_bounds_by_seconds = extend_range_bounds_by_seconds

    @property
    def video_bitrate(self):
        return str(int(self.bitrate_megabits * 1e6))


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

SAMPLE_RATE = 44100


def peak_ranges(audio, config: VideoMontageConfig):
    """
        max_distance_sec: max distance between peaks (in seconds) which can be used to increase count of peaks range
        min_count: min count of peaks in range to include it in result

        :returns array of valid ranges
    """
    peaks, _ = find_peaks(audio, height=config.peak_height, threshold=config.peak_threshold)

    max_distance = config.max_seconds_between_peaks * SAMPLE_RATE

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
                if count >= config.min_count_of_peaks:
                    ranges.append((start, last))
                start = x
                last = x
                count = 1

    if start != -1 and last != -1:
        ranges.append((start, last))

    return ranges


def plot_audio(filename, start, end):
    start, end = audio_range(SAMPLE_RATE, start, end)
    audio = ap.extract_audio(filename, SAMPLE_RATE)[start:end]

    plt.figure(1)

    ranges = peak_ranges(audio, VideoMontageConfig(None,
                                                   None,
                                                   peak_height=0.7,
                                                   peak_threshold=0.15,
                                                   max_seconds_between_peaks=2,
                                                   min_count_of_peaks=3))

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
    plot_b.specgram(audio, NFFT=1024, Fs=SAMPLE_RATE, noverlap=100)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

    plt.show()


def time_to_sec(time):
    """ time is tuple of (minutes:seconds) """
    return time[0] * 60 + time[1]


def audio_range(sample_rate, start_time=(0, 0), end_time=(0, 10)):
    """ time is tuple of (minutes:seconds) """
    return int(sample_rate * time_to_sec(start_time)), int(sample_rate * time_to_sec(end_time))


def filter_ranges(ranges, config: VideoMontageConfig):
    min_length = config.min_duration_of_valid_range * SAMPLE_RATE

    goods = [x for x in ranges if x[1] - x[0] > min_length]

    return [(x[0] - config.extend_range_bounds_by_seconds,
             x[1] + config.extend_range_bounds_by_seconds)
            for x in goods]


def sec_to_time(sec):
    return int(sec / 60), sec % 60


def cut_ranges(filename, ranges):
    """ ranges are in seconds """
    raise AssertionError('Deprecated! Look at concat_ranges!')

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


def concat_ranges(filename, out_filename, ranges, config: VideoMontageConfig):
    """ ranges are in seconds """

    assert os.path.isfile(filename)

    input_vid = ffmpeg.input(filename)

    total_duration = sum([x[1] - x[0] for x in ranges])
    print(f'Processing {out_filename} ({len(ranges)} ranges -> {total_duration:.0f} seconds)')

    streams = []

    for r in ranges:
        start = int(r[0])
        end = math.floor(r[1])

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
        full_aud = ffmpeg.filter([aud, mic], 'amix', duration='shortest', weights=f'1 {config.mic_volume_multiplier}')

        streams.append(vid)
        streams.append(full_aud)

    joined = ffmpeg.concat(*streams, v=1, a=1)
    output = ffmpeg.output(joined, out_filename, vcodec='hevc_nvenc', video_bitrate=config.video_bitrate)
    output = output.global_args('-loglevel', 'error')
    output = ffmpeg.overwrite_output(output)

    start_time = time.time()
    # print(' '.join([f'"{x}"' for x in ffmpeg.compile(output, cmd=ffmpeg_cmd)]).replace('/', '\\'))
    output.run(cmd=ffmpeg_cmd)
    elapsed = time.time() - start_time
    print(f'Elapsed {elapsed:.2f} seconds\n')


def make_sec_ranges(filename, config: VideoMontageConfig):
    audio = ap.extract_audio(filename, SAMPLE_RATE)

    ranges = peak_ranges(audio, config=config)
    ranges = filter_ranges(ranges, config=config)

    sec_ranges = [(x[0] / SAMPLE_RATE, x[1] / SAMPLE_RATE) for x in ranges]
    return sec_ranges


def cut_video_into_parts(filename, config: VideoMontageConfig):
    sec_ranges = make_sec_ranges(filename, config=config)
    cut_ranges(filename, sec_ranges)


def print_log(msg):
    print(msg)
    # with open("vids/skipped.txt", "a") as file:
    #     file.write(msg)


def log_video_ranges(ranges, filename, log):
    log.write(filename + '\n')
    for r in ranges:
        log.write(str(r) + '\n')


def cut_video_into_single(filename, config: VideoMontageConfig):
    out_filename = os.path.join(config.output_dir, os.path.basename(filename))

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)

    if os.path.isfile(out_filename):
        print_log(f"Already exists '{out_filename}'")
        return

    sec_ranges = make_sec_ranges(filename, config=config)

    # with open("vids/ranges.txt", "w") as ranges_log:
    #     log_video_ranges(sec_ranges, filename, ranges_log)

    if len(sec_ranges) == 0:
        print_log(f"No ranges for file '{filename}'")
    else:
        print_log(f"Found {len(sec_ranges)} for '{filename}'")

        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        concat_ranges(filename, out_filename, sec_ranges, config=config)


# video_file_1 = "vids/Desktop 08.28.2017 - 16.41.29.05.DVR.mp4"  # 1:05 - 1:20 silenced scar
# video_file_2 = "vids/Desktop 08.31.2017 - 22.50.22.05.DVR.mp4"  # 4:25 - 4:40 akm

# cut_video_into_parts(video_file_1)
# cut_video_into_parts(video_file_2)

# plot_audio(video_file_1, (4, 20), (4, 40), sample_rate)
# plot_audio(video_file_2, (0, 50), (0, 60), sample_rate)

def file_list_from_dir(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def run_file(input_file, config: VideoMontageConfig):
    cut_video_into_single(filename=input_file, config=config)


def run_directory(config: VideoMontageConfig):
    for file in file_list_from_dir(config.input_dir):
        run_file(input_file=file, config=config)


if __name__ == "__main__":
    pubg = VideoMontageConfig(
        input_dir='E:/shadow play/replays/PLAYERUNKNOWN\'S BATTLEGROUNDS',
        output_dir='vids/pubg',
        bitrate_megabits=50,
        mic_volume_multiplier=3,
        peak_height=0.9,
        peak_threshold=0.2)

    apex = VideoMontageConfig(
        input_dir='E:/shadow play/replays/Apex Legends',
        output_dir='vids/apex',
        bitrate_megabits=50,
        mic_volume_multiplier=3,
        peak_height=0.6,
        peak_threshold=0.2,
        max_seconds_between_peaks=4,
        min_count_of_peaks=1,
        extend_range_bounds_by_seconds=1,
        min_duration_of_valid_range=0
    )

    run_directory(config=apex)
