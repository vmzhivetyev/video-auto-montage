import math
import os
import re
import time

import ffmpeg
import numpy as np
from ffmpeg import Error
from matplotlib import mlab
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

ffmpeg_cmd = 'bin/ffmpeg.exe'


class VideoMontageConfig:
    def __init__(self,
                 input_dir,
                 output_dir,
                 bitrate_megabits=50.0,
                 mic_volume_multiplier=3,
                 peak_height=0.9,
                 peak_threshold=0.15,
                 max_seconds_between_peaks: float = 4,
                 min_count_of_peaks=2,
                 min_duration_of_valid_range: float = 1,
                 extend_range_bounds_by_seconds: float = 2,
                 mix_mic_audio_track: bool = True):
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
        :param mix_mic_audio_track:
            if True - mix second audio track into resulting video
            (shadow play can record your mic into separate audio track)
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
        self.mix_mic_audio_track = mix_mic_audio_track

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


def make_fft(x, NFFT=None, Fs=None, Fc=None, detrend=None,
             window=None, noverlap=None,
             cmap=None, xextent=None, pad_to=None, sides=None,
             scale_by_freq=None, mode=None, scale=None,
             vmin=None, vmax=None, **kwargs):
    if NFFT is None:
        NFFT = 256  # same default as in mlab.specgram()
    if Fc is None:
        Fc = 0  # same default as in mlab._spectral_helper()
    if noverlap is None:
        noverlap = 128  # same default as in mlab.specgram()
    if Fs is None:
        Fs = 2  # same default as in mlab._spectral_helper()

    if mode == 'complex':
        raise ValueError('Cannot plot a complex specgram')

    if scale is None or scale == 'default':
        if mode in ['angle', 'phase']:
            scale = 'linear'
        else:
            scale = 'dB'
    elif mode in ['angle', 'phase'] and scale == 'dB':
        raise ValueError('Cannot use dB scale with angle or phase mode')

    spec, _, _ = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs,
                               detrend=detrend, window=window,
                               noverlap=noverlap, pad_to=pad_to,
                               sides=sides,
                               scale_by_freq=scale_by_freq,
                               mode=mode)

    return spec


def fft_of_lows(audio):
    speq = make_fft(audio, NFFT=256, Fs=SAMPLE_RATE, noverlap=100)
    speq = np.array([x[:40] for x in speq.T]).T
    low_freq_volumes = np.array([sum(x) * 1000 for x in speq.T]).T
    return speq, low_freq_volumes


def peak_ranges_of_lows(audio, config: VideoMontageConfig):
    speq, lows = fft_of_lows(audio)
    return peak_ranges(lows, config, mult=len(lows) / len(audio))


def peak_ranges(audio, config: VideoMontageConfig, mult=1.0):
    """
        max_distance_sec: max distance between peaks (in seconds) which can be used to increase count of peaks range
        min_count: min count of peaks in range to include it in result

        :returns array of valid ranges
    """
    peaks, _ = find_peaks(audio, height=config.peak_height, threshold=config.peak_threshold)
    # peaks_new, _ = peakdet(audio, delta=config.peak_threshold)
    # peaks_new = [int(x[0]) for x in peaks if x[1] >= config.peak_height]

    max_distance = int(config.max_seconds_between_peaks * SAMPLE_RATE * mult)

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
            if dist <= max_distance:
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

    peaks = [int(x / mult) for x in peaks]
    ranges = [(int(x / mult), int(y / mult)) for x, y in ranges]

    return peaks, ranges


def plot_audio(filename, start, end, config: VideoMontageConfig):
    start, end = audio_range(SAMPLE_RATE, start, end)
    audio = ap.extract_audio(filename, SAMPLE_RATE)[start:end]

    plt.figure(1)

    def show_data_and_peaks(plot, data, peaks, ranges):
        plot.plot(data)
        plot.plot(peaks, data[peaks], "x")
        starts = [x[0] for x in ranges]
        ends = [x[1] for x in ranges]
        plot.plot(starts, [max(data) * 1.1] * len(starts), "g+")
        plot.plot(ends, [max(data) * 1.1] * len(ends), "r+")

    plot_a = plt.subplot(411)
    show_data_and_peaks(plot_a, audio, *peak_ranges(audio, config=config))
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('energy')

    plot_b = plt.subplot(412)
    plot_b.specgram(audio, NFFT=256, Fs=SAMPLE_RATE, noverlap=100)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

    plot_c = plt.subplot(413)
    speq, lows = fft_of_lows(audio)
    plot_c.imshow(np.log(speq), cmap='viridis', aspect='auto')

    plot_d = plt.subplot(414)
    show_data_and_peaks(plot_d, lows, *peak_ranges(lows, config=config))

    plt.show()


def time_to_sec(time):
    """ time is tuple of (minutes:seconds) """
    return time[0] * 60 + time[1]


def audio_range(sample_rate, start_time=(0, 0), end_time=(0, 10)):
    """ time is tuple of (minutes:seconds) """
    return int(sample_rate * time_to_sec(start_time)), int(sample_rate * time_to_sec(end_time))


def filter_ranges(ranges, config: VideoMontageConfig):
    min_length = int(config.min_duration_of_valid_range * SAMPLE_RATE)

    return [x for x in ranges if x[1] - x[0] >= min_length]


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


def custom_ffmpeg_run(output, cmd):
    full_cmd = ffmpeg.compile(output, cmd=cmd)
    # print(' '.join([f'"{x}"' for x in full_cmd]))

    filter_str = None
    for i in range(0, len(full_cmd)):
        x = full_cmd[i]
        if x == '-filter_complex':
            full_cmd[i] = '-filter_complex_script'
            filter_str = full_cmd[i + 1]
            full_cmd[i + 1] = 'filter.txt'

    with open('filter.txt', 'w', encoding='utf8') as f:
        f.write(filter_str)

    # print(' '.join([f'"{x}"' for x in full_cmd]))
    import subprocess
    args = full_cmd
    process = subprocess.Popen(args)
    out, err = process.communicate(input)
    retcode = process.poll()
    if retcode:
        raise Error('ffmpeg', out, err)


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

        if config.mix_mic_audio_track:
            mic = (
                input_vid['a:1']
                    .filter_('atrim', start=start, end=end)
                    .filter_('asetpts', 'PTS-STARTPTS')
            )
            aud = ffmpeg.filter([aud, mic], 'amix', duration='shortest', weights=f'1 {config.mic_volume_multiplier}')

        streams.append(vid)
        streams.append(aud)

    joined = ffmpeg.concat(*streams, v=1, a=1)
    output = ffmpeg.output(joined, out_filename, vcodec='hevc_nvenc', video_bitrate=config.video_bitrate)
    output = output.global_args('-loglevel', 'error')
    output = ffmpeg.overwrite_output(output)

    start_time = time.time()

    custom_ffmpeg_run(output, ffmpeg_cmd)

    elapsed = time.time() - start_time
    print(f'Elapsed {elapsed:.2f} seconds\n')


def make_sec_ranges(filename, config: VideoMontageConfig):
    audio = ap.extract_audio(filename, SAMPLE_RATE)

    _, ranges = peak_ranges_of_lows(audio, config=config)
    ranges = filter_ranges(ranges, config=config)

    sec_ranges = [(x[0] / SAMPLE_RATE, x[1] / SAMPLE_RATE) for x in ranges]

    sec_ranges = [[x[0] - config.extend_range_bounds_by_seconds,
                   x[1] + config.extend_range_bounds_by_seconds]
                  for x in sec_ranges]

    def weld_overlapping_ranges():
        i = 0
        dropped = []
        while i < len(sec_ranges) - 1:
            if sec_ranges[i][1] > sec_ranges[i + 1][0]:
                sec_ranges[i][1] = sec_ranges[i + 1][1]
                dropped.append(i + 1)
                i += 1
            i += 1

        result = [x for idx, x in enumerate(sec_ranges) if idx not in dropped]
        return result

    new_ranges = weld_overlapping_ranges()
    while len(new_ranges) < len(sec_ranges):
        sec_ranges = new_ranges
        new_ranges = weld_overlapping_ranges()

    return new_ranges


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
    out_filename = str(os.path.join(config.output_dir, os.path.basename(filename)))
    out_filename = re.sub(r'\.[^.]+?$', '.mp4', out_filename)

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

        os.makedirs(config.output_dir, exist_ok=True)

        concat_ranges(filename, out_filename, sec_ranges, config=config)


def file_list_from_dir(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def run_file(input_file, config: VideoMontageConfig):
    # plot_audio(input_file, (0, 00), (2, 00), config=config)
    cut_video_into_single(filename=input_file, config=config)


def run_directory(config: VideoMontageConfig):
    for file in file_list_from_dir(config.input_dir):
        run_file(input_file=file, config=config)


if __name__ == "__main__":
    apex = VideoMontageConfig(
        input_dir='D:\Videos\Apex Legends',
        output_dir='vids\Apex Legends',
        bitrate_megabits=50,
        mic_volume_multiplier=3,
        peak_height=1.3,
        peak_threshold=0.1,
        max_seconds_between_peaks=2,
        min_count_of_peaks=1,
        extend_range_bounds_by_seconds=1,
        min_duration_of_valid_range=0
    )

    run_directory(config=apex)

    # video_file_1 = "E:/shadow play/replays/Apex Legends/Apex Legends 2021.03.09 - 20.39.17.02.mp4"  # 1:05 - 1:20 silenced scar
    # plot_audio(video_file_1, (2, 20), (2, 30), config=apex)
