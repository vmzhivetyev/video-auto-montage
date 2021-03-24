import os
import re

from video_montage.ffmpeg_processor import FFmpegProcessor
from video_montage.segments_builder import SegmentsBuilder
from video_montage.video_montage_config import VideoMontageConfig


def cut_video_into_single(filename, config: VideoMontageConfig):
    out_filename = str(os.path.join(config.output_dir, os.path.basename(filename)))
    out_filename = re.sub(r'\.[^.]+?$', '.mp4', out_filename)

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)

    if os.path.isfile(out_filename):
        print(f"Already exists '{out_filename}'")
        return
    else:
        print(f"Start processing file: '{out_filename}'")

    ffmpeg_processor = FFmpegProcessor()
    segments_builder = SegmentsBuilder(config=config)

    audio = ffmpeg_processor.extract_audio(filename, config.sample_rate)
    sec_ranges = segments_builder.make_sec_ranges(audio)

    if len(sec_ranges) == 0:
        print(f"No ranges for file '{filename}'")
    else:
        print(f"Found {len(sec_ranges)} for '{filename}'")

        os.makedirs(config.output_dir, exist_ok=True)

        ffmpeg_processor.montage(filename, out_filename, ranges=sec_ranges, config=config)


def file_list_from_dir(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def run_file(input_file, config: VideoMontageConfig):
    # plot_audio(input_file, (0, 00), (2, 00), config=config)
    cut_video_into_single(filename=input_file, config=config)


def run_directory(config: VideoMontageConfig):
    # pool = Pool(2)
    # pool.starmap(run_file, zip(file_list_from_dir(config.input_dir), repeat(config)), chunksize=1)
    # pool.close()
    # pool.join()
    #
    for file in file_list_from_dir(config.input_dir):
        run_file(file, config)


if __name__ == "__main__":
    apex = VideoMontager(VideoMontageConfig(
        input_dir='D:\Videos\Apex Legends',
        output_dir='vids\Apex Legends',
        bitrate_megabits=50,
        mic_volume_multiplier=3,
        freq_range=(0, 40),
        peak_height=1.3,
        peak_threshold=0.1,
        max_seconds_between_peaks=2,
        min_count_of_peaks=1,
        extend_range_bounds_by_seconds=1,
        min_duration_of_valid_range=0
    )

    run_directory(config=apex)

    quake = VideoMontageConfig(
        input_dir='D:\Videos\Quake Champions',
        output_dir='vids\Quake Champions',
        bitrate_megabits=50,
        mic_volume_multiplier=1,
        peak_height=1.3,
        peak_threshold=0.1,
        max_seconds_between_peaks=2,
        min_count_of_peaks=1,
        extend_range_bounds_by_seconds=1,
        mix_mic_audio_track=False,
    )

    run_directory(config=quake)

