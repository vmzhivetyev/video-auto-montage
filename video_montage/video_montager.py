import os
import re
from glob import glob

from video_montage.ffmpeg_processor import FFmpegProcessor
from video_montage.segments_builder import SegmentsBuilder
from video_montage.video_montage_config import MontageConfig


class VideoMontager:
    def __init__(self, config: MontageConfig):
        self.config = config
        self.ffmpeg_processor = FFmpegProcessor()
        self.segments_builder = SegmentsBuilder(config)

    def cut_video_into_single(self, input_filepath):
        out_filename = str(os.path.join(self.config.output_dir, os.path.basename(input_filepath)))
        out_filename = re.sub(r'\.[^.]+?$', '.mp4', out_filename)

        os.makedirs(os.path.dirname(out_filename), exist_ok=True)

        if os.path.isfile(out_filename):
            print(f"Already exists '{out_filename}'")
            return
        else:
            print(f"Start processing file: '{out_filename}'")

        ffmpeg_processor = FFmpegProcessor()
        segments_builder = SegmentsBuilder(config=self.config)

        audio = ffmpeg_processor.extract_audio(input_filepath, self.config.sample_rate)
        sec_ranges = segments_builder.make_sec_ranges(audio)

        if len(sec_ranges) == 0:
            print(f"No ranges for file '{input_filepath}'")
        else:
            print(f"Found {len(sec_ranges)} for '{input_filepath}'")

            os.makedirs(self.config.output_dir, exist_ok=True)

            ffmpeg_processor.montage(input_filepath, out_filename, ranges=sec_ranges, config=self.config)

    def run_file(self, filepath):
        self.cut_video_into_single(input_filepath=filepath)

    def run_directory(self):
        for filepath in glob(os.path.join(self.config.input_dir, '*')):
            self.run_file(filepath)
