import math
import os
import subprocess
import time

import ffmpeg
import numpy as np
from ffmpeg import Error

from video_montage.video_montage_config import VideoMontageConfig


class FFmpegProcessor:
    def __init__(
            self,
            path_to_ffmpeg_bin: str = 'bin/ffmpeg.exe',
    ):
        self.path_to_ffmpeg_bin = path_to_ffmpeg_bin

    def extract_audio(self, filename: str, sample_rate, format: str = 'f32le', acodec: str = 'pcm_f32le'):
        out, err = (
            ffmpeg
                .input(filename)
                .output('-', format=format, acodec=acodec, ac=1, ar=str(sample_rate))
                .run(cmd=self.path_to_ffmpeg_bin, capture_stdout=True, capture_stderr=True)
        )

        return np.frombuffer(out, np.float32)

    def _run(self, ffmpeg_output):
        full_cmd = ffmpeg.compile(ffmpeg_output, cmd=self.path_to_ffmpeg_bin)

        filter_str = None
        for i in range(len(full_cmd)):
            if full_cmd[i] == '-filter_complex':
                full_cmd[i] += '_script'
                filter_str = full_cmd[i + 1]
                full_cmd[i + 1] = 'filter.txt'
                break

        with open('filter.txt', 'w', encoding='utf8') as f:
            f.write(filter_str)

        args = full_cmd
        process = subprocess.Popen(args)
        out, err = process.communicate(input)
        retcode = process.poll()
        if retcode:
            raise Error('ffmpeg', out, err)

    def montage(self, filename, out_filename, ranges, config: VideoMontageConfig):
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
                aud = ffmpeg.filter([aud, mic], 'amix', duration='shortest',
                                    weights=f'1 {config.mic_volume_multiplier}')
            streams.append(vid)
            streams.append(aud)

        joined = ffmpeg.concat(*streams, v=1, a=1)
        output = ffmpeg.output(joined, out_filename, vcodec='hevc_nvenc', video_bitrate=config.video_bitrate)
        output = output.global_args('-loglevel', 'error')
        output = ffmpeg.overwrite_output(output)

        start_time = time.time()

        self._run(ffmpeg_output=output)

        elapsed = time.time() - start_time
        print(f'Elapsed {elapsed:.2f} seconds\n')
