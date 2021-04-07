from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VideoConfig:
    bitrate_megabits: float = 50

    @property
    def bitrate(self):
        return str(int(self.bitrate_megabits * 1e6))


@dataclass(frozen=True)
class PeakDetectionConfig:
    peak_height: float = 0.9
    peak_threshold: float = 0.15
    freq_range: Tuple[int, int] = (0, 40)
    min_peaks_in_a_row: int = 2


@dataclass(frozen=True)
class RangeConfig:
    min_duration: float = 1
    min_distance: float = 4
    extend_before_start: float = 2
    extend_after_end: float = 2


@dataclass(frozen=True)
class MicrophoneConfig:
    mic_volume_multiplier: float = 3
    mix_mic_audio_track: bool = True


@dataclass(frozen=True)
class MusicConfig:
    chance: float = 0.5
    volume: float = 0.3
    random_seed_by_file: bool = False


@dataclass(frozen=True)
class MontageConfig:
    sample_rate = 44100
    input_dir: str
    output_dir: str
    video: VideoConfig
    detection: PeakDetectionConfig
    range: RangeConfig
    microphone: MicrophoneConfig
    music: MusicConfig
    '''
    :param input_dir:
    :param output_dir:
    :param mic_volume_multiplier:
    :param freq_range: range of frequencies to sum volume of
    :param peak_height:
    :param peak_threshold:
    :param min_distance: distance between peaks to unite them to single time range
    :param min_peaks_in_a_row: if count of peaks in range is less than this value than the range is ignored
    :param min_duration
    :param extend_range_bounds_by_seconds
    :param mix_mic_audio_track:
        if True - mix second audio track into resulting video
        (shadow play can record your mic into separate audio track)
    '''
