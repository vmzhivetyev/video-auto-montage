class VideoMontageConfig:
    sample_rate = 44100

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 bitrate_megabits: float = 50.0,
                 mic_volume_multiplier: float = 3,
                 peak_height: float = 0.9,
                 peak_threshold: float = 0.15,
                 max_seconds_between_peaks: float = 4,
                 min_count_of_peaks: int = 2,
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
