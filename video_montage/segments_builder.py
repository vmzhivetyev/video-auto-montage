import numpy as np
from matplotlib import mlab
from scipy.signal import find_peaks

from video_montage.video_montage_config import VideoMontageConfig


class SegmentsBuilder:
    nfft = 256
    noverlap = 100

    def __init__(self, config: VideoMontageConfig):
        self.config = config

    def _make_fft(self, x):
        return mlab.specgram(x=x, NFFT=self.nfft, Fs=self.config.sample_rate)[0]

    def _get_lows_from_fft(self, speq):
        speq = np.array([x[:40] for x in speq.T]).T
        low_freq_volumes = np.array([sum(x) * 1000 for x in speq.T]).T
        return low_freq_volumes

    def fft_of_lows(self, audio):
        speq = self._make_fft(audio)
        return self._get_lows_from_fft(speq)

    def peak_ranges_of_lows(self, audio):
        lows = self.fft_of_lows(audio)
        return self.peak_ranges(lows, mult=len(lows) / len(audio))

    def peak_ranges(self, audio, mult=1.0):
        """
            max_distance_sec: max distance between peaks (in seconds) which can be used to increase count of peaks range
            min_count: min count of peaks in range to include it in result

            :returns array of valid ranges
        """
        peaks, _ = find_peaks(audio, height=self.config.peak_height, threshold=self.config.peak_threshold)

        max_distance = int(self.config.max_seconds_between_peaks * self.config.sample_rate * mult)

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
                    if count >= self.config.min_count_of_peaks:
                        ranges.append((start, last))
                    start = x
                    last = x
                    count = 1

        if start != -1 and last != -1:
            ranges.append((start, last))

        peaks = [int(x / mult) for x in peaks]
        ranges = [(int(x / mult), int(y / mult)) for x, y in ranges]

        return peaks, ranges

    def _filter_ranges(self, ranges):
        min_length = int(self.config.min_duration_of_valid_range * self.config.sample_rate)

        return [x for x in ranges if x[1] - x[0] >= min_length]

    def _weld_overlapping_ranges(self, sec_ranges):
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

    def make_sec_ranges(self, audio):
        _, ranges = self.peak_ranges_of_lows(audio)
        ranges = self._filter_ranges(ranges)

        sec_ranges = [(x[0] / self.config.sample_rate, x[1] / self.config.sample_rate) for x in ranges]

        sec_ranges = [[x[0] - self.config.extend_range_bounds_by_seconds,
                       x[1] + self.config.extend_range_bounds_by_seconds]
                      for x in sec_ranges]

        new_ranges = self._weld_overlapping_ranges(sec_ranges)
        while len(new_ranges) < len(sec_ranges):
            sec_ranges = new_ranges
            new_ranges = self._weld_overlapping_ranges(new_ranges)

        return new_ranges
