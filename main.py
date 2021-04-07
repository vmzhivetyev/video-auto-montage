from video_montage import *

if __name__ == "__main__":
    apex = MontageConfig(
        input_dir='E:/shadow play/replays/Apex Legends',
        output_dir='vids/apex',
        video=VideoConfig(
            bitrate_megabits=50
        ),
        detection=PeakDetectionConfig(
            peak_height=1.3,
            peak_threshold=0.1,
            freq_range=(0, 40),
            min_peaks_in_a_row=1
        ),
        range=RangeConfig(
            min_distance=2,
            extend_before_start=1,
            extend_after_end=1.5,
            min_duration=0
        ),
        microphone=MicrophoneConfig(
            mic_volume_multiplier=3,
            # When True - FFMPEG will crash if source video file has only one audio track.
            mix_mic_audio_track=True
        ),
        music=MusicConfig(
            chance=0.9,
            volume=0.3,
            random_seed_by_file=True
        ))

    apex = VideoMontager(apex)

    # apex.run_directory()
    apex.run_file('E:/shadow play/replays/Apex Legends\Apex Legends 2021.04.06 - 00.41.08.28.DVR.mp4')

    # quake = VideoMontager(MontageConfig(
    #     input_dir='D:\Videos\Quake Champions',
    #     output_dir='vids\Quake Champions',
    #     bitrate_megabits=50,
    #     mic_volume_multiplier=1,
    #     freq_range=(0, 40),
    #     peak_height=1.3,
    #     peak_threshold=0.1,
    #     min_distance=2,
    #     min_peaks_in_a_row=1,
    #     extend_range_bounds_by_seconds=1,
    #     min_duration=0,
    #     mix_mic_audio_track=False
    # ))
    # quake.run_directory()
