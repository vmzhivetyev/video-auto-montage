from video_montage import VideoMontager, VideoMontageConfig

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
    ))
    apex.run_directory()

    quake = VideoMontager(VideoMontageConfig(
        input_dir='D:\Videos\Quake Champions',
        output_dir='vids\Quake Champions',
        bitrate_megabits=50,
        mic_volume_multiplier=1,
        peak_height=1.3,
        peak_threshold=0.1,
        max_seconds_between_peaks=2,
        min_count_of_peaks=1,
        extend_range_bounds_by_seconds=1,
        min_duration_of_valid_range=0
        mix_mic_audio_track=False,
    quake.run_directory()
