# video-auto-montage
Generic script which extracts potentialy interesting moment out of capture game video based on audio peaks.

# Installation (Windows)

Download ffmpeg build from https://github.com/BtbN/FFmpeg-Builds and place binaries into `bin`

# Usage

1) Edit `main.py`

    Edit path to the directory containing videos to be processed:
    ```
    files = file_list_from_dir("E:/shadow play/replays/Apex Legends/")
    ```

    Edit output directory:
    ```
    cut_video_into_single(file, 'vids/apex')
    ```

2) Run `main.py`
3) ???
4) PROFIT
