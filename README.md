# video-auto-montage
This script extracts potentialy interesting moments (based on audio peaks) out of a game screen capture video.

# Installation (Windows)

Download ffmpeg build from https://github.com/BtbN/FFmpeg-Builds and place binaries into `bin` folder

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
