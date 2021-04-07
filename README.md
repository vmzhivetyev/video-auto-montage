# video-auto-montage
This script extracts potentialy interesting moments (based on audio peaks) out of a game screen capture video.

# Installation (Windows)

Download ffmpeg build from https://github.com/BtbN/FFmpeg-Builds and place binaries into `bin` folder

**Make sure you are using 64-bit Python** or you will experience memory overflow errors.

# Usage

1) Edit `main.py`

    Look for config `apex = VideoMontageConfig(` in **main.py** and tune it for your needs or create a new one and use in `run_directory(config=apex)`.
    
    Config for PUBG is outdated but may work for a version tagged **1.2** or older.
    

2) Run `main.py`
3) ???
4) PROFIT

# Questions?

Feel free to create an issue if you have any problems or questions.
