# SIFT Object Tracking in Video

This project implements object tracking in a video using Scale-Invariant Feature Transform (SIFT) with OpenCV. The application allows users to upload an image and a video. It then detects instances of the uploaded image appearing in the video, providing the start time, duration, and number of occurrences.
Made as a project for the annual Hackathon,2024([HackVita 2.0](https://gdsc.community.dev/events/details/developer-student-clubs-jorhat-engineering-college-jorhat-presents-hackvita-20/)) organised by **Google-DSC**, Jorhat Engineering College.<br>
**Highest rated model of the competition.**

## How to Use

1. **Upload Files**: Users can upload an image (`png` or `jpg`) and a video (`mp4`).
2. **Detection Process**: The application uses SIFT to detect keypoints and descriptors in the uploaded image and the frames of the video.
3. **Output**: For each occurrence found, it displays:
   - Start time: Time in seconds when the occurrence starts.
   - Duration: Duration in seconds for which the image appears continuously.
   - Number of Occurrences: Total instances where the image appears in the video.

## Links

- [Kaggle Notebook](https://www.kaggle.com/code/hillol10/sift-object-tracking/notebook)
- [SnapShot](https://huggingface.co/spaces/hillol7/SnapShot)

This application is hosted on Hugging Face Spaces for interactive usage.

