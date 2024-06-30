import streamlit as st
import cv2
import numpy as np

st.title("Object Tracking in Video using SIFT")

# Upload files
uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg'])
uploaded_video = st.file_uploader("Upload a video", type=['mp4'])

if uploaded_image and uploaded_video:
    # Convert uploaded files to OpenCV-compatible formats
    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    input_image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

    video_bytes = np.asarray(bytearray(uploaded_video.read()), dtype=np.uint8)
    cap = cv2.VideoCapture()
    cap.open(uploaded_video.name)

    # Initialize SIFT
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # Detect keypoints and descriptors for the input image
    keypoints_input, descriptors_input = sift.detectAndCompute(input_image, None)

    occurrences = 0
    occurrence_start = 0
    occurrence_duration = 0
    prev_matches = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("End of video reached.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors for the video frame
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

        # Match descriptors
        matches = bf.knnMatch(descriptors_input, descriptors_frame, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 6:
            if not prev_matches:
                occurrence_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                occurrences += 1

            prev_matches = good_matches
            occurrence_duration = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) - occurrence_start
        else:
            if prev_matches:
                st.write(f"Occurrence {occurrences}: Start time: {occurrence_start:.2f}s, Duration: {occurrence_duration:.2f}s")
                prev_matches = []

    cap.release()

