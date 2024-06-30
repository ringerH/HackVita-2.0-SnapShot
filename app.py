import streamlit as st
import cv2
import numpy as np
def main():
    st.title("SnapShot: One Stop Solution For Surveillance")
    st.sidebar.title("Upload Files")
    
    uploaded_image = st.sidebar.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg"])
    uploaded_video = st.sidebar.file_uploader("Upload a video (MP4)", type=["mp4"])

    if uploaded_image and uploaded_video:
        st.sidebar.success("Files successfully uploaded!")
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        video = cv2.VideoCapture(uploaded_video)

        st.header("Uploaded Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.header("Uploaded Video")
        st.video(uploaded_video)

        # Perform object tracking with SIFT
        sift = cv2.SIFT_create()
        keypoints_input, descriptors_input = sift.detectAndCompute(image, None)
        bf = cv2.BFMatcher()

        occurrences = 0
        occurrence_start = 0
        occurrence_duration = 0
        prev_matches = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                st.write("End of video reached.")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

            matches = bf.knnMatch(descriptors_input, descriptors_frame, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 6:
                if not prev_matches:
                    occurrence_start = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    occurrences += 1

                prev_matches = good_matches
                occurrence_duration = (video.get(cv2.CAP_PROP_POS_MSEC) / 1000) - occurrence_start
            else:
                if prev_matches:
                    st.write(f"Occurrence {occurrences}: Start time: {occurrence_start:.2f}s, Duration: {occurrence_duration:.2f}s")
                    prev_matches = []

        video.release()

    else:
        st.sidebar.warning("Please upload both an image and a video.")

if __name__ == "__main__":
    main()
