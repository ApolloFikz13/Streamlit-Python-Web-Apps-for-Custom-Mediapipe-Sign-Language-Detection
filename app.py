import numpy as np
import streamlit as st
import mediapipe as mp
from tensorflow.keras.models import load_model
import streamlit_webrtc as webrtc

mp_holistic = mp.solutions.holistic  # Holistic model

def mediapipe_detection(image, model):
    image = image[:, :, ::-1]  # Convert BGR to RGB
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = image[:, :, ::-1]  # Convert RGB back to BGR
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

actions = ['aku', 'apa', 'bagaimana', 'berapa', 'di', 'dia', 'F', 'halo', 'I', 'J', 'K', 'kamu', 'kapan', 'ke', 'kita', 'makan', 'mana', 'mereka', 'minum', 'nama', 'R', 'saya', 'siapa', 'Y', 'yang', 'Z']

model = load_model('realtimeV9.h5')

sequence = []
sentence = []
predictions = []
threshold = 0.5

st.set_page_config(layout="wide")

def main():
    st.markdown(
        """
        <style>
        .thin-bar {
            background-color: rgb(114, 134, 211);
            height: 35px;
        }
        .video-container {
            display: flex;
            justify-content: center;
        }
        .video-placeholder {
            width: 600px;
            height: 400px;
            object-fit: contain;
        }
        .text-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .text {
            font-size: 24px;
            font-weight: bold;
            color: white;
            text-align: center;
            background-color: white;
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="thin-bar"></div>', unsafe_allow_html=True)
    st.title("Sign Language Detection")

    webrtc_ctx = webrtc.StreamlitWebRTC(
        key="example",
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if webrtc_ctx.video_receiver:
        while True:
            frame = webrtc_ctx.video_frame.copy()

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                confidence = res[np.argmax(res)]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            text = ' '.join(sentence)
            text_html = f'<div class="text"><span>{text}</span></div>'
            st.markdown(text_html, unsafe_allow_html=True)

            st.image(image, channels="BGR")

if __name__ == "__main__":
    main()
