import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic  # Holistic model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    # Custom CSS style for the thin colored bar
    st.markdown(
        """
        <style>
        .thin-bar {
            background-color: rgb(114, 134, 211);
            height: 35px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add the thin colored bar before the st.title
    st.markdown('<div class="thin-bar"></div>', unsafe_allow_html=True)

    st.title("Sign Language Detection")

    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

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

        height, width, channels = image.shape
        text_size, _ = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_DUPLEX, 1, 2)
        text_width = text_size[0]
        x = (width - text_width) // 2
        cv2.putText(image, ' '.join(sentence), (x, height - 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_placeholder.image(image_rgb, channels="RGB")


    cap.release()
    cv2.destroyAllWindows()