import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


#Page Setup
st.set_page_config(page_title="Visuotactile Body Illusion",layout='centered', page_icon=':leg:')
st.title("Visuotactile Body Illusion Demo")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.top = 200
        self.bottom = 0
        self.running = False
        #Tweakables
        self.extra = 10
        self.scale = 1
        self.annotate = True


    def recv(self, frame):
        with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            image = frame.to_ndarray(format="bgr24")
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.annotate:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    self.bottom = int(min(landmarks[8].y, landmarks[4].y)*image_height) + self.extra//2
                    if not self.running and self.check_thumbs_up(landmarks):
                        self.running = True
                        self.top = int(min([landmarks[8].y, landmarks[7].y, landmarks[6].y, landmarks[5].y])\
                                                  *image_height) - self.extra//2
                    elif self.check_stop(landmarks):
                        self.running = False
                        self.top = 0
            if self.running:
                section = self.bottom-self.top
                for j in range(section):
                    image = self.expand(image,(self.bottom-(section-j)*self.scale*2),self.scale)
                if self.annotate:
                    cv2.rectangle(image,(0,self.top),(image_width, self.top),(255,0,0),3)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def expand(self, image, section, scale):
        np_leg = np.array(image)
        original_shape = np_leg.shape
        #print(np_leg.shape)
        expanded = np.concatenate([np_leg[section:section+1] for i in range(scale)])
        output = np.concatenate([np_leg[:section,:],expanded,np_leg[section:,:]])
        #Crop image to maintain original shape
        output = output[:original_shape[0],:original_shape[1]]
        return output

    def check_thumbs_up(self, landmarks):
        rest_of_hand = landmarks[:3] + landmarks[5:]
        return landmarks[4].y < landmarks[3].y and all(landmarks[3].y < landmark.y for landmark in rest_of_hand)

    def check_stop(self, landmarks):
        thumb  = landmarks[4].y  < landmarks[3].y  and landmarks[4].y  < landmarks[2].y
        index  = landmarks[8].y  < landmarks[6].y  and landmarks[8].y  < landmarks[5].y
        middle = landmarks[12].y < landmarks[10].y and landmarks[12].y  < landmarks[9].y
        ring   = landmarks[16].y < landmarks[14].y and landmarks[16].y  < landmarks[13].y
        pinky  = landmarks[20].y < landmarks[18].y and landmarks[20].y  < landmarks[17].y
        upward = landmarks[20].y < landmarks[4].y  and landmarks[12].y  < landmarks[20].y
        return sum([thumb, index, middle, ring, pinky]) == 5 and upward

webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
