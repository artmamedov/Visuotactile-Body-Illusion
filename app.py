import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings
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

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.top = 200
        self.vert = 0
        self.bottom = 0
        self.side_end = 0
        self.running = False
        #Tweakables
        self.annotate = False
        self.side = "None"
        self.direction = "Vertical"
        self.scale = 1
        self.extra = 5
        self.edge_scale = 1.15
        self.smoothing_factor = 10

    def recv(self, frame):
        with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
            image = frame.to_ndarray(format="bgr24")
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape
            startx = 0
            endx   = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.annotate:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    self.bottom = int(min(landmarks[8].y, landmarks[4].y)*image_height + self.extra//2*self.scale)
                    self.side_end = int(min(landmarks[8].x, landmarks[4].x)*image_height + self.extra//2*self.scale)

                    if not self.running and self.check_thumbs_up(landmarks):
                        self.running = True
                        if self.direction ==  "Vertical":
                            self.top = int(min([landmarks[8].y, landmarks[7].y, landmarks[6].y, landmarks[5].y])\
                                                  *image_height - self.extra//2*self.scale)
                        if self.direction == "Horizontal":
                            self.vert = int(min([landmarks[8].x, landmarks[7].x, landmarks[6].x, landmarks[5].x])\
                                                  *image_width - self.extra//2*self.scale)
                    elif self.check_two(landmarks):
                        self.running = False
                        self.top = 0

            if self.running:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = hand_landmarks.landmark
                        startx = int(min(landmarks[4].x, landmarks[8].x, landmarks[12].x,
                                         landmarks[16].x, landmarks[20].x)*image_width/self.edge_scale)
                        endx   = int(max(landmarks[4].x, landmarks[8].x, landmarks[12].x,
                                         landmarks[16].x, landmarks[20].x)*image_width*self.edge_scale)
                        starty = int(min(landmarks[4].y, landmarks[8].y, landmarks[12].y,
                                         landmarks[16].y, landmarks[20].y)*image_height/self.edge_scale)
                        endy   = int(max(landmarks[4].y, landmarks[8].y, landmarks[12].y,
                                         landmarks[16].y, landmarks[20].y)*image_height*self.edge_scale)

                    if self.direction == "Vertical":
                        section = self.bottom-self.top
                        for j in range(int(section*self.scale)):
                            smooth_startx = int(startx-j//(self.smoothing_factor/10))
                            smooth_endx   = int(endx-j//(self.smoothing_factor/10))
                            image = self.expand_vertical(image,int((self.bottom-(section-j)*2)), smooth_startx, smooth_endx)
                    elif self.direction == "Horizontal":
                        section = self.side_end-self.vert
                        for j in range(int(section*self.scale)):
                            smooth_starty = int(starty-j//(self.smoothing_factor/10))
                            smooth_endy   = int(endy-j//(self.smoothing_factor/10))
                            image =self. expand_horizontal(image,int((self.side_end-(section+j)*2)), smooth_starty, smooth_endy)
                if self.annotate:
                    if self.direction == "Vertical":
                        cv2.rectangle(image,(0,self.top),(image_width, self.top),(255,0,0),3)
                    elif self.direction == "Horizontal":
                        cv2.rectangle(image,(self.vert,0),(self.vert,image_height),(255,0,0),3)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def set_annotations(self, value):
        if value == "Yes":
            self.annotate = True
        else:
            self.annotate = False

    def set_side(self, value):
        self.side = value

    def set_scale(self,value):
        self.scale = value

    def set_extra(self,value):
        self.extra = value

    def set_edge_scale(self,value):
        self.edge_scale = value

    def set_direction(self,value):
        self.direction = value

    def set_smoothing_factor(self,value):
        self.smoothing_factor = value

    def expand_vertical(self,image, section, startx, endx):
        np_leg = np.array(image)
        original_shape = np_leg.shape
        expanded = np_leg[section:section+1]
        output = np.concatenate([np_leg[:section,:],expanded,np_leg[section:,:]])

        #Crop image to maintain original shape
        output = output[:original_shape[0],:original_shape[1]]

        #Which side is being altered?
        if self.side == "Right":
            output[:,:startx] = np_leg[:,:startx]
        if self.side == "Left":
            output[:,endx:]   = np_leg[:,endx:]
        return output

    def expand_horizontal(self,image, section, startx, endx):
        np_leg = np.array(image)
        original_shape = np_leg.shape

        expanded = np_leg[:,section:section+1]
        output = np.concatenate([np_leg[:,:section],expanded,np_leg[:,section:]],axis=1)

        #Crop image to maintain original shape
        output = output[:original_shape[0],:original_shape[1]]

        #Which side is being altered?
        if self.side == "Down":
            output[:startx,:] = np_leg[:startx,:]
        if self.side == "Up":
            output[endx:,:]   = np_leg[endx:,:]
        return output

    def check_thumbs_up(self, landmarks):
        rest_of_hand = landmarks[:3] + landmarks[5:]
        return landmarks[4].y < landmarks[3].y and all(landmarks[3].y < landmark.y for landmark in rest_of_hand)

    def check_two(self,landmarks):
        #Up
        index  = landmarks[8].y  < landmarks[6].y  and landmarks[8].y  < landmarks[7].y and landmarks[8].y  < landmarks[5].y
        middle = landmarks[12].y < landmarks[10].y and landmarks[12].y  < landmarks[11].y and landmarks[12].y  < landmarks[9].y
        middle_higher = landmarks[12].y < landmarks[8].y
        rest_of_hand = landmarks[:5] + landmarks[13:]
        upwards = all(landmarks[6].y < landmark.y for landmark in rest_of_hand) and all(landmarks[10].y < landmark.y for landmark in rest_of_hand)
        check_up = index and middle and middle_higher and upwards

        #Left
        index  = landmarks[8].x  < landmarks[6].x  and landmarks[8].x  < landmarks[7].x and landmarks[8].x  < landmarks[5].x
        middle = landmarks[12].x < landmarks[10].x and landmarks[12].x  < landmarks[11].x and landmarks[12].x  < landmarks[9].x
        middle_higher = landmarks[12].x < landmarks[8].x
        rest_of_hand = landmarks[:5] + landmarks[13:]
        leftwards = all(landmarks[6].x < landmark.x for landmark in rest_of_hand) and all(landmarks[10].x < landmark.x for landmark in rest_of_hand)
        check_left = index and middle and middle_higher and leftwards

        #Right
        index  = landmarks[8].x  > landmarks[6].x  and landmarks[8].x  > landmarks[7].x and landmarks[8].x  > landmarks[5].x
        middle = landmarks[12].x > landmarks[10].x and landmarks[12].x  > landmarks[11].x and landmarks[12].x  > landmarks[9].x
        middle_higher = landmarks[12].x > landmarks[8].x
        rest_of_hand = landmarks[:5] + landmarks[13:]
        rightwards = all(landmarks[6].x > landmark.x for landmark in rest_of_hand) and all(landmarks[10].x > landmark.x for landmark in rest_of_hand)
        check_right = index and middle and middle_higher and rightwards
        return check_up or check_left or check_right

site = webrtc_streamer(key="example", video_processor_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS)

if site.video_processor:
    #Side Bar
    st.sidebar.header("Customize the Settings!")
    site.video_processor.set_annotations(st.sidebar.radio("Show annotations?", ["Yes","No"]))
    site.video_processor.set_side(st.sidebar.radio("Which side to morph?", ["None","Left","Right"]))
    site.video_processor.set_direction(st.sidebar.radio("Vertical or Horizontal?", ["Vertical","Horizontal"]))
    site.video_processor.set_scale(st.sidebar.slider("Leg Scaling Ratio:", min_value = 0.0, max_value = 1.0, value=0.5, step=0.1))
    site.video_processor.set_smoothing_factor(st.sidebar.number_input("Extra pixels for smoothing (above line)", min_value=0, value=5, step=1,format='%i'))
    site.video_processor.set_extra(st.sidebar.number_input("What is the smoothing intensity?", min_value=0, value=10, step=1,format='%i'))
    site.video_processor.set_edge_scale(st.sidebar.number_input("How much to the left/right of hand to use?", min_value=0.0, max_value=2.0, value=1.15, step=0.01))
