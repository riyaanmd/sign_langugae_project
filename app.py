import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import google.generativeai as genai
import os
import streamlit as st
import time
from datetime import datetime
from gtts import gTTS
import tempfile

# Setup
GOOGLE_API_KEY = "AIzaSyCxCJoOU1A5JPDAwtmpt5nr-Q97jTqLNzg"
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat = chat_model.start_chat(history=[])

# Load the trained ASL model and other initializations (keeping original code)
model_path = './model.p'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the model is available.")
    st.stop()
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Label dictionary (keeping original)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ', 37: '.'
}

def generate_speech(text):
    """Generate speech from text using gTTS and return the audio file path."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def get_gemini_response(user_message):
    """Get response from Google Gemini AI."""
    try:
        response = st.session_state.chat.send_message(user_message)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Keep original process_hand_landmarks function
def process_hand_landmarks(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []
    
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)
    
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))
    
    while len(data_aux) < 42:
        data_aux.append(0)
    return data_aux[:42]

# Keep original process_camera function
def process_camera():
    """Function to capture video, detect ASL signs, and save detected words/sentences."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access the camera. Please ensure it is connected and try again.")
        return
    
    frame_width = 1920
    frame_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    window_name = 'ASL Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    instructions = [
        "Place hand in green box",
        "Press SPACE to capture each sign (letter)",
        "Press ENTER to add the captured word to the sentence",
        "Press Q when the sentence is complete"
    ]
    
    word_buffer = ""
    sentence_buffer = ""
    last_capture_time = 0
    capture_interval = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera. Exiting.")
            break

        height, width = frame.shape[:2]
        box_size = int(min(width, height) * 0.7)
        center_x = width // 2
        center_y = height // 2
        cv2.rectangle(frame, 
                     (center_x - box_size//2, center_y - box_size//2),
                     (center_x + box_size//2, center_y + box_size//2), 
                     (0, 255, 0), 3)

        overlay = np.zeros((200, width, 3), dtype=np.uint8)
        frame[0:200, 0:width] = cv2.addWeighted(overlay, 0.5, frame[0:200, 0:width], 0.5, 0)
        
        for idx, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 40 + idx * 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        detected_char = '?'
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            prediction = model.predict([np.asarray(process_hand_landmarks(hand_landmarks))])
            predicted_index = int(prediction[0])
            detected_char = labels_dict.get(predicted_index, '?')

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Detected: {detected_char}", (10, 250), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Word: {word_buffer}", (10, 300), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence_buffer}", (10, 350), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        if key == ord(' '):
            if current_time - last_capture_time > capture_interval:
                word_buffer += detected_char
                last_capture_time = current_time
                flash = np.ones_like(frame) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(50)
        elif key == 13:
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                word_buffer = ""
        elif key == ord('q'):
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                word_buffer = ""
            
            try:
                with open('detected_text.txt', 'w') as f:
                    f.write(sentence_buffer)
            except Exception as e:
                st.error(f"Error saving text: {str(e)}")
                return
            
            st.session_state.completed_detection = True
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.set_page_config(layout="wide", page_title="ASL Chatbot")
    
    # Custom CSS for styling
    st.markdown("""<style>
        .stApp { background-color: #2c2f33 }
        .stTitle, .stMarkdown { color: white }
        .stButton > button {
            background-color: #7289da;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
        }
        .chat-message {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        .user-message {
            background-color: #7289da;
            color: white;
        }
        .bot-message {
            background-color: #424549;
            color: white;
        }
    </style>""", unsafe_allow_html=True)

    st.title("ASL Chat Interface")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">You: {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">Bot: {message["content"]}</div>', 
                       unsafe_allow_html=True)
            if "audio_path" in message:
                with open(message["audio_path"], "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")

    # Start Detection button
    if st.button("Start Signing"):
        process_camera()
        
        # After detection, read the file and process the message
        if os.path.exists('detected_text.txt'):
            with open('detected_text.txt', 'r') as f:
                user_message = f.read().strip()
                if user_message:
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Get and process bot response
                    bot_response = get_gemini_response(user_message)
                    audio_path = generate_speech(bot_response)
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": bot_response,
                        "audio_path": audio_path
                    })
                    
                    # Clean up the file with proper error handling
                    try:
                        # Make sure file handle is closed
                        with open('detected_text.txt', 'r') as f:
                            pass
                        # Try to remove the file
                        os.remove('detected_text.txt')
                    except PermissionError:
                        # If file is locked, we'll try to remove it next time
                        pass
                    except Exception as e:
                        st.error(f"Error cleaning up file: {str(e)}")
                    
                    # Rerun to update the display
                    st.rerun()

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.chat = chat_model.start_chat(history=[])
        try:
            if os.path.exists('detected_text.txt'):
                # Make sure file handle is closed
                with open('detected_text.txt', 'r') as f:
                    pass
                os.remove('detected_text.txt')
        except PermissionError:
            # If file is locked, we'll try again next time
            pass
        except Exception as e:
            st.error(f"Error clearing chat history: {str(e)}")
        st.rerun()

if __name__ == "__main__":
    main()