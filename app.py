import cv2
import html
import mimetypes
import os
import pickle
import tempfile
import time

import google.generativeai as genai
import mediapipe as mp
import numpy as np
import streamlit as st
import whisper
from datetime import datetime
from gtts import gTTS

# Setup
GOOGLE_API_KEY = "AIzaSyCxCJoOU1A5JPDAwtmpt5nr-Q97jTqLNzg"
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state for chat and UI metadata
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat' not in st.session_state:
    st.session_state.chat = chat_model.start_chat(history=[])
if 'bookmarked_messages' not in st.session_state:
    st.session_state.bookmarked_messages = []
if 'live_detection' not in st.session_state:
    st.session_state.live_detection = {"char": "", "word": "", "sentence": ""}
if 'generated_audio_files' not in st.session_state:
    st.session_state.generated_audio_files = []
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = ""
if 'last_transcript' not in st.session_state:
    st.session_state.last_transcript = ""

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


def current_timestamp():
    """Return the current UTC timestamp formatted for display."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def generate_speech(text):
    """Generate speech from text using gTTS and return the audio file path."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.session_state.generated_audio_files.append(fp.name)
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


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load and cache the Whisper model for speech recognition."""
    return whisper.load_model("base")


def save_uploaded_audio(uploaded_audio):
    """Persist an uploaded audio clip to a temporary file and return its path."""
    suffix = mimetypes.guess_extension(uploaded_audio.type) or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_audio.getbuffer())
        return tmp_file.name


def transcribe_audio(file_path):
    """Transcribe an audio file using Whisper."""
    try:
        with st.spinner("Transcribing audio message..."):
            model = load_whisper_model()
            result = model.transcribe(file_path)
        return result.get("text", "").strip()
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return ""


def build_transcript(history):
    """Create a plain-text transcript from the chat history."""
    lines = []
    for message in history:
        role = message.get("role", "assistant")
        speaker = "Assistant"
        if role == "user":
            speaker = "User"
        mode = message.get("metadata", {}).get("mode")
        if mode == "sign":
            speaker += " (sign)"
        elif mode == "speech":
            speaker += " (speech)"
        timestamp = message.get("timestamp", "")
        timestamp_prefix = f"[{timestamp}] " if timestamp else ""
        lines.append(f"{timestamp_prefix}{speaker}: {message.get('content', '')}")
    return "\n".join(lines)


def summarize_conversation(history):
    """Summarize the conversation using Gemini."""
    transcript = build_transcript(history)
    if not transcript.strip():
        return "Conversation is empty.", False

    prompt = (
        "Provide a concise summary of the following conversation between an ASL "
        "user and a hearing user. Highlight any requested follow-ups or next steps.\n\n"
        f"{transcript}"
    )

    try:
        summary_model = genai.GenerativeModel('gemini-1.5-flash')
        with st.spinner("Summarizing conversation..."):
            response = summary_model.generate_content(prompt)
        return response.text, True
    except Exception as e:
        return f"Summary error: {str(e)}", False


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


# Updated process_camera function to support live UI updates
def process_camera(live_char_placeholder, live_word_placeholder, live_sentence_placeholder):
    """Capture video, detect ASL signs, and stream live transcription updates."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access the camera. Please ensure it is connected and try again.")
        return ""

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
    final_sentence = ""

    st.session_state.live_detection = {"char": "", "word": "", "sentence": ""}
    live_char_placeholder.markdown("**Detected character:** ` `")
    live_word_placeholder.markdown("**Current word:** ` `")
    live_sentence_placeholder.markdown("**Sentence preview:** ` `")

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

        st.session_state.live_detection = {
            "char": detected_char if detected_char != '?' else "",
            "word": word_buffer,
            "sentence": sentence_buffer
        }
        live_char_placeholder.markdown(f"**Detected character:** `{st.session_state.live_detection['char'] or ' '}`")
        live_word_placeholder.markdown(f"**Current word:** `{word_buffer}`")
        live_sentence_placeholder.markdown(f"**Sentence preview:** `{sentence_buffer}`")

        cv2.putText(frame, f"Word: {word_buffer}", (10, 300),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence_buffer}", (10, 350),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()

        if key == ord(' '):
            if current_time - last_capture_time > capture_interval:
                word_buffer += detected_char if detected_char != '?' else ''
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

            final_sentence = sentence_buffer.strip()
            break

    cap.release()
    cv2.destroyAllWindows()

    st.session_state.live_detection = {"char": "", "word": "", "sentence": final_sentence}
    live_char_placeholder.markdown("**Detected character:** ` `")
    live_word_placeholder.markdown("**Current word:** ` `")
    live_sentence_placeholder.markdown(f"**Sentence preview:** `{final_sentence}`")

    return final_sentence


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

    transcript_text = build_transcript(st.session_state.chat_history)

    with st.sidebar:
        st.header("Conversation tools")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download transcript",
            data=transcript_text or "No messages yet.",
            file_name=f"asl_conversation_{timestamp}.txt",
            mime="text/plain"
        )

        if st.button("Summarize conversation"):
            summary, success = summarize_conversation(st.session_state.chat_history)
            st.session_state.last_summary = summary
            if success:
                st.success("Conversation summarized.")
            else:
                st.warning(summary)

        if st.session_state.last_summary:
            st.markdown("**Latest summary**")
            st.markdown(st.session_state.last_summary)

        if st.session_state.bookmarked_messages:
            st.markdown("**Bookmarked exchanges**")
            for idx in st.session_state.bookmarked_messages:
                if idx < len(st.session_state.chat_history):
                    entry = st.session_state.chat_history[idx]
                    role = entry.get("role", "assistant")
                    label = "Assistant"
                    if role == "user":
                        label = "User"
                        mode = entry.get("metadata", {}).get("mode")
                        if mode == "sign":
                            label += " (sign)"
                        elif mode == "speech":
                            label += " (speech)"
                    st.markdown(f"- **{label}**: {entry.get('content', '')}")

        if st.button("Clear chat history"):
            for audio_path in st.session_state.generated_audio_files:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            st.session_state.generated_audio_files = []
            st.session_state.chat_history = []
            st.session_state.bookmarked_messages = []
            st.session_state.last_summary = ""
            st.session_state.last_transcript = ""
            st.session_state.live_detection = {"char": "", "word": "", "sentence": ""}
            st.session_state.chat = chat_model.start_chat(history=[])
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Live ASL transcription")
        live_char_placeholder = st.empty()
        live_word_placeholder = st.empty()
        live_sentence_placeholder = st.empty()

        live_state = st.session_state.live_detection
        live_char_placeholder.markdown(f"**Detected character:** `{live_state.get('char', '') or ' '}`")
        live_word_placeholder.markdown(f"**Current word:** `{live_state.get('word', '')}`")
        live_sentence_placeholder.markdown(f"**Sentence preview:** `{live_state.get('sentence', '')}`")

        if st.button("Start Signing", use_container_width=True):
            final_sentence = process_camera(live_char_placeholder, live_word_placeholder, live_sentence_placeholder)
            if final_sentence:
                st.success(f"Captured sentence: {final_sentence}")
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": final_sentence,
                    "timestamp": current_timestamp(),
                    "metadata": {"mode": "sign"}
                })
                bot_response = get_gemini_response(final_sentence)
                audio_path = generate_speech(bot_response)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": bot_response,
                    "timestamp": current_timestamp(),
                    "audio_path": audio_path
                })
                st.session_state.last_summary = ""
                st.rerun()
            else:
                st.info("No sentence captured from the signing session.")

        st.markdown("### Speech backchannel")
        with st.form("speech_input_form"):
            audio_input = st.audio_input("Record a spoken message")
            submitted = st.form_submit_button("Send speech message")

        if submitted:
            if audio_input is None:
                st.warning("Please record an audio message before submitting.")
            else:
                temp_audio_path = save_uploaded_audio(audio_input)
                transcript = transcribe_audio(temp_audio_path)
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass

                if transcript:
                    st.session_state.last_transcript = transcript
                    st.success(f"Recognized speech: {transcript}")
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": transcript,
                        "timestamp": current_timestamp(),
                        "metadata": {"mode": "speech"}
                    })
                    bot_response = get_gemini_response(transcript)
                    audio_path = generate_speech(bot_response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": bot_response,
                        "timestamp": current_timestamp(),
                        "audio_path": audio_path
                    })
                    st.session_state.last_summary = ""
                    st.rerun()
                else:
                    st.error("Could not transcribe the provided audio.")

        if st.session_state.last_transcript:
            st.caption(f"Last speech transcript: {st.session_state.last_transcript}")

    with col2:
        st.subheader("Conversation")
        for idx, message in enumerate(st.session_state.chat_history):
            role = message.get("role", "assistant")
            css_class = "bot-message"
            label = "Bot"
            if role == "user":
                css_class = "user-message"
                label = "You"
                mode = message.get("metadata", {}).get("mode")
                if mode == "sign":
                    label += " (sign)"
                elif mode == "speech":
                    label += " (speech)"
            timestamp = message.get("timestamp")
            bookmark_icon = " ⭐" if idx in st.session_state.bookmarked_messages else ""

            escaped_content = html.escape(message.get("content", "")).replace("\n", "<br>")
            st.markdown(
                f'<div class="chat-message {css_class}"><strong>{label}{bookmark_icon}:</strong> '
                f'{escaped_content}</div>',
                unsafe_allow_html=True
            )
            if timestamp:
                st.caption(timestamp)

            if role == "assistant" and message.get("audio_path"):
                audio_path = message["audio_path"]
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")

            bookmark_label = "Remove bookmark" if idx in st.session_state.bookmarked_messages else "Bookmark"
            if st.button(bookmark_label, key=f"bookmark_{idx}"):
                if idx in st.session_state.bookmarked_messages:
                    st.session_state.bookmarked_messages.remove(idx)
                else:
                    st.session_state.bookmarked_messages.append(idx)
                st.session_state.bookmarked_messages = sorted(set(st.session_state.bookmarked_messages))
                st.rerun()


if __name__ == "__main__":
    main()
