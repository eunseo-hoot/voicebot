import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from datetime import datetime
import openai
import av
import numpy as np
import tempfile
import os
from gtts import gTTS
import uuid

##### 오디오 처리 클래스 #####
class AudioProcessor:
    def __init__(self) -> None:
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame)
        return frame

    def get_audio_data(self):
        if not self.audio_frames:
            return None

        audio = b""
        for frame in self.audio_frames:
            for p in frame.planes:
                audio += p.to_bytes()

        sample_rate = frame.sample_rate
        temp_audio_path = tempfile.mktemp(suffix=".wav")

        with open(temp_audio_path, "wb") as f:
            f.write(audio)

        return temp_audio_path

##### STT 함수 #####
def STT(audio_path):
    audio_file = open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

##### GPT 응답 함수 #####
def ask_gpt(messages, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

##### TTS 함수 #####
def TTS(text):
    tts = gTTS(text=text, lang='ko')
    file_path = tempfile.mktemp(suffix=".mp3")
    tts.save(file_path)
    st.audio(file_path, format="audio/mp3")

##### 메인 #####
def main():
    st.set_page_config(page_title="음성 비서 프로그램", layout="wide")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korean."}
        ]

    st.header("음성 비서 프로그램")
    st.markdown("---")

    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.write("""
        - UI는 Streamlit 기반
        - STT: Whisper AI
        - GPT 응답: OpenAI GPT 모델
        - TTS: Google TTS
        """)

    with st.sidebar:
        openai.api_key = st.text_input("OPENAI API 키", type="password")
        model = st.radio("GPT 모델", ["gpt-4", "gpt-3.5-turbo"])
        if st.button("초기화"):
            st.session_state.chat = []
            st.session_state.messages = [
                {"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korean."}
            ]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("질문하기")
        ctx = webrtc_streamer(
            key="speech_audio_input_unique_key",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": False, "audio": True},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            audio_processor_factory=AudioProcessor,
            async_processing=False,
        )

        if ctx.audio_processor:
            if st.button("녹음 종료 및 분석"):
                audio_path = ctx.audio_processor.get_audio_data()
                if audio_path:
                    question = STT(audio_path)
                    now = datetime.now().strftime("%H:%M")
                    st.session_state.chat.append(("user", now, question))
                    st.session_state.messages.append({"role": "user", "content": question})

    with col2:
        st.subheader("질문/답변")
        if st.session_state.chat and st.session_state.chat[-1][0] == "user":
            response = ask_gpt(st.session_state.messages, model)
            now = datetime.now().strftime("%H:%M")
            st.session_state.chat.append(("bot", now, response))
            st.session_state.messages.append({"role": "system", "content": response})

            for sender, time, message in st.session_state.chat:
                st.markdown(f"**[{sender}] {time}**: {message}")
            TTS(response)

if __name__ == "__main__":
    main()
