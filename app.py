import whisper
from transformers import pipeline
import edge_tts
from pydub import AudioSegment
import asyncio
import streamlit as st

# Step 1: Voice-to-Text Conversion
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    return result["text"]

# Step 2: Text Input into LLM
def generate_response(text_input):
    llm = pipeline("text-generation", model="google/flan-t5-small")  # Load LLM
    response = llm(text_input, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

# Step 3: Text-to-Speech Conversion
async def text_to_speech(text, output_path="output.wav"):
    tts = edge_tts.Communicate(text, voice="en-US-JennyNeural")  # Example voice
    await tts.save(output_path)

# Streamlit Web Interface
def main():
    st.title("AI Voice Assistant")

    # File upload for audio input
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        # Save the uploaded file temporarily
        with open("input.wav", "wb") as f:
            f.write(audio_file.getbuffer())

        # Process the uploaded audio file
        st.write("Processing audio...")
        transcribed_text = transcribe_audio("input.wav")
        st.write(f"Transcribed Text: {transcribed_text}")

        st.write("Generating response...")
        response = generate_response(transcribed_text)
        st.write(f"Generated Response: {response}")

        st.write("Converting text to speech...")
        asyncio.run(text_to_speech(response, "output.wav"))

        # Play the audio response
        st.audio("output.wav", format="audio/wav")

if __name__ == "__main__":
    main()

