import sounddevice as sd
import soundfile as sf
import numpy as np
import queue

import sys
import os

# Import our engine from the other file
from chat_engine import LlamaChatEngine


def record_audio_from_mic(filename="temp_user_input.wav", samplerate=44100):
    """
    Records audio from the default microphone.
    Recording starts on the first key press (Enter) and stops on the second.
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """Called continuously and pushes audio data into the queue."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n" + "=" * 50)
    input("🎤 Press [ENTER] to START voice recording...")
    print("🔴 Recording... (Speak now. Press [ENTER] again to STOP)")

    # Open audio stream (1 channel = mono, 16kHz is typically enough for speech)
    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=callback)
    with stream:
        input()  # Blocks until the user presses Enter again

    print("⏳ Processing audio...")

    # Pull all data from the queue and combine it
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())

    audio_data = np.concatenate(audio_data, axis=0)

    # Save as WAV file
    sf.write(filename, audio_data, samplerate)
    return filename


def main():
    # Initialize chat engine
    print("🤖 Starting LlamaChatEngine...")
    chat_engine = LlamaChatEngine(
        server_url="http://192.168.80.7:8080",
        system_prompt=(
            "You are a friendly conversational AI assistant. The user is speaking to you "
            "through a microphone. You MUST always use the respond_to_user tool to reply. "
            "First transcribe what the user said, then write your response."
            # "You are an AI translator. The user speaks to you through a microphone. "
            # "First transcribe what the user said, then write a translation. "
            # "Your target language is English."
        ),
        save_messages=True,
        choose_tool="respond_to_user",
    )

    start_msg = "Hello! I am ready. Follow the instructions to speak with me."
    print(f"\nKI: {start_msg}")

    audio_file = "./samples/temp-user-input.wav"

    try:
        while True:
            record_audio_from_mic(filename=audio_file)
            print("🧠 Sending audio to Llama.cpp (model is thinking)...")
            ergebnis = chat_engine.send_message(audio_path=audio_file)

            if isinstance(ergebnis, dict) and "transcription" in ergebnis:
                # Successful tool call / JSON format
                transkription = ergebnis.get("transcription", "")
                antwort = ergebnis.get("response", "")

                print(f"\n🗣️  You said: '{transkription}'")
                print(f"🤖 AI replies:  '{antwort}'")
            else:
                print(f"\n⚠️ Received unexpected format from server: {ergebnis}")

    except KeyboardInterrupt:
        # Exit with Ctrl+C
        print("\n\n👋 Chatbot is shutting down. Goodbye!")
        if os.path.exists(audio_file):
            os.remove(audio_file)  # Clean up temporary audio file
        sys.exit(0)


if __name__ == "__main__":
    main()
