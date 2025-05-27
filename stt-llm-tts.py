import os
import json
import base64
import threading
import pyaudio
import websocket
import pyaudio
import requests
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(override=True)  # Load environment variables from .env

# Load Transcribe configuration from environment variables
# WebSocket endpoint for OpenAI Realtime API (transcription model)
url = f"{os.environ.get("AZURE_OPENAI_ENDPOINT_STT").replace("https", "wss")}/openai/realtime?api-version={os.environ.get("AZURE_OPENAI_API_VERSION_STT")}&intent=transcription"
print("Connecting to:", url)
headers = {"api-key": os.environ.get("AZURE_OPENAI_API_KEY_STT")}
# Audio stream parameters (16-bit PCM, 16kHz mono)
RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

is_playing_audio = threading.Event()

# Initialize PyAudio for audio input
audio_interface = pyaudio.PyAudio()
stream = audio_interface.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)

# Load Azure OpenAI configuration from environment variables
AOAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AOAI_DEPLOYMENT_NAME_STT = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_STT"]
AOAI_DEPLOYMENT_NAME_TTS = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_TTS"]

aoai_client = AzureOpenAI(azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                          api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                          api_version=os.environ.get("AZURE_OPENAI_API_VERSION"))

# Load TTS configuration from environment variables
tts_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_TTS"),
                         api_key=os.getenv("AZURE_OPENAI_API_KEY_TTS"),
                         api_version=os.getenv("AZURE_OPENAI_API_VERSION_TTS"))

# Function to call the RAG API
def call_rag(question):
    url = "https://answergen.azurewebsites.net/api/answergen"
    params = {
        "code": "P9DP4SiEWK_dLf5Q6GyDfr-YwdQDSIOtWBaFZFrmy6unAzFuOo2zWQ=="
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "question": question,
        "index": "rag-index-docs",
        "verbose": "no"
    }
    print(f"\nCalling RAG API with question: {question}")
    response = requests.post(url, params=params, headers=headers, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())

# Function to call to AOAI
# Send a call to the model deployed on Azure OpenAI
def call_aoai(aoai_client, aoai_model_name, system_prompt, user_prompt, temperature, max_tokens):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}]
    try:
        response = aoai_client.chat.completions.create(
            model=aoai_model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        json_response = json.loads(response.model_dump_json())
        response = json_response['choices'][0]['message']['content']
    except Exception as ex:
        print(f'ERROR call_aoai: {ex}')
        response = None
    
    return response

def on_open(ws):
    print("Connected! Start speaking...")
    session_config = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": AOAI_DEPLOYMENT_NAME_STT,
                "prompt": "Respond in the same language than the text."
            },
            "input_audio_noise_reduction": {"type": "near_field"},
            "turn_detection": {"type": "server_vad"}
        }
    }
    ws.send(json.dumps(session_config))

    def stream_microphone():
        try:
            while ws.keep_running:
                if not is_playing_audio.is_set():
                    audio_data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_base64
                    }))
                else:
                    time.sleep(0.1) # Espera breve antes de volver a comprobar

        except Exception as e:
            print("Audio streaming error:", e)
            ws.close()

    threading.Thread(target=stream_microphone, daemon=True).start()

def play_audio(response):
    is_playing_audio.set()  # Pausar micrófono
    pcm_bytes = b"".join(chunk for chunk in response.iter_bytes())
    # Play the audio using PyAudio
    audio_stream = audio_interface.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        output=True)
    audio_stream.write(pcm_bytes)
    audio_stream.stop_stream()
    audio_stream.close()
    is_playing_audio.clear()  # Reanudar micrófono

def on_message(ws, message):
    try:
        data = json.loads(message)
        event_type = data.get("type", "")
        print("\tEvent type:", event_type)
        #print(data)   
        # Stream live incremental transcripts
        if event_type == "conversation.item.input_audio_transcription.delta":
            transcript_piece = data.get("delta", "")
            if transcript_piece:
                print(transcript_piece, end=' ', flush=True)
        if event_type == "conversation.item.input_audio_transcription.completed":
            print(f"\n>> {data["transcript"]}\n")
            #call_rag(data["transcript"])
            print("Calling AOAI...")
            answer = call_aoai(aoai_client, AOAI_DEPLOYMENT_NAME, "You are a helpful assistant.", data["transcript"], 0.7, 1000)
            print("Response from AOAI:", answer)

            # Call TTS API to convert text to speech
            print("Calling TTS API...")
            with tts_client.audio.speech.with_streaming_response.create(
                model=AOAI_DEPLOYMENT_NAME_TTS,
                voice="coral",
                input=(answer),
                instructions="Affect/personality: A cheerful guide\\n\\nTone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable.\\n\\nPronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow.\\n\\nPause: Brief, purposeful pauses after key instructions (e.g., \"cross the street\" and \"turn right\") to allow time for the listener to process the information and follow along.\\n\\nEmotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey.",
                response_format="pcm",
            ) as response:
                play_audio(response)

            print("You can continue Start speaking...")
        if event_type == "item":
            transcript = data.get("item", "")
            if transcript:
                print("\nFinal transcript:", transcript)

    except Exception as e:
        print("Error:", e)
        pass  # Ignore unrelated events


def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("Disconnected from server.")
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

print("Connecting to OpenAI Realtime API...")
ws_app = websocket.WebSocketApp(
    url,
    header=headers,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

ws_app.run_forever()