"""
Speech-to-AOAI-to-TTS with streaming
Real-time transcription
Streaming response from Azure OpenAI model
Streaming TTS and instant playback
"""

import os
import json
import base64
import threading
import queue
import re
import time

import websocket
import pyaudio
import requests                          # (lo usa websocket-client)
from dotenv import load_dotenv
from openai import AzureOpenAI

# Loading environment variables
load_dotenv(override=True)

# Audio constants
RATE            = 24_000                 # 24 kHz → matches Azure voices
CHANNELS        = 1
FORMAT          = pyaudio.paInt16        # 16-bit little-endian
CHUNK           = 1024
WAV_HEADER_LEN  = 44                     # RIFF/WAV header bytes
VOICE = "ballad"

PROMPT_STT = "Your response **MUST** be in the same language than the user's question."
SYSTEM_PROMPT_CHAT = "You are a helpful assistant. Respond in the same language than the user's question." # Respond in Spanish.

is_playing_audio = threading.Event()     # Activates while TTS is playing

# Websocket STT (transcription)
ws_url = (
    f'{os.environ["AZURE_OPENAI_ENDPOINT_STT"].replace("https", "wss")}'
    f'/openai/realtime?api-version={os.environ["AZURE_OPENAI_API_VERSION_STT"]}&intent=transcription'
)
ws_headers = {"api-key": os.environ["AZURE_OPENAI_API_KEY_STT"]}

# PyAudio – Microphone and speaker
audio = pyaudio.PyAudio()

mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

# Reusable speaker output (prevents clicks when opening/closing)
speaker_out = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    frames_per_buffer=CHUNK,
)

# Azure OpenAI Clients
aoai_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

tts_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_TTS"],
    api_key=os.environ["AZURE_OPENAI_API_KEY_TTS"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION_TTS"],
)

AOAI_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
AOAI_DEPLOYMENT_NAME_STT = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_STT"]
AOAI_DEPLOYMENT_NAME_TTS = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_TTS"]

# Utils Functions
def play_audio_streaming(pcm_iter):
    """
    Plays the audio received via streaming:
    1. Discards the WAV header sent by TTS.
    2. Ensures that byte pairs (16-bit) are always written.
    3. Uses a single output stream to avoid clicks.
    """
    first_chunk = True
    leftover    = b""               # Unpaired byte remaining between calls

    for chunk in pcm_iter:
        if not chunk:
            continue

        # --- 1) Remove WAV header the first time -------------------
        if first_chunk:
            first_chunk = False
            if chunk.startswith(b"RIFF"):
                chunk = chunk[WAV_HEADER_LEN:]   # skip header
                if not chunk:
                    continue

        # --- 2) Align to 16-bit (multiple of 2 bytes) ---------------
        chunk = leftover + chunk
        if len(chunk) & 1:                       # Odd-sized buffer
            leftover, chunk = chunk[-1:], chunk[:-1]
        else:
            leftover = b""

        # --- 3) Playback -----------------------------------------
        if chunk:
            speaker_out.write(chunk)

    # If a byte is left over, discard it; half a sample can’t be played

# ----------------------------------------------------------------------------
# AOAI model + TTS – everything in streaming
# ----------------------------------------------------------------------------
def assistant_stream(question: str):
    """
    Receives the response from AOAI model in streaming
    and sends it to TTS sentence by sentence.
    """
    # Text Queue → TTS
    tts_queue: queue.Queue[str | None] = queue.Queue()

    # --- Worker that consumes the queue and plays each chunk ----------
    def tts_worker():
        is_playing_audio.set()                      # Pauses microphone input
        #print("\nAssistant:", end=" ", flush=True)
        while True:
            fragment = tts_queue.get() 
            #print(fragment, end=" ", flush=True)
            if fragment is None or fragment == "":
                break

            # TTS in streaming
            with tts_client.audio.speech.with_streaming_response.create(
                model=AOAI_DEPLOYMENT_NAME_TTS,
                voice=VOICE,                    # voz
                input=fragment,
                instructions=(
                    "Affect/personality: A cheerful guide\n\n"
                    "Tone: Friendly, clear, and reassuring.\n"
                    "Pause: Brief pauses after key instructions.\n"
                    "Emotion: Warm and supportive."
                ),
                response_format="pcm",
            ) as tts_response:
                play_audio_streaming(tts_response.iter_bytes())

        print('\n____________________________________________________')
        print("Say something else!")
        is_playing_audio.clear()                   # Resume the microphone
  
    threading.Thread(target=tts_worker, daemon=True).start()

    # ---  Request chat in streaming ---------------------------------
    buffer = ""
    print("\nAssistant:\n", end=" ", flush=True)

    for chunk in aoai_client.chat.completions.create(
        model=AOAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_CHAT},
            {"role": "user",   "content": question},
        ],
        temperature=0.7,
        max_tokens=1000,
        stream=True,
    ):
        if not chunk.choices:
            continue

        for choice in chunk.choices:
            delta_obj     = getattr(choice, "delta", None)
            content_piece = getattr(delta_obj, "content", None)
            if not content_piece:
                continue

            # 1) Display on screen
            print(content_piece, end="", flush=True)

            # 2) Accumulate and split by sentence end
            buffer += content_piece
            if re.search(r"[.!?\n]$", buffer):
                tts_queue.put(buffer.strip())
                buffer = ""

    # Any remaining text
    if buffer.strip():
        tts_queue.put(buffer.strip())

    # End signal to the TTS worker
    tts_queue.put(None)

# ---------------------------------------------------------------------------
# Callbacks websocket STT
# ---------------------------------------------------------------------------
def on_open(ws):
    print("Connected. Say something!")

    session_cfg = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": AOAI_DEPLOYMENT_NAME_STT,
                "prompt": PROMPT_STT,
            },
            "input_audio_noise_reduction": {"type": "near_field"},
            "turn_detection": {"type": "server_vad"},
        },
    }
    ws.send(json.dumps(session_cfg))

    # Thread that sends microphone audio
    def mic_sender():
        try:
            while ws.keep_running:
                if is_playing_audio.is_set():      # If TTS is playing: pause mic
                    time.sleep(0.05)
                    continue
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
                ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(data).decode(),
                        }
                    )
                )
        except Exception as exc:
            print("Error sending audio:", exc)
            ws.close()

    threading.Thread(target=mic_sender, daemon=True).start()

def on_message(ws, message):
    try:
        ev = json.loads(message)
        etype = ev.get("type", "")

        if etype == "conversation.item.input_audio_transcription.delta":
            print(ev.get("delta", ""), end=" ", flush=True)

        elif etype == "conversation.item.input_audio_transcription.completed":
            transcript = ev["transcript"]
            print(f"\n>> {transcript}\n")

            # Launches AOAI model + TTS in a separate thread to avoid blocking the websocket
            threading.Thread(
                target=assistant_stream, args=(transcript,), daemon=True
            ).start()

    except Exception as exc:
        print("on_message error:", exc)

def on_error(ws, error):
    print("Websocket error:", error)

def on_close(ws, code, reason):
    print("Websocket closed:", code, reason)
    mic_stream.stop_stream()
    mic_stream.close()
    speaker_out.stop_stream()
    speaker_out.close()
    audio.terminate()

# ---------------------------------------------------------------------------
# Starting…
# ---------------------------------------------------------------------------
print("Connected to:", ws_url)
ws_app = websocket.WebSocketApp(
    ws_url,
    header=ws_headers,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

ws_app.run_forever()  