"""  
STT y TTS con Azure Speech  –  LLM con Azure OpenAI (GPT-4o-mini)  
-----------------------------------------------------------------  
Micrófono ─► STT (Azure Speech) ─► GPT-4o-mini ─► TTS (Azure Speech) ─► Altavoz  
"""  
import os, sys, time, threading, queue, re  
import pyaudio  
from dotenv import load_dotenv  
import azure.cognitiveservices.speech as speechsdk  
from openai import AzureOpenAI  
  
load_dotenv(override=True)  
  
# ---------------------------------------------------------------------------#  
# Parámetros de audio  
# ---------------------------------------------------------------------------#  
RATE, CHANNELS, CHUNK = 24_000, 1, 1024          # 24 kHz mono  
FORMAT          = pyaudio.paInt16                # 16-bit  
VOICE           = "es-MX-DaliaNeural"            # voz TTS  
WAV_HEADER_LEN  = 44                             # cabecera RIFF/WAV  
  
PROMPT_STT         = "Your response MUST be in the same language as the user."  
SYSTEM_PROMPT_CHAT = "You are a helpful assistant. Respond in the same language."  
  
is_playing_audio = threading.Event()             # se activa mientras suena TTS  
  
# ---------------------------------------------------------------------------#  
# Clientes y configuración Azure  
# ---------------------------------------------------------------------------#  
# -- Speech (STT + TTS) ------------------------------------------------------  
speech_key   = os.environ["AZURE_SPEECH_KEY"]  
speech_reg   = os.environ["AZURE_SPEECH_REGION"]  
speech_lang  = os.environ.get("AZURE_SPEECH_LANGUAGE", "es-ES")  
  
#   STT  
speech_config_stt = speechsdk.SpeechConfig(subscription=speech_key, region=speech_reg)  
speech_config_stt.speech_recognition_language = speech_lang  
  
stream_format = speechsdk.audio.AudioStreamFormat(  
    samples_per_second=RATE, bits_per_sample=16, channels=CHANNELS  
)  
push_stream   = speechsdk.audio.PushAudioInputStream(stream_format)  
audio_config_stt = speechsdk.audio.AudioConfig(stream=push_stream)  
speech_recognizer = speechsdk.SpeechRecognizer(  
    speech_config=speech_config_stt, audio_config=audio_config_stt  
)  
  
#   TTS  
speech_config_tts = speechsdk.SpeechConfig(subscription=speech_key, region=speech_reg)  
speech_config_tts.speech_synthesis_voice_name = VOICE  
speech_config_tts.set_speech_synthesis_output_format(  
    speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm  
)  
  
# -- GPT-4o-mini ------------------------------------------------------------  
aoai_client = AzureOpenAI(  
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],  
    api_key        = os.environ["AZURE_OPENAI_API_KEY"],  
    api_version    = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01-preview"),  
)
aoai_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
  
# ---------------------------------------------------------------------------#  
# PyAudio – micro y altavoz  
# ---------------------------------------------------------------------------#  
audio = pyaudio.PyAudio()  
mic_stream = audio.open(  
    format=FORMAT, channels=CHANNELS, rate=RATE,  
    input=True, frames_per_buffer=CHUNK,  
)  
speaker_out = audio.open(  
    format=FORMAT, channels=CHANNELS, rate=RATE,  
    output=True, frames_per_buffer=CHUNK,  
)  
  
# ---------------------------------------------------------------------------#  
# Utilidades  
# ---------------------------------------------------------------------------#  
def play_pcm_bytes(data: bytes, leftover_holder: dict):  
    """  
    Escribe datos PCM (16 bit, 24 kHz) al altavoz garantizando  
    que siempre se envían pares de bytes.  
    """  
    chunk = leftover_holder.get("leftover", b"") + data  
    if len(chunk) & 1:  
        leftover_holder["leftover"], chunk = chunk[-1:], chunk[:-1]  
    else:  
        leftover_holder["leftover"] = b""  
    if chunk:  
        speaker_out.write(chunk)  
  
def tts_speak_streaming(text: str):  
    """  
    Convierte 'text' en audio con Azure Speech y lo reproduce en streaming.  
    """  
    leftover = {"leftover": b""}  
    done_evt = threading.Event()  
  
    # Nueva instancia de sintetizador (independiente por llamada)  
    synthesizer = speechsdk.SpeechSynthesizer(  
        speech_config=speech_config_tts, audio_config=None  
    )  
  
    def on_syn(evt: speechsdk.SpeechSynthesisEventArgs):  
        # evt.result.audio_data trae el fragmento generado hasta el momento  
        if evt.result.audio_data:  
            play_pcm_bytes(evt.result.audio_data, leftover)  
  
    def on_end(evt): done_evt.set()  
    def on_cancel(evt):  
        print("[TTS cancelado]", evt.reason, evt.error_details)  
        done_evt.set()  
  
    synthesizer.synthesizing.connect(on_syn)  
    synthesizer.synthesis_completed.connect(on_end)  
    synthesizer.synthesis_canceled.connect(on_cancel)  
  
    synthesizer.speak_text_async(text)  
    done_evt.wait()                     # bloqueo hasta terminar  
    #synthesizer.close()  
  
# ---------------------------------------------------------------------------#  
# GPT-4o-mini  ➜  TTS (Azure Speech)  
# ---------------------------------------------------------------------------#  
def assistant_stream(user_text: str):  
    """Pide respuesta a GPT-4o-mini y la locuta con Azure Speech TTS."""  
    tts_queue: queue.Queue[str | None] = queue.Queue()  
  
    # Worker que consume cola y habla  
    def tts_worker():  
        is_playing_audio.set()  
        while True:  
            fragment = tts_queue.get()  
            if fragment in (None, ""):  
                break  
            tts_speak_streaming(fragment)  
        print("\n____________________________________________________")  
        print("¡Dime algo más!")  
        is_playing_audio.clear()  
  
    threading.Thread(target=tts_worker, daemon=True).start()  
  
    buffer = ""  
    print("\nAssistant:\n", end="", flush=True)  
    for chunk in aoai_client.chat.completions.create(  
        model=aoai_model,  
        stream=True,  
        messages=[  
            {"role": "system", "content": SYSTEM_PROMPT_CHAT},  
            {"role": "user",   "content": user_text},  
        ],  
        temperature=0.7,  
        max_tokens=1000,  
    ):  
        if not chunk.choices:  
            continue  
        piece = getattr(chunk.choices[0].delta, "content", None)  
        if not piece:  
            continue  
        print(piece, end="", flush=True)  
        buffer += piece  
        if re.search(r"[.!?\n]$", buffer):  
            tts_queue.put(buffer.strip())  
            buffer = ""  
  
    if buffer.strip():  
        tts_queue.put(buffer.strip())  
    tts_queue.put(None)                 # fin para el worker  
  
# ---------------------------------------------------------------------------#  
# Reconocimiento continuo con Azure Speech  
# ---------------------------------------------------------------------------#  
def start_stt():  
    def on_recognizing(evt):  
        if evt.result.text:
            print('.', end=" ", flush=True)
            #print(evt.result.text, end=" ", flush=True)  
  
    def on_recognized(evt):  
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:  
            txt = evt.result.text  
            if txt:  
                print(f"\n>> {txt}\n")  
                threading.Thread(  
                    target=assistant_stream, args=(txt,), daemon=True  
                ).start()  
  
    def on_canceled(evt):  
        print("\n[STT cancelado]", evt.reason, evt.error_details)  
  
    speech_recognizer.recognizing.connect(on_recognizing)  
    speech_recognizer.recognized.connect(on_recognized)  
    speech_recognizer.canceled.connect(on_canceled)  
  
    speech_recognizer.session_started.connect(  
        lambda _: print("🟢 STT iniciado – habla cuando quieras"))  
    speech_recognizer.session_stopped.connect(  
        lambda _: print("🔴 STT detenido"))  
  
    speech_recognizer.start_continuous_recognition()  
  
    # Hilo que envía audio del micro al PushAudioInputStream  
    def mic_sender():  
        try:  
            while True:  
                if is_playing_audio.is_set():  
                    time.sleep(0.05)  
                    continue  
                data = mic_stream.read(CHUNK, exception_on_overflow=False)  
                push_stream.write(data)  
        except Exception as ex:  
            print("Error en mic_sender:", ex)  
            push_stream.close()  
  
    threading.Thread(target=mic_sender, daemon=True).start()  
  
# ---------------------------------------------------------------------------#  
# Main  
# ---------------------------------------------------------------------------#  
if __name__ == "__main__":  
    try:  
        start_stt()  
        while True:                       # mantiene vivo el hilo principal  
            time.sleep(0.5)  
    except KeyboardInterrupt:  
        print("\nCerrando…")  
    finally:  
        speech_recognizer.stop_continuous_recognition()  
        push_stream.close()  
        mic_stream.stop_stream(); mic_stream.close()  
        speaker_out.stop_stream(); speaker_out.close()  
        audio.terminate()  
        sys.exit(0)  