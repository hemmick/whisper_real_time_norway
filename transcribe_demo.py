#! python3.8

import argparse
import os
import numpy as np
import speech_recognition as sr
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
import logging
import threading
from logging.handlers import QueueHandler, QueueListener
from collections import deque
from queue import Queue
from time import sleep,time
from sys import platform
import scipy.signal
import subprocess
from torchaudio._extension.utils import _init_dll_path

def setup_logging(verbose_level):
    log_queue = Queue()
    queue_handler = QueueHandler(log_queue)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(queue_handler)
    match verbose_level:
        case 1:
            root.setLevel(logging.DEBUG)
        case 2:
            root.setLevel(logging.INFO)
        case _:
            root.setLevel(logging.ERROR)
    listener = QueueListener(log_queue, handler)
    listener.start()
    log_level = root.level
    return log_level, listener

sample_rate = 40000

def setup_microphone(args):
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return None
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    return sr.Microphone(sample_rate=sample_rate, device_index=index)
    else:
        return sr.Microphone(sample_rate=sample_rate)

def transcriber(model_prefix, model_name, device):
    processor = AutoProcessor.from_pretrained(model_prefix + model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_prefix + model_name).to(device)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        generate_kwargs={'num_beams': 5, 'task': 'transcribe', 'language': "no"},
        batch_size=16,
        torch_dtype=torch.float32,
        device=device,
    )

def record_callback(_, audio: sr.AudioData, ring_buffer) -> None:
    data = audio.get_raw_data()
    ring_buffer.append(data)

def apply_bandpass_filter(audio_np, sample_rate, lowcut=100.0, highcut=12000.0):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1")
    b, a = scipy.signal.butter(2, [low, high], btype='band')
    filtered_audio = scipy.signal.lfilter(b, a, audio_np)
    return filtered_audio

def transcribe_worker(pipe, ring_buffer, phrase_timeout, transcription, log_level, noise_floor, stop_event):
    phrase_time = None
    _init_dll_path()

    output_file_path = f"audio/.rec-{int(time())}.raw"
    with open(output_file_path, "ab") as output_file:
        while not stop_event.is_set():
            try:
                now = time()
                if ring_buffer:
                    phrase_complete = False
                    if phrase_time and now - phrase_time > phrase_timeout:
                        phrase_complete = True
                    phrase_time = now

                    audio_data = b''.join(ring_buffer)
                    ring_buffer.clear()

                    # Append raw audio to output binary file
                    output_file.write(audio_data)

                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    filtered_audio = apply_bandpass_filter(audio_np, sample_rate)  # Apply bandpass filter
                    attention_mask = np.where(filtered_audio > noise_floor, 1, 0)  # Create an attention mask based on the noise floor
                    inputs = {"raw": filtered_audio, "sampling_rate": sample_rate, "attention_mask": attention_mask}
                    text = pipe(inputs)

                    if log_level <= logging.DEBUG:
                        logging.debug(f"{text}")
                        print(f"{text}")
                    elif log_level <= logging.INFO:
                        print(f"{text['text'].strip()}")

                    if phrase_complete:
                        transcription.append(text["text"].strip())
                    else:
                        transcription[-1] = text["text"].strip()
                else:
                    sleep(0.15)
            except KeyboardInterrupt:
                break

    # Convert the output binary file to an OGG file and delete the raw file
    oggpath = output_file_path.replace(".raw", ".ogg")
    subprocess.run(["ffmpeg", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", 
                    output_file_path, "-c:a", "libvorbis", "-ab", "32k", oggpath])
    os.remove(output_file_path)

def start_transcription(args, log_level, log_listener):
    source = setup_microphone(args)
    if source is None:
        return

    device = "mps" if torch.mps.is_available() else "cpu"
    pipe = transcriber("NbAiLab/nb-whisper-", args.model, device)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    noise_floor = args.noise_floor if args.noise_floor > 0 else 0.02

    transcription = ['']
    ring_buffer = deque(maxlen=10)  # Ring buffer to store raw audio samples

    os.makedirs(os.path.dirname("audio/"), exist_ok=True)

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    with source:
        recorder.adjust_for_ambient_noise(source)

    recorder.listen_in_background(source, 
                                  lambda _, 
                                  audio: record_callback(_, audio, ring_buffer), 
                                  phrase_time_limit=record_timeout)

    if log_level <= logging.INFO:
        print("Ready to transcribe. Press Ctrl+C to stop.")

    stop_event = threading.Event()
    transcribe_thread = threading.Thread(target=transcribe_worker, 
                                         args=(pipe, 
                                               ring_buffer, 
                                               phrase_timeout, 
                                               transcription, 
                                               log_level, 
                                               noise_floor, 
                                               stop_event), 
                                               daemon=True)
    transcribe_thread.start()

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        transcribe_thread.join()

    log_listener.stop()

    for line in transcription:
        print(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-distil-turbo-beta", help="Model to use",
                        choices=["large", "medium", "medium-semantic", "large-semantic"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--noise_floor", default=0.02, help="Noise floor energy level for attention mask.", type=float)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        default=0, choices=[0, 1, 2], type=int)
    
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    log_level, log_listener = setup_logging(args.verbose)
    start_transcription(args, log_level, log_listener)

if __name__ == "__main__":
    main()
