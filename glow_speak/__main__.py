#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
import onnxruntime

from espeak_phonemizer import Phonemizer
from phonemes2ids import load_phoneme_ids, load_phoneme_map

from . import (
    PhonemeGuesser,
    audio_to_wav,
    ids_to_mels,
    init_denoiser,
    mels_to_audio,
    text_to_ids,
)

_LOGGER = logging.getLogger("glow_speak")


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text", nargs="*", help="Text to convert to speech (default: stdin)"
    )
    parser.add_argument("-v", "--voice", required=True, help="espeak-ng voice")
    parser.add_argument("--tts", required=True, help="Path to TTS model directory")
    parser.add_argument(
        "--vocoder", required=True, help="Path to vocoder model directory"
    )
    parser.add_argument(
        "--denoiser",
        type=float,
        default=0.005,
        help="Strength of denoiser (0 to disable)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.333,
        help="Noise scale (default: 0.333, GlowTTS only)",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Length scale (default: 1.0, GlowTTS only)",
    )
    parser.add_argument("--play", help="Command used to play audio (WAV on stdin)")
    parser.add_argument(
        "--queue-lines",
        type=int,
        default=5,
        help="Number of lines to process ahead of playback",
    )
    parser.add_argument(
        "--process-on-blank-line",
        action="store_true",
        help="Process text only after encountering a blank line",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    args.tts = Path(args.tts)
    args.vocoder = Path(args.vocoder)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    wav_queue = None
    play_thread = None

    if args.play:
        play_command = shlex.split(args.play)
        wav_queue = Queue(maxsize=args.queue_lines)

        def play_proc():
            while True:
                try:
                    wav_bytes = wav_queue.get()
                    if wav_bytes is None:
                        break

                    _LOGGER.debug(play_command)
                    subprocess.run(play_command, input=wav_bytes, check=True)
                except Exception:
                    _LOGGER.exception("play_proc")

        play_thread = threading.Thread(target=play_proc, daemon=True)
        play_thread.start()

    # -------------------------------------------------------------------------

    # Load TTS/vocoder models in parallel
    sess_options = onnxruntime.SessionOptions()

    with ThreadPoolExecutor() as executor:
        tts_future = executor.submit(
            onnxruntime.InferenceSession,
            str(args.tts / "generator.onnx"),
            sess_options=sess_options,
        )
        vocoder_future = executor.submit(
            onnxruntime.InferenceSession,
            str(args.vocoder / "generator.onnx"),
            sess_options=sess_options,
        )

    tts = tts_future.result()
    _LOGGER.debug("Loaded TTS model from %s", args.tts)

    vocoder = vocoder_future.result()
    _LOGGER.debug("Loaded vocoder model from %s", args.vocoder)

    # Load audio settings and initialize denoiser
    bias_spec = None
    with open(args.vocoder / "config.json") as vocoder_config_file:
        vocoder_config = json.load(vocoder_config_file)
        vocoder_audio = vocoder_config["audio"]
        num_mels = int(vocoder_audio["num_mels"])
        sample_rate = int(vocoder_audio["sampling_rate"])
        channels = int(vocoder_audio["channels"])
        sample_bytes = int(vocoder_audio["sample_bytes"])

        if args.denoiser > 0:
            _LOGGER.debug("Initializing denoiser")
            bias_spec = init_denoiser(vocoder, num_mels)

    # Load phoneme -> id map
    with open(args.tts / "phonemes.txt") as phonemes_file:
        phoneme_to_id = load_phoneme_ids(phonemes_file)

    # Load phoneme -> phoneme map
    phoneme_map = None
    phoneme_map_path = args.tts / "phoneme_map.txt"
    if phoneme_map_path.is_file():
        with open(phoneme_map_path) as phoneme_map_file:
            phoneme_map = load_phoneme_map(phoneme_map_file)

    phoneme_guesser = PhonemeGuesser(phoneme_to_id, phoneme_map)

    # Initialize eSpeak phonemizer
    phonemizer = Phonemizer(default_voice=args.voice)

    # -------------------------------------------------------------------------

    # Read text from stdin or arguments
    if args.text:
        # Use arguments
        texts = args.text
    else:
        # Use stdin
        texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    if args.process_on_blank_line:
        # Combine text until a blank line is encountered.
        # Good for line-wrapped books where
        # sentences are broken
        # up across multiple
        # lines.
        def process_on_blank_line(lines):
            text = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if text:
                        yield text

                    text = ""
                    continue

                text += " " + line

        texts = process_on_blank_line(texts)

    audios = []
    try:
        for text in texts:
            text = text.strip()
            if not text:
                continue

            _LOGGER.debug(text)
            text_ids = text_to_ids(
                text=text,
                phonemizer=phonemizer,
                phoneme_to_id=phoneme_to_id,
                phoneme_map=phoneme_map,
                missing_func=phoneme_guesser.guess_ids,
            )

            mels_start_time = time.perf_counter()
            mels = ids_to_mels(
                ids=text_ids,
                tts_model=tts,
                noise_scale=args.noise_scale,
                length_scale=args.length_scale,
            )
            _LOGGER.debug("Mels in %s second(s)", time.perf_counter() - mels_start_time)

            audio_start_time = time.perf_counter()
            audio = mels_to_audio(
                mels=mels,
                vocoder_model=vocoder,
                denoiser_strength=args.denoiser,
                bias_spec=bias_spec,
            )
            _LOGGER.debug(
                "Audio in %s second(s)", time.perf_counter() - audio_start_time
            )

            infer_sec = time.perf_counter() - mels_start_time
            audio_sec = len(audio) / sample_rate
            real_time_factor = infer_sec / audio_sec if audio_sec > 0 else 0.0
            _LOGGER.debug(
                "Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
                real_time_factor,
                infer_sec,
                audio_sec,
            )

            if wav_queue is not None:
                wav_bytes = audio_to_wav(
                    audio=audio,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_bytes=sample_bytes,
                )

                wav_queue.put(wav_bytes)
            else:
                # Combine into single WAV
                audios.append(audio)
    except KeyboardInterrupt:
        pass
    finally:
        if wav_queue is not None:
            wav_queue.put(None)

        if play_thread is not None:
            play_thread.join()

        if audios:
            # Write combined audio to WAV on stdout
            wav_bytes = audio_to_wav(
                audio=np.concatenate(audios),
                sample_rate=sample_rate,
                channels=channels,
                sample_bytes=sample_bytes,
            )

            sys.stdout.buffer.write(wav_bytes)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
