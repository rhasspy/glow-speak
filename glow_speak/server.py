#!/usr/bin/env python3
"""
Does text to speech for clients on a Unix domain socket.

Reads lines of text from clients.
Prints WAV paths back to clients.

WAV files are cached in the system temp directory of wherever --cache-dir
points to.
"""
import argparse
import json
import logging
import os
import socket
import tempfile
import time
import typing
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnxruntime

from espeak_phonemizer import Phonemizer
from phonemes2ids import load_phoneme_ids, load_phoneme_map

from . import PhonemeGuesser, ids_to_mels, init_denoiser, mels_to_audio, text_to_ids

_LOGGER = logging.getLogger("glow_speak.server")


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--voice", required=True, help="espeak-ng voice")
    parser.add_argument("--tts", required=True, help="Path to TTS model directory")
    parser.add_argument(
        "--vocoder", required=True, help="Path to vocoder model directory"
    )
    parser.add_argument("--socket", required=True, help="Path to Unix domain socket")
    parser.add_argument(
        "--cache-dir", help="Set directory for WAV cache (default: use tempfile)"
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
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    args.tts = Path(args.tts)
    args.vocoder = Path(args.vocoder)

    # Unix domain socket
    args.socket = Path(args.socket)

    try:
        args.socket.unlink()
    except FileNotFoundError:
        pass

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
    print("Ready")

    # text -> WAV path
    path_cache: typing.Dict[str, Path] = {}

    # Create domain socket
    with tempfile.TemporaryDirectory(prefix="glow-tts_") as temp_dir:
        if not args.cache_dir:
            args.cache_dir = temp_dir

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(args.socket))
        sock.listen()

        while True:
            try:
                connection, _client_address = sock.accept()
                with connection, connection.makefile(mode="rw") as conn_file:
                    # Read line-by-line
                    for text in conn_file:
                        text = text.strip()
                        if not text:
                            continue

                        _LOGGER.debug(text)

                        # Check WAV cache first
                        cached_wav_path = path_cache.get(text)
                        if (
                            cached_wav_path
                            and os.path.isfile(cached_wav_path)
                            and (os.path.getsize(cached_wav_path) > 0)
                        ):
                            # Cache hit
                            _LOGGER.debug(
                                "Using WAV file from cache: %s", cached_wav_path
                            )
                            print(cached_wav_path, file=conn_file, flush=True)
                            continue

                        # Cache miss
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
                        _LOGGER.debug(
                            "Mels in %s second(s)",
                            time.perf_counter() - mels_start_time,
                        )

                        audio_start_time = time.perf_counter()
                        audio = mels_to_audio(
                            mels=mels,
                            vocoder_model=vocoder,
                            denoiser_strength=args.denoiser,
                            bias_spec=bias_spec,
                        )
                        _LOGGER.debug(
                            "Audio in %s second(s)",
                            time.perf_counter() - audio_start_time,
                        )

                        infer_sec = time.perf_counter() - mels_start_time
                        audio_sec = len(audio) / sample_rate
                        real_time_factor = (
                            infer_sec / audio_sec if audio_sec > 0 else 0.0
                        )
                        _LOGGER.debug(
                            "Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
                            real_time_factor,
                            infer_sec,
                            audio_sec,
                        )

                        # Save to cache directory
                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".wav", dir=temp_dir, delete=False
                        ) as temp_file:
                            wav_out: wave.Wave_write = wave.open(temp_file, "wb")
                            with wav_out:
                                wav_out.setframerate(sample_rate)
                                wav_out.setnchannels(channels)
                                wav_out.setsampwidth(sample_bytes)
                                wav_out.writeframes(audio.tobytes())

                            wav_path = temp_file.name

                            if len(audio) > 0:
                                # Only cache if audio is preset
                                path_cache[text] = wav_path

                        print(wav_path, file=conn_file, flush=True)
            except Exception:
                _LOGGER.exception("main")


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
