#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shlex
import string
import subprocess
import sys
import threading
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from .const import VocoderQuality

_LOGGER = logging.getLogger("glow_speak")

# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text", nargs="*", help="Text to convert to speech (default: stdin)"
    )
    parser.add_argument("-v", "--voice", help="Name of voice or language")
    parser.add_argument("--voices", action="store_true", help="List voices and exit")
    parser.add_argument(
        "--voices-dir",
        help="Directory with voices (default: $HOME/.local/share/glow-speak/voices)",
    )
    parser.add_argument("--tts", help="Path to TTS model directory")
    parser.add_argument(
        "--quality",
        default="high",
        choices=list(VocoderQuality),
        help="Quality of vocoder (default: high)",
    )
    parser.add_argument(
        "--vocoder", help="Path to vocoder model directory (default: hifi-gan)"
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
        "-o",
        "--output-file",
        help="Write entire WAV output to a file (use '-' for stdout)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write entire WAV output to stdout (alias of --output-file -)",
    )
    parser.add_argument(
        "--output-dir",
        help="Write WAV file(s) to a directory (one for each input line, see --output-naming)",
    )
    parser.add_argument(
        "--output-naming",
        default="{time}_{text:.100s}",
        help="Format string used for file names with --output-dir (receives {time} for timestamp and {text} for filtered transcription)",
    )
    parser.add_argument(
        "--text-is-phoneme-ids",
        action="store_true",
        help="Input 'text' is actually whitespace-separated phoneme ids (see phonemes.txt)",
    )
    parser.add_argument(
        "--text-language",
        help="eSpeak voice for input text (if different than voice language)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    from .download import find_voice

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    if args.voices:
        # List voices and exit
        list_voices(args.voices_dir)
        return

    if args.output_file == "-":
        args.stdout = True
        args.output_file = None

    if args.tts:
        # Directory with voice (must have generator.onnx)
        args.tts = Path(args.tts)
    else:
        if not args.voice:
            # List voices and exit
            print("Missing --voice", file=sys.stderr)
            list_voices(args.voices_dir)
            return

        args.tts = find_voice(args.voice, voices_dir=args.voices_dir)
        assert args.tts is not None, f"Voice not found: {args.voice}"

    if args.vocoder:
        # User-supplied vocoder path (must have generator.onnx)
        args.vocoder = Path(args.vocoder)
    else:
        # Use built-in hifi-gan
        from . import get_vocoder_dir

        args.vocoder = get_vocoder_dir(args.quality)

    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Test format string
        args.output_naming.format(time=0, text="Sample text")

    if args.output_file:
        args.output_file = Path(args.output_file)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------

    wav_queue = None
    play_thread = None
    play_proc: typing.Optional[subprocess.Popen] = None
    play_running = True

    if args.play:
        # Execute external command to play audio in a separate thread (from a queue)
        play_command = shlex.split(args.play)
        wav_queue = Queue(maxsize=args.queue_lines)

        def play_target():
            nonlocal play_proc

            while play_running:
                try:
                    wav_bytes = wav_queue.get()
                    if (wav_bytes is None) or (not play_running):
                        break

                    _LOGGER.debug(play_command)
                    play_proc = subprocess.Popen(play_command, stdin=subprocess.PIPE)
                    play_proc.stdin.write(wav_bytes)
                    play_proc.stdin.flush()
                    play_proc.wait()

                    play_proc = None
                except Exception:
                    _LOGGER.exception("play_proc")

        play_thread = threading.Thread(target=play_target, daemon=True)
        play_thread.start()

    # -------------------------------------------------------------------------

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

    if not args.text_language:
        lang_path = args.tts / "LANGUAGE"
        assert (
            lang_path.is_file()
        ), "Missing --voice or LANGUAGE file in voice directory"

        args.text_language = lang_path.read_text().strip()
        _LOGGER.debug("Text language: %s", args.text_language)

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
    phonemizer = Phonemizer(default_voice=args.text_language)

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
        #
        # And paragraphs separate text instead.
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

    # -------------------------------------------------------------------------
    # Main loop

    audios = []
    try:
        for text in texts:
            text = text.strip()
            if not text:
                continue

            _LOGGER.debug(text)

            if args.text_is_phoneme_ids:
                # Text *is* phoneme ids already
                text_ids = np.array([int(n) for n in text.split()], dtype=np.int64)
            else:
                # Text to phoneme ids
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

            wav_bytes: typing.Optional[bytes] = None

            if (wav_queue is not None) or args.output_dir:
                wav_bytes = audio_to_wav(
                    audio=audio,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_bytes=sample_bytes,
                )

            if wav_queue is not None:
                # Audio will play
                assert wav_bytes is not None
                wav_queue.put(wav_bytes)

            if args.output_dir:
                # Output WAV(s) to directory
                assert wav_bytes is not None

                text_filtered = text.strip().replace(" ", "_")
                text_filtered = text_filtered.translate(
                    str.maketrans("", "", string.punctuation.replace("_", ""))
                )
                output_name = (
                    args.output_naming.format(time=int(time.time()), text=text_filtered)
                    + ".wav"
                )

                output_path = args.output_dir / output_name
                output_path.write_bytes(wav_bytes)
                _LOGGER.debug("Write %s byte(s) to %s", len(wav_bytes), output_path)
            elif args.output_file or args.stdout:
                # Combine into single WAV for output later
                audios.append(audio)
    except KeyboardInterrupt:
        pass
    finally:
        play_running = False

        if wav_queue is not None:
            # Drain queue
            while not wav_queue.empty():
                wav_queue.get()

            wav_queue.put(None)

        if play_proc is not None:
            play_proc.terminate()

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

            if args.output_file:
                # Write to WAV file
                args.output_file.write_bytes(wav_bytes)
                _LOGGER.debug(
                    "Wrote %s byte(s) to %s", len(wav_bytes), args.output_file
                )
            elif args.stdout:
                # Write to stdout
                sys.stdout.buffer.write(wav_bytes)


# -----------------------------------------------------------------------------


def list_voices(voices_dir: typing.Optional[typing.Union[str, Path]] = None):
    """List available voices"""
    from .download import LANG_VOICES, OTHER_VOICES, find_voice

    downloaded = "[downloaded]"
    empty = " " * len(downloaded)

    voices = set(LANG_VOICES.values())
    voices.update(OTHER_VOICES)

    for voice in sorted(voices):
        maybe_voice_dir = find_voice(voice, voices_dir=voices_dir)

        if maybe_voice_dir is None:
            print(empty, voice, sep="\t")
        else:
            print(downloaded, voice, sep="\t")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
