#!/usr/bin/env python3
"""Glow-speak web server"""
import argparse
import asyncio
import functools
import hashlib
import io
import json
import logging
import os
import signal
import string
import tempfile
import time
import typing
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

import hypercorn
import numpy as np
import onnxruntime
import quart_cors
from quart import (
    Quart,
    Response,
    helpers,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from swagger_ui import quart_api_doc

from espeak_phonemizer import Phonemizer
from phonemes2ids import load_phoneme_ids, load_phoneme_map

from . import (
    PhonemeGuesser,
    VocoderQuality,
    get_vocoder_dir,
    ids_to_mels,
    init_denoiser,
    mels_to_audio,
    text_to_ids,
)

_MISSING = object()

_DIR = Path(__file__).parent
_TEMPLATE_DIR = _DIR / "templates"
_TEMP_DIR: typing.Optional[str] = None

_LOGGER = logging.getLogger("glow_speak.http_server")
_LOOP = asyncio.get_event_loop()

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tts-dir", required=True, help="Directory with TTS models (one per directory)"
)
parser.add_argument(
    "--host", default="0.0.0.0", help="Host of HTTP server (default: 0.0.0.0)"
)
parser.add_argument(
    "--port", type=int, default=5002, help="Port of HTTP server (default: 5002)"
)
parser.add_argument(
    "--cache-dir",
    nargs="?",
    default=_MISSING,
    help="Enable WAV cache with optional directory (default: /tmp)",
)
parser.add_argument("--logfile", help="Path to logging file (default: stderr)")
parser.add_argument(
    "--debug", action="store_true", help="Print DEBUG messages to console"
)
args = parser.parse_args()

# Set up logging
log_args = {}

if args.debug:
    log_args["level"] = logging.DEBUG
else:
    log_args["level"] = logging.INFO

if args.logfile:
    log_args["filename"] = args.logfile

logging.basicConfig(**log_args)  # type: ignore
_LOGGER.debug(args)

args.tts_dir = Path(args.tts_dir)

# -----------------------------------------------------------------------------

app = Quart("glow-speak", template_folder=str(_TEMPLATE_DIR))
app.secret_key = str(uuid4())

if args.debug:
    app.config["TEMPLATES_AUTO_RELOAD"] = True

app = quart_cors.cors(app)

# -----------------------------------------------------------------------------

_SESSION_OPTIONS = onnxruntime.SessionOptions()


@dataclass
class TTSInfo:
    model: typing.Any
    phoneme_to_id: typing.Dict[str, int]
    phoneme_guesser: PhonemeGuesser
    language: typing.Optional[str] = None
    phoneme_map: typing.Optional[typing.Dict[str, typing.List[str]]] = None


_TTS_INFO: typing.Dict[str, TTSInfo] = {}
_TTS_INFO_LOCK = asyncio.Lock()


@dataclass
class VocoderInfo:
    model: typing.Any
    sample_rate: int
    channels: int
    sample_bytes: int
    bias_spec: typing.Optional[np.ndarray] = None


# quality -> onnx model
_VOCODER_INFO: typing.Dict[VocoderQuality, VocoderInfo] = {}
_VOCODER_INFO_LOCK = asyncio.Lock()

# language -> phonemizer
_PHONEMIZERS: typing.Dict[str, Phonemizer] = {}
_PHONEMIZERS_LOCK = asyncio.Lock()


@dataclass(frozen=True)  # must be hashable
class TextToWavParams:
    text: str
    voice: str
    text_language: str
    vocoder_quality: VocoderQuality
    denoiser_strength: float
    noise_scale: float
    length_scale: float


# params -> Path
_WAV_CACHE: typing.Dict[TextToWavParams, Path] = {}

# -----------------------------------------------------------------------------


async def text_to_wav(
    text: str,
    voice: str,
    text_language: typing.Optional[str] = None,
    vocoder_quality: VocoderQuality = VocoderQuality.HIGH,
    denoiser_strength: float = 0.005,
    noise_scale: float = 0.333,
    length_scale: float = 1.0,
    no_cache: bool = False,
) -> bytes:
    """Runs TTS for each line and accumulates all audio into a single WAV."""
    # Load TTS model from cache or disk
    async with _TTS_INFO_LOCK:
        maybe_tts = _TTS_INFO.get(voice)
        if maybe_tts is None:
            voice_dir = helpers.safe_join(args.tts_dir, voice)
            _LOGGER.debug("Loading TTS model from %s", voice_dir)

            tts_model = await _LOOP.run_in_executor(
                None,
                functools.partial(
                    onnxruntime.InferenceSession,
                    str(voice_dir / "generator.onnx"),
                    sess_options=_SESSION_OPTIONS,
                ),
            )

            # Load model language
            model_language: typing.Optional[str] = None
            lang_path = voice_dir / "LANGUAGE"
            if lang_path.is_file():
                model_language = lang_path.read_text().strip()

            # Load phoneme -> id map
            with open(voice_dir / "phonemes.txt") as phonemes_file:
                phoneme_to_id = load_phoneme_ids(phonemes_file)

            # Load phoneme -> phoneme map
            phoneme_map = None
            phoneme_map_path = voice_dir / "phoneme_map.txt"
            if phoneme_map_path.is_file():
                with open(phoneme_map_path) as phoneme_map_file:
                    phoneme_map = load_phoneme_map(phoneme_map_file)

            phoneme_guesser = PhonemeGuesser(phoneme_to_id, phoneme_map)

            # Add to cache
            tts_info = TTSInfo(
                model=tts_model,
                phoneme_to_id=phoneme_to_id,
                phoneme_map=phoneme_map,
                phoneme_guesser=phoneme_guesser,
                language=model_language,
            )

            _TTS_INFO[voice] = tts_info
        else:
            tts_info = maybe_tts

    if text_language is None:
        text_language = tts_info.language

    assert (
        text_language is not None
    ), "Text language not set (missing LANGUAGE file in voice directory)"

    text_to_wav_params: typing.Optional[TextToWavParams] = None
    if _TEMP_DIR and (not no_cache):
        # Look up in cache
        text_to_wav_params = TextToWavParams(
            text=text,
            voice=voice,
            text_language=text_language,
            vocoder_quality=vocoder_quality,
            noise_scale=noise_scale,
            length_scale=length_scale,
            denoiser_strength=denoiser_strength,
        )

        maybe_wav_path = _WAV_CACHE.get(text_to_wav_params)
        if (maybe_wav_path is not None) and maybe_wav_path.is_file():
            _LOGGER.debug("Loading WAV from cache: %s", maybe_wav_path)
            wav_bytes = maybe_wav_path.read_bytes()
            return wav_bytes

    # Load language info
    async with _PHONEMIZERS_LOCK:
        maybe_phonemizer = _PHONEMIZERS.get(text_language)
        if maybe_phonemizer is None:
            # Initialize eSpeak phonemizer
            phonemizer = Phonemizer(default_voice=text_language)
        else:
            phonemizer = maybe_phonemizer

    # Load vocoder from cache or disk
    async with _VOCODER_INFO_LOCK:
        maybe_vocoder = _VOCODER_INFO.get(vocoder_quality)
        if maybe_vocoder is None:
            vocoder_dir = get_vocoder_dir(vocoder_quality)
            _LOGGER.debug("Loading vocoder model from %s", vocoder_dir)

            vocoder_model = await _LOOP.run_in_executor(
                None,
                functools.partial(
                    onnxruntime.InferenceSession,
                    str(vocoder_dir / "generator.onnx"),
                    sess_options=_SESSION_OPTIONS,
                ),
            )

            bias_spec = None

            # Load audio config
            with open(vocoder_dir / "config.json") as vocoder_config_file:
                vocoder_config = json.load(vocoder_config_file)
                vocoder_audio = vocoder_config["audio"]
                num_mels = int(vocoder_audio["num_mels"])
                sample_rate = int(vocoder_audio["sampling_rate"])
                channels = int(vocoder_audio["channels"])
                sample_bytes = int(vocoder_audio["sample_bytes"])

                if denoiser_strength > 0:
                    _LOGGER.debug("Initializing denoiser")
                    bias_spec = await _LOOP.run_in_executor(
                        None, functools.partial(init_denoiser, vocoder_model, num_mels)
                    )

                vocoder_info = VocoderInfo(
                    model=vocoder_model,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_bytes=sample_bytes,
                    bias_spec=bias_spec,
                )

                _VOCODER_INFO[vocoder_quality] = vocoder_info
        else:
            vocoder_info = maybe_vocoder

    # Synthesize each line separately.
    # Accumulate into a single WAV file.
    _LOGGER.info(
        "Synthesizing with %s (lang=%s, quality=%s) (%s char(s))...",
        voice,
        text_language,
        vocoder_quality.value,
        len(text),
    )
    audios = []
    start_time = time.time()

    with io.StringIO(text) as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            text_ids = text_to_ids(
                text=line,
                phonemizer=phonemizer,
                phoneme_to_id=tts_info.phoneme_to_id,
                phoneme_map=tts_info.phoneme_map,
                missing_func=tts_info.phoneme_guesser.guess_ids,
            )

            mels_start_time = time.perf_counter()
            mels = ids_to_mels(
                ids=text_ids,
                tts_model=tts_info.model,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )
            _LOGGER.debug("Mels in %s second(s)", time.perf_counter() - mels_start_time)

            audio_start_time = time.perf_counter()
            audio = mels_to_audio(
                mels=mels,
                vocoder_model=vocoder_info.model,
                denoiser_strength=denoiser_strength,
                bias_spec=vocoder_info.bias_spec,
            )
            _LOGGER.debug(
                "Audio in %s second(s)", time.perf_counter() - audio_start_time
            )

            infer_sec = time.perf_counter() - mels_start_time
            audio_sec = len(audio) / vocoder_info.sample_rate
            real_time_factor = infer_sec / audio_sec if audio_sec > 0 else 0.0
            _LOGGER.debug(
                "Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
                real_time_factor,
                infer_sec,
                audio_sec,
            )

            audios.append(audio)

    with io.BytesIO() as wav_io:
        wav_out: wave.Wave_write = wave.open(wav_io, "wb")
        with wav_out:
            wav_out.setframerate(vocoder_info.sample_rate)
            wav_out.setnchannels(vocoder_info.channels)
            wav_out.setsampwidth(vocoder_info.sample_bytes)
            wav_out.writeframes(audio.tobytes())

        wav_bytes = wav_io.getvalue()

    if _TEMP_DIR and (text_to_wav_params is not None) and (not no_cache):
        try:
            # Save to cache
            text_filtered = text.strip().replace(" ", "_")
            text_filtered = text_filtered.translate(
                str.maketrans("", "", string.punctuation.replace("_", ""))
            )

            param_hash = hashlib.md5()
            param_hash.update(str(text_to_wav_params).encode("utf-8"))

            output_name = "{text:.100s}_{hash}.wav".format(
                text=text_filtered, hash=param_hash.hexdigest()
            )
            output_path = os.path.join(_TEMP_DIR, output_name)

            with open(output_path, mode="wb") as output_file:
                output_file.write(wav_bytes)
                _WAV_CACHE[text_to_wav_params] = Path(output_path)
                _LOGGER.debug(
                    "Wrote %s byte(s) to cache file: %s", len(wav_bytes), output_path
                )
        except Exception:
            _LOGGER.exception("text_to_wav")

    end_time = time.time()
    _LOGGER.info(
        "Synthesized %s byte(s) in %s second(s)", len(wav_bytes), end_time - start_time
    )

    return wav_bytes


def get_voices() -> typing.Dict[str, str]:
    """Get dict of voice names/descriptions"""
    voices: typing.Dict[str, str] = {}

    for maybe_voice_dir in args.tts_dir.iterdir():
        if (not maybe_voice_dir.is_dir()) or (
            not (maybe_voice_dir / "generator.onnx").is_file()
        ):
            # Not a voice directory
            continue

        voice_name = maybe_voice_dir.name
        description = ""

        # Load description, if available
        description_path = maybe_voice_dir / "DESCRIPTION"
        if description_path.is_file():
            description = description_path.read_text().strip()

        voices[voice_name] = description

    return voices


# -----------------------------------------------------------------------------
# HTTP Endpoints
# -----------------------------------------------------------------------------


@app.route("/api/voices")
async def app_voices() -> Response:
    """Get available voices."""
    return jsonify(get_voices())


@app.route("/api/tts", methods=["GET", "POST"])
async def app_say() -> Response:
    """Speak text to WAV."""
    voice = request.args.get("voice", "")
    assert voice, "No voice provided"

    # TTS settings
    tts_args: typing.Dict[str, typing.Any] = {}
    noise_scale = request.args.get("noiseScale")
    if noise_scale is not None:
        tts_args["noise_scale"] = float(noise_scale)

    length_scale = request.args.get("lengthScale")
    if length_scale is not None:
        tts_args["length_scale"] = float(length_scale)

    # Text can come from POST body or GET ?text arg
    if request.method == "POST":
        text = (await request.data).decode()
    else:
        text = request.args.get("text", "")

    assert text, "No text provided"

    # Vocoder settings
    quality_str = request.args.get("quality")
    if quality_str:
        tts_args["vocoder_quality"] = VocoderQuality(quality_str)

    denoiser_strength = request.args.get("denoiserStrength")
    if denoiser_strength is not None:
        tts_args["denoiser_strength"] = float(denoiser_strength)

    # Phonemizer settings
    text_language = request.args.get("textLanguage")
    if text_language:
        tts_args["text_language"] = text_language

    # Cache settings
    no_cache = request.args.get("noCache")
    if no_cache is not None:
        tts_args["no_cache"] = no_cache.strip().lower() in {"true", "1", "yes"}

    wav_bytes = await text_to_wav(text, voice, **tts_args)

    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/api/phonemes", methods=["GET"])
async def api_phonemes():
    """Get phonemes for voice"""
    voice = request.args.get("voice", "")
    assert voice, "No voice provided"

    voice_dir = helpers.safe_join(args.tts_dir, voice)
    with open(voice_dir / "phonemes.txt", "r") as phonemes_file:
        phoneme_ids = load_phoneme_ids(phonemes_file)

    return jsonify(phoneme_ids)


# -----------------------------------------------------------------------------

# MaryTTS compatibility layer
@app.route("/process", methods=["GET", "POST"])
async def api_process():
    """MaryTTS-compatible /process endpoint"""
    if request.method == "POST":
        data = parse_qs((await request.data).decode())
        text = data.get("INPUT_TEXT", [""])[0]
        voice = data.get("VOICE", [""])[0]
    else:
        text = request.args.get("INPUT_TEXT", "")
        voice = request.args.get("VOICE", "")

    # <VOICE>;<VOCODER_QUALITY>
    tts_args = {}
    if ";" in voice:
        voice, quality_str = voice.split(";", maxsplit=1)
        if quality_str:
            tts_args["vocoder_quality"] = VocoderQuality(quality_str)

    wav_bytes = await text_to_wav(text, voice, **tts_args)

    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/voices", methods=["GET"])
async def api_voices():
    """MaryTTS-compatible /voices endpoint"""
    lines = []
    for voice_id in get_voices():
        lines.append(voice_id)

    return "\n".join(lines)


# -----------------------------------------------------------------------------

_CSS_DIR = _DIR / "css"


@app.route("/")
async def app_index():
    """Main page."""
    return await render_template("index.html")


@app.route("/css/<path:filename>", methods=["GET"])
async def css(filename) -> Response:
    """CSS static endpoint."""
    return await send_from_directory(_CSS_DIR, filename)


# Swagger UI
quart_api_doc(
    app,
    config_path=str(_DIR / "swagger.yaml"),
    url_prefix="/openapi",
    title="Glow-Speak",
)


@app.errorhandler(Exception)
async def handle_error(err) -> typing.Tuple[str, int]:
    """Return error as text."""
    _LOGGER.exception(err)
    return (f"{err.__class__.__name__}: {err}", 500)


# -----------------------------------------------------------------------------
# Run Web Server
# -----------------------------------------------------------------------------

hyp_config = hypercorn.config.Config()
hyp_config.bind = [f"{args.host}:{args.port}"]

# Create shutdown event for Hypercorn
shutdown_event = asyncio.Event()


def _signal_handler(*_: typing.Any) -> None:
    """Signal shutdown to Hypercorn"""
    shutdown_event.set()


_LOOP.add_signal_handler(signal.SIGTERM, _signal_handler)

try:
    # Need to type cast to satisfy mypy
    shutdown_trigger = typing.cast(
        typing.Callable[..., typing.Awaitable[None]], shutdown_event.wait
    )

    with tempfile.TemporaryDirectory(prefix="glow-speak") as temp_dir:
        if args.cache_dir != _MISSING:
            if args.cache_dir is None:
                # Use temporary directory
                _TEMP_DIR = temp_dir
            else:
                # Use user-supplied cache directory
                os.makedirs(args.cache_dir, exist_ok=True)
                _TEMP_DIR = args.cache_dir

        if _TEMP_DIR:
            _LOGGER.debug("Cache directory: %s", _TEMP_DIR)

        _LOOP.run_until_complete(
            hypercorn.asyncio.serve(app, hyp_config, shutdown_trigger=shutdown_trigger)
        )
except KeyboardInterrupt:
    _LOOP.call_soon(shutdown_event.set)
