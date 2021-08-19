#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import tempfile
import typing
import urllib.request
from pathlib import Path

# language -> voice name
LANG_VOICES = {
    "en": "en-us_mary_ann",
    "en-us": "en-us_mary_ann",
    "de": "de_thorsten",
    "el": "el_rapunzelina",
    "es": "es_tux",
    "fi": "fi_harri_tapani_ylilammi",
    "hu": "hu_diana_majlinger",
    "it": "it_riccardo_fasol",
    "ko": "ko_kss",
    "nl": "nl_rdh",
    "ru": "ru_nikolaev",
    "sv": "sv_talesyntese",
    "sw": "sw_biblia_takatifu",
    "vi": "vi_vais1000",
    "cmn": "cmn_jing_li",
    "zh-CN": "cmn_jing_li",
    "zh-cn": "cmn_jing_li",
}

OTHER_VOICES = ["en-us_ljspeech"]


_LOGGER = logging.getLogger("glow_speak.download")

# -----------------------------------------------------------------------------


def find_voice(
    voice: str, voices_dir: typing.Optional[typing.Union[str, Path]] = None
) -> typing.Optional[Path]:
    """Find voice by name"""
    resolved_voice = LANG_VOICES.get(voice)

    # 1. voices_dir parameter
    # 2. $GLOW_SPEAK_VOICES environment variable
    # 3. $XDG_DATA_HOME/glow-speak/voices
    # 4. $PWD/local
    vars_voices_dirs = [
        (True, voices_dir),
        ("GLOW_SPEAK_VOICES", "${GLOW_SPEAK_VOICES}"),
        ("XDG_DATA_HOME", "${XDG_DATA_HOME}/glow-speak/voices"),
        ("HOME", "${HOME}/.local/share/glow-speak/voices"),
        (True, "${PWD}/local"),
    ]

    for env_var, dir_path in vars_voices_dirs:
        if dir_path is None:
            continue

        if isinstance(env_var, str) and (os.getenv(env_var) is None):
            # Environment variable must be defined
            continue

        maybe_voice_dir = os.path.join(os.path.expandvars(dir_path), voice)
        maybe_generator = os.path.join(maybe_voice_dir, "generator.onnx")
        _LOGGER.debug("Looking for voice in %s", maybe_voice_dir)

        if os.path.isfile(maybe_generator):
            return Path(maybe_voice_dir)

        # Try with resolved name
        if resolved_voice is not None:
            maybe_voice_dir = os.path.join(os.path.expandvars(dir_path), resolved_voice)
            maybe_generator = os.path.join(maybe_voice_dir, "generator.onnx")
            _LOGGER.debug("Looking for voice in %s", maybe_voice_dir)

            if os.path.isfile(maybe_generator):
                return Path(maybe_voice_dir)

    return None


# -----------------------------------------------------------------------------


def download_voice(voice: str, voices_dir: typing.Union[str, Path], link: str) -> Path:
    """Download and extract a voice (or vocoder)"""
    from tqdm.auto import tqdm

    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)

    response = urllib.request.urlopen(link)

    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".tar.gz") as temp_file:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=voice,
            total=int(response.headers.get("content-length", 0)),
        ) as pbar:
            chunk = response.read(4096)
            while chunk:
                temp_file.write(chunk)
                pbar.update(len(chunk))
                chunk = response.read(4096)

        temp_file.seek(0)

        # Extract
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            _LOGGER.debug("Extracting %s to %s", temp_file.name, temp_dir_str)
            shutil.unpack_archive(temp_file.name, temp_dir_str)

            voice_dir = Path(next(temp_dir.iterdir()))

            dest_voice_dir = voices_dir / voice_dir.name
            if dest_voice_dir.is_dir():
                # Delete existing files
                shutil.rmtree(str(dest_voice_dir))

            # Move files
            _LOGGER.debug("Moving %s to %s", voice_dir, dest_voice_dir)
            shutil.move(str(voice_dir), str(dest_voice_dir))

            return dest_voice_dir


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("voice", nargs="+", help="Voice name(s) to download")
    parser.add_argument(
        "--url-format",
        default="http://github.com/rhasspy/glow-speak/releases/download/v1.0/{voice}.tar.gz",
    )
    parser.add_argument(
        "--voices-dir",
        help="Directory to extract voices into (default: $HOME/.local/share/glow-speak/voices)",
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

    if not args.voices_dir:
        maybe_data_dir = os.getenv("XDG_DATA_HOME")
        if maybe_data_dir is None:
            args.voices_dir = os.path.expandvars(
                "${HOME}/.local/share/glow-speak/voices"
            )
        else:
            args.voices_dir = os.path.join(maybe_data_dir, "glow-speak", "voices")

    os.makedirs(args.voices_dir, exist_ok=True)

    for voice in args.voice:
        url = args.url_format.format(voice=voice)
        _LOGGER.debug("Downloading %s to %s", url, args.voices_dir)
        voice_dir = download_voice(voice, args.voices_dir, url)
        print(voice, voice_dir)


# -----------------------------------------------------------------------------


def list_voices(voices_dir: typing.Optional[typing.Union[str, Path]] = None):
    """List available voices"""
    downloaded = "[downloaded]"
    empty = "(missing)   "

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
