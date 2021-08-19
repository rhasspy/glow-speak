#!/usr/bin/env python3
import logging
import os
import typing
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
        _LOGGER.debug("Looking for voice in %s", maybe_voice_dir)

        if os.path.isdir(maybe_voice_dir):
            return Path(maybe_voice_dir)

        # Try with resolved name
        if resolved_voice is not None:
            maybe_voice_dir = os.path.join(os.path.expandvars(dir_path), resolved_voice)
            _LOGGER.debug("Looking for voice in %s", maybe_voice_dir)

            if os.path.isdir(maybe_voice_dir):
                return Path(maybe_voice_dir)

    return None


# -----------------------------------------------------------------------------


def main():
    pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
