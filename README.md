# Glow-Speak

glow-speak is a fast, local, neural text to speech system that uses [eSpeak-ng](https://github.com/espeak-ng/espeak-ng) as a text/phoneme front-end.

## Installation

```sh
git clone https://github.com/rhasspy/glow-speak.git
cd glow-speak/

python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools wheel
pip3 install -f 'https://synesthesiam.github.io/prebuilt-apps/' -r requirements.txt

python3 setup.py develop
glow-speak --version
```

## Voices

The following languages/voices are supported:

* German
    * de\_thorsten
* Chinese
    * cmn\_jing\_li
* Greek
    * el\_rapunzelina
* English
    * en-us\_ljspeech
    * en-us\_mary\_ann
* Spanish
    * es\_tux
* Finnish
    * fi\_harri\_tapani\_ylilammi
* French
    * fr\_siwis
* Hungarian
    * hu\_diana\_majlinger
* Italian
    * it\_riccardo\_fasol
* Korean
    * ko\_kss
* Dutch
    * nl\_rdh
* Russian
    * ru\_nikolaev
* Swedish
    * sv\_talesyntese
* Swahili
    * sw\_biblia\_takatifu
* Vietnamese
    * vi\_vais1000

## Usage

### Download Voices

``` sh
glow-speak-download de_thorsten
```

### Command-Line Synthesis

``` sh
glow-speak -v en-us_mary_ann 'This is a test.' --output-file test.wav
```

### HTTP Server

``` sh
glow-speak-http-server --debug
```

Visit http://localhost:5002

## Socket Server

Start the server:

``` sh
glow-speak-socket-server --voice en-us_mary_ann --socket /tmp/glow-speak.sock
```

From a separate terminal:

``` sh
echo 'This is a test.' | bin/glow-speak-socket-client --socket /tmp/glow-speak.sock | xargs aplay
```

Lines from client to server are synthesized, and the path to the WAV file is returned (usually in `/tmp`). 
