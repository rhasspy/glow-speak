#!/usr/bin/env python3
"""
Communicates with glow-tts server over a Unix domain socket.

Reads lines of text from stdin.
Prints WAV paths to stdout.
"""
import argparse
import os
import socket
import sys


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True, help="Path to Unix domain socket")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket)

    if os.isatty(sys.stdin.fileno()):
        print("Reading text from stdin...", file=sys.stderr)

    with sock.makefile(mode="rw") as conn_file:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            print(line, file=conn_file, flush=True)

            wav_path = conn_file.readline().strip()
            print(wav_path, flush=True)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
