#!/usr/bin/env python3
"""
audio_to_fir_coeff.py: Convert audio files (WAV, etc.) to FIR coefficient files using SoX + AWK.

Usage: python audio_to_fir_coeff.py [glob_pattern1 [glob_pattern2 ...]]
- Default: Processes all *.wav in current directory.
- Supports any audio format SoX can read (WAV, FLAC, MP3, etc.).
- Outputs: For each <base>.ext → <base>.fir with coefficients from `sox <file> -t dat - channels 1 | awk 'NR>2 {print $2}'`.
- Prints: Full path of each created .fir file to stdout (one per line).
- Requires: SoX and AWK in PATH.

Examples:
  python audio_to_fir_coeff.py *.wav
  python audio_to_fir_coeff.py "path/to/impulses/*.wav"
"""

import glob
import os
import sys
import subprocess
import shlex

def process_audio_to_fir(input_file):
    """
    Process a single audio file to FIR using: sox <input> -t dat - channels 1 | awk 'NR>2 {print $2}' > <fir>
    Returns the output FIR path on success, None on failure.
    """
    if not os.path.isfile(input_file):
        return None

    base_name = os.path.splitext(input_file)[0]
    fir_file = f"{base_name}.fir"

    # Exact command components
    sox_cmd = ["sox", input_file, "-t", "dat", "-", "channels", "1"]
    awk_cmd = ["awk", "NR>2 {print $2}"]

    try:
        # Pipe SoX → AWK → FIR file
        with open(fir_file, "w") as f:
            proc_sox = subprocess.Popen(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            proc_awk = subprocess.Popen(awk_cmd, stdin=proc_sox.stdout, stdout=f, stderr=subprocess.PIPE, text=True)
            proc_sox.stdout.close()  # Signal EOF to AWK

            _, sox_err = proc_sox.communicate()
            _, awk_err = proc_awk.communicate()

        # Check return codes
        if proc_sox.returncode != 0:
            sys.stderr.write(f"SoX error for {input_file}: {sox_err.strip()}\n")
            if os.path.exists(fir_file):
                os.remove(fir_file)
            return None
        if proc_awk.returncode != 0:
            sys.stderr.write(f"AWK error for {input_file}: {awk_err.strip()}\n")
            if os.path.exists(fir_file):
                os.remove(fir_file)
            return None

        # Verify non-empty FIR
        if os.path.exists(fir_file) and os.path.getsize(fir_file) > 0:
            return os.path.abspath(fir_file)
        else:
            sys.stderr.write(f"Empty FIR for {input_file}\n")
            if os.path.exists(fir_file):
                os.remove(fir_file)
            return None

    except Exception as e:
        sys.stderr.write(f"Error processing {input_file}: {str(e)}\n")
        if os.path.exists(fir_file):
            os.remove(fir_file)
        return None

def main():
    patterns = sys.argv[1:] if len(sys.argv) > 1 else ["*.wav"]
    created_files = []

    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            sys.stderr.write(f"No files matched: {pattern}\n")
            continue

        for audio_file in matches:
            fir_path = process_audio_to_fir(audio_file)
            if fir_path:
                created_files.append(fir_path)
                print(fir_path)  # Output FIR filename

    if not created_files:
        sys.stderr.write("No FIR files created. Check SoX/AWK installation and audio formats.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
