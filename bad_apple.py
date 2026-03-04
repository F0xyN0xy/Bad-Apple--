import cv2
import numpy as np
import argparse
import sys
import os
import time
import shutil
import subprocess
import threading

DEFAULT_CHARS = " .,:;i1tfLCG08@"


def get_terminal_size():
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines


def frame_to_ascii(gray_frame: np.ndarray, cols: int, rows: int, chars: str) -> str:
    """Resize frame to (cols, rows) and map each pixel to an ASCII character."""
    resized = cv2.resize(gray_frame, (cols, rows), interpolation=cv2.INTER_AREA)
    indices = (resized / 255 * (len(chars) - 1)).astype(int)
    lines = ["".join(chars[i] for i in row) for row in indices]
    return "\n".join(lines)


def play_audio(input_path: str, audio_proc_holder: list):
    """Launch ffplay in a background thread to play audio. Stores the process so we can kill it on exit."""
    proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-vn", input_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    audio_proc_holder.append(proc)
    proc.wait()


def play_video(input_path: str, width: int | None, chars: str, fps_override: float | None, no_audio: bool):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{input_path}'")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fps = fps_override or src_fps
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    term_w, term_h = get_terminal_size()

    cols = min(width or term_w, term_w)
    aspect = src_h / src_w
    rows = min(int(cols * aspect * 0.45), term_h - 1)

    frame_duration = 1.0 / fps

    print(f"Playing: {input_path}")
    print(f"Terminal: {term_w}x{term_h}  |  Render: {cols}x{rows}  |  FPS: {fps:.1f}")
    if no_audio:
        print("Audio: disabled")
    else:
        print("Audio: enabled (requires ffmpeg)")
    print("Press Ctrl+C to quit.\n")
    time.sleep(1.5)

    audio_proc_holder = []
    if not no_audio:
        audio_thread = threading.Thread(target=play_audio, args=(input_path, audio_proc_holder), daemon=True)
        audio_thread.start()

    sys.stdout.write("\033[?25l")  
    sys.stdout.write("\033[2J")    
    sys.stdout.flush()

    frame_idx = 0
    skipped = 0

    try:
        start_time = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            elapsed = time.perf_counter() - start_time
            expected_frame = int(elapsed * fps)

            if frame_idx < expected_frame - 1:
                skipped += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ascii_frame = frame_to_ascii(gray, cols, rows, chars)

            sys.stdout.write("\033[H")
            sys.stdout.write(ascii_frame)
            sys.stdout.flush()

            next_frame_time = start_time + frame_idx * frame_duration
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if audio_proc_holder:
            audio_proc_holder[0].terminate()
        sys.stdout.write("\033[?25h")
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print(f"Stopped after {frame_idx}/{total} frames ({skipped} skipped to stay in sync).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a video as ASCII art in your terminal."
    )
    parser.add_argument("input", help="Path to video file (e.g. bad_apple.mp4)")
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=None,
        help="Character columns to use (default: full terminal width)",
    )
    parser.add_argument(
        "--chars", "-c",
        type=str,
        default=DEFAULT_CHARS,
        help=f'ASCII ramp dark→light (default: "{DEFAULT_CHARS}")',
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override playback FPS (default: use video's native FPS)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert brightness mapping (good for dark-on-light videos)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found: '{args.input}'")
        sys.exit(1)

    chars = args.chars[::-1] if args.invert else args.chars

    play_video(
        input_path=args.input,
        width=args.width,
        chars=chars,
        fps_override=args.fps,
        no_audio=args.no_audio,
    )