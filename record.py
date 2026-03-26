# ~/meeting_tools/record.py
"""
마이크로 회의를 녹음하고 WAV 파일로 저장한다.
사용법: python record.py [출력파일경로] [--device 장치명 또는 id]
종료: Enter 키
"""
import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000  # Whisper 최적 샘플레이트
AUTO_SAVE_INTERVAL_SEC = 5
PREFERRED_INPUT_KEYWORDS = [
    "jabra",
    "speak",
    "speakerphone",
    "conference",
    "meeting",
    "mic",
    "마이크",
]


def _partial_output_path(output_path: Path) -> Path:
    """녹음 중 임시 저장 파일 경로를 만든다."""
    return output_path.with_name(f"{output_path.stem}_inprogress.wav")


def save_recording(chunks: list[np.ndarray], output_path: Path, final: bool = False) -> Path:
    """현재까지 녹음된 청크를 wav로 저장한다."""
    target_path = output_path if final else _partial_output_path(output_path)
    audio = np.concatenate(chunks, axis=0)
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(str(target_path), SAMPLE_RATE, audio_int16)

    if final:
        _partial_output_path(output_path).unlink(missing_ok=True)

    return target_path


def find_input_devices(devices: list[dict]) -> list[dict]:
    """입력 가능한 장치만 추린다."""
    return [device for device in devices if device.get("max_input_channels", 0) > 0]


def choose_input_device(devices: list[dict], requested_device: Optional[str] = None) -> dict:
    """요청 장치 또는 회의용 마이크 우선순위로 입력 장치를 고른다."""
    input_devices = find_input_devices(devices)
    if not input_devices:
        raise ValueError("사용 가능한 입력 장치를 찾을 수 없습니다.")

    if requested_device:
        if requested_device.isdigit():
            device_index = int(requested_device)
            if 0 <= device_index < len(input_devices):
                return input_devices[device_index]
            raise ValueError(f"입력 장치 번호가 유효하지 않습니다: {requested_device}")

        requested_lower = requested_device.lower()
        for device in input_devices:
            if requested_lower in device["name"].lower():
                return device
        raise ValueError(f"요청한 입력 장치를 찾을 수 없습니다: {requested_device}")

    def score(device: dict) -> int:
        name = device["name"].lower()
        for index, keyword in enumerate(PREFERRED_INPUT_KEYWORDS):
            if keyword in name:
                return len(PREFERRED_INPUT_KEYWORDS) - index
        return 0

    return max(input_devices, key=score) if max(score(d) for d in input_devices) > 0 else input_devices[0]


def list_input_devices() -> list[dict]:
    """현재 시스템의 입력 장치 목록을 반환한다."""
    devices = sd.query_devices()
    normalized = []
    for index, device in enumerate(devices):
        item = dict(device)
        item["id"] = index
        normalized.append(item)
    return find_input_devices(normalized)


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="회의 녹음 도구")
    parser.add_argument("output", nargs="?", help="출력 파일 경로")
    parser.add_argument("--device", help="입력 장치 id 또는 이름 일부")
    parser.add_argument("--list-devices", action="store_true", help="사용 가능한 입력 장치 목록 출력")
    return parser.parse_args(argv)


def record(output_path: Path, requested_device: Optional[str] = None) -> None:
    chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(f"경고: {status}")
        chunks.append(indata.copy())

    print(f"녹음 시작... (종료하려면 Enter 키를 누르세요)")
    print(f"저장 경로: {output_path}")

    devices = list_input_devices()
    selected = choose_input_device(devices, requested_device=requested_device)
    print(f"입력 장치: {selected['name']} (id: {selected['id']})")

    stop_event = threading.Event()
    previous_handlers = {}

    def wait_for_enter():
        try:
            with open("/dev/tty") as tty:
                tty.readline()
        except Exception:
            pass
        stop_event.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    def handle_signal(signum, frame):
        print(f"\n종료 신호 감지({signum}), 저장 준비 중...")
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    last_saved_chunk_count = 0
    last_auto_save_at = time.monotonic()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
            device=selected["id"],
        ):
            try:
                while not stop_event.wait(timeout=0.5):
                    if not chunks:
                        continue
                    if len(chunks) == last_saved_chunk_count:
                        continue
                    now = time.monotonic()
                    if now - last_auto_save_at < AUTO_SAVE_INTERVAL_SEC:
                        continue

                    partial_path = save_recording(chunks, output_path, final=False)
                    last_saved_chunk_count = len(chunks)
                    last_auto_save_at = now
                    print(f"임시 저장: {partial_path}")
            except KeyboardInterrupt:
                print("\nCtrl+C 감지, 저장 준비 중...")
                stop_event.set()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if not chunks:
        print("녹음된 데이터가 없습니다.")
        return

    print("녹음 종료 중...")
    final_path = save_recording(chunks, output_path, final=True)
    print(f"저장 완료: {final_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.list_devices:
        for device in list_input_devices():
            print(f"{device['id']}: {device['name']} (input={device['max_input_channels']})")
        sys.exit(0)

    if args.output:
        out = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out = Path.home() / "meetings" / f"audio_{ts}.wav"

    out.parent.mkdir(parents=True, exist_ok=True)
    record(out, requested_device=args.device)
