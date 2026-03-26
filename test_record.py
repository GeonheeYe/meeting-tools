import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import wavfile

import record


class RecordDeviceSelectionTest(unittest.TestCase):
    def test_find_input_devices_filters_output_only_devices(self):
        devices = [
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Mic A", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Mic B", "max_input_channels": 2, "max_output_channels": 2},
        ]

        result = record.find_input_devices(devices)

        self.assertEqual([item["name"] for item in result], ["Mic A", "Mic B"])

    def test_choose_input_device_prefers_meeting_microphone_keywords(self):
        devices = [
            {"name": "MacBook Pro 마이크", "max_input_channels": 1},
            {"name": "Jabra Speak 710", "max_input_channels": 1},
            {"name": "AirPods Pro", "max_input_channels": 1},
        ]

        result = record.choose_input_device(devices)

        self.assertEqual(result["name"], "Jabra Speak 710")

    def test_choose_input_device_uses_requested_device_id(self):
        devices = [
            {"name": "MacBook Pro 마이크", "max_input_channels": 1},
            {"name": "Jabra Speak 710", "max_input_channels": 1},
        ]

        result = record.choose_input_device(devices, requested_device="1")

        self.assertEqual(result["name"], "Jabra Speak 710")

    def test_choose_input_device_matches_requested_name(self):
        devices = [
            {"name": "MacBook Pro 마이크", "max_input_channels": 1},
            {"name": "Jabra Speak 710", "max_input_channels": 1},
        ]

        result = record.choose_input_device(devices, requested_device="jabra")

        self.assertEqual(result["name"], "Jabra Speak 710")

    def test_choose_input_device_falls_back_to_first_input(self):
        devices = [
            {"name": "MacBook Pro 마이크", "max_input_channels": 1},
            {"name": "AirPods Pro", "max_input_channels": 1},
        ]

        result = record.choose_input_device(devices)

        self.assertEqual(result["name"], "MacBook Pro 마이크")


class RecordArgumentParsingTest(unittest.TestCase):
    def test_parse_args_supports_list_devices_and_device(self):
        args = record.parse_args(["--list-devices", "--device", "Jabra Speak 710"])

        self.assertTrue(args.list_devices)
        self.assertEqual(args.device, "Jabra Speak 710")


class RecordPersistenceTest(unittest.TestCase):
    def test_save_recording_writes_partial_wav_with_inprogress_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.wav"
            chunks = [np.array([[0.25], [-0.25]], dtype=np.float32)]

            saved_path = record.save_recording(chunks, output_path, final=False)

            self.assertEqual(saved_path, output_path.with_name("meeting_inprogress.wav"))
            self.assertTrue(saved_path.exists())
            sample_rate, audio = wavfile.read(saved_path)
            self.assertEqual(sample_rate, record.SAMPLE_RATE)
            self.assertEqual(audio.tolist(), [8191, -8191])

    def test_save_recording_finalizes_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "meeting.wav"
            partial_path = output_path.with_name("meeting_inprogress.wav")
            chunks = [np.array([[0.1], [0.0]], dtype=np.float32)]

            partial_path.write_bytes(b"partial")

            saved_path = record.save_recording(chunks, output_path, final=True)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(saved_path.exists())
            self.assertFalse(partial_path.exists())


if __name__ == "__main__":
    unittest.main()
