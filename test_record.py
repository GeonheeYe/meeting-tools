import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
