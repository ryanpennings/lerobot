#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Ryan Pennings. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on Pollen Robotics Hand Tracking Demo
# https://github.com/google-ai-edge/mediapipe/issues/5612 - see for issues with numpy and media pipe

import logging
from collections import deque
from typing import Any, Deque

from lerobot.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)  # ? are these needed for camera?
from lerobot.motors.motors_bus import MotorNormMode  # ? check if needed

from ..teleoperator import Teleoperator
from .camera_tracker import CameraTracker
from .config_hopejr_camtracker import HopejrCamtrackConfig, HopejrCamtrackHandConfig

logger = logging.getLogger(__name__)

# TODO (ryanpennings): pass left/right hand properlly

# TODO (ryanpennings): not sure if inversions are correct / needed
LEFT_HAND_INVERSIONS = [
    "thumb_cmc",
    "index_dip",
    "middle_mcp_abduction",
    "middle_dip",
    "pinky_mcp_abduction",
    "pinky_dip",
]

RIGHT_HAND_INVERSIONS = [
    "thumb_mcp",
    "thumb_cmc",
    "thumb_pip",
    "thumb_dip",
    "index_mcp_abduction",
    # "index_dip",
    "middle_mcp_abduction",
    # "middle_dip",
    "ring_mcp_abduction",
    "ring_mcp_flexion",
    # "ring_dip",
    "pinky_mcp_abduction",
]


class HopejrCamtrackHand(Teleoperator):
    """
    Hand Tracking via MediaPipe based on Pollen Robotics Amazing Hand Handtrack Demo
    """

    config_class = HopejrCamtrackHandConfig
    name = "hopejr_camtracker_hand"

    def __init__(self, config: HopejrCamtrackHandConfig):
        super().__init__(config)

        # TODO (ryanpennings): init camera_trakcer.cam_track here?

        self.joints = {
            "thumb_cmc": MotorNormMode.RANGE_0_100,
            "thumb_mcp": MotorNormMode.RANGE_0_100,
            "thumb_pip": MotorNormMode.RANGE_0_100,
            "thumb_dip": MotorNormMode.RANGE_0_100,
            "index_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "index_mcp_flexion": MotorNormMode.RANGE_0_100,
            "index_dip": MotorNormMode.RANGE_0_100,
            "middle_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "middle_mcp_flexion": MotorNormMode.RANGE_0_100,
            "middle_dip": MotorNormMode.RANGE_0_100,
            "ring_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "ring_mcp_flexion": MotorNormMode.RANGE_0_100,
            "ring_dip": MotorNormMode.RANGE_0_100,
            "pinky_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "pinky_mcp_flexion": MotorNormMode.RANGE_0_100,
            "pinky_dip": MotorNormMode.RANGE_0_100,
        }
        # self.inverted_joints = RIGHT_HAND_INVERSIONS if config.side == "right" else LEFT_HAND_INVERSIONS

        n = 10
        # EMA Parameters
        self.n: int = n
        self.alpha: float = 2 / (n + 1)
        # one deque *per joint* so we can inspect raw history if needed
        self._buffers: dict[str, Deque[int]] = {joint: deque(maxlen=n) for joint in self.joints}
        # running EMA value per joint – lazily initialised on first read
        self._ema: dict[str, float | None] = dict.fromkeys(self._buffers)

        self.camera_tracker = CameraTracker(HopejrCamtrackConfig, logger, self.joints)
        self._hand_state: dict[str, float] | None = None

        # self._state: dict[str, float] | None = None
        # self.new_state_event = threading.Event()
        # self.stop_event = threading.Event()
        # self.thread = threading.Thread(target=self._read_loop, daemon=True, name=f"{self} _read_loop")
        # self.state_lock = threading.Lock()

    @property
    def action_features(self) -> dict:
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return False  # TODO (ryanpennings): add this logic

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.camera_tracker.start()

        # self.thread.start()

        # wait for the thread to ramp up & 1st state to be ready
        # if not self.new_state_event.wait(timeout=2):
        # raise TimeoutError(f"{self}: Timed out waiting for state after 2s.")

        logger.info(f"{self} connected.")
        # TODO (ryanpennings): whatever media pipe connection is

    # TODO (ryanpennings) - do we need calibration? pass if not
    @property
    def is_calibrated(self) -> bool:
        # return self.calibration_fpath.is_file()
        pass

    def calibrate(self) -> None:
        # TODO (ryanpennings): add calibration if needed
        pass

    def configure(self) -> None:
        pass

    def _normalise(self, values: dict[str, float]) -> dict[str, float]:
        """Normalise angle values (radians) to 0-1024 range"""
        # TODO (ryanpennings): update this to use calibrated range?
        normalised = {}
        for joint, value in values.items():
            # Clamp value to reasonable angle range (0 to π radians)
            # Adjust these ranges based on your actual angle ranges
            clamped_value = max(0.0, min(value, 3.14159))  # 0 to π radians

            # Normalize to 0-1024 range
            normalised_value = (clamped_value / 3.14159) * 1024
            normalised[joint] = normalised_value

        return normalised

    def _apply_ema(self, raw: dict[str, int]) -> dict[str, int]:
        """Update buffers & running EMA values; return smoothed dict as integers."""
        smoothed: dict[str, int] = {}
        for joint, value in raw.items():
            # maintain raw history
            self._buffers[joint].append(value)

            # initialise on first run
            if self._ema[joint] is None:
                self._ema[joint] = float(value)
            else:
                self._ema[joint] = self.alpha * value + (1 - self.alpha) * self._ema[joint]

            # Convert back to int for compatibility with normalization
            smoothed[joint] = int(round(self._ema[joint]))
        return smoothed

    # * gets the joint positions (in glove runs via joint_translations)
    def get_action(self) -> dict[str, Any]:
        _hand_positions = self.camera_tracker.get_hand_data()

        # TODO (ryanpennings): figure out better logic for no tracking - atm send middle positions
        if _hand_positions is None:
            # * for the events where we either start without tracking or lose tracking
            """
            if we start without - set to max values (will require calibration then)
            if we lose tracking - set motors to same value, maybe set
            all of the finger to same value if we lose tracking on any finger joint
            or (might not even be able to do that) in that case, set all to wait
            add a timeout so that hand opens after a certain amount of time
            (for arm would probably have to define a safe position)
            """

            # joint_positions = {joint: 0 for joint, _ in zip(self.joints, self.joints, strict=True)}
            # ! TEMP
            joint_positions = {
                "index_mcp.pos": 511,
                "index_dip.pos": 511,
                "middle_mcp.pos": 511,
                "middle_dip.pos": 511,
                "ring_mcp.pos": 511,
                "ring_dip.pos": 511,
                "pinky_mcp.pos": 511,
                "pinky_dip.pos": 511,
                "thumb_cmc.pos": 511,
                "thumb_mcp.pos": 511,
                "thumb_pip.pos": 511,
                "thumb_dip.pos": 511,
            }
        else:
            joint_positions = self._normalise(_hand_positions)
            # joint: int(pos) for joint, pos in zip(self.joints, _hand_positions, strict=True)
        # ? do EMA and NORM
        # ? do joint translations
        # pprint(joint_positions)
        return joint_positions

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.camera_tracker.stop()
        # self.stop_event.set()
        # self.thread.join(timeout=1)
        logger.info(f"{self} disconnected.")
