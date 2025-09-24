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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("hopejr_camtracker_hand")
@dataclass
class HopejrCamtrackHandConfig(TeleoperatorConfig):
    side: str  # Left or Right hand/arm to track and teleoperate [lowercase]

    # TODO (ryanpennings): return later
    def __post_init__(self):
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


# TODO (ryanpennings): adjust this???
@TeleoperatorConfig.register_subclass("hopejr_camtracker_arm")
@dataclass
class HopejrCamtrackArmConfig(TeleoperatorConfig):
    side: str  # Left or Right hand/arm to track and teleoperate [lowercase]

    def __post_init__(self):
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@TeleoperatorConfig.register_subclass("hopejr_camtracker")
@dataclass
class HopejrCamtrackConfig(TeleoperatorConfig):
    side: str  # Left or Right hand/arm to track and teleoperate [lowercase]
    camera_id: int = 0  # ID of camera to use for tracking
    display_camera: bool = True  # display camera on screen or not
    min_detection_confidence: float = 0.5  # TODO (ryanpennings): comment / do I need?
    min_tracking_confidence: float = 0.5  # TODO (rpryanpenningsennings): comment / do I need?
    handedness_threshold: float = 0.8  # TODO (ryanpennings): comment / do I need?
    model_complexity: int = 0  # TODO (ryanpennings): comment / do I need?

    def __post_init__(self):
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)
