#!/usr/bin/env python

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

# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

import threading
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from lerobot.teleoperators.hopejr_camtracker.joints_translation import camtracker_to_hope_jr_hand

from .config_hopejr_camtracker import HopejrCamtrackConfig

# from scipy.spatial.transform import Rotation # ? not used?

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# TODO (ryanpennings): use config
# TODO: - add process_arm_img
# TODO: - move annotations into separate function that is run after
# how to run


class CameraTracker:
    """
    Camera Tracker as IPC Server
    """

    def __init__(
        self,
        config: HopejrCamtrackConfig,
        logger,
        joints,
        camera_id: int = 0,
        fps_limit: int = 30,
    ):
        self.camera_id = camera_id
        self.fps_limit = fps_limit
        self.frame_time = 1.0 / self.fps_limit  # ? needed?
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.logger = logger

        self.new_state_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True, name=f"{self} _tracking_loop")
        self.thread_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.cam_open = False
        self._hand_state: dict[str, float] | None = None
        self._arm_state: dict[str, float] | None = None

    def start(self):
        """Start the camera tracker in a separate process"""
        # TODO (ryanpennings): add check to not start if already started
        self.thread.start()
        # TODO (ryanpennings): add error check??

        # wait for the thread to ramp up & 1st state to be ready
        if not self.new_state_event.wait(timeout=100):
            raise TimeoutError(f"{self}: Timed out waiting for state after 2s.")

        self.logger.info(f"{self} connected.")

    def stop(self):
        """Stop the camera tracking thread"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.cam_open = False
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.logger.info(f"{self} disconnected.")
        pass

    def get_hand_data(self, timeout: float = 2) -> dict[str, Any] | None:
        """Get latest hand tracking data"""
        if not self.new_state_event.wait(timeout=timeout):
            raise TimeoutError(f"{self}: Timed out waiting for state after {timeout}s.")

        with self.state_lock:
            state = self._hand_state

        self.new_state_event.clear()

        if state is None:
            # raise RuntimeError(f"{self} Internal error: Event set but no state available.")
            return None

        # TODO (ryanpennings): add separate state locks for hand and arm
        return self._hand_state

    def angle_between_vectors(self, v1, v2):
        """Calculate the angle between 2 vectors with dot product"""
        # Normalise the vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Dot Product
        dot_product = np.dot(v1_norm, v2_norm)

        # Clamp
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate
        angle_rad = np.arccos(dot_product)

        return angle_rad

    def project_vector_onto_plane(self, vector, plane_normal):
        """
        Project a vector onto a plane defined by its normal vector.
        """
        # Normalize the plane normal
        normal_unit = plane_normal / np.linalg.norm(plane_normal)

        # Calculate the component of vector perpendicular to the plane
        perpendicular_component = np.dot(vector, normal_unit) * normal_unit

        # Subtract the perpendicular component to get the projection
        projected_vector = vector - perpendicular_component

        return projected_vector

    def calculate_mcp_abduction_in_palm_plane(self, finger1_vec, finger2_vec, palm_normal):
        """
        Calculate abduction angle between two fingers in the palm plane
        """
        # Project both finger vectors onto the palm plane
        finger1_projected = self.project_vector_onto_plane(finger1_vec, palm_normal)
        finger2_projected = self.project_vector_onto_plane(finger2_vec, palm_normal)

        # Calculate angle between projected vectors
        angle = self.angle_between_vectors(finger1_projected, finger2_projected)

        # For signed angle (to know direction of spread)
        cross_product = np.cross(finger1_projected, finger2_projected)
        # The sign of the dot product with palm normal tells us the direction
        sign = np.sign(np.dot(cross_product, palm_normal))

        return sign * angle

    def calculate_pure_mcp_flexion(self, wrist_to_mcp, mcp_to_pip, palm_normal):
        """
        Calculate MCP flexion in the finger's sagittal plane (bending plane)
        to avoid influence from abduction
        """
        # Create the finger's sagittal plane normal
        # This is perpendicular to both the base finger direction and palm normal
        finger_sagittal_normal = np.cross(wrist_to_mcp, palm_normal)
        finger_sagittal_normal = finger_sagittal_normal / np.linalg.norm(finger_sagittal_normal)

        # Project both vectors onto the sagittal plane
        base_projected = self.project_vector_onto_plane(wrist_to_mcp, finger_sagittal_normal)
        pip_projected = self.project_vector_onto_plane(mcp_to_pip, finger_sagittal_normal)

        # Calculate angle between projected vectors
        flexion_angle = self.angle_between_vectors(base_projected, pip_projected)

        return flexion_angle

    def process_hand_img(self, hand_proc, image):
        """
        Process camera image frame and return image and positions
        """
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hand_proc.process(image)

        # Draw Annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        res = None
        # print("before if statement")
        if results.multi_hand_landmarks:
            for index, handedness_classif in enumerate(results.multi_handedness):
                if (
                    handedness_classif.classification[0].label
                    == "Right"  # self.config.side  #  # TODO (ryanpennings): get from config
                    and handedness_classif.classification[0].score
                    > 0.8  # self.config.handedness_threshold # TODO (ryanpennings): get from config
                ):
                    # hand_landmarks = results.multi_hand_world_landmarks[index]  # metric
                    hand_landmarks_norm = results.multi_hand_landmarks[index]  # normalized

                    # **** Create Coordinate System
                    origin = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.WRIST].z,
                        ]
                    )
                    mid_mcp = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,
                        ]
                    )
                    unit_z = mid_mcp - origin
                    unit_z = unit_z / np.linalg.norm(unit_z)
                    pinky_mcp = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].z,
                        ]
                    )

                    # print(f"ORIGIN: {origin} MID: {mid_mcp}")
                    vec_towards_y = pinky_mcp - origin
                    unit_x = np.cross(vec_towards_y, unit_z)
                    unit_x = unit_x / np.linalg.norm(unit_x)
                    unit_y = np.cross(unit_z, unit_x)
                    palm_normal = unit_y

                    # * Index Finger
                    # vector from origin to index mcp
                    index_mcp = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,
                        ]
                    )
                    index_pip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z,
                        ]
                    )
                    index_tip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
                        ]
                    )

                    index_wrist_to_mcp = index_mcp - origin
                    index_mcp_to_pip = index_pip - index_mcp
                    index_pip_to_tip = index_tip - index_pip

                    # * Middle Finger

                    middle_mcp = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,
                        ]
                    )
                    middle_pip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z,
                        ]
                    )
                    middle_tip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,
                        ]
                    )

                    middle_wrist_to_mcp = middle_mcp - origin
                    middle_mcp_to_pip = middle_pip - middle_mcp
                    middle_pip_to_tip = middle_tip - middle_pip

                    # * Ring Finger

                    ring_mcp = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z,
                        ]
                    )
                    ring_pip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z,
                        ]
                    )
                    ring_tip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z,
                        ]
                    )

                    ring_wrist_to_mcp = ring_mcp - origin
                    ring_mcp_to_pip = ring_pip - ring_mcp
                    ring_pip_to_tip = ring_tip - ring_pip

                    # * Pinky Finger

                    pinky_mcp = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_MCP].z,
                        ]
                    )
                    pinky_pip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_PIP].z,
                        ]
                    )
                    pinky_tip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.PINKY_TIP].z,
                        ]
                    )

                    pinky_wrist_to_mcp = pinky_mcp - origin
                    pinky_mcp_to_pip = pinky_pip - pinky_mcp
                    pinky_pip_to_tip = pinky_tip - pinky_pip

                    # * Thumb

                    # Thumb positions (already have CMC, MCP, IP, TIP calculations)
                    thumb_cmc = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_CMC].z,
                        ]
                    )
                    thumb_mcp_pos = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_MCP].z,
                        ]
                    )
                    thumb_ip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_IP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_IP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_IP].z,
                        ]
                    )
                    thumb_tip = np.array(
                        [
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                            hand_landmarks_norm.landmark[mp_hands.HandLandmark.THUMB_TIP].z,
                        ]
                    )

                    # Thumb vectors
                    thumb_wrist_to_cmc = thumb_cmc - origin
                    thumb_cmc_to_mcp = thumb_mcp_pos - thumb_cmc
                    thumb_mcp_to_ip = thumb_ip - thumb_mcp_pos
                    thumb_ip_to_tip = thumb_tip - thumb_ip

                    # * draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks_norm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # * Calculate angles for all fingers
                    # INDEX FINGER
                    index_pipdip = self.angle_between_vectors(index_mcp_to_pip, index_pip_to_tip)
                    index_mcp_flexion = self.calculate_pure_mcp_flexion(
                        index_wrist_to_mcp, index_mcp_to_pip, palm_normal
                    )
                    index_mcp_abduction = self.calculate_mcp_abduction_in_palm_plane(
                        index_wrist_to_mcp, middle_wrist_to_mcp, palm_normal
                    )

                    # MIDDLE FINGER
                    middle_pipdip = self.angle_between_vectors(middle_mcp_to_pip, middle_pip_to_tip)
                    middle_mcp_flexion = self.calculate_pure_mcp_flexion(
                        middle_wrist_to_mcp, middle_mcp_to_pip, palm_normal
                    )
                    # Middle finger abduction relative to a neutral reference (could use average of index and ring)
                    middle_mcp_abduction = self.calculate_mcp_abduction_in_palm_plane(
                        middle_wrist_to_mcp, unit_z, palm_normal
                    )  # unit_z as reference

                    # RING FINGER
                    ring_pipdip = self.angle_between_vectors(ring_mcp_to_pip, ring_pip_to_tip)
                    ring_mcp_flexion = self.calculate_pure_mcp_flexion(
                        ring_wrist_to_mcp, ring_mcp_to_pip, palm_normal
                    )
                    ring_mcp_abduction = self.calculate_mcp_abduction_in_palm_plane(
                        ring_wrist_to_mcp, middle_wrist_to_mcp, palm_normal
                    )
                    # PINKY FINGER
                    pinky_pipdip = self.angle_between_vectors(pinky_mcp_to_pip, pinky_pip_to_tip)
                    pinky_mcp_flexion = self.calculate_pure_mcp_flexion(
                        pinky_wrist_to_mcp, pinky_mcp_to_pip, palm_normal
                    )
                    pinky_mcp_abduction = self.calculate_mcp_abduction_in_palm_plane(
                        pinky_wrist_to_mcp, ring_wrist_to_mcp, palm_normal
                    )

                    # THUMB (special case - different joint structure)
                    thumb_cmc_angle = self.calculate_mcp_abduction_in_palm_plane(
                        thumb_wrist_to_cmc, unit_z, palm_normal
                    )
                    thumb_mcp_angle = self.angle_between_vectors(thumb_wrist_to_cmc, thumb_cmc_to_mcp)
                    thumb_pip_angle = self.angle_between_vectors(thumb_cmc_to_mcp, thumb_mcp_to_ip)
                    thumb_dip_angle = self.angle_between_vectors(thumb_mcp_to_ip, thumb_ip_to_tip)

                    joint_positions = {
                        "thumb_cmc.pos": thumb_cmc_angle,
                        "thumb_mcp.pos": thumb_mcp_angle,
                        "thumb_pip.pos": thumb_pip_angle,
                        "thumb_dip.pos": thumb_dip_angle,
                        "index_mcp_abduction.pos": index_mcp_abduction,
                        "index_mcp_flexion.pos": index_mcp_flexion,
                        "index_dip.pos": index_pipdip,
                        "middle_mcp_abduction.pos": middle_mcp_abduction,
                        "middle_mcp_flexion.pos": middle_mcp_flexion,
                        "middle_dip.pos": middle_pipdip,
                        "ring_mcp_abduction.pos": ring_mcp_abduction,
                        "ring_mcp_flexion.pos": ring_mcp_flexion,
                        "ring_dip.pos": ring_pipdip,
                        "pinky_mcp_abduction.pos": pinky_mcp_abduction,
                        "pinky_mcp_flexion.pos": pinky_mcp_flexion,
                        "pinky_dip.pos": pinky_pipdip,
                    }

                    # * Do Joint Translations here
                    # res = camtracker_to_hope_jr_hand(
                    #     {f"{joint}.pos": pos for joint, pos in joint_positions.items()}
                    # )
                    res = camtracker_to_hope_jr_hand(joint_positions)
        # print("about to return")
        return image, res

    def _tracking_loop(self) -> None:
        """Tracking Loop"""

        # * start cap if not already started
        # ! add alt code if getting video feed from zeromq (using arm at same time)
        if not self.cam_open:
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                self.cam_open = True
            else:
                self.cam_open = False
                pass

            with mp_hands.Hands(
                model_complexity=self.config.model_complexity,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            ) as hands:
                # print("outside loop")
                while not self.stop_event.is_set():
                    ret, frame = self.cap.read()  # ! needs to move for arm code / replace with copy
                    # print("1")
                    if not ret:
                        print("no result")
                        break
                        # ? error?
                    # print("2")
                    frame = cv2.flip(frame, 1)
                    # print("3")
                    # process
                    frame, res = self.process_hand_img(hands, frame)
                    # print("4")
                    # print("do we make it?")
                    if res is not None:
                        with self.state_lock:
                            self._hand_state = res
                        self.new_state_event.set()
                    # TODO (ryanpennings): add arm tracking too - split

                    # with self.state_lock:
                    #     self._hand_state = res  # joint_positions
                    # self.new_state_event.set()
                    # time.sleep(1)
                    # cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
                    cv2.imshow("MediaPipe Hands", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        # break
                        print("cv2 exit")
                        # TODO (ryanpennings): call stop? or remove
