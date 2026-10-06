import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import requests


LOGGER = logging.getLogger("ppe_monitor")
ROBOFLOW_INFERENCE_URL = "https://serverless.roboflow.com"
REQUIRED_GEAR = ("helmet", "mask")
GEAR_ALIASES = {
    "helmet": {"hard_hat", "hardhat", "helmet"},
    "mask": {"mask", "face_mask"},
}


@dataclass(frozen=True)
class Config:
    roboflow_api_key: str
    roboflow_workspace: str
    person_workflow_id: str
    ppe_workflow_id: str
    camera_index: int
    inference_width: int
    confidence_threshold: float


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overriding existing environment values."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def load_config() -> Config:
    required = (
        "ROBOFLOW_API_KEY",
        "ROBOFLOW_WORKSPACE",
        "PERSON_WORKFLOW_ID",
        "PPE_WORKFLOW_ID",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(
            "Missing configuration values: "
            + ", ".join(missing)
            + ". Fill in the .env file before starting."
        )

    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
    inference_width = int(os.getenv("INFERENCE_WIDTH", "640"))
    if not 0 <= threshold <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1.")
    if inference_width < 1:
        raise ValueError("INFERENCE_WIDTH must be a positive integer.")

    return Config(
        roboflow_api_key=os.environ["ROBOFLOW_API_KEY"],
        roboflow_workspace=os.environ["ROBOFLOW_WORKSPACE"],
        person_workflow_id=os.environ["PERSON_WORKFLOW_ID"],
        ppe_workflow_id=os.environ["PPE_WORKFLOW_ID"],
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
        inference_width=inference_width,
        confidence_threshold=threshold,
    )


def parse_workflow_detections(
    result: Any, confidence_threshold: float
) -> list[Detection]:
    detections: list[Detection] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            label_value = node.get("class", node.get("class_name", node.get("label")))
            has_center_box = all(
                key in node for key in ("x", "y", "width", "height")
            )
            has_corner_box = all(
                key in node for key in ("x1", "y1", "x2", "y2")
            )
            if label_value is not None and (has_center_box or has_corner_box):
                confidence = float(node.get("confidence", node.get("probability", 0)))
                if confidence >= confidence_threshold:
                    if has_center_box:
                        x, y = float(node["x"]), float(node["y"])
                        half_width = float(node["width"]) / 2
                        half_height = float(node["height"]) / 2
                        x1, y1 = x - half_width, y - half_height
                        x2, y2 = x + half_width, y + half_height
                    else:
                        x1, y1, x2, y2 = (
                            float(node["x1"]),
                            float(node["y1"]),
                            float(node["x2"]),
                            float(node["y2"]),
                        )
                    detections.append(
                        Detection(
                            str(label_value).strip().lower(),
                            confidence,
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                    )
                return

            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(result)
    return detections


def roboflow_workflow_infer(
    frame: Any,
    workflow_id: str,
    classes: str,
    config: Config,
) -> list[Detection]:
    original_height, original_width = frame.shape[:2]
    scale = min(1.0, config.inference_width / original_width)
    if scale < 1.0:
        inference_frame = cv2.resize(
            frame,
            (config.inference_width, max(1, round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        inference_frame = frame

    encoded_ok, image_buffer = cv2.imencode(
        ".jpg", inference_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
    )
    if not encoded_ok:
        raise RuntimeError("Could not encode webcam frame for Roboflow workflow.")

    endpoint = (
        f"{ROBOFLOW_INFERENCE_URL}/"
        f"{quote(config.roboflow_workspace, safe='')}/workflows/"
        f"{quote(workflow_id, safe='')}"
    )
    payload = {
        "use_cache": True,
        "inputs": {
            "image": {
                "type": "InferenceImage",
                "value": base64.b64encode(image_buffer.tobytes()).decode("ascii"),
            },
            "classes": classes,
        },
    }
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": config.roboflow_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(
            f"Roboflow workflow {workflow_id} returned invalid JSON."
        ) from exc
    except (requests.RequestException, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Roboflow workflow {workflow_id} failed: {exc}") from exc
    detections = parse_workflow_detections(result, config.confidence_threshold)
    if scale < 1.0:
        detections = [
            Detection(
                detection.label,
                detection.confidence,
                detection.x1 / scale,
                detection.y1 / scale,
                detection.x2 / scale,
                detection.y2 / scale,
            )
            for detection in detections
        ]
    return detections


def normalized_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def select_center_person(
    people: list[Detection], frame_width: int, frame_height: int
) -> Detection | None:
    """Choose one person in the central check-in area, prioritizing image center."""
    center_x = frame_width / 2
    center_y = frame_height / 2
    zone_left = frame_width * 0.25
    zone_right = frame_width * 0.75
    zone_top = frame_height * 0.08
    zone_bottom = frame_height * 0.95
    candidates = [
        person
        for person in people
        if zone_left <= person.center[0] <= zone_right
        and zone_top <= person.center[1] <= zone_bottom
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda person: (
            (person.center[0] - center_x) / frame_width
        ) ** 2
        + ((person.center[1] - center_y) / frame_height) ** 2,
    )


def person_violations(
    person_detections: list[Detection], ppe_detections: list[Detection]
) -> list[dict[str, Any]]:
    people = [
        detection
        for detection in person_detections
        if normalized_label(detection.label) == "person"
    ]
    violations = []
    for person_index, person in enumerate(people, start=1):
        contained_gear = {
            normalized_label(detection.label)
            for detection in ppe_detections
            if person.x1 <= detection.center[0] <= person.x2
            and person.y1 <= detection.center[1] <= person.y2
        }
        missing = [
            gear
            for gear in REQUIRED_GEAR
            if not contained_gear.intersection(GEAR_ALIASES[gear])
        ]
        if missing:
            violations.append(
                {
                    "person_index": person_index,
                    "person_bbox": {
                        "x1": round(person.x1, 1),
                        "y1": round(person.y1, 1),
                        "x2": round(person.x2, 1),
                        "y2": round(person.y2, 1),
                    },
                    "person_confidence": round(person.confidence, 4),
                    "missing_items": missing,
                    "gear_confidences": {
                        gear: [
                            round(detection.confidence, 4)
                            for detection in ppe_detections
                            if normalized_label(detection.label)
                            in GEAR_ALIASES[gear]
                            and person.x1
                            <= detection.center[0]
                            <= person.x2
                            and person.y1
                            <= detection.center[1]
                            <= person.y2
                        ]
                        for gear in REQUIRED_GEAR
                    },
                }
            )
    return violations


def draw_scan_zone(frame: Any) -> None:
    height, width = frame.shape[:2]
    left = int(width * 0.25)
    right = int(width * 0.75)
    top = int(height * 0.08)
    bottom = int(height * 0.95)
    guide_color = (255, 210, 80)
    cv2.rectangle(frame, (left, top), (right, bottom), guide_color, 1)
    cv2.putText(
        frame,
        "ONE PERSON - STAND/SIT IN CENTER",
        (left + 8, min(bottom - 8, top + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        guide_color,
        1,
        cv2.LINE_AA,
    )


def draw_results(
    frame: Any, people: list[Detection], violations: list[dict[str, Any]]
) -> None:
    violation_by_box = {
        (
            int(item["person_bbox"]["x1"]),
            int(item["person_bbox"]["y1"]),
            int(item["person_bbox"]["x2"]),
            int(item["person_bbox"]["y2"]),
        ): item
        for item in violations
    }
    for person in people:
        box = (int(person.x1), int(person.y1), int(person.x2), int(person.y2))
        violation = next(
            (
                item
                for violating, item in violation_by_box.items()
                if all(abs(box[index] - violating[index]) <= 1 for index in range(4))
            ),
            None,
        )
        missing_items = violation["missing_items"] if violation else []
        color = (
            (0, 0, 255)
            if len(missing_items) == len(REQUIRED_GEAR)
            else (0, 255, 255)
            if missing_items
            else (0, 200, 0)
        )
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = (
            f"MISSING: {', '.join(missing_items)}"
            if missing_items
            else "HELMET + MASK OK"
        )
        cv2.putText(
            frame,
            f"{label} ({person.confidence:.2f})",
            (box[0], max(20, box[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_checkin_display(
    frame: Any,
    status: str,
    passed_count: int,
    failed_count: int,
    most_missed_item: str,
    most_missed_count: int,
    inference_age: float | None,
    next_message: str,
    last_failure_reason: str,
) -> None:
    color = (
        (0, 0, 255)
        if status == "DO NOT PASS"
        else (0, 200, 0)
        if status == "GO AHEAD" or status == "NEXT PERSON - READY"
        else (0, 255, 255)
        if status == "SCANNING - PLEASE WAIT"
        else (80, 80, 80)
    )
    panel_width = min(680, frame.shape[1] - 20)
    panel_height = 220
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, panel_height),
        (24, 28, 34),
        -1,
    )
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (10, 10), (16, panel_height), color, -1)
    cv2.putText(
        frame,
        "PPE CHECK-IN",
        (30, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 195, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        color,
        2,
        cv2.LINE_AA,
    )
    counter_y = 115
    cv2.rectangle(frame, (30, 94), (panel_width - 10, 130), (39, 45, 52), -1)
    cv2.putText(
        frame,
        f"PASSED  {passed_count}",
        (44, counter_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (100, 230, 145),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"NOT PASSED  {failed_count}",
        (220, counter_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (110, 135, 255),
        1,
        cv2.LINE_AA,
    )
    missed_text = (
        "MOST MISSED  None yet"
        if most_missed_count == 0
        else f"MOST MISSED  {most_missed_item} ({most_missed_count})"
    )
    cv2.putText(
        frame,
        missed_text,
        (30, 151),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (110, 205, 255),
        1,
        cv2.LINE_AA,
    )
    reason_line = f"Last failed check: {last_failure_reason}"
    if len(reason_line) > 76:
        reason_line = reason_line[:73] + "..."
    cv2.putText(
        frame,
        reason_line,
        (30, 179),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (235, 205, 155),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        next_message,
        (30, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 200, 210),
        1,
        cv2.LINE_AA,
    )
    if inference_age is not None:
        cv2.putText(
            frame,
            f"{inference_age:.1f}s",
            (panel_width - 52, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (155, 170, 185),
            1,
            cv2.LINE_AA,
        )


class InferenceWorker:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._latest_frame: Any = None
        self._latest_result: tuple[
            list[Detection], list[dict[str, Any]], float
        ] | None = None
        self._error: str | None = None
        self._phase = "READY"
        self.passed_count = 0
        self.failed_count = 0
        self.last_failure_reason = "None yet"
        self.missing_item_counts = {item: 0 for item in REQUIRED_GEAR}
        self._thread = threading.Thread(
            target=self._run,
            name="roboflow-inference",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def submit_frame(self, frame: Any) -> None:
        with self._lock:
            self._latest_frame = frame.copy()
            self._new_frame.set()

    def get_status(
        self,
    ) -> tuple[
        tuple[list[Detection], list[dict[str, Any]], float] | None,
        str | None,
        str,
        int,
        int,
        str,
        dict[str, int],
    ]:
        with self._lock:
            return (
                self._latest_result,
                self._error,
                self._phase,
                self.passed_count,
                self.failed_count,
                self.last_failure_reason,
                dict(self.missing_item_counts),
            )

    def stop(self) -> None:
        self._stop.set()
        self._new_frame.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        next_presence_scan_at = 0.0
        consecutive_clear_scans = 0
        while not self._stop.is_set():
            self._new_frame.wait()
            if self._stop.is_set():
                break
            with self._lock:
                self._new_frame.clear()
                frame = self._latest_frame
                if frame is not None:
                    frame = frame.copy()
            if frame is None:
                continue

            wait_seconds = next_presence_scan_at - time.monotonic()
            if wait_seconds > 0 and self._stop.wait(wait_seconds):
                break
            next_presence_scan_at = time.monotonic() + 0.35

            try:
                person_detections = roboflow_workflow_infer(
                    frame,
                    self.config.person_workflow_id,
                    "person",
                    self.config,
                )
                people = [
                    detection
                    for detection in person_detections
                    if normalized_label(detection.label) == "person"
                ]
                selected_person = select_center_person(
                    people, frame.shape[1], frame.shape[0]
                )
                selected_people = [selected_person] if selected_person else []

                with self._lock:
                    phase = self._phase

                if phase == "NEXT":
                    if selected_person is not None:
                        consecutive_clear_scans = 0
                        continue
                    consecutive_clear_scans += 1
                    if consecutive_clear_scans >= 2:
                        consecutive_clear_scans = 0
                        with self._lock:
                            self._phase = "READY"
                            self._latest_result = None
                            self._error = None
                    continue

                if selected_person is None:
                    continue

                with self._lock:
                    self._phase = "SCANNING"
                ppe_detections = roboflow_workflow_infer(
                    frame,
                    self.config.ppe_workflow_id,
                    "helmet, mask",
                    self.config,
                )
                violations = person_violations(selected_people, ppe_detections)
                with self._lock:
                    self._latest_result = (
                        selected_people,
                        violations,
                        time.monotonic(),
                    )
                    failed_indexes = {
                        int(item["person_index"]) - 1 for item in violations
                    }
                    self.failed_count += len(failed_indexes)
                    self.passed_count += len(selected_people) - len(failed_indexes)
                    if violations:
                        for item in violations:
                            for missing_item in item["missing_items"]:
                                if missing_item in self.missing_item_counts:
                                    self.missing_item_counts[missing_item] += 1
                        self.last_failure_reason = "; ".join(
                            f"Missing {', '.join(item['missing_items'])}"
                            for item in violations
                        )
                    self._phase = "NEXT"
                    self._error = None
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                LOGGER.error("Background inference failed: %s", exc)
                with self._lock:
                    self._error = str(exc)
                    if self._phase == "SCANNING":
                        self._phase = "READY"


def run_monitor(config: Config) -> None:
    camera = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open webcam at CAMERA_INDEX={config.camera_index}."
        )

    worker = InferenceWorker(config)
    worker.start()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            worker.submit_frame(frame)
            draw_scan_zone(frame)
            (
                result,
                error,
                phase,
                passed_count,
                failed_count,
                last_failure_reason,
                missing_item_counts,
            ) = worker.get_status()
            highest_missed_count = max(missing_item_counts.values(), default=0)
            most_missed_items = [
                item
                for item, count in missing_item_counts.items()
                if count == highest_missed_count and count > 0
            ]
            most_missed_item = (
                " / ".join(most_missed_items) if most_missed_items else "None yet"
            )
            if result is not None:
                people, violations, result_time = result
                draw_results(frame, people, violations)
                primary_status = (
                    "DO NOT PASS" if violations else "GO AHEAD"
                ) if phase == "NEXT" else "NEXT PERSON - READY"
                inference_age = time.monotonic() - result_time
            elif phase == "SCANNING":
                primary_status = "SCANNING - PLEASE WAIT"
                inference_age = None
            else:
                primary_status = "NEXT PERSON - READY"
                inference_age = None
            next_message = (
                "NEXT - waiting for person to leave"
                if phase == "NEXT"
                else "Step into view for one-time PPE check"
                if phase == "READY"
                else "Checking helmet and mask..."
            )
            draw_checkin_display(
                frame,
                primary_status,
                passed_count,
                failed_count,
                most_missed_item,
                highest_missed_count,
                inference_age,
                next_message,
                last_failure_reason,
            )
            if error is not None:
                cv2.putText(
                    frame,
                    "INFERENCE ERROR - see terminal",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("PPE Safety Monitor (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    load_env_file()
    config = load_config()
    run_monitor(config)


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        LOGGER.error("PPE monitor stopped: %s", exc)
        raise SystemExit(1) from exc
