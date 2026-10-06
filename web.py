import hmac
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from main import InferenceWorker, load_config, load_env_file


LOGGER = logging.getLogger("ppe_monitor.web")
MAX_FRAME_BYTES = 2 * 1024 * 1024


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    load_env_file()
    config = load_config()

    app_username = os.getenv("APP_USERNAME", "")
    app_password = os.getenv("APP_PASSWORD", "")
    if bool(app_username) != bool(app_password):
        raise ValueError("APP_USERNAME and APP_PASSWORD must either both be set or both be empty.")

    worker = InferenceWorker(config)
    worker.start()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES

    @app.before_request
    def require_authentication() -> Response | None:
        if not app_username or request.path == "/healthz":
            return None
        credentials = request.authorization
        if (
            credentials is not None
            and credentials.type == "basic"
            and credentials.username is not None
            and credentials.password is not None
            and hmac.compare_digest(
                credentials.username.encode("utf-8"), app_username.encode("utf-8")
            )
            and hmac.compare_digest(
                credentials.password.encode("utf-8"), app_password.encode("utf-8")
            )
        ):
            return None
        return Response(
            "Authentication required.",
            401,
            {"WWW-Authenticate": 'Basic realm="PPE Safety Monitor"'},
        )

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/healthz")
    def healthz() -> tuple[Any, int]:
        return jsonify({"status": "ok"}), 200

    @app.post("/api/frame")
    def submit_frame() -> tuple[Any, int]:
        try:
            image_bytes = request.get_data(cache=False)
            if not image_bytes:
                LOGGER.warning("Rejected empty frame request.")
                return (
                    jsonify({"error": "The request body must contain a JPEG frame."}),
                    400,
                )

            frame_data = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if frame is None:
                LOGGER.warning(
                    "Rejected frame request containing an invalid image (%d bytes).",
                    len(image_bytes),
                )
                return jsonify({"error": "The request body is not a valid image."}), 400

            worker.submit_frame(frame)
            return jsonify({"accepted": True}), 202
        except RequestEntityTooLarge:
            raise
        except Exception:
            LOGGER.exception("Failed to process incoming frame request.")
            return jsonify({"error": "The frame could not be processed."}), 500

    @app.get("/api/status")
    def get_status() -> tuple[Any, int]:
        (
            result,
            error,
            phase,
            passed_count,
            failed_count,
            last_failure_reason,
            missing_item_counts,
        ) = worker.get_status()

        people: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        inference_age = None
        if result is not None:
            detected_people, violations, result_time = result
            inference_age = max(0.0, time.monotonic() - result_time)
            people = [
                {
                    "label": person.label,
                    "confidence": person.confidence,
                    "x1": person.x1,
                    "y1": person.y1,
                    "x2": person.x2,
                    "y2": person.y2,
                }
                for person in detected_people
            ]

        if phase == "SCANNING":
            status = "SCANNING - PLEASE WAIT"
        elif phase == "NEXT":
            status = "DO NOT PASS" if violations else "GO AHEAD"
        else:
            status = "NEXT PERSON - READY"

        return jsonify(
            {
                "status": status,
                "phase": phase,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "last_failure_reason": last_failure_reason,
                "missing_item_counts": missing_item_counts,
                "inference_age": inference_age,
                "people": people,
                "violations": violations,
                "error": error,
            }
        ), 200

    @app.errorhandler(413)
    def request_too_large(_error: Any) -> tuple[Any, int]:
        return jsonify({"error": "The uploaded frame exceeds the 2 MB limit."}), 413

    return app


app = create_app()
