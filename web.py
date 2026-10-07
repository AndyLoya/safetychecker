import atexit
import hmac
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from main import InferenceWorker, load_config, load_env_file


MAX_FRAME_BYTES = 2 * 1024 * 1024


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    load_env_file(Path(__file__).resolve().parent / ".env")
    config = load_config()

    app_username = os.getenv("APP_USERNAME", "")
    app_password = os.getenv("APP_PASSWORD", "")
    require_auth_value = os.getenv("REQUIRE_AUTH", "false").strip().lower()
    if require_auth_value not in {"true", "false"}:
        raise ValueError("REQUIRE_AUTH must be either true or false.")
    require_auth = require_auth_value == "true"
    if bool(app_username) != bool(app_password):
        raise ValueError(
            "APP_USERNAME and APP_PASSWORD must either both be set or both be empty."
        )
    if require_auth and not app_username:
        raise ValueError(
            "APP_USERNAME and APP_PASSWORD are required when REQUIRE_AUTH is true."
        )

    worker = InferenceWorker(config)
    worker.start()
    atexit.register(worker.stop)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES

    @app.before_request
    def authenticate() -> Response | None:
        if not require_auth or request.path == "/healthz":
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
        image_bytes = request.get_data(cache=False)
        if not image_bytes:
            return (
                jsonify({"error": "The request body must contain a JPEG frame."}),
                400,
            )

        frame_data = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "The request body is not a valid image."}), 400

        worker.submit_frame(frame)
        return jsonify({"accepted": True}), 202

    @app.post("/api/reset")
    def reset_checkin() -> tuple[Any, int]:
        worker.reset_checkin()
        return jsonify({"reset": True}), 200

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

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error: RequestEntityTooLarge) -> tuple[Any, int]:
        return jsonify({"error": "The uploaded frame exceeds the 2 MB limit."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
