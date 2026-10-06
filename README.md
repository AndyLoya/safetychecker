# PPE Safety Monitor

The monitor checks a browser camera feed for people wearing a helmet and face mask. Frames are sent to Roboflow for inference. When Roboflow-only mode is disabled, the app validates the supervisor phone number on startup and sends violation alerts using the configured RapidAPI endpoints.

## Deploy on Render

1. Push this project to a GitHub repository.
2. In Render, choose **New +** → **Blueprint**, then select the repository containing `render.yaml`.
3. Enter the requested environment values, including a unique `APP_USERNAME` and strong `APP_PASSWORD`. Render keeps values marked `sync: false` out of source control.
4. Use the generated `onrender.com` HTTPS URL. Open it in a supported browser and select **Start camera**; allow camera access when prompted.

Render starts the Flask app with Gunicorn and checks `/healthz`. The app expects one active check-in station: its counters and inference state live in the web process, so keep the Render service to one instance and one Gunicorn worker.

### Required environment values

- `ROBOFLOW_API_KEY`: API key for the Roboflow workspace.
- `ROBOFLOW_WORKSPACE`: Roboflow workspace slug.
- `PERSON_WORKFLOW_ID` and `PPE_WORKFLOW_ID`: IDs of the person and PPE workflows.
- With `ROBOFLOW_ONLY=false`, also set `RAPIDAPI_KEY`, `RAPIDAPI_PHONE_VALIDATION_URL`, `RAPIDAPI_PHONE_VALIDATION_HOST`, `RAPIDAPI_ALERT_URL`, `RAPIDAPI_ALERT_HOST`, and `SUPERVISOR_PHONE`.

To run only the Roboflow checks without phone validation and SMS alerts, set `ROBOFLOW_ONLY=true`; the RapidAPI and supervisor values are then not required. Set `CONFIDENCE_THRESHOLD`, `ALERT_COOLDOWN_SECONDS`, and `INFERENCE_WIDTH` only if you want to change their defaults.

The service processes camera frames sent from the user's browser; Render does not access a webcam attached to the user's computer. Browser camera access requires HTTPS, which Render provides for its hosted URL. The hosted service uses HTTP Basic authentication to limit access to the camera station and its inference/alert endpoints.

## Run locally

Create a virtual environment, install dependencies, copy `.env.example` to `.env`, and fill in the required values:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env -NoClobber
python -m flask --app web run
```

The copy command leaves an existing `.env` file untouched. Fill the required settings if creating a new one. Open the local URL in the browser. For camera access outside `localhost`, use HTTPS. If `APP_USERNAME` and `APP_PASSWORD` are left blank, the local server does not require authentication.
