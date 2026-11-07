# Compatibility shim: the backend now lives in the installable package
# `delyrism.server` (delyrism/server/).  Kept so `uvicorn app.main:app
# --app-dir web/backend` (Dockerfile, README, older scripts) keeps working.
