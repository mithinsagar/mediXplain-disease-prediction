# MediXplain — HTTP API

> Author: Mithin Sagar S · GitHub: https://github.com/mithinsagar

MediXplain exposes two blueprints:

- `web` — the browser-facing UI at `/` and `/about`.
- `api` — a small JSON API prefixed with `/api`.

## GET /api/health

Returns a static heartbeat, useful for uptime probes.

```json
{
  "status": "ok",
  "author": "Mithin Sagar S",
  "github": "https://github.com/mithinsagar",
  "project": "MediXplain"
}
```

## POST /api/predict

Accepts a JSON body with a symptom list and returns the same result
structure the Flask UI uses internally.

### Request

```json
{
  "symptoms": ["fever", "cough", "sore throat"],
  "top_k": 3
}
```

- `symptoms` (required): array of strings, minimum length 3.
- `top_k` (optional, default 3): number of differential predictions to
  return.

### Response

```json
{
  "prediction": "influenza",
  "confidence": 82.35,
  "retrieval_score": 0.6421,
  "top_predictions": [
    {"disease": "influenza", "confidence": 82.35},
    {"disease": "common cold", "confidence": 9.10},
    {"disease": "pharyngitis", "confidence": 4.02}
  ],
  "evidence": [
    {
      "disease": "influenza",
      "paragraph": "Influenza is commonly associated with fever, cough...",
      "score": 0.6421
    }
  ],
  "explanation": "The model predicts influenza based on the symptom pattern...",
  "best_model_name": "logistic_regression"
}
```

### Errors

- `400 Bad Request` — Payload validation failed. Includes an `error` field.

## Rate limiting and auth

The prototype ships without authentication or rate limiting. For production
deployment place the app behind a reverse proxy that enforces both.
