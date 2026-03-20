# Holiday Homes Rent Price Prediction

Professional Flask application for estimating short-term rental prices based on city, neighbourhood, property details, guest capacity, review score, and selected amenities.

## Overview

This project provides an interactive pricing dashboard for holiday-home listings. It combines a trained machine learning pipeline with business-layer calibration rules to produce practical nightly price estimates and minimum-stay totals.

The application includes:

- Secure login gate for dashboard access
- Interactive pricing form with location map
- Support for multiple property types such as apartments, houses, condominiums, and villas
- Amenity-aware pricing adjustments
- Human-readable explanation of price drivers
- Optional AI assistant integration through an OpenAI API key

## Project Structure

```text
.
|-- model/
|   `-- pipeline_Holiday_Homes.joblib
|-- src/
|   |-- app_flask.py
|   `-- metadata.json
|-- static/
|   |-- styles.css
|   `-- rent-price-logo.svg
|-- templates/
|   `-- index.html
|-- tests/
|   `-- test_app_flask.py
|-- Procfile
|-- README.md
|-- render.yaml
|-- requirements.txt
`-- runtime.txt
```

## Features

- Prediction dashboard for nightly and total minimum-stay pricing
- Location-based defaults and neighbourhood mapping
- Property-type correction for larger properties such as houses and villas
- Logical amenity pricing rules that prevent unrealistic negative price movement
- Responsive UI styled for the Holiday Homes brand
- Production-ready Gunicorn startup command

## Tech Stack

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- LightGBM
- Gunicorn

## Local Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `src/.env` and add the variables you need:

```env
FLASK_SECRET_KEY=replace-with-a-secure-secret
PROJECT_LOGIN_PASSWORD=replace-with-your-password
OPENAI_API_KEY=optional
OPENAI_CHAT_MODEL=gpt-4o-mini
```

Notes:

- `PROJECT_LOGIN_PASSWORD` secures dashboard access.
- `OPENAI_API_KEY` is optional and only needed for the assistant/chat feature.
- If no OpenAI key is provided, the pricing dashboard still works.

### 4. Run the app

```powershell
python src/app_flask.py
```

The app will start on:

```text
http://127.0.0.1:5000
```

## Running Tests

```powershell
python -m unittest tests.test_app_flask
```

## Deployment

### Deploy on Render

This repository includes:

- `Procfile` with the production start command
- `runtime.txt` for the Python runtime
- `render.yaml` for Render Blueprint deployment

Recommended Render environment variables:

- `FLASK_SECRET_KEY`
- `PROJECT_LOGIN_PASSWORD`
- `OPENAI_API_KEY` (optional)
- `OPENAI_CHAT_MODEL` (optional)

Render start command used by this app:

```text
gunicorn src.app_flask:app
```

## GitHub Publishing

To publish this project:

1. Create a new empty GitHub repository.
2. Initialize Git locally if needed.
3. Add the GitHub repository as `origin`.
4. Push the `main` branch.

Example:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Security Notes

- Do not commit `src/.env`
- Do not commit secrets or private API keys
- Change the default Flask secret in production

## Model Notes

The trained model file is preserved as-is. Pricing corrections for special property types and amenities are handled in the application layer, so the deployed system can be improved without retraining the model artifact.

## License

Add your preferred license before public release.
