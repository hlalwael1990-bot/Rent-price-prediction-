import os
import json
import math
import random
import warnings
import urllib.error
import urllib.request
import numpy as np
import pandas as pd
import joblib

try:
    import shap
except ImportError:
    shap = None

from flask import Flask, has_request_context, jsonify, redirect, render_template, request, session, url_for


# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR = os.path.join(BASE_DIR, "model")
ENV_PATH = os.path.join(SRC_DIR, ".env")

METADATA_PATH = os.path.join(SRC_DIR, "metadata.json")
MODEL_PATH = os.path.join(MODEL_DIR, "pipeline_Holiday_Homes.joblib")


# =========================================================
# Environment
# =========================================================
def load_env_file(path):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env_file(ENV_PATH)


# =========================================================
# Flask app
# =========================================================
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")
app.config["SESSION_PERMANENT"] = False

def normalize_api_key(raw_value: str) -> str:
    value = str(raw_value or "").strip().strip('"').strip("'")
    for prefix in ["OPENAI_API_KEY=", "API_OPENAI_KEY="]:
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    return value


LOGIN_PASSWORD = os.getenv("PROJECT_LOGIN_PASSWORD", "").strip()
OPENAI_API_KEY = normalize_api_key(
    os.getenv("OPENAI_API_KEY", "") or os.getenv("API_OPENAI_KEY", "")
)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
CHAT_HISTORY_LIMIT = 12


# =========================================================
# Helpers
# =========================================================
def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_tf(value, default="f"):
    if value is None:
        return default

    value = str(value).strip().lower()

    if value in {"true", "t", "1", "yes", "y", "on"}:
        return "t"
    if value in {"false", "f", "0", "no", "n", "off"}:
        return "f"

    return default


def pretty_error(exc):
    return str(exc).replace("\n", " ").strip()


def is_logged_in():
    return bool(session.get("authenticated"))


def login_is_configured():
    return bool(LOGIN_PASSWORD)


def chatbot_is_configured():
    return bool(OPENAI_API_KEY)


def get_chat_history() -> list[dict]:
    if not has_request_context():
        return []

    history = session.get("chat_history", [])
    if not isinstance(history, list):
        return []

    cleaned = []
    for item in history[-CHAT_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})

    return cleaned


def set_chat_history(history: list[dict]):
    if not has_request_context():
        return

    session["chat_history"] = history[-CHAT_HISTORY_LIMIT:]
    session.modified = True


def clear_chat_history():
    if not has_request_context():
        return

    session.pop("chat_history", None)
    session.modified = True


def build_listing_snapshot(form_like) -> dict:
    if form_like is None:
        return {}

    amenities = []
    try:
        amenities = list(form_like.getlist("amenities"))
    except Exception:
        raw_amenities = form_like.get("amenities", [])
        if isinstance(raw_amenities, list):
            amenities = raw_amenities

    city = str(form_like.get("city", "")).strip()
    neighbourhood = str(form_like.get("neighbourhood", "")).strip()

    return {
        "city": city,
        "neighbourhood": neighbourhood,
        "property_type": str(form_like.get("property_type", "")).strip(),
        "room_type": str(form_like.get("room_type", "")).strip(),
        "accommodates": str(form_like.get("accommodates", "")).strip(),
        "bedrooms": str(form_like.get("bedrooms", "")).strip(),
        "minimum_nights": str(form_like.get("minimum_nights", "")).strip(),
        "maximum_nights": str(form_like.get("maximum_nights", "")).strip(),
        "review_scores_rating": str(form_like.get("review_scores_rating", "")).strip(),
        "distance_from_center_km": (
            round(get_distance_from_center(city, neighbourhood), 2)
            if city and neighbourhood else ""
        ),
        "amenities": [str(item).strip() for item in amenities if str(item).strip()],
    }


def get_prediction_snapshot() -> dict:
    if not has_request_context():
        return {}

    snapshot = session.get("prediction_snapshot", {})
    return snapshot if isinstance(snapshot, dict) else {}


def set_prediction_snapshot(snapshot: dict):
    if not has_request_context():
        return

    session["prediction_snapshot"] = snapshot
    session.modified = True


def get_project_explanation_snapshot() -> dict:
    if not has_request_context():
        return {}

    snapshot = session.get("project_explanation_snapshot", {})
    return snapshot if isinstance(snapshot, dict) else {}


def set_project_explanation_snapshot(snapshot: dict):
    if not has_request_context():
        return

    session["project_explanation_snapshot"] = snapshot
    session.modified = True


def get_preferred_chat_language() -> str:
    if not has_request_context():
        return ""

    value = str(session.get("preferred_chat_language", "")).strip()
    return value


def set_preferred_chat_language(language: str):
    if not has_request_context():
        return

    session["preferred_chat_language"] = str(language).strip()
    session.modified = True


def detect_requested_language(user_message: str) -> str:
    lowered = str(user_message or "").strip().lower()

    language_patterns = {
        "Arabic": ["arabic", "speak arabic", "translate to arabic", "in arabic", "بالعربية", "عربي"],
        "English": ["english", "speak english", "translate to english", "in english"],
        "French": ["french", "speak french", "translate to french", "in french", "francais", "français"],
        "Spanish": ["spanish", "speak spanish", "translate to spanish", "in spanish", "espanol", "español"],
        "German": ["german", "speak german", "translate to german", "in german", "deutsch"],
        "Turkish": ["turkish", "speak turkish", "translate to turkish", "in turkish"],
        "Italian": ["italian", "speak italian", "translate to italian", "in italian"],
        "Portuguese": ["portuguese", "speak portuguese", "translate to portuguese", "in portuguese"],
        "Chinese": ["chinese", "speak chinese", "translate to chinese", "in chinese", "mandarin"],
        "Hindi": ["hindi", "speak hindi", "translate to hindi", "in hindi"],
    }

    trigger_terms = ["translate", "speak", "reply", "answer", "write", "talk", "in "]
    for language, patterns in language_patterns.items():
        for pattern in patterns:
            if pattern in lowered and any(term in lowered for term in trigger_terms):
                return language

    return ""


def get_chatbot_welcome_messages() -> list[str]:
    language = get_preferred_chat_language() or "English"

    welcome_messages = {
        "Arabic": [
            "Ø£Ù‡Ù„Ù‹Ø§ Ø¨Ùƒ ÙÙŠ Ù…Ø³Ø§Ø¹Ø¯ Ø§Ù„ØªÙ†Ø¨Ø¤ Ø¨Ø£Ø³Ø¹Ø§Ø± Ø¨ÙŠÙˆØª Ø§Ù„Ø¹Ø·Ù„Ø§Øª. ÙƒÙŠÙ ÙŠÙ…ÙƒÙ†Ù†ÙŠ Ù…Ø³Ø§Ø¹Ø¯ØªÙƒ Ø§Ù„ÙŠÙˆÙ…ØŸ",
            "Ù…Ø±Ø­Ø¨Ø§Ù‹ Ø¨Ùƒ. ÙŠÙ…ÙƒÙ†Ù†ÙŠ Ù…Ø³Ø§Ø¹Ø¯ØªÙƒ ÙÙŠ ÙÙ‡Ù… Ø§Ù„Ø£Ø³Ø¹Ø§Ø± ÙˆØ§Ù„Ù…Ø¯Ù† ÙˆØ§Ù„Ø¹ÙˆØ§Ù…Ù„ Ø§Ù„Ù…Ø¤Ø«Ø±Ø©.",
            "Ø§Ø³Ø£Ù„Ù†ÙŠ Ø¹Ù† Ø§Ù„Ø³Ø¹Ø± Ø§Ù„Ù…ØªÙˆÙ‚Ø¹ Ø£Ùˆ Ø§Ù„Ù…ÙˆÙ‚Ø¹ Ø£Ùˆ Ø§Ù„Ù…Ø±Ø§ÙÙ‚ ÙˆØ³Ø£Ø³Ø§Ø¹Ø¯Ùƒ.",
        ],
        "French": [
            "Bienvenue dans l'assistant de prediction des prix des maisons de vacances. Comment puis-je vous aider aujourd'hui ?",
            "Bonjour, je peux vous aider avec les prix, les villes et les facteurs de prediction.",
            "Posez-moi une question sur le prix estime, les equipements ou la localisation.",
        ],
        "Spanish": [
            "Te doy la bienvenida al asistente de prediccion de precios para casas vacacionales. Como puedo ayudarte hoy?",
            "Hola, puedo ayudarte con precios estimados, ciudades y factores que influyen en el valor.",
            "Preguntame sobre el precio, las comodidades o la ubicacion.",
        ],
        "German": [
            "Willkommen beim Assistenten fuer die Preisvorhersage von Ferienhaeusern. Wie kann ich Ihnen heute helfen?",
            "Hallo, ich kann Ihnen bei Preisen, Staedten und Einflussfaktoren helfen.",
            "Fragen Sie mich nach Preis, Ausstattung oder Standortvergleich.",
        ],
        "Turkish": [
            "Tatil evi fiyat tahmin asistanina hos geldiniz. Bugun size nasil yardimci olabilirim?",
            "Merhaba, fiyat tahmini, sehirler ve etkileyen ozelliklerde yardimci olabilirim.",
            "Bana fiyat, olanaklar veya konum hakkinda soru sorabilirsiniz.",
        ],
        "Italian": [
            "Benvenuto nell'assistente per la previsione dei prezzi delle case vacanza. Come posso aiutarti oggi?",
            "Ciao, posso aiutarti con prezzi stimati, citta e fattori che influenzano il valore.",
            "Chiedimi pure di prezzi, servizi o posizione.",
        ],
        "Portuguese": [
            "Bem-vindo ao assistente de previsao de preco para casas de ferias. Como posso ajudar voce hoje?",
            "Ola, posso ajudar com estimativas de preco, cidades e fatores que afetam o valor.",
            "Pergunte sobre preco, comodidades ou localizacao.",
        ],
        "Chinese": [
            "Welcome to the holiday homes price prediction assistant. How can I help you today?",
            "Ask me about estimated prices, amenities, or which location may cost more.",
            "I can help explain how city, room type, and reviews affect the predicted price.",
        ],
        "Hindi": [
            "Welcome to the holiday homes price prediction assistant. How can I help you today?",
            "Ask me about price estimates, amenities, or city comparisons for this project.",
            "I can help explain what affects the predicted nightly price.",
        ],
        "English": [
            "Welcome to the chatbot prediction price assistant. How can I help you today?",
            "Ask me about estimated prices, cities, amenities, or what changes the prediction.",
            "I can help compare locations, explain price factors, or guide you through the dashboard.",
        ],
    }

    return welcome_messages.get(language, welcome_messages["English"])


def get_chatbot_welcome_message() -> str:
    return random.choice(get_chatbot_welcome_messages())

    language = get_preferred_chat_language() or "Arabic"

    welcome_messages = {
        "Arabic": "أهلًا بك في مساعد التنبؤ بأسعار بيوت العطلات. كيف يمكنني مساعدتك اليوم؟",
        "French": "Bienvenue dans l'assistant de prediction des prix des maisons de vacances. Comment puis-je vous aider aujourd'hui ?",
        "Spanish": "Te doy la bienvenida al asistente de prediccion de precios para casas vacacionales. Como puedo ayudarte hoy?",
        "German": "Willkommen beim Assistenten fuer die Preisvorhersage von Ferienhaeusern. Wie kann ich Ihnen heute helfen?",
        "Turkish": "Tatil evi fiyat tahmin asistanina hos geldiniz. Bugun size nasil yardimci olabilirim?",
        "Italian": "Benvenuto nell'assistente per la previsione dei prezzi delle case vacanza. Come posso aiutarti oggi?",
        "Portuguese": "Bem-vindo ao assistente de previsao de preco para casas de ferias. Como posso ajudar voce hoje?",
        "Chinese": "Welcome to the holiday homes price prediction assistant. How can I help you today?",
        "Hindi": "Welcome to the holiday homes price prediction assistant. How can I help you today?",
        "English": "Welcome to the chatbot prediction price assistant. How can I help you today?",
    }

    return welcome_messages.get(language, welcome_messages["Arabic"])


def get_chatbot_welcome_messages() -> list[str]:
    language = get_preferred_chat_language() or "English"

    welcome_messages = {
        "Arabic": [
            "\u0623\u0647\u0644\u0627\u064b \u0628\u0643 \u0641\u064a \u0645\u0633\u0627\u0639\u062f \u0627\u0644\u062a\u0646\u0628\u0624 \u0628\u0623\u0633\u0639\u0627\u0631 \u0628\u064a\u0648\u062a \u0627\u0644\u0639\u0637\u0644\u0627\u062a. \u0643\u064a\u0641 \u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643 \u0627\u0644\u064a\u0648\u0645\u061f",
            "\u0645\u0631\u062d\u0628\u0627\u064b \u0628\u0643. \u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643 \u0641\u064a \u0641\u0647\u0645 \u0627\u0644\u0623\u0633\u0639\u0627\u0631 \u0648\u0627\u0644\u0645\u062f\u0646 \u0648\u0627\u0644\u0639\u0648\u0627\u0645\u0644 \u0627\u0644\u0645\u0624\u062b\u0631\u0629.",
            "\u0627\u0633\u0623\u0644\u0646\u064a \u0639\u0646 \u0627\u0644\u0633\u0639\u0631 \u0627\u0644\u0645\u062a\u0648\u0642\u0639 \u0623\u0648 \u0627\u0644\u0645\u0648\u0642\u0639 \u0623\u0648 \u0627\u0644\u0645\u0631\u0627\u0641\u0642 \u0648\u0633\u0623\u0633\u0627\u0639\u062f\u0643.",
        ],
        "French": [
            "Bienvenue dans l'assistant de prediction des prix des maisons de vacances. Comment puis-je vous aider aujourd'hui ?",
            "Bonjour, je peux vous aider avec les prix, les villes et les facteurs de prediction.",
            "Posez-moi une question sur le prix estime, les equipements ou la localisation.",
        ],
        "Spanish": [
            "Te doy la bienvenida al asistente de prediccion de precios para casas vacacionales. Como puedo ayudarte hoy?",
            "Hola, puedo ayudarte con precios estimados, ciudades y factores que influyen en el valor.",
            "Preguntame sobre el precio, las comodidades o la ubicacion.",
        ],
        "German": [
            "Willkommen beim Assistenten fuer die Preisvorhersage von Ferienhaeusern. Wie kann ich Ihnen heute helfen?",
            "Hallo, ich kann Ihnen bei Preisen, Staedten und Einflussfaktoren helfen.",
            "Fragen Sie mich nach Preis, Ausstattung oder Standortvergleich.",
        ],
        "Turkish": [
            "Tatil evi fiyat tahmin asistanina hos geldiniz. Bugun size nasil yardimci olabilirim?",
            "Merhaba, fiyat tahmini, sehirler ve etkileyen ozelliklerde yardimci olabilirim.",
            "Bana fiyat, olanaklar veya konum hakkinda soru sorabilirsiniz.",
        ],
        "Italian": [
            "Benvenuto nell'assistente per la previsione dei prezzi delle case vacanza. Come posso aiutarti oggi?",
            "Ciao, posso aiutarti con prezzi stimati, citta e fattori che influenzano il valore.",
            "Chiedimi pure di prezzi, servizi o posizione.",
        ],
        "Portuguese": [
            "Bem-vindo ao assistente de previsao de preco para casas de ferias. Como posso ajudar voce hoje?",
            "Ola, posso ajudar com estimativas de preco, cidades e fatores que afetam o valor.",
            "Pergunte sobre preco, comodidades ou localizacao.",
        ],
        "Chinese": [
            "Welcome to the holiday homes price prediction assistant. How can I help you today?",
            "Ask me about estimated prices, amenities, or which location may cost more.",
            "I can help explain how city, room type, and reviews affect the predicted price.",
        ],
        "Hindi": [
            "Welcome to the holiday homes price prediction assistant. How can I help you today?",
            "Ask me about price estimates, amenities, or city comparisons for this project.",
            "I can help explain what affects the predicted nightly price.",
        ],
        "English": [
            "Welcome to the chatbot prediction price assistant. How can I help you today?",
            "Ask me about estimated prices, cities, amenities, or what changes the prediction.",
            "I can help compare locations, explain price factors, or guide you through the dashboard.",
        ],
    }

    return welcome_messages.get(language, welcome_messages["English"])


def get_chatbot_welcome_message() -> str:
    return random.choice(get_chatbot_welcome_messages())


def build_chatbot_system_prompt(listing_snapshot: dict | None = None, response_language: str = "English") -> str:
    snapshot = listing_snapshot or {}
    filtered_snapshot = {
        k: v
        for k, v in snapshot.items()
        if v is not None and v != "" and v != []
    }
    prediction_snapshot = get_prediction_snapshot()
    explanation_snapshot = get_project_explanation_snapshot()
    snapshot_text = json.dumps(filtered_snapshot, ensure_ascii=False)
    prediction_text = json.dumps(prediction_snapshot, ensure_ascii=False)
    explanation_text = json.dumps(explanation_snapshot, ensure_ascii=False)
    metadata_text = json.dumps(
        {
            "prediction_target": PREDICTION_TARGET,
            "output_currency": OUTPUT_CURRENCY,
            "cities": CITY_OPTIONS,
            "property_types": PROPERTY_TYPE_OPTIONS,
            "room_types": ROOM_TYPE_OPTIONS,
            "top_amenities": TOP_AMENITIES,
            "defaults": DEFAULTS,
            "model_knowledge": MODEL_KNOWLEDGE,
        },
        ensure_ascii=False,
    )

    return (
        "You are the in-app AI assistant for a Holiday Homes Rent Price Prediction project. "
        f"Reply in {response_language}. "
        "If the user asks you to switch languages, follow that request and answer fully in that language. "
        "Be professional, warm, and concise. "
        "You help with anything related to this project: pricing logic, listing inputs, amenities, location effects, review score impact, minimum stay, dashboard usage, prediction interpretation, and project behavior. "
        "Stay grounded in the provided project data and configuration. "
        "When the user asks for cheapest, best, or comparison questions, answer based on the trained model, metadata, and the current dashboard context only. "
        "Do not claim to retrieve real listings from a listings table or dataset file. "
        "When explaining a price, explicitly consider and mention distance from center, property type, room type, guest capacity, review score, amenities, and output currency whenever they are available in context. "
        "If the user asks something unrelated to the project, answer briefly and then steer back to project context when appropriate. "
        "Do not claim you were trained on private project data; instead, use the project context provided to you. "
        f"Current listing form context: {snapshot_text}. "
        f"Latest prediction context: {prediction_text}. "
        f"Latest explanation context: {explanation_text}. "
        f"Project metadata context: {metadata_text}."
    )


def request_openai_chat_response(user_message: str, history: list[dict], listing_snapshot: dict | None = None) -> str:
    if not chatbot_is_configured():
        return "The chatbot is not configured yet. Add `API_OPENAI_KEY` or `OPENAI_API_KEY` to `src/.env` to enable it."

    requested_language = detect_requested_language(user_message)
    if requested_language:
        set_preferred_chat_language(requested_language)

    response_language = requested_language or get_preferred_chat_language() or "English"

    messages = [
        {
            "role": "system",
            "content": build_chatbot_system_prompt(listing_snapshot, response_language=response_language),
        },
    ]

    for item in history[-8:]:
        messages.append(
            {
                "role": item["role"],
                "content": item["content"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    http_request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(http_request, timeout=45) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            error_message = error_payload.get("error", {}).get("message", pretty_error(exc))
        except Exception:
            error_message = pretty_error(exc)
        return f"The assistant could not generate a response: {error_message}"
    except Exception as exc:
        return f"The assistant could not connect to the API: {pretty_error(exc)}"

    choices = response_data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return "The assistant returned an empty response. Please try again."


def predict_with_model(input_df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        return model.predict(input_df)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def clean_feature_name(name: str) -> str:
    cleaned = str(name)
    for prefix in ["remainder__", "num__", "cat__", "onehot__", "ordinal__", "scaler__"]:
        cleaned = cleaned.replace(prefix, "")
    return cleaned


def distance_band(km):
    if km <= 3:
        return "central"
    elif km <= 7:
        return "mid"
    elif km <= 12:
        return "far"
    return "very_far"


def amenity_col_name(name):
    return (
        "amenity_"
        + str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .lower()
    )


def normalize_property_type_for_model(value: str) -> str:
    value = str(value).strip()
    if value.lower() == "other":
        return "other"
    return value


def property_type_key(value: str) -> str:
    normalized = str(value or "").strip().lower()

    if "villa" in normalized:
        return "villa"
    if "house" in normalized:
        return "house"
    if "condominium" in normalized or "condo" in normalized:
        return "condominium"
    if "apartment" in normalized:
        return "apartment"
    return "other"


def compute_property_type_multiplier(
    property_type: str,
    room_type: str,
    accommodates: int,
    bedrooms: float,
) -> float:
    if str(room_type or "").strip().lower() != "entire place":
        return 1.0

    key = property_type_key(property_type)
    accommodates = max(int(accommodates), 1)
    bedrooms = max(float(bedrooms), 0.0)

    multiplier = 1.0
    if key == "villa":
        multiplier *= 3.8
        multiplier *= 1.0 + min(max(accommodates - 4, 0), 8) * 0.07
        multiplier *= 1.0 + min(max(bedrooms - 2, 0.0), 5.0) * 0.10
    elif key == "house":
        multiplier *= 3.0
        multiplier *= 1.0 + min(max(accommodates - 4, 0), 8) * 0.055
        multiplier *= 1.0 + min(max(bedrooms - 2, 0.0), 5.0) * 0.08
    elif key == "condominium":
        multiplier *= 1.06
        multiplier *= 1.0 + min(max(accommodates - 3, 0), 6) * 0.01

    return min(multiplier, 8.0)


def compute_property_type_amenity_factor(property_type: str, room_type: str) -> float:
    return 1.0


def compute_min_amenity_increment_local(city: str, property_type: str, amenity: str) -> float:
    key = property_type_key(property_type)
    amenity_key = str(amenity or "").strip().lower()
    base_weight = AMENITY_PRICE_WEIGHTS.get(amenity_key, 0.005)
    rate = get_currency_rate(city) if OUTPUT_CURRENCY == "USD_after_prediction" else 1.0

    villa_usd_map = {
        "wifi": 4.0,
        "kitchen": 15,
        "tv": 5,
        "hot water": 20,
        "washer": 15,
        "dedicated workspace": 15,
        "heating": 25,
        "air conditioning": 20,
    }
    house_usd_map = {
        "wifi": 3.0,
        "kitchen": 10,
        "tv": 1.5,
        "hot water": 7,
        "washer": 10,
        "dedicated workspace": 12,
        "heating": 15,
        "air conditioning": 10,
    }

    if key == "villa":
        usd_increment = villa_usd_map.get(amenity_key, min(1.5 + base_weight * 70.0, 5.0))
    elif key == "house":
        usd_increment = house_usd_map.get(amenity_key, min(1.2 + base_weight * 55.0, 4.0))
    else:
        usd_increment = min(1.0 + base_weight * 45.0, 3.0)

    if OUTPUT_CURRENCY == "USD_after_prediction" and rate > 0:
        return round(usd_increment / rate, 2)
    return round(usd_increment, 2)


def compute_property_type_amenity_bonus_local(city: str, property_type: str, selected_amenities: list[str]) -> float:
    normalized = []
    seen = set()

    for item in selected_amenities:
        amenity = str(item).strip().lower()
        if not amenity or amenity in seen:
            continue
        seen.add(amenity)
        normalized.append(amenity)

    return round(
        sum(compute_min_amenity_increment_local(city, property_type, amenity) for amenity in normalized),
        2,
    )


def compute_property_type_floor_price(
    base_inputs: dict,
    selected_amenities: list[str],
    review_score: int,
    city: str,
    property_type: str,
    room_type: str,
    accommodates: int,
    bedrooms: float,
) -> float:
    key = property_type_key(property_type)

    if str(room_type or "").strip().lower() != "entire place":
        return 0.0

    if key not in {"house", "villa"}:
        return 0.0

    apartment_anchor_inputs = dict(base_inputs)
    apartment_anchor_inputs["property_type"] = "Entire apartment"
    apartment_anchor_inputs["amenities"] = []
    apartment_anchor_inputs["review_scores_rating"] = 0

    apartment_anchor_df = build_feature_row_from_inputs(apartment_anchor_inputs)
    apartment_anchor_base_price = predict_base_price_local(apartment_anchor_df)
    apartment_anchor_price = apply_light_calibration(
        base_price_local=apartment_anchor_base_price,
        city=city,
        selected_amenities=[],
        review_score=0,
        property_type="Entire apartment",
        room_type=room_type,
        accommodates=accommodates,
        bedrooms=bedrooms,
    )

    if key == "house":
        floor_multiplier = 3.0 + min(max(accommodates - 4, 0), 8) * 0.08 + min(max(bedrooms - 2, 0.0), 5.0) * 0.14
    else:
        floor_multiplier = 4.1 + min(max(accommodates - 4, 0), 8) * 0.10 + min(max(bedrooms - 2, 0.0), 5.0) * 0.18

    review_uplift = compute_review_uplift(review_score)
    property_amenity_bonus_local = compute_property_type_amenity_bonus_local(
        city=city,
        property_type=property_type,
        selected_amenities=selected_amenities,
    )

    floor_price = apartment_anchor_price * floor_multiplier
    floor_price *= 1.0 + review_uplift
    floor_price += property_amenity_bonus_local

    return round(floor_price, 2)


def get_currency_rate(city: str) -> float:
    rate = EXCHANGE_RATES.get(city)
    if rate is None:
        return 1.0
    return float(rate)


def output_currency_label() -> str:
    if OUTPUT_CURRENCY == "USD_after_prediction":
        return "USD"
    return "Local currency"


def get_fixed_bonus_amenities(selected_amenities: list[str]) -> list[str]:
    normalized_to_label = {}
    for item in selected_amenities:
        label = str(item).strip()
        if not label:
            continue
        normalized_to_label[label.lower()] = label

    ordered_bonus_keys = [
        "kitchen",
        "dedicated workspace",
        "washer",
        "hot water",
        "tv",
    ]

    return [
        normalized_to_label[key]
        for key in ordered_bonus_keys
        if key in normalized_to_label
    ]


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# =========================================================
# Load metadata
# =========================================================
metadata = load_json(METADATA_PATH)

MODEL_TARGET_TRANSFORM = metadata.get("model_target_transform", "log1p")
PREDICTION_TARGET = metadata.get("prediction_target", "price_local")
OUTPUT_CURRENCY = metadata.get("output_currency", "price_local")

CATEGORICAL_COLUMNS = metadata.get("categorical_columns", [])
NUMERICAL_COLUMNS = metadata.get("numerical_columns", [])

CITY_OPTIONS = metadata.get("city_options", [])
CITY_NEIGHBOURHOOD_MAP = metadata.get("city_neighbourhood_map", {})
NEIGHBOURHOOD_COORDINATES = metadata.get("neighbourhood_coordinates", {})
CITY_CENTERS = metadata.get("city_centers", {})
PROPERTY_TYPE_OPTIONS = metadata.get("property_type_options", [])
ROOM_TYPE_OPTIONS = metadata.get("room_type_options", [])
INSTANT_BOOKABLE_OPTIONS = metadata.get("instant_bookable_options", ["Yes", "No"])
TOP_AMENITIES = metadata.get("top_amenities", [])
DEFAULTS = metadata.get("defaults", {})
EXCHANGE_RATES = metadata.get("exchange_rates", {})


def build_model_knowledge() -> dict:
    knowledge = {
        "available": model is not None,
        "pipeline_steps": [],
        "categorical_features": CATEGORICAL_COLUMNS,
        "numerical_features": NUMERICAL_COLUMNS,
        "trained_property_types": [],
        "trained_room_types": ROOM_TYPE_OPTIONS,
        "trained_cities": CITY_OPTIONS,
        "top_model_features": [],
    }

    if model is None or not hasattr(model, "named_steps"):
        return knowledge

    try:
        knowledge["pipeline_steps"] = list(model.named_steps.keys())

        preprocessor = model.named_steps.get("preprocessing")
        estimator = list(model.named_steps.values())[-1]

        if preprocessor is not None and hasattr(preprocessor, "transformers_"):
            for name, transformer, cols in preprocessor.transformers_:
                if name != "cat":
                    continue

                cat_columns = list(cols)
                if hasattr(transformer, "named_steps"):
                    onehot = transformer.named_steps.get("onehot")
                    if onehot is not None and hasattr(onehot, "categories_"):
                        if "property_type" in cat_columns:
                            idx = cat_columns.index("property_type")
                            knowledge["trained_property_types"] = [
                                str(item) for item in list(onehot.categories_[idx])
                            ]
                        if "room_type" in cat_columns:
                            idx = cat_columns.index("room_type")
                            knowledge["trained_room_types"] = [
                                str(item) for item in list(onehot.categories_[idx])
                            ]
                        if "city" in cat_columns:
                            idx = cat_columns.index("city")
                            knowledge["trained_cities"] = [
                                str(item) for item in list(onehot.categories_[idx])
                            ]
                break

        feature_names = []
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            feature_names = [clean_feature_name(item) for item in preprocessor.get_feature_names_out()]

        importances = getattr(estimator, "feature_importances_", None)
        if importances is not None and len(feature_names) == len(importances):
            ranked = sorted(
                zip(feature_names, importances),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            knowledge["top_model_features"] = [
                {"feature": name, "importance": round(float(score), 4)}
                for name, score in ranked[:12]
            ]
    except Exception as exc:
        print("MODEL KNOWLEDGE ERROR:", pretty_error(exc))

    return knowledge


AMENITY_PRICE_WEIGHTS = {
    "wifi": 0.030,
    "essentials": 0.008,
    "kitchen": 0.040,
    "long term stays allowed": 0.025,
    "hangers": 0.004,
    "tv": 0.012,
    "iron": 0.004,
    "dedicated workspace": 0.013,
    "hair dryer": 0.006,
    "hot water": 0.008,
    "washer": 0.030,
    "heating": 0.012,
    "shampoo": 0.004,
    "smoke alarm": 0.005,
    "air conditioning": 0.0,
}

FIXED_ONLY_AMENITIES = {
}

DEFAULTS.setdefault("city", CITY_OPTIONS[0] if CITY_OPTIONS else "")
DEFAULTS.setdefault("neighbourhood", "")
DEFAULTS.setdefault("property_type", PROPERTY_TYPE_OPTIONS[0] if PROPERTY_TYPE_OPTIONS else "")
DEFAULTS.setdefault("room_type", ROOM_TYPE_OPTIONS[0] if ROOM_TYPE_OPTIONS else "")
DEFAULTS.setdefault("instant_bookable", "No")
DEFAULTS.setdefault("latitude", 48.8566)
DEFAULTS.setdefault("longitude", 2.3522)
DEFAULTS.setdefault("distance_from_center_km", 2.0)
DEFAULTS.setdefault("accommodates", 2)
DEFAULTS.setdefault("bedrooms", 1)
DEFAULTS.setdefault("minimum_nights", 2)
DEFAULTS.setdefault("maximum_nights", 30)
DEFAULTS.setdefault("review_scores_rating", 90)

EXPECTED_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS


# =========================================================
# Location helpers
# =========================================================
def get_neighbourhood_lat_lon(city: str, neighbourhood: str):
    city_data = NEIGHBOURHOOD_COORDINATES.get(city, {})
    coords = city_data.get(neighbourhood)

    if not coords or len(coords) != 2:
        return None, None

    return float(coords[0]), float(coords[1])


def get_distance_from_center(city: str, neighbourhood: str) -> float:
    city_center = CITY_CENTERS.get(city)
    if not city_center or len(city_center) != 2:
        return float(DEFAULTS["distance_from_center_km"])

    n_lat, n_lon = get_neighbourhood_lat_lon(city, neighbourhood)
    if n_lat is None or n_lon is None:
        return float(DEFAULTS["distance_from_center_km"])

    center_lat, center_lon = float(city_center[0]), float(city_center[1])
    return round(haversine_km(n_lat, n_lon, center_lat, center_lon), 2)


# =========================================================
# Load model
# =========================================================
model = None
model_error = None

try:
    model = joblib.load(MODEL_PATH)
    print("MODEL LOADED SUCCESSFULLY")
except Exception as exc:
    print("MODEL ERROR:", exc)
    model_error = pretty_error(exc)

MODEL_KNOWLEDGE = build_model_knowledge()


# =========================================================
# Optional: inspect trained property types
# =========================================================
def inspect_trained_property_types():
    print("Property types from metadata:", PROPERTY_TYPE_OPTIONS)

    if model is None:
        print("Model not available for category inspection.")
        return

    try:
        if not hasattr(model, "named_steps"):
            return

        preprocessor = model.named_steps.get("preprocessing")
        if preprocessor is None:
            return

        cat_transformer = None
        cat_columns = None

        for name, transformer, cols in preprocessor.transformers_:
            if name == "cat":
                cat_transformer = transformer
                cat_columns = list(cols)
                break

        if cat_transformer is None or "property_type" not in cat_columns:
            return

        onehot = cat_transformer.named_steps.get("onehot")
        idx = cat_columns.index("property_type")
        trained_categories = list(onehot.categories_[idx])

        print("Property types learned by model:", trained_categories)

    except Exception as exc:
        print("Could not inspect trained property types:", pretty_error(exc))


inspect_trained_property_types()


# =========================================================
# SHAP setup
# =========================================================
shap_transformer = None
shap_estimator = None
shap_explainer = None


def build_shap_components():
    global shap_transformer, shap_estimator, shap_explainer

    shap_transformer = None
    shap_estimator = None
    shap_explainer = None

    if model is None or shap is None:
        return

    try:
        if hasattr(model, "named_steps"):
            step_items = list(model.named_steps.items())
            step_values = list(model.named_steps.values())

            shap_estimator = step_values[-1]

            if len(step_items) > 1:
                from sklearn.pipeline import Pipeline
                shap_transformer = Pipeline(step_items[:-1])
        else:
            shap_estimator = model
            shap_transformer = None

        try:
            shap_explainer = shap.TreeExplainer(shap_estimator)
        except Exception:
            try:
                shap_explainer = shap.Explainer(shap_estimator)
            except Exception:
                shap_explainer = None

    except Exception as exc:
        print("SHAP SETUP ERROR:", exc)
        shap_transformer = None
        shap_estimator = None
        shap_explainer = None


build_shap_components()


def generate_shap_text(input_df: pd.DataFrame):
    if shap_explainer is None or shap_estimator is None:
        return []

    try:
        if shap_transformer is not None:
            transformed = shap_transformer.transform(input_df)

            if hasattr(transformed, "toarray"):
                transformed_dense = transformed.toarray()
            else:
                transformed_dense = np.asarray(transformed)

            if hasattr(shap_transformer, "get_feature_names_out"):
                feature_names = [clean_feature_name(x) for x in shap_transformer.get_feature_names_out()]
            else:
                feature_names = [f"feature_{i}" for i in range(transformed_dense.shape[1])]

            transformed_frame = pd.DataFrame(transformed_dense, columns=feature_names)
        else:
            transformed_dense = input_df.values
            feature_names = [clean_feature_name(x) for x in input_df.columns]
            transformed_frame = input_df.copy()
            transformed_frame.columns = feature_names

        shap_output = shap_explainer(transformed_frame)

        if hasattr(shap_output, "values"):
            shap_values = np.asarray(shap_output.values)
        else:
            shap_values = np.asarray(shap_output)

        if shap_values.ndim >= 2:
            shap_row = shap_values[0]
        else:
            shap_row = shap_values

        shap_row = np.array(shap_row).ravel()
        actual_vals = np.array(transformed_dense[0]).ravel()

        contributions = []
        for feat, val, actual_val in zip(feature_names, shap_row, actual_vals):
            feat = clean_feature_name(feat)

            if (
                ("city_" in feat or "neighbourhood_" in feat or "property_type_" in feat or "room_type_" in feat or "distance_band_" in feat)
                and float(actual_val) == 0.0
            ):
                continue

            contributions.append((feat, float(val), float(actual_val)))

        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_contributions = contributions[:6]

        lines = []
        for feat, val, actual_val in top_contributions:
            direction = "increased" if val > 0 else "decreased"

            if any(prefix in feat for prefix in ["city_", "neighbourhood_", "property_type_", "room_type_", "distance_band_"]):
                lines.append(f"{feat} {direction} the base model prediction.")
            else:
                lines.append(f"{feat} (value={actual_val:g}) {direction} the base model prediction.")

        return lines

    except Exception as exc:
        print("SHAP ERROR:", exc)
        return []


# =========================================================
# Light calibration
# =========================================================
def compute_amenities_uplift(selected_amenities: list[str]) -> float:
    normalized = {
        str(item).strip().lower()
        for item in selected_amenities
        if str(item).strip()
    }
    uplift_amenities = {item for item in normalized if item not in FIXED_ONLY_AMENITIES}

    weighted_score = 0.0
    for amenity in uplift_amenities:
        weighted_score += AMENITY_PRICE_WEIGHTS.get(amenity, 0.005)

    premium_count = sum(
        1
        for amenity in uplift_amenities
        if amenity in {
            "wifi",
            "kitchen",
            "washer",
            "tv",
            "long term stays allowed",
        }
    )
    standard_count = max(len(uplift_amenities) - premium_count, 0)

    premium_bonus = min(premium_count, 4) * 0.009
    standard_bonus = min(standard_count, 6) * 0.0025
    blended_score = weighted_score + premium_bonus + standard_bonus

    # Keep the uplift realistic and avoid overpricing from checkbox stacking.
    return min(blended_score, 0.28)


def compute_review_uplift(review_score: int) -> float:
    score = max(0, min(int(review_score), 100))

    if score >= 95:
        return 0.082 + ((score - 95) / 5.0) * 0.018
    if score >= 90:
        return 0.064 + ((score - 90) / 5.0) * 0.018
    if score >= 85:
        return 0.050 + ((score - 85) / 5.0) * 0.014
    if score >= 75:
        return 0.030 + ((score - 75) / 10.0) * 0.020
    if score >= 60:
        return 0.012 + ((score - 60) / 15.0) * 0.018
    if score >= 40:
        return 0.004 + ((score - 40) / 20.0) * 0.008
    return (score / 40.0) * 0.004


def compute_fixed_amenity_bonus_local(city: str, selected_amenities: list[str]) -> float:
    normalized = {str(item).strip().lower() for item in selected_amenities if str(item).strip()}
    bonus_local = 0.0
    rate = get_currency_rate(city) if OUTPUT_CURRENCY == "USD_after_prediction" else 1.0

    if "kitchen" in normalized:
        if OUTPUT_CURRENCY == "USD_after_prediction":
            if rate > 0:
                bonus_local += 10.0 / rate
        else:
            bonus_local += 10.0

    if "dedicated workspace" in normalized:
        if OUTPUT_CURRENCY == "USD_after_prediction":
            if rate > 0:
                bonus_local += 10.0 / rate
        else:
            bonus_local += 10.0

    if "washer" in normalized:
        if OUTPUT_CURRENCY == "USD_after_prediction":
            if rate > 0:
                bonus_local += 5.0 / rate
        else:
            bonus_local += 5.0

    if "hot water" in normalized:
        if OUTPUT_CURRENCY == "USD_after_prediction":
            if rate > 0:
                bonus_local += 5.0 / rate
        else:
            bonus_local += 5.0

    if "tv" in normalized:
        if OUTPUT_CURRENCY == "USD_after_prediction":
            if rate > 0:
                bonus_local += 2.0 / rate
        else:
            bonus_local += 2.0

    return round(bonus_local, 2)


def apply_light_calibration(
    base_price_local: float,
    city: str,
    selected_amenities: list[str],
    review_score: int,
    property_type: str = "",
    room_type: str = "",
    accommodates: int = 1,
    bedrooms: float = 0.0,
) -> float:
    price = float(base_price_local)
    amenities_uplift = compute_amenities_uplift(selected_amenities)
    review_uplift = compute_review_uplift(review_score)
    fixed_bonus_local = compute_fixed_amenity_bonus_local(city, selected_amenities)
    property_type_multiplier = compute_property_type_multiplier(
        property_type=property_type,
        room_type=room_type,
        accommodates=accommodates,
        bedrooms=bedrooms,
    )
    amenity_factor = compute_property_type_amenity_factor(
        property_type=property_type,
        room_type=room_type,
    )

    amenities_uplift = min(amenities_uplift * amenity_factor, 0.95)

    price *= 1.0 + amenities_uplift
    price *= 1.0 + review_uplift
    price *= property_type_multiplier
    price += fixed_bonus_local

    price = max(price, 15.0)
    price = min(price, 5000.0)

    return round(price, 2)


def build_prediction_inputs(form) -> dict:
    return {
        "city": str(form.get("city", DEFAULTS["city"])).strip(),
        "neighbourhood": str(form.get("neighbourhood", DEFAULTS["neighbourhood"])).strip(),
        "property_type": str(form.get("property_type", DEFAULTS["property_type"])).strip(),
        "room_type": str(form.get("room_type", DEFAULTS["room_type"])).strip(),
        "instant_bookable": str(form.get("instant_bookable", DEFAULTS["instant_bookable"])).strip(),
        "accommodates": safe_int(form.get("accommodates"), DEFAULTS["accommodates"]),
        "bedrooms": safe_float(form.get("bedrooms"), DEFAULTS["bedrooms"]),
        "minimum_nights": safe_int(form.get("minimum_nights"), DEFAULTS["minimum_nights"]),
        "maximum_nights": safe_int(form.get("maximum_nights"), DEFAULTS["maximum_nights"]),
        "review_scores_rating": safe_int(form.get("review_scores_rating"), DEFAULTS["review_scores_rating"]),
        "amenities": list(form.getlist("amenities")),
    }


def remove_fixed_only_amenities(selected_amenities: list[str]) -> list[str]:
    return [
        item
        for item in selected_amenities
        if str(item).strip().lower() not in FIXED_ONLY_AMENITIES
    ]


def build_feature_row_from_inputs(inputs: dict) -> pd.DataFrame:
    row = {col: np.nan for col in EXPECTED_COLUMNS}

    city = str(inputs.get("city", DEFAULTS["city"])).strip()
    neighbourhood = str(inputs.get("neighbourhood", DEFAULTS["neighbourhood"])).strip()
    property_type_display = str(inputs.get("property_type", DEFAULTS["property_type"])).strip()
    property_type_model = normalize_property_type_for_model(property_type_display)
    room_type = str(inputs.get("room_type", DEFAULTS["room_type"])).strip()
    instant_bookable_form = str(inputs.get("instant_bookable", DEFAULTS["instant_bookable"])).strip()

    auto_lat, auto_lon = get_neighbourhood_lat_lon(city, neighbourhood)
    latitude = auto_lat if auto_lat is not None else float(DEFAULTS["latitude"])
    longitude = auto_lon if auto_lon is not None else float(DEFAULTS["longitude"])
    distance_km = get_distance_from_center(city, neighbourhood)

    accommodates = safe_int(inputs.get("accommodates"), DEFAULTS["accommodates"])
    bedrooms = safe_float(inputs.get("bedrooms"), DEFAULTS["bedrooms"])
    minimum_nights = safe_int(inputs.get("minimum_nights"), DEFAULTS["minimum_nights"])
    maximum_nights = safe_int(inputs.get("maximum_nights"), DEFAULTS["maximum_nights"])
    review_scores_rating = safe_int(inputs.get("review_scores_rating"), DEFAULTS["review_scores_rating"])

    selected_amenities = list(inputs.get("amenities", []))
    currency_rate = get_currency_rate(city)
    property_type_normalized = property_type_key(property_type_display)

    amenities_count = 0
    for amenity in TOP_AMENITIES:
        col = amenity_col_name(amenity)
        value = 1 if amenity in selected_amenities else 0

        if col in row:
            row[col] = value

        amenities_count += value

    if "city" in row:
        row["city"] = city

    if "neighbourhood" in row:
        row["neighbourhood"] = neighbourhood

    if "property_type" in row:
        row["property_type"] = property_type_model

    if "room_type" in row:
        row["room_type"] = room_type

    if "instant_bookable" in row:
        row["instant_bookable"] = normalize_tf(instant_bookable_form, default="f")

    if "distance_band" in row:
        row["distance_band"] = distance_band(distance_km)

    if "latitude" in row:
        row["latitude"] = latitude

    if "longitude" in row:
        row["longitude"] = longitude

    if "distance_from_center_km" in row:
        row["distance_from_center_km"] = distance_km

    if "log_distance_from_center" in row:
        row["log_distance_from_center"] = np.log1p(distance_km)

    if "is_central" in row:
        row["is_central"] = int(distance_km <= 3)

    if "is_far" in row:
        row["is_far"] = int(distance_km >= 10)

    if "accommodates" in row:
        row["accommodates"] = accommodates

    if "bedrooms" in row:
        row["bedrooms"] = bedrooms

    if "minimum_nights" in row:
        row["minimum_nights"] = minimum_nights

    if "maximum_nights" in row:
        row["maximum_nights"] = maximum_nights

    if "review_scores_rating" in row:
        row["review_scores_rating"] = review_scores_rating

    if "host_years" in row:
        row["host_years"] = np.nan

    safe_bedrooms = bedrooms if bedrooms > 0 else 0.5
    if "guests_per_bedroom" in row:
        row["guests_per_bedroom"] = min(accommodates / safe_bedrooms, 20)

    if "amenities_count" in row:
        row["amenities_count"] = amenities_count

    if "many_amenities" in row:
        row["many_amenities"] = int(amenities_count >= 6)

    if "premium_amenities" in row:
        row["premium_amenities"] = int(amenities_count >= 9)

    if "guest_x_amenities" in row:
        row["guest_x_amenities"] = accommodates * amenities_count

    if "rating_x_amenities" in row:
        row["rating_x_amenities"] = review_scores_rating * amenities_count

    if "distance_x_rating" in row:
        row["distance_x_rating"] = distance_km * review_scores_rating

    if "currency_rate" in row:
        row["currency_rate"] = currency_rate

    if "amenities_x_rate" in row:
        row["amenities_x_rate"] = amenities_count * currency_rate

    if "is_villa" in row:
        row["is_villa"] = int(property_type_normalized == "villa")

    if "is_house" in row:
        row["is_house"] = int(property_type_normalized == "house")

    if "is_apartment" in row:
        row["is_apartment"] = int(property_type_normalized == "apartment")

    if "is_condominium" in row:
        row["is_condominium"] = int(property_type_normalized == "condominium")

    if "large_capacity" in row:
        row["large_capacity"] = int(accommodates >= 6)

    if "many_bedrooms" in row:
        row["many_bedrooms"] = int(bedrooms >= 3)

    if "is_large_property" in row:
        row["is_large_property"] = int(
            property_type_normalized in {"villa", "house"} or accommodates >= 6 or bedrooms >= 3
        )

    if "villa_x_accommodates" in row:
        row["villa_x_accommodates"] = int(property_type_normalized == "villa") * accommodates

    if "villa_x_bedrooms" in row:
        row["villa_x_bedrooms"] = int(property_type_normalized == "villa") * bedrooms

    if "villa_x_amenities" in row:
        row["villa_x_amenities"] = int(property_type_normalized == "villa") * amenities_count

    if "villa_x_rating" in row:
        row["villa_x_rating"] = int(property_type_normalized == "villa") * review_scores_rating

    if "house_x_accommodates" in row:
        row["house_x_accommodates"] = int(property_type_normalized == "house") * accommodates

    if "house_x_bedrooms" in row:
        row["house_x_bedrooms"] = int(property_type_normalized == "house") * bedrooms

    if "house_x_amenities" in row:
        row["house_x_amenities"] = int(property_type_normalized == "house") * amenities_count

    if "house_x_rating" in row:
        row["house_x_rating"] = int(property_type_normalized == "house") * review_scores_rating

    return pd.DataFrame([row], columns=EXPECTED_COLUMNS)


def predict_base_price_local(input_df: pd.DataFrame) -> float:
    raw_pred = predict_with_model(input_df)[0]

    if MODEL_TARGET_TRANSFORM == "log1p":
        return max(float(np.expm1(raw_pred)), 0.0)
    return max(float(raw_pred), 0.0)


def compute_monotonic_price(base_inputs: dict, enforce_amenity_monotonic: bool = True) -> tuple[float, float]:
    city = str(base_inputs.get("city", DEFAULTS["city"])).strip()
    property_type = str(base_inputs.get("property_type", DEFAULTS["property_type"])).strip()
    room_type = str(base_inputs.get("room_type", DEFAULTS["room_type"])).strip()
    accommodates = safe_int(base_inputs.get("accommodates"), DEFAULTS["accommodates"])
    bedrooms = safe_float(base_inputs.get("bedrooms"), DEFAULTS["bedrooms"])
    selected_amenities = list(base_inputs.get("amenities", []))
    review_score = safe_int(base_inputs.get("review_scores_rating"), DEFAULTS["review_scores_rating"])
    property_type_normalized = property_type_key(property_type)
    is_large_entire_property = property_type_normalized in {"house", "villa"} and str(room_type or "").strip().lower() == "entire place"

    model_selected_inputs = dict(base_inputs)
    model_selected_inputs["amenities"] = remove_fixed_only_amenities(selected_amenities)
    selected_df = build_feature_row_from_inputs(model_selected_inputs)
    selected_base_price = predict_base_price_local(selected_df)
    selected_calibrated_price = apply_light_calibration(
        base_price_local=selected_base_price,
        city=city,
        selected_amenities=selected_amenities,
        review_score=review_score,
        property_type=property_type,
        room_type=room_type,
        accommodates=accommodates,
        bedrooms=bedrooms,
    )

    no_amenities_inputs = dict(base_inputs)
    no_amenities_inputs["amenities"] = []
    no_amenities_df = build_feature_row_from_inputs(no_amenities_inputs)
    no_amenities_base_price = predict_base_price_local(no_amenities_df)
    no_amenities_calibrated_price = apply_light_calibration(
        base_price_local=no_amenities_base_price,
        city=city,
        selected_amenities=[],
        review_score=review_score,
        property_type=property_type,
        room_type=room_type,
        accommodates=accommodates,
        bedrooms=bedrooms,
    )

    review_anchor_inputs = dict(base_inputs)
    review_anchor_inputs["review_scores_rating"] = 0
    review_anchor_df = build_feature_row_from_inputs(review_anchor_inputs)
    review_anchor_base_price = predict_base_price_local(review_anchor_df)

    monotonic_anchor_inputs = dict(base_inputs)
    monotonic_anchor_inputs["amenities"] = []
    monotonic_anchor_inputs["review_scores_rating"] = 0
    monotonic_anchor_df = build_feature_row_from_inputs(monotonic_anchor_inputs)
    monotonic_anchor_base_price = predict_base_price_local(monotonic_anchor_df)

    amenities_uplift = compute_amenities_uplift(selected_amenities)
    review_uplift = compute_review_uplift(review_score)
    fixed_bonus_local = compute_fixed_amenity_bonus_local(city, selected_amenities)

    amenities_floor_price = no_amenities_base_price * (1.0 + amenities_uplift) * (1.0 + review_uplift) + fixed_bonus_local
    review_floor_price = review_anchor_base_price * (1.0 + compute_amenities_uplift(selected_amenities)) * (1.0 + review_uplift) + fixed_bonus_local
    monotonic_floor_price = monotonic_anchor_base_price * (1.0 + amenities_uplift) * (1.0 + review_uplift) + fixed_bonus_local
    property_type_floor_price = compute_property_type_floor_price(
        base_inputs=base_inputs,
        selected_amenities=selected_amenities,
        review_score=review_score,
        city=city,
        property_type=property_type,
        room_type=room_type,
        accommodates=accommodates,
        bedrooms=bedrooms,
    )

    if is_large_entire_property:
        logical_amenity_bonus_local = compute_property_type_amenity_bonus_local(
            city=city,
            property_type=property_type,
            selected_amenities=selected_amenities,
        )
        selected_calibrated_price = max(
            no_amenities_calibrated_price + logical_amenity_bonus_local,
            round(property_type_floor_price, 2),
        )

    final_price_local = max(
        selected_calibrated_price,
        round(amenities_floor_price, 2),
        round(review_floor_price, 2),
        round(monotonic_floor_price, 2),
        round(property_type_floor_price, 2),
    )

    final_price_local = max(final_price_local, 15.0)
    final_price_local = min(final_price_local, 5000.0)

    if enforce_amenity_monotonic and selected_amenities:
        subset_prices = []
        subset_required_prices = []
        normalized_seen = set()

        for amenity_to_remove in selected_amenities:
            reduced_amenities = []
            removed = False

            for amenity in selected_amenities:
                if not removed and amenity == amenity_to_remove:
                    removed = True
                    continue
                reduced_amenities.append(amenity)

            subset_key = tuple(sorted(str(item).strip().lower() for item in reduced_amenities))
            if subset_key in normalized_seen:
                continue
            normalized_seen.add(subset_key)

            reduced_inputs = dict(base_inputs)
            reduced_inputs["amenities"] = reduced_amenities
            _, reduced_final_price = compute_monotonic_price(
                reduced_inputs,
                enforce_amenity_monotonic=False,
            )
            subset_prices.append(reduced_final_price)

            if property_type_key(property_type) in {"house", "villa"} and str(room_type or "").strip().lower() == "entire place":
                min_increment_local = compute_min_amenity_increment_local(
                    city=city,
                    property_type=property_type,
                    amenity=amenity_to_remove,
                )
                required_price = reduced_final_price + min_increment_local
                subset_required_prices.append(required_price)

        if subset_prices:
            final_price_local = max(final_price_local, max(subset_prices))
        if subset_required_prices:
            final_price_local = max(final_price_local, max(subset_required_prices))

    return round(selected_base_price, 2), round(final_price_local, 2)


# =========================================================
# Default form values
# =========================================================
def get_default_form_values():
    default_city = str(DEFAULTS["city"])
    default_neighbourhood = str(DEFAULTS["neighbourhood"])

    auto_lat, auto_lon = get_neighbourhood_lat_lon(default_city, default_neighbourhood)
    auto_distance = get_distance_from_center(default_city, default_neighbourhood)

    return {
        "city": [default_city],
        "neighbourhood": [default_neighbourhood],
        "property_type": [str(DEFAULTS["property_type"])],
        "room_type": [str(DEFAULTS["room_type"])],
        "instant_bookable": [str(DEFAULTS["instant_bookable"])],
        "latitude": [str(auto_lat if auto_lat is not None else DEFAULTS["latitude"])],
        "longitude": [str(auto_lon if auto_lon is not None else DEFAULTS["longitude"])],
        "distance_from_center_km": [str(auto_distance)],
        "accommodates": [str(DEFAULTS["accommodates"])],
        "bedrooms": [str(DEFAULTS["bedrooms"])],
        "minimum_nights": [str(DEFAULTS["minimum_nights"])],
        "maximum_nights": [str(DEFAULTS["maximum_nights"])],
        "review_scores_rating": [str(DEFAULTS["review_scores_rating"])],
        "amenities": [],
    }


# =========================================================
# Validation
# =========================================================
def validate_form(form):
    if model is None:
        return f"Model loading failed: {model_error}"

    city = str(form.get("city", "")).strip()
    neighbourhood = str(form.get("neighbourhood", "")).strip()
    property_type = str(form.get("property_type", "")).strip()
    room_type = str(form.get("room_type", "")).strip()

    accommodates = safe_int(form.get("accommodates"), DEFAULTS["accommodates"])
    bedrooms = safe_float(form.get("bedrooms"), DEFAULTS["bedrooms"])
    minimum_nights = safe_int(form.get("minimum_nights"), DEFAULTS["minimum_nights"])
    maximum_nights = safe_int(form.get("maximum_nights"), DEFAULTS["maximum_nights"])
    review_scores_rating = safe_int(form.get("review_scores_rating"), DEFAULTS["review_scores_rating"])

    if not city:
        return "Please select a city."

    if CITY_OPTIONS and city not in CITY_OPTIONS:
        return "Invalid city selected."

    allowed_neighbourhoods = CITY_NEIGHBOURHOOD_MAP.get(city, [])
    if not neighbourhood:
        return "Please select a neighbourhood."

    if allowed_neighbourhoods and neighbourhood not in allowed_neighbourhoods:
        return "Invalid neighbourhood for selected city."

    if PROPERTY_TYPE_OPTIONS and property_type not in PROPERTY_TYPE_OPTIONS:
        return f"Invalid property type selected. Allowed values: {PROPERTY_TYPE_OPTIONS}"

    if ROOM_TYPE_OPTIONS and room_type not in ROOM_TYPE_OPTIONS:
        return "Invalid room type selected."

    if accommodates < 1 or accommodates > 20:
        return "Accommodates must be between 1 and 20."

    if bedrooms < 0 or bedrooms > 10:
        return "Bedrooms must be between 0 and 10."

    if minimum_nights < 1 or minimum_nights > 30:
        return "Minimum nights must be between 1 and 30."

    if maximum_nights < 1 or maximum_nights > 30:
        return "Maximum nights must be between 1 and 30."

    if maximum_nights < minimum_nights:
        return "Maximum nights must be greater than or equal to minimum nights."

    if review_scores_rating < 0 or review_scores_rating > 100:
        return "Review score rating must be between 0 and 100."

    return None


# =========================================================
# Build row aligned with notebook + metadata
# =========================================================
def build_feature_row(form) -> pd.DataFrame:
    return build_feature_row_from_inputs(build_prediction_inputs(form))


# =========================================================
# Human explanation
# =========================================================
def build_human_explanation(
    form,
    base_price_local: float,
    final_price_local: float,
    final_price_output: float,
    amenities_count: int
) -> list[str]:
    lines = []

    city = str(form.get("city", "")).strip()
    neighbourhood = str(form.get("neighbourhood", "")).strip()
    room_type = str(form.get("room_type", "")).strip()
    property_type = str(form.get("property_type", "")).strip()
    instant_bookable = str(form.get("instant_bookable", "")).strip()
    distance_km = get_distance_from_center(city, neighbourhood)
    review_score = safe_int(form.get("review_scores_rating"), DEFAULTS["review_scores_rating"])
    accommodates = safe_int(form.get("accommodates"), DEFAULTS["accommodates"])
    currency_rate = get_currency_rate(city)
    selected_amenities = form.getlist("amenities")
    amenities_uplift = compute_amenities_uplift(selected_amenities)
    review_uplift = compute_review_uplift(review_score)
    fixed_bonus_local = compute_fixed_amenity_bonus_local(city, selected_amenities)
    fixed_bonus_amenities = get_fixed_bonus_amenities(selected_amenities)

    if distance_km <= 3:
        lines.append("The listing is close to the city center, which usually supports a higher price.")
    elif distance_km >= 10:
        lines.append("The listing is relatively far from the city center, which can reduce the price.")

    if room_type == "Entire place":
        lines.append("Entire place usually increases the nightly price.")
    elif room_type == "Shared room":
        lines.append("Shared room usually lowers the nightly price.")

    if instant_bookable.lower() in {"yes", "t", "true", "1"}:
        lines.append("Instant booking is enabled.")

    lines.append(
        f"The review score of {review_score} added an estimated {(review_uplift * 100):.1f}% quality uplift."
    )

    if amenities_count >= 1:
        lines.append(
            f"{amenities_count} selected amenities contributed an estimated {(amenities_uplift * 100):.1f}% service uplift."
        )

    if fixed_bonus_local > 0 and fixed_bonus_amenities:
        amenity_summary = ", ".join(fixed_bonus_amenities)
        lines.append(
            f"Fixed amenity premium applied for: {amenity_summary}."
        )

    if accommodates >= 2:
        lines.append(f"The model considered capacity for {accommodates} guests.")

    lines.append(f"Property type selected: {property_type}.")
    lines.append(f"Base model estimate in local currency: {base_price_local:,.2f}.")
    lines.append(f"Calibrated estimate in local currency: {final_price_local:,.2f}.")

    if OUTPUT_CURRENCY == "USD_after_prediction":
        lines.append(f"Converted to USD using city exchange rate ({currency_rate:.6f}).")
        lines.append(f"Final estimated nightly price in USD: ${final_price_output:,.2f}.")
    else:
        lines.append(f"Final estimated nightly price: {final_price_output:,.2f}.")

    return lines


# =========================================================
# Route
# =========================================================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    total_price_text = None
    error_message = None
    login_error = None
    form_values = get_default_form_values()
    explanation_lines = None
    shap_explanation_lines = None

    if not is_logged_in():
        if request.method == "POST":
            password = str(request.form.get("password", "")).strip()
            if not login_is_configured():
                login_error = "Login password is not configured in the environment file."
            elif password == LOGIN_PASSWORD:
                session["authenticated"] = True
                session["fresh_login"] = True
                return redirect(url_for("home"))
            else:
                login_error = "Invalid password."

        return render_template(
            "index.html",
            page_title="Holiday Homes",
            login_required=True,
            login_error=login_error,
            model_ready=model is not None,
            model_error=model_error,
            chat_history=[],
            chatbot_ready=chatbot_is_configured(),
            chatbot_welcome_message=get_chatbot_welcome_message(),
            chatbot_welcome_messages=get_chatbot_welcome_messages(),
        )

    if request.method == "POST":
        form_values = request.form.to_dict(flat=False)
        if "amenities" not in form_values:
            form_values["amenities"] = []

        city = str(request.form.get("city", DEFAULTS["city"])).strip()
        neighbourhood = str(request.form.get("neighbourhood", DEFAULTS["neighbourhood"])).strip()

        auto_lat, auto_lon = get_neighbourhood_lat_lon(city, neighbourhood)
        auto_distance = get_distance_from_center(city, neighbourhood)

        form_values["latitude"] = [str(auto_lat if auto_lat is not None else DEFAULTS["latitude"])]
        form_values["longitude"] = [str(auto_lon if auto_lon is not None else DEFAULTS["longitude"])]
        form_values["distance_from_center_km"] = [str(auto_distance)]

        error_message = validate_form(request.form)

        if error_message is None:
            try:
                prediction_inputs = build_prediction_inputs(request.form)
                input_df = build_feature_row_from_inputs(prediction_inputs)
                base_price_local, final_price_local = compute_monotonic_price(prediction_inputs)
                selected_amenities = request.form.getlist("amenities")
                amenities_count = len(selected_amenities)
                review_score = safe_int(
                    request.form.get("review_scores_rating"),
                    DEFAULTS["review_scores_rating"]
                )

                currency_rate = get_currency_rate(city)

                if OUTPUT_CURRENCY == "USD_after_prediction":
                    final_price_output = round(final_price_local * currency_rate, 2)
                    currency_symbol = "$"
                    prediction_text = f"Estimated price per night ({output_currency_label()}): {currency_symbol}{final_price_output:,.2f}"
                else:
                    final_price_output = final_price_local
                    prediction_text = f"Estimated price per night ({output_currency_label()}): {final_price_output:,.2f}"

                minimum_nights = safe_int(
                    request.form.get("minimum_nights"),
                    DEFAULTS["minimum_nights"]
                )
                total_price_output = round(final_price_output * minimum_nights, 2)

                if OUTPUT_CURRENCY == "USD_after_prediction":
                    total_price_text = (
                        f"Estimated total price for minimum stay ({minimum_nights} night(s)) "
                        f"in {output_currency_label()}: ${total_price_output:,.2f}"
                    )
                else:
                    total_price_text = (
                        f"Estimated total price for minimum stay ({minimum_nights} night(s)): "
                        f"{total_price_output:,.2f}"
                    )

                explanation_lines = build_human_explanation(
                    request.form,
                    base_price_local=base_price_local,
                    final_price_local=final_price_local,
                    final_price_output=final_price_output,
                    amenities_count=amenities_count
                )

                set_prediction_snapshot(
                    {
                        "prediction_text": prediction_text,
                        "total_price_text": total_price_text,
                        "base_price_local": round(base_price_local, 2),
                        "final_price_local": round(final_price_local, 2),
                        "final_price_output": round(final_price_output, 2),
                        "minimum_nights": minimum_nights,
                        "city": city,
                        "neighbourhood": neighbourhood,
                        "property_type": str(request.form.get("property_type", "")).strip(),
                        "room_type": str(request.form.get("room_type", "")).strip(),
                        "distance_from_center_km": round(get_distance_from_center(city, neighbourhood), 2),
                        "review_score": review_score,
                        "selected_amenities": selected_amenities,
                    }
                )

                shap_explanation_lines = generate_shap_text(input_df)
                set_project_explanation_snapshot(
                    {
                        "human_explanation_lines": explanation_lines or [],
                        "shap_explanation_lines": shap_explanation_lines or [],
                        "distance_from_center_km": round(get_distance_from_center(city, neighbourhood), 2),
                        "property_type": str(request.form.get("property_type", "")).strip(),
                        "room_type": str(request.form.get("room_type", "")).strip(),
                    }
                )

            except Exception as exc:
                error_message = f"Prediction failed: {pretty_error(exc)}"

    just_logged_in = bool(session.pop("fresh_login", False))

    return render_template(
        "index.html",
        page_title="Holiday Homes",
        login_required=False,
        just_logged_in=just_logged_in,
        prediction_text=prediction_text,
        total_price_text=total_price_text,
        error_message=error_message,
        login_error=login_error,
        city_options=CITY_OPTIONS,
        property_options=PROPERTY_TYPE_OPTIONS,
        room_options=ROOM_TYPE_OPTIONS,
        instant_bookable_options=INSTANT_BOOKABLE_OPTIONS,
        city_neighbourhood_map=CITY_NEIGHBOURHOOD_MAP,
        neighbourhood_coordinates=NEIGHBOURHOOD_COORDINATES,
        city_centers=CITY_CENTERS,
        top_amenities=TOP_AMENITIES,
        defaults=DEFAULTS,
        form_values=form_values,
        model_ready=model is not None,
        model_error=model_error,
        explanation_lines=explanation_lines,
        shap_explanation_lines=shap_explanation_lines,
        output_currency=OUTPUT_CURRENCY,
        prediction_target=PREDICTION_TARGET,
        chat_history=get_chat_history(),
        chatbot_ready=chatbot_is_configured(),
        chatbot_welcome_message=get_chatbot_welcome_message(),
        chatbot_welcome_messages=get_chatbot_welcome_messages(),
    )


@app.route("/chatbot", methods=["POST"])
def chatbot():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()
    if not user_message:
        return jsonify({"ok": False, "error": "Message is required."}), 400

    listing_snapshot = payload.get("listing_snapshot")
    if not isinstance(listing_snapshot, dict):
        listing_snapshot = {}

    history = get_chat_history()
    assistant_reply = request_openai_chat_response(
        user_message=user_message,
        history=history,
        listing_snapshot=listing_snapshot,
    )

    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    set_chat_history(updated_history)

    return jsonify(
        {
            "ok": True,
            "reply": assistant_reply,
            "history": get_chat_history(),
            "chatbot_ready": chatbot_is_configured(),
        }
    )


@app.route("/chatbot/reset", methods=["POST"])
def chatbot_reset():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    clear_chat_history()
    session.pop("preferred_chat_language", None)
    session.modified = True

    return jsonify({"ok": True})


@app.route("/logout", methods=["POST"])
def logout():
    clear_chat_history()
    session.pop("prediction_snapshot", None)
    session.pop("project_explanation_snapshot", None)
    session.pop("preferred_chat_language", None)
    session.clear()
    return redirect(url_for("home"))


@app.route("/session-exit", methods=["POST"])
def session_exit():
    clear_chat_history()
    session.pop("prediction_snapshot", None)
    session.pop("project_explanation_snapshot", None)
    session.pop("preferred_chat_language", None)
    session.clear()
    return ("", 204)


@app.route("/force-relogin", methods=["GET"])
def force_relogin():
    clear_chat_history()
    session.pop("prediction_snapshot", None)
    session.pop("project_explanation_snapshot", None)
    session.pop("preferred_chat_language", None)
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    app.run(host=host, port=port, debug=debug_mode)
