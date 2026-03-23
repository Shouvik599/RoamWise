# main.py

import os
import json
import logging
import base64
from io import BytesIO
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# ──────────────────────────────────────────────
# Configuration & Initialization
# ──────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")

CONTINENTS = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

# Global NVIDIA client
client: Optional[OpenAI] = None
nvidia_configured: bool = False

# In-memory chat sessions  (session_id → list[dict])
chat_sessions: dict[str, list[dict]] = {}


def _init_nvidia_client():
    """Initialise the NVIDIA OpenAI-compatible client once."""
    global client, nvidia_configured
    if NVIDIA_API_KEY:
        try:
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=NVIDIA_API_KEY,
            )
            nvidia_configured = True
            logger.info("NVIDIA API configured successfully.")
        except Exception as e:
            logger.error("Error configuring NVIDIA API: %s", e)
    else:
        logger.warning("No NVIDIA API key found. AI features disabled.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    _init_nvidia_client()
    yield
    # cleanup if needed


app = FastAPI(
    title="RoamWise API",
    description="AI-powered travel companion – REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at root (add BEFORE the existing @app.get("/") route)
# Option 1: Serve from a "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse, tags=["UI"], include_in_schema=False)
def serve_ui():
    return FileResponse("static/index.html")



# ──────────────────────────────────────────────
# Pydantic Models – Requests
# ──────────────────────────────────────────────

class TravelPlanRequest(BaseModel):
    country: str = Field(..., examples=["Japan"])


class ItineraryRequest(BaseModel):
    country: str = Field(..., examples=["Japan"])
    num_days: int = Field(7, ge=1, le=30)
    travel_style: str = Field("cultural", examples=["adventure", "relaxation", "cultural", "family", "romantic", "solo backpacking"])
    budget_level: str = Field("moderate", examples=["budget", "moderate", "luxury"])


class BudgetRequest(BaseModel):
    country: str
    num_days: int = Field(7, ge=1, le=60)
    travel_style: str = Field("mid-range")
    num_travelers: int = Field(2, ge=1, le=10)


class AdvisoryRequest(BaseModel):
    country: str
    nationality: str = Field("Indian")


class ChatRequest(BaseModel):
    country: str
    message: str
    session_id: str = Field("default", description="Unique session id for conversation continuity")


class PackingListRequest(BaseModel):
    country: str
    num_days: int = Field(7, ge=1, le=60)
    travel_style: str = Field("leisure")
    travel_dates: Optional[str] = None


class CompareRequest(BaseModel):
    countries: list[str] = Field(..., min_length=2, max_length=4)
    criteria: list[str] = Field(
        default=["Cost of Living", "Safety", "Food Scene"],
        examples=[["Cost of Living", "Safety", "Weather", "Food Scene", "Nightlife", "Cultural Attractions"]],
    )


class PhrasesRequest(BaseModel):
    country: str


class WeatherActivitiesRequest(BaseModel):
    country: str
    month: str = Field(..., examples=["January"])


# ──────────────────────────────────────────────
# Pydantic Models – Responses
# ──────────────────────────────────────────────

class HealthComponent(BaseModel):
    status: str
    details: str
    working: bool


class HealthResponse(BaseModel):
    nvidia_api: HealthComponent
    exchange_api: HealthComponent
    rest_countries_api: HealthComponent
    nvidia_client: HealthComponent
    timestamp: str
    all_ok: bool


class CountryInfoResponse(BaseModel):
    capital: Optional[str] = None
    currency_code: Optional[str] = None
    currency_name: Optional[str] = None
    error: Optional[str] = None


class ConversionResponse(BaseModel):
    from_currency: Optional[str] = Field(None, alias="from")
    to_currency: str = Field("INR", alias="to")
    rate: Optional[float] = None
    error: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

class ChatResponse(BaseModel):
    session_id: str
    response: Optional[str] = None
    history_length: int = 0
    error: Optional[str] = None


class GenericAIResponse(BaseModel):
    """Wraps any JSON blob returned by the AI."""
    data: Optional[dict | list] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────
import re
def clean_json_response(text: str) -> str:
    # Find the first '{' and last '}' to strip any surrounding text or markdown
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()


def _require_nvidia():
    if not nvidia_configured or client is None:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA API client is not configured. Set NVIDIA_API_KEY.",
        )


# ──────────────────────────────────────────────
# Core API helpers (external services)
# ──────────────────────────────────────────────

def fetch_country_info(country: str) -> dict:
    try:
        rc = requests.get(
            f"https://restcountries.com/v3.1/name/{country}",
            params={"fullText": "true"},
            timeout=8,
        )
        if rc.status_code != 200:
            return {"error": f"REST Countries API returned {rc.status_code}"}
        cdata = rc.json()[0]
        capital = cdata.get("capital", ["Unknown"])[0]
        currencies = cdata.get("currencies", {})
        if currencies:
            currency_code = list(currencies.keys())[0]
            currency_name = currencies[currency_code].get("name", "")
        else:
            currency_code = None
            currency_name = "None"
        return {
            "capital": capital,
            "currency_code": currency_code,
            "currency_name": currency_name,
            "error": None,
        }
    except Exception as e:
        logger.error("fetch_country_info(%s): %s", country, e)
        return {"error": str(e)}


def fetch_currency_conversion(currency_code: str) -> dict:
    info: dict = {"from": currency_code, "to": "INR", "rate": None, "error": None}
    if not currency_code:
        info["error"] = "No currency code provided."
        return info
    try:
        params: dict = {"from": currency_code, "to": "INR", "amount": 1}
        if EXCHANGE_API_KEY:
            params["access_key"] = EXCHANGE_API_KEY
        resp = requests.get(
            "https://api.exchangerate.host/convert",
            params=params,
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("success") is False:
            info["error"] = str(data.get("error", "Unknown exchange API error"))
            return info
        rate = None
        if isinstance(data, dict):
            if data.get("result") is not None:
                rate = data["result"]
            elif data.get("quotes"):
                for k, v in data["quotes"].items():
                    if k.endswith("INR"):
                        rate = v
                        break
            elif data.get("rates"):
                rate = data["rates"].get("INR")
        info["rate"] = rate
        if rate is None:
            info["error"] = "API returned no exchange rate."
    except Exception as e:
        info["error"] = f"Conversion failed: {e}"
    return info


def fetch_countries_for_continent(continent: str) -> list[str]:
    try:
        resp = requests.get(
            f"https://restcountries.com/v3.1/region/{continent}",
            params={"fields": "name"},
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return sorted(c.get("name", {}).get("common", "") for c in data)
    except Exception as e:
        logger.error("fetch_countries_for_continent: %s", e)
        return []


# ──────────────────────────────────────────────
# NVIDIA LLM helpers
# ──────────────────────────────────────────────

def call_nvidia_llm(
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    _require_nvidia()
    completion = client.chat.completions.create(  # type: ignore[union-attr]
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )
    return completion.choices[0].message.content


def call_nvidia_vision_llm(
    prompt: str,
    image_bytes: bytes,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    _require_nvidia()
    img_b64 = base64.b64encode(image_bytes).decode()
    image_url = f"data:image/jpeg;base64,{img_b64}"
    completion = client.chat.completions.create(  # type: ignore[union-attr]
        model="meta/llama-3.2-90b-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


# ──────────────────────────────────────────────
# Prompt-based generation functions
# These functions construct specific prompts for different features and call the NVIDIA LLM.
# ──────────────────────────────────────────────

def _generate_travel_plan(country: str) -> dict:
    prompt = (
        f"Generate a travel guide for {country} as a single, valid JSON object. "
        "Do not include any introductory text, closing text, or markdown formatting like ```json. "
        "The JSON object should have the following keys:\n"
        "1. 'cities': An array of 4-5 must-visit city objects. Each city object should have:\n"
        "   - 'name': The city's name (string).\n"
        "   - 'reason': A brief, compelling reason to visit (string).\n"
        "2. 'activities': An object where each key is a city name from the 'cities' list. "
        "The value for each city key should be an array of 3 activity objects. Each activity object should have:\n"
        "   - 'name': The activity name (string).\n"
        "   - 'description': A short description of the activity (string).\n"
        "   - 'price_inr': An estimated price in Indian Rupees (INR) as an integer. Use 0 for free activities.\n"
        "3. 'foods': An array of 5 must-try food objects. Each food object should have:\n"
        "   - 'name': The food's name (string).\n"
        "   - 'description': A brief description (string).\n"
        "   - 'image_query': A search query for finding an image of this food (string).\n"
        "4. 'tips': An array of 5-6 essential travel tip strings for visitors.\n"
    )
    raw = call_nvidia_llm(prompt, max_tokens=4096)
    logging.info("Travel plan prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _generate_itinerary(country: str, num_days: int, style: str, budget: str) -> dict:
    prompt = f"""Create a detailed {num_days}-day travel itinerary for {country}.

Travel Style: {style}
Budget Level: {budget}

Return a JSON object with this structure:
{{
    "itinerary": [
        {{
            "day": 1,
            "title": "Day title/theme",
            "city": "City name",
            "morning": {{"activity":"...","description":"...","duration":"2 hours","cost_inr":500}},
            "afternoon": {{"activity":"...","description":"...","duration":"3 hours","cost_inr":1000}},
            "evening": {{"activity":"...","description":"...","duration":"2 hours","cost_inr":800}},
            "meals": {{"breakfast":"...","lunch":"...","dinner":"..."}},
            "accommodation": "Hotel/area suggestion",
            "travel_tip": "Specific tip for this day"
        }}
    ],
    "total_estimated_cost_inr": 50000,
    "packing_essentials": ["item1","item2"],
    "best_time_to_visit": "Month or season"
}}
Only output valid JSON, no markdown."""
    raw = call_nvidia_llm(prompt, max_tokens=4096)
    logging.info("Itinerary prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _generate_budget(country: str, days: int, style: str, travelers: int) -> dict:
    prompt = f"""Create a detailed travel budget for {travelers} traveler(s)
visiting {country} for {days} days.  Travel style: {style}

Return JSON with all costs in INR:
{{
    "summary": {{"total_per_person":0,"total_trip_cost":0,"daily_average_per_person":0}},
    "breakdown": {{
        "accommodation": {{"total":0,"daily_rate":0,"hotel_type":"","tips":""}},
        "food": {{"total":0,"daily_rate":0,"breakdown":{{"breakfast":0,"lunch":0,"dinner":0,"snacks":0}},"tips":""}},
        "transportation": {{"total":0,"local_transport_daily":0,"intercity_estimate":0,"tips":""}},
        "activities": {{"total":0,"popular_activities":[{{"name":"","cost":0}}],"tips":""}},
        "miscellaneous": {{"total":0,"includes":["Tips","Souvenirs","Emergency fund"]}}
    }},
    "money_saving_tips": [],
    "hidden_costs_warning": [],
    "best_value_period": ""
}}
Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=3000)
    logging.info("Budget prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _get_advisory(country: str, nationality: str) -> dict:
    prompt = f"""Provide comprehensive travel advisory for {nationality} travelers visiting {country}.

Return JSON:
{{
    "safety_rating":"Safe/Moderate Caution/Exercise Caution/Reconsider Travel",
    "safety_score":8,
    "visa_requirements":{{
        "visa_required":true,"visa_type":"","duration_allowed":"","processing_time":"",
        "approximate_cost_inr":0,"documents_required":[],"apply_link":""
    }},
    "health_advisories":[{{"type":"","details":"","mandatory":false}}],
    "safety_tips":[{{"category":"","tips":[]}}],
    "areas_to_avoid":[],
    "emergency_numbers":{{"police":"","ambulance":"","tourist_helpline":"","indian_embassy":""}},
    "local_laws_to_know":[],
    "scams_to_watch":[]
}}
Only output valid JSON."""
    raw = call_nvidia_llm(prompt)
    logging.info("Advisory prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _identify_landmark(image_bytes: bytes, country_hint: Optional[str]) -> dict:
    hint = f"The image is likely from {country_hint}" if country_hint else ""
    prompt = f"""Analyze this image and identify any landmarks, tourist attractions,
or notable locations visible.
{hint}

Return JSON:
{{
    "identified":true,
    "landmark_name":"Name of the landmark",
    "location":"City, Country",
    "description":"Brief history and significance",
    "visitor_info":{{"best_time_to_visit":"","typical_visit_duration":"","entry_fee_inr":0,"tips":[]}},
    "nearby_attractions":[],
    "photo_tips":""
}}
If no landmark is identifiable, set identified to false and provide a general description.
Only output valid JSON."""
    raw = call_nvidia_vision_llm(prompt, image_bytes)
    logging.info("Vision prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _generate_packing_list(country: str, days: int, style: str, dates: Optional[str]) -> dict:
    date_ctx = f"Travel dates: {dates}" if dates else "General packing advice"
    prompt = f"""Generate a comprehensive packing list for a {days}-day trip to {country}.
Travel style: {style}
{date_ctx}

Return JSON:
{{
    "weather_summary":"Expected weather conditions",
    "categories":{{
        "clothing":[{{"item":"","quantity":2,"notes":""}}],
        "toiletries":[...],
        "electronics":[...],
        "documents":[...],
        "health_safety":[...],
        "accessories":[...],
        "country_specific":[...]
    }},
    "pro_tips":[],
    "items_to_avoid":[]
}}
Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=2048)
    logging.info("Packing list prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _compare_destinations(countries: list[str], criteria: list[str]) -> dict:
    prompt = f"""Compare these travel destinations: {', '.join(countries)}
Compare based on these criteria: {', '.join(criteria)}

Return ONLY a valid JSON object. 
    IMPORTANT: Every nested object must be closed with a curly brace '}}'. 
    Do NOT use square brackets '[]' unless you are defining an array.{{
    "comparison_table":{{
        "criteria_name":{{
            "Country1":{{"score":8,"details":"explanation"}}
        }}
    }},
    "overall_winner":"Country name",
    "winner_reason":"Why this country wins overall",
    "best_for":{{"budget_travelers":"","families":"","adventure_seekers":"","foodies":"","culture_lovers":""}},
    "summary":"Brief overall comparison summary"
}}
Score each country 1-10. Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=3000)
    logging.info("Comparison prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _get_phrases(country: str) -> dict:
    prompt = f"""For travelers visiting {country}, provide essential phrases.

Return JSON:
{{
    "primary_language":"Language name",
    "greeting_culture":"Brief note on greeting customs",
    "categories":{{
        "greetings":[{{"english":"Hello","local":"translation","pronunciation":"phonetic"}}],
        "directions":[...],
        "dining":[...],
        "emergencies":[...]
    }},
    "cultural_notes":[],
    "common_mistakes":[]
}}
Include 5-8 phrases per category. Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=2048)
    logging.info("Phrases prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


def _get_weather_activities(country: str, month: str) -> dict:
    prompt = f"""For {country} in {month}, provide weather-based activity recommendations.

Return JSON:
{{
    "weather_summary":{{"temperature_range":"","rainfall":"","humidity":"","general_conditions":""}},
    "is_peak_season":true,
    "tourist_crowd_level":"High/Medium/Low",
    "recommended_activities":[{{"activity":"","why_this_month":"","best_locations":[],"what_to_pack":[]}}],
    "activities_to_avoid":[{{"activity":"","reason":""}}],
    "regional_differences":[{{"region":"","weather":"","best_activities":[]}}],
    "festivals_events":[{{"name":"","date":"","location":"","description":""}}],
    "packing_for_weather":[]
}}
Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=2048)
    logging.info("Weather activities prompt response: %s", clean_json_response(raw))
    return json.loads(clean_json_response(raw))


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

# ---------- Health ----------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Run all system health checks."""

    def _check_nvidia_api() -> dict:
        if NVIDIA_API_KEY:
            return {"status": "✅ OK", "details": "NVIDIA API key is configured", "working": True}
        return {"status": "❌ Error", "details": "NVIDIA API key not found", "working": False}

    def _check_exchange_api() -> dict:
        try:
            r = requests.get("https://api.exchangerate.host/latest", params={"base": "USD"}, timeout=5)
            if r.status_code == 200:
                return {"status": "✅ OK", "details": "Exchange API is accessible", "working": True}
            return {"status": "⚠️ Warning", "details": f"Status {r.status_code}", "working": False}
        except Exception:
            return {"status": "❌ Error", "details": "Exchange API unreachable", "working": False}

    def _check_countries_api() -> dict:
        try:
            r = requests.get("https://restcountries.com/v3.1/all", timeout=5)
            if r.status_code == 200:
                return {"status": "✅ OK", "details": "REST Countries API accessible", "working": True}
            return {"status": "⚠️ Warning", "details": f"Status {r.status_code}", "working": False}
        except Exception:
            return {"status": "❌ Error", "details": "REST Countries API unreachable", "working": False}

    def _check_nvidia_client() -> dict:
        if nvidia_configured and client:
            return {"status": "✅ OK", "details": "NVIDIA client initialized", "working": True}
        return {"status": "⚠️ Warning", "details": "NVIDIA client not initialized", "working": False}

    nv = _check_nvidia_api()
    ex = _check_exchange_api()
    rc = _check_countries_api()
    nc = _check_nvidia_client()

    return HealthResponse(
        nvidia_api=HealthComponent(**nv),
        exchange_api=HealthComponent(**ex),
        rest_countries_api=HealthComponent(**rc),
        nvidia_client=HealthComponent(**nc),
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        all_ok=all(c["working"] for c in [nv, ex, rc, nc]),
    )


# ---------- Geography ----------

@app.get("/continents", tags=["Geography"])
def list_continents():
    """Return the list of supported continents."""
    return {"continents": CONTINENTS}


@app.get("/countries/{continent}", tags=["Geography"])
def list_countries(continent: str):
    """Return countries for a continent."""
    if continent not in CONTINENTS:
        raise HTTPException(status_code=400, detail=f"Invalid continent. Choose from {CONTINENTS}")
    countries = fetch_countries_for_continent(continent)
    return {"continent": continent, "countries": countries, "count": len(countries)}


# ---------- Country Info ----------

@app.get("/country/{country}/info", response_model=CountryInfoResponse, tags=["Country Info"])
def country_info(country: str):
    """Capital, currency code & name for a country."""
    data = fetch_country_info(country)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    return CountryInfoResponse(**data)


@app.get("/currency/convert/{currency_code}", tags=["Country Info"])
def currency_to_inr(currency_code: str):
    """Live exchange rate from *currency_code* → INR."""
    data = fetch_currency_conversion(currency_code)
    if data.get("error") and data.get("rate") is None:
        raise HTTPException(status_code=502, detail=data["error"])
    return data


# ---------- AI Travel Features ----------

@app.post("/travel/plan", response_model=GenericAIResponse, tags=["AI Travel"])
def travel_plan(req: TravelPlanRequest):
    """Generate a comprehensive travel plan (cities, activities, foods, tips)."""
    _require_nvidia()
    try:
        data = _generate_travel_plan(req.country)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/itinerary", response_model=GenericAIResponse, tags=["AI Travel"])
def travel_itinerary(req: ItineraryRequest):
    """Generate a day-by-day itinerary."""
    _require_nvidia()
    try:
        data = _generate_itinerary(req.country, req.num_days, req.travel_style, req.budget_level)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/budget", response_model=GenericAIResponse, tags=["AI Travel"])
def travel_budget(req: BudgetRequest):
    """Generate a detailed trip budget breakdown."""
    _require_nvidia()
    try:
        data = _generate_budget(req.country, req.num_days, req.travel_style, req.num_travelers)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/advisory", response_model=GenericAIResponse, tags=["AI Travel"])
def travel_advisory(req: AdvisoryRequest):
    """Safety info, visa requirements & health advisories."""
    _require_nvidia()
    try:
        data = _get_advisory(req.country, req.nationality)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/chat", response_model=ChatResponse, tags=["AI Travel"])
def travel_chat(req: ChatRequest):
    """Conversational AI assistant for a destination."""
    _require_nvidia()

    history = chat_sessions.setdefault(req.session_id, [])
    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-5:]
    )

    prompt = f"""You are an expert travel assistant for {req.country}.
Answer the user's question helpfully and concisely.

Previous conversation:
{history_text}

User's new question: {req.message}

Provide a helpful, accurate response. If you're unsure about specific current
information (prices, hours), mention that the user should verify locally."""

    try:
        answer = call_nvidia_llm(prompt)
        history.append({"user": req.message, "assistant": answer})
        return ChatResponse(session_id=req.session_id, response=answer, history_length=len(history))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/travel/chat/{session_id}", tags=["AI Travel"])
def clear_chat(session_id: str):
    """Clear chat history for a session."""
    chat_sessions.pop(session_id, None)
    return {"message": f"Chat session '{session_id}' cleared."}


@app.post("/travel/packing-list", response_model=GenericAIResponse, tags=["AI Travel"])
def packing_list(req: PackingListRequest):
    """Generate a personalised packing list."""
    _require_nvidia()
    try:
        data = _generate_packing_list(req.country, req.num_days, req.travel_style, req.travel_dates)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/compare", response_model=GenericAIResponse, tags=["AI Travel"])
def compare_destinations_endpoint(req: CompareRequest):
    """Compare 2-4 destinations across chosen criteria."""
    _require_nvidia()
    try:
        data = _compare_destinations(req.countries, req.criteria)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/phrases", response_model=GenericAIResponse, tags=["AI Travel"])
def essential_phrases(req: PhrasesRequest):
    """Essential travel phrases in the local language."""
    _require_nvidia()
    try:
        data = _get_phrases(req.country)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/travel/weather-activities", response_model=GenericAIResponse, tags=["AI Travel"])
def weather_activities(req: WeatherActivitiesRequest):
    """Weather-based activity recommendations for a given month."""
    _require_nvidia()
    try:
        data = _get_weather_activities(req.country, req.month)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Landmark Recognition (image upload) ----------

@app.post("/landmark/identify", response_model=GenericAIResponse, tags=["AI Vision"])
async def identify_landmark_endpoint(
    image: UploadFile = File(..., description="Image file (jpg/jpeg/png/webp)"),
    country_hint: Optional[str] = Form(None, description="Optional country hint"),
):
    """Upload an image to identify landmarks via NVIDIA Vision model."""
    _require_nvidia()

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.content_type}")

    image_bytes = await image.read()

    # Ensure it's a valid image and convert to JPEG bytes
    try:
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    try:
        data = _identify_landmark(jpeg_bytes, country_hint)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Composite endpoint (mirrors the Streamlit "Get Travel Plan" button) ----------

@app.get("/travel/full/{country}", tags=["AI Travel"])
def full_travel_info(country: str):
    """
    All-in-one: country info + currency conversion + AI travel plan.
    Mirrors the Streamlit 'Get Travel Plan' button.
    """
    info = fetch_country_info(country)
    conversion = (
        fetch_currency_conversion(info["currency_code"])
        if info.get("currency_code")
        else {"from": None, "to": "INR", "rate": None, "error": "No currency code"}
    )

    plan: dict | None = None
    plan_error: str | None = None
    if nvidia_configured:
        try:
            plan = _generate_travel_plan(country)
        except Exception as e:
            plan_error = str(e)

    return {
        "country": country,
        "info": info,
        "conversion": conversion,
        "travel_plan": plan,
        "travel_plan_error": plan_error,
    }


# ──────────────────────────────────────────────
# Root
# ──────────────────────────────────────────────

# Move the old root endpoint to /api
@app.get("/api", tags=["System"])
def api_root():
    return {
        "app": "RoamWise API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=True)