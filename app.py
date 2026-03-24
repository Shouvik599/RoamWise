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

# Indian states and union territories with major cities
INDIAN_STATES = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Tirupati", "Guntur", "Nellore", "Kakinada"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Ziro", "Pasighat", "Bomdila"],
    "Assam": ["Guwahati", "Jorhat", "Silchar", "Dibrugarh", "Tezpur", "Nagaon"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Nalanda", "Rajgir"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Jagdalpur", "Korba", "Durg"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", "Kutch"],
    "Haryana": ["Gurugram", "Faridabad", "Chandigarh", "Karnal", "Panipat", "Kurukshetra"],
    "Himachal Pradesh": ["Shimla", "Manali", "Dharamshala", "Kullu", "Dalhousie", "Kasauli"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Deoghar", "Hazaribagh"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hampi", "Mangaluru", "Coorg", "Hubli"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Munnar", "Alleppey", "Kozhikode", "Wayanad"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Khajuraho", "Ujjain", "Gwalior", "Orchha"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Aurangabad", "Nashik", "Lonavala", "Mahabaleshwar"],
    "Manipur": ["Imphal", "Loktak Lake", "Churachandpur", "Ukhrul"],
    "Meghalaya": ["Shillong", "Cherrapunji", "Dawki", "Tura", "Mawlynnong"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai", "Serchhip"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Mon", "Zunheboto"],
    "Odisha": ["Bhubaneswar", "Puri", "Konark", "Cuttack", "Chilika", "Gopalpur"],
    "Punjab": ["Amritsar", "Chandigarh", "Ludhiana", "Jalandhar", "Patiala"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur", "Jaisalmer", "Pushkar", "Mount Abu", "Bikaner"],
    "Sikkim": ["Gangtok", "Pelling", "Namchi", "Lachung", "Ravangla", "Yuksom"],
    "Tamil Nadu": ["Chennai", "Madurai", "Ooty", "Kodaikanal", "Pondicherry", "Mahabalipuram", "Rameswaram"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
    "Tripura": ["Agartala", "Udaipur", "Neermahal", "Unakoti"],
    "Uttar Pradesh": ["Agra", "Varanasi", "Lucknow", "Mathura", "Allahabad", "Ayodhya", "Jhansi"],
    "Uttarakhand": ["Dehradun", "Rishikesh", "Haridwar", "Nainital", "Mussoorie", "Auli", "Jim Corbett"],
    "West Bengal": ["Kolkata", "Darjeeling", "Sundarbans", "Siliguri", "Shantiniketan", "Digha"],
    "Andaman and Nicobar Islands": ["Port Blair", "Havelock Island", "Neil Island", "Baratang"],
    "Chandigarh": ["Chandigarh"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Daman", "Diu", "Silvassa"],
    "Delhi": ["New Delhi", "Old Delhi", "Mehrauli", "Chandni Chowk"],
    "Jammu and Kashmir": ["Srinagar", "Gulmarg", "Pahalgam", "Jammu", "Sonamarg", "Leh"],
    "Ladakh": ["Leh", "Nubra Valley", "Pangong Lake", "Kargil", "Tso Moriri"],
    "Lakshadweep": ["Kavaratti", "Agatti", "Bangaram", "Minicoy"],
    "Puducherry": ["Pondicherry", "Karaikal", "Mahe", "Yanam"],
}

INDIAN_REGIONS = {
    "North India": ["Delhi", "Uttar Pradesh", "Uttarakhand", "Himachal Pradesh", "Punjab", "Haryana",
                     "Jammu and Kashmir", "Ladakh", "Chandigarh"],
    "South India": ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana", "Puducherry",
                     "Lakshadweep"],
    "East India": ["West Bengal", "Odisha", "Bihar", "Jharkhand", "Andaman and Nicobar Islands"],
    "West India": ["Maharashtra", "Gujarat", "Goa", "Rajasthan",
                    "Dadra and Nagar Haveli and Daman and Diu"],
    "Central India": ["Madhya Pradesh", "Chhattisgarh"],
    "Northeast India": ["Assam", "Meghalaya", "Arunachal Pradesh", "Nagaland", "Manipur", "Mizoram",
                         "Tripura", "Sikkim"],
}

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
    travel_style: str = Field("cultural")
    budget_level: str = Field("moderate")


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
    session_id: str = Field("default")


class PackingListRequest(BaseModel):
    country: str
    num_days: int = Field(7, ge=1, le=60)
    travel_style: str = Field("leisure")
    travel_dates: Optional[str] = None


class CompareRequest(BaseModel):
    countries: list[str] = Field(..., min_length=2, max_length=4)
    criteria: list[str] = Field(default=["Cost of Living", "Safety", "Food Scene"])


class PhrasesRequest(BaseModel):
    country: str


class WeatherActivitiesRequest(BaseModel):
    country: str
    month: str = Field(..., examples=["January"])


# ── Indian Travel Models ──

class IndiaTravelPlanRequest(BaseModel):
    state: str = Field(..., examples=["Rajasthan"])
    city: Optional[str] = Field(None, examples=["Jaipur"])


class IndiaItineraryRequest(BaseModel):
    state: str = Field(..., examples=["Rajasthan"])
    city: Optional[str] = Field(None, examples=["Jaipur"])
    num_days: int = Field(7, ge=1, le=30)
    travel_style: str = Field("cultural")
    budget_level: str = Field("moderate")


class IndiaBudgetRequest(BaseModel):
    state: str
    city: Optional[str] = None
    num_days: int = Field(7, ge=1, le=60)
    travel_style: str = Field("mid-range")
    num_travelers: int = Field(2, ge=1, le=10)


class IndiaAdvisoryRequest(BaseModel):
    state: str
    city: Optional[str] = None


class IndiaChatRequest(BaseModel):
    state: str
    city: Optional[str] = None
    message: str
    session_id: str = Field("default")


class IndiaPackingListRequest(BaseModel):
    state: str
    city: Optional[str] = None
    num_days: int = Field(7, ge=1, le=60)
    travel_style: str = Field("leisure")
    travel_dates: Optional[str] = None


class IndiaCompareRequest(BaseModel):
    destinations: list[str] = Field(..., min_length=2, max_length=4)
    criteria: list[str] = Field(default=["Cost of Living", "Safety", "Food Scene"])


class IndiaPhrasesRequest(BaseModel):
    state: str


class IndiaWeatherActivitiesRequest(BaseModel):
    state: str
    city: Optional[str] = None
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
    data: Optional[dict | list] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────
import re

def clean_json_response(text: str) -> str:
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


def _india_destination_str(state: str, city: Optional[str] = None) -> str:
    """Build a human-readable destination string.
    
    When a city is provided the AI should focus on that city;
    when only a state is given the guide should cover the whole state.
    """
    if city:
        return f"the city of {city} in {state}, India"
    return f"the state of {state}, India"


def _india_destination_short(state: str, city: Optional[str] = None) -> str:
    """Short label used in UI-facing responses."""
    if city:
        return f"{city}, {state}"
    return state


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
    completion = client.chat.completions.create(
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
    completion = client.chat.completions.create(
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
# Prompt-based generation functions (International)
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
    return json.loads(clean_json_response(raw))


# ──────────────────────────────────────────────
# Prompt-based generation functions (India Domestic)
# ──────────────────────────────────────────────

def _generate_india_travel_plan(state: str, city: Optional[str]) -> dict:
    destination = _india_destination_str(state, city)
    
    if city:
        # City-focused plan
        places_instruction = (
            f"Focus specifically on {city}. "
            "1. 'places': An array of 4-5 must-visit places/attractions within or very near this city. "
        )
        foods_instruction = (
            f"3. 'foods': An array of 5 must-try local food items specifically famous in {city}. "
        )
        transport_instruction = (
            f"7. 'local_transport': An array of local transport options available in {city} (strings).\n"
        )
        reach_instruction = (
            f"5. 'how_to_reach': An object describing how to reach {city} specifically, "
            "with keys 'by_air', 'by_train', 'by_road' each being a brief description string.\n"
        )
    else:
        # State-wide plan
        places_instruction = (
            "1. 'places': An array of 4-5 must-visit places/attractions across the state. "
        )
        foods_instruction = (
            "3. 'foods': An array of 5 must-try local food items famous in this state. "
        )
        transport_instruction = (
            "7. 'local_transport': An array of local transport options available in the state (strings).\n"
        )
        reach_instruction = (
            "5. 'how_to_reach': An object describing how to reach this state, "
            "with keys 'by_air', 'by_train', 'by_road' each being a brief description string.\n"
        )

    prompt = (
        f"Generate a domestic travel guide for {destination} as a single, valid JSON object. "
        "This is for an Indian traveler exploring within India. "
        "Do not include any introductory text, closing text, or markdown formatting like ```json. "
        "The JSON object should have the following keys:\n"
        f"{places_instruction}"
        "Each place object should have:\n"
        "   - 'name': The place name (string).\n"
        "   - 'type': Type like 'Temple', 'Fort', 'Beach', 'Hill Station', 'Market', 'Museum', 'Nature' etc. (string).\n"
        "   - 'reason': A brief, compelling reason to visit (string).\n"
        "2. 'activities': An object where each key is a place name from the 'places' list. "
        "The value for each place key should be an array of 3 activity objects. Each activity object should have:\n"
        "   - 'name': The activity name (string).\n"
        "   - 'description': A short description of the activity (string).\n"
        "   - 'price_inr': An estimated price in Indian Rupees (INR) as an integer. Use 0 for free activities.\n"
        f"{foods_instruction}"
        "Each food object should have:\n"
        "   - 'name': The food's name (string).\n"
        "   - 'description': A brief description (string).\n"
        "   - 'where_to_try': A specific place or area to try this food (string).\n"
        "   - 'price_range_inr': Approximate price range as a string like '50-150' (string).\n"
        "4. 'tips': An array of 5-6 essential travel tip strings for visitors.\n"
        f"{reach_instruction}"
        "6. 'best_time_to_visit': A string describing the best months/season to visit.\n"
        f"{transport_instruction}"
    )
    raw = call_nvidia_llm(prompt, max_tokens=4096)
    return json.loads(clean_json_response(raw))


def _generate_india_itinerary(state: str, city: Optional[str], num_days: int, style: str, budget: str) -> dict:
    destination = _india_destination_str(state, city)
    
    if city:
        scope_instruction = (
            f"IMPORTANT: Focus the entire itinerary specifically on {city} and its immediate surroundings. "
            f"All activities, meals, and accommodation should be in or around {city}. "
            f"Do NOT spread the itinerary across the whole state of {state}. "
            f"Use specific neighborhoods, areas, and localities within {city}."
        )
        area_label = f"Area/neighborhood within {city}"
    else:
        scope_instruction = (
            f"Cover multiple cities and regions across {state}. "
            f"Include travel between different destinations within the state."
        )
        area_label = "City/town within the state"

    prompt = f"""Create a detailed {num_days}-day domestic travel itinerary for {destination}.
This is for an Indian traveler. All costs in INR.

{scope_instruction}

Travel Style: {style}
Budget Level: {budget}

Return a JSON object with this structure:
{{
    "itinerary": [
        {{
            "day": 1,
            "title": "Day title/theme",
            "area": "{area_label}",
            "morning": {{"activity":"...","description":"...","duration":"2 hours","cost_inr":500}},
            "afternoon": {{"activity":"...","description":"...","duration":"3 hours","cost_inr":1000}},
            "evening": {{"activity":"...","description":"...","duration":"2 hours","cost_inr":800}},
            "meals": {{"breakfast":"local dish suggestion","lunch":"local dish suggestion","dinner":"local dish suggestion"}},
            "accommodation": "Hotel/homestay suggestion",
            "travel_tip": "Specific tip for this day"
        }}
    ],
    "total_estimated_cost_inr": 15000,
    "packing_essentials": ["item1","item2"],
    "best_time_to_visit": "Month or season",
    "local_transport_tips": "How to get around locally"
}}
Only output valid JSON, no markdown."""
    raw = call_nvidia_llm(prompt, max_tokens=4096)
    return json.loads(clean_json_response(raw))


def _generate_india_budget(state: str, city: Optional[str], days: int, style: str, travelers: int) -> dict:
    destination = _india_destination_str(state, city)
    
    if city:
        scope = f"Focus costs and recommendations specifically on {city}."
    else:
        scope = f"Cover costs across major destinations in {state}."

    prompt = f"""Create a detailed domestic travel budget for {travelers} traveler(s)
visiting {destination} for {days} days. Travel style: {style}
This is domestic Indian travel. All costs in INR.
{scope}

Return JSON:
{{
    "summary": {{"total_per_person":0,"total_trip_cost":0,"daily_average_per_person":0}},
    "breakdown": {{
        "accommodation": {{"total":0,"daily_rate":0,"stay_type":"","tips":""}},
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
    return json.loads(clean_json_response(raw))


def _get_india_advisory(state: str, city: Optional[str]) -> dict:
    destination = _india_destination_str(state, city)
    
    if city:
        scope = f"Focus safety information specifically on {city}."
    else:
        scope = f"Cover safety across {state}."

    prompt = f"""Provide comprehensive domestic travel advisory for Indian travelers visiting {destination}.
No visa or passport information needed as this is domestic travel.
{scope}

Return JSON:
{{
    "safety_rating":"Safe/Moderate Caution/Exercise Caution",
    "safety_score":8,
    "health_advisories":[{{"type":"","details":"","mandatory":false}}],
    "safety_tips":[{{"category":"","tips":[]}}],
    "areas_to_avoid":[],
    "emergency_numbers":{{"police":"100","ambulance":"108","tourist_helpline":"1363","women_helpline":"1091","fire":"101"}},
    "local_laws_to_know":[],
    "scams_to_watch":[],
    "connectivity":{{"mobile_network":"","internet":"","atm_availability":""}},
    "cultural_etiquette":[]
}}
Only output valid JSON."""
    raw = call_nvidia_llm(prompt)
    return json.loads(clean_json_response(raw))


def _generate_india_packing_list(state: str, city: Optional[str], days: int, style: str, dates: Optional[str]) -> dict:
    destination = _india_destination_str(state, city)
    date_ctx = f"Travel dates: {dates}" if dates else "General packing advice"
    
    if city:
        scope = f"Tailor the packing list for the weather and activities specific to {city}."
    else:
        scope = f"Cover packing needs for travel across {state}."

    prompt = f"""Generate a comprehensive packing list for a {days}-day domestic trip to {destination}.
Travel style: {style}
{date_ctx}
This is domestic Indian travel.
{scope}

Return JSON:
{{
    "weather_summary":"Expected weather conditions",
    "categories":{{
        "clothing":[{{"item":"","quantity":2,"notes":""}}],
        "toiletries":[...],
        "electronics":[...],
        "documents":[{{"item":"Aadhaar Card/ID proof","quantity":1,"notes":"Required for hotels"}}],
        "health_safety":[...],
        "accessories":[...],
        "region_specific":[...]
    }},
    "pro_tips":[],
    "items_to_avoid":[]
}}
Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=2048)
    return json.loads(clean_json_response(raw))


def _compare_india_destinations(destinations: list[str], criteria: list[str]) -> dict:
    prompt = f"""Compare these Indian travel destinations: {', '.join(destinations)}
Compare based on these criteria: {', '.join(criteria)}
This is domestic Indian travel comparison. All costs in INR.

Return ONLY a valid JSON object.
{{
    "comparison_table":{{
        "criteria_name":{{
            "Destination1":{{"score":8,"details":"explanation"}}
        }}
    }},
    "overall_winner":"Destination name",
    "winner_reason":"Why this destination wins overall",
    "best_for":{{"budget_travelers":"","families":"","adventure_seekers":"","foodies":"","culture_lovers":"","honeymooners":""}},
    "summary":"Brief overall comparison summary"
}}
Score each destination 1-10. Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=3000)
    return json.loads(clean_json_response(raw))


def _get_india_phrases(state: str) -> dict:
    prompt = f"""For travelers visiting {state}, India, provide essential local language phrases.
Include the primary regional language of this state.

Return JSON:
{{
    "primary_language":"Language name",
    "secondary_languages":["Hindi","English"],
    "greeting_culture":"Brief note on greeting customs in this state",
    "categories":{{
        "greetings":[{{"english":"Hello","local":"translation","pronunciation":"phonetic","hindi_equivalent":"Namaste"}}],
        "directions":[...],
        "dining":[...],
        "emergencies":[...],
        "shopping":[...]
    }},
    "cultural_notes":[],
    "common_mistakes":[]
}}
Include 5-8 phrases per category. Only output valid JSON."""
    raw = call_nvidia_llm(prompt, max_tokens=2048)
    return json.loads(clean_json_response(raw))


def _get_india_weather_activities(state: str, city: Optional[str], month: str) -> dict:
    destination = _india_destination_str(state, city)
    
    if city:
        scope = f"Focus weather and activity recommendations specifically on {city}."
    else:
        scope = f"Cover weather patterns across {state}."

    prompt = f"""For {destination} in {month}, provide weather-based activity recommendations.
This is for domestic Indian travel.
{scope}

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
    return json.loads(clean_json_response(raw))


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

# ---------- Health ----------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
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
    return {"continents": CONTINENTS}


@app.get("/countries/{continent}", tags=["Geography"])
def list_countries(continent: str):
    if continent not in CONTINENTS:
        raise HTTPException(status_code=400, detail=f"Invalid continent. Choose from {CONTINENTS}")
    countries = fetch_countries_for_continent(continent)
    return {"continent": continent, "countries": countries, "count": len(countries)}


# ---------- India Geography ----------

@app.get("/india/regions", tags=["India Geography"])
def list_india_regions():
    return {"regions": list(INDIAN_REGIONS.keys())}


@app.get("/india/states/{region}", tags=["India Geography"])
def list_india_states(region: str):
    if region not in INDIAN_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region. Choose from {list(INDIAN_REGIONS.keys())}")
    states = INDIAN_REGIONS[region]
    return {"region": region, "states": sorted(states), "count": len(states)}


@app.get("/india/states", tags=["India Geography"])
def list_all_india_states():
    return {"states": sorted(INDIAN_STATES.keys()), "count": len(INDIAN_STATES)}


@app.get("/india/cities/{state}", tags=["India Geography"])
def list_india_cities(state: str):
    if state not in INDIAN_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state. Check /india/states for the list.")
    cities = INDIAN_STATES[state]
    return {"state": state, "cities": cities, "count": len(cities)}


# ---------- Country Info ----------

@app.get("/country/{country}/info", response_model=CountryInfoResponse, tags=["Country Info"])
def country_info(country: str):
    data = fetch_country_info(country)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
    return CountryInfoResponse(**data)


@app.get("/currency/convert/{currency_code}", tags=["Country Info"])
def currency_to_inr(currency_code: str):
    data = fetch_currency_conversion(currency_code)
    if data.get("error") and data.get("rate") is None:
        raise HTTPException(status_code=502, detail=data["error"])
    return data


# ---------- AI Travel Features (International) ----------

@app.post("/travel/plan", response_model=GenericAIResponse, tags=["AI Travel"])
def travel_plan(req: TravelPlanRequest):
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
    chat_sessions.pop(session_id, None)
    return {"message": f"Chat session '{session_id}' cleared."}


@app.post("/travel/packing-list", response_model=GenericAIResponse, tags=["AI Travel"])
def packing_list(req: PackingListRequest):
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
    _require_nvidia()
    try:
        data = _get_weather_activities(req.country, req.month)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- AI Travel Features (India Domestic) ----------

@app.post("/india/travel/plan", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_travel_plan(req: IndiaTravelPlanRequest):
    _require_nvidia()
    try:
        data = _generate_india_travel_plan(req.state, req.city)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/india/travel/itinerary", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_travel_itinerary(req: IndiaItineraryRequest):
    _require_nvidia()
    try:
        data = _generate_india_itinerary(req.state, req.city, req.num_days, req.travel_style, req.budget_level)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/india/travel/budget", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_travel_budget(req: IndiaBudgetRequest):
    _require_nvidia()
    try:
        data = _generate_india_budget(req.state, req.city, req.num_days, req.travel_style, req.num_travelers)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/india/travel/advisory", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_travel_advisory(req: IndiaAdvisoryRequest):
    _require_nvidia()
    try:
        data = _get_india_advisory(req.state, req.city)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/india/travel/chat", response_model=ChatResponse, tags=["AI India Travel"])
def india_travel_chat(req: IndiaChatRequest):
    _require_nvidia()
    destination = _india_destination_str(req.state, req.city)
    history = chat_sessions.setdefault(req.session_id, [])
    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-5:]
    )

    if req.city:
        scope = f"Focus your answers specifically on {req.city} in {req.state}."
    else:
        scope = f"Cover the entire state of {req.state}."

    prompt = f"""You are an expert domestic travel assistant for {destination}.
You are helping an Indian traveler explore within India.
{scope}
Answer the user's question helpfully and concisely.
Focus on local insights, hidden gems, local food, and practical domestic travel tips.

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


@app.post("/india/travel/packing-list", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_packing_list(req: IndiaPackingListRequest):
    _require_nvidia()
    try:
        data = _generate_india_packing_list(req.state, req.city, req.num_days, req.travel_style, req.travel_dates)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/india/travel/compare", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_compare_destinations(req: IndiaCompareRequest):
    _require_nvidia()
    try:
        data = _compare_india_destinations(req.destinations, req.criteria)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/india/travel/phrases", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_essential_phrases(req: IndiaPhrasesRequest):
    _require_nvidia()
    try:
        data = _get_india_phrases(req.state)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/india/travel/weather-activities", response_model=GenericAIResponse, tags=["AI India Travel"])
def india_weather_activities(req: IndiaWeatherActivitiesRequest):
    _require_nvidia()
    try:
        data = _get_india_weather_activities(req.state, req.city, req.month)
        return GenericAIResponse(data=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/india/travel/full/{state}", tags=["AI India Travel"])
def india_full_travel_info(state: str, city: Optional[str] = Query(None)):
    if state not in INDIAN_STATES:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    plan: dict | None = None
    plan_error: str | None = None
    if nvidia_configured:
        try:
            plan = _generate_india_travel_plan(state, city)
        except Exception as e:
            plan_error = str(e)

    destination_label = _india_destination_short(state, city)

    return {
        "state": state,
        "city": city,
        "destination_label": destination_label,
        "info": {
            "state": state,
            "city": city,
            "cities": INDIAN_STATES.get(state, []),
            "region": next((r for r, states in INDIAN_REGIONS.items() if state in states), "Unknown"),
            "currency": "INR (Indian Rupee ₹)",
        },
        "travel_plan": plan,
        "travel_plan_error": plan_error,
    }


# ---------- Landmark Recognition ----------

@app.post("/landmark/identify", response_model=GenericAIResponse, tags=["AI Vision"])
async def identify_landmark_endpoint(
    image: UploadFile = File(...),
    country_hint: Optional[str] = Form(None),
):
    _require_nvidia()
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.content_type}")
    image_bytes = await image.read()
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


# ---------- Composite endpoint (international) ----------

@app.get("/travel/full/{country}", tags=["AI Travel"])
def full_travel_info(country: str):
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