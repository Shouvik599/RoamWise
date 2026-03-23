import streamlit as st
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
from io import BytesIO
import logging
import json
from PIL import Image
from datetime import datetime
import plotly.express as px

# --- Configuration and Initialization ---

# Load .env for local development
load_dotenv()

def get_secret(key, default=None):
    """
    Get secret from Streamlit secrets (cloud) or environment variables (local).
    Priority: Streamlit Secrets > Environment Variables > Default
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    return default

# Get API Keys
NVIDIA_API_KEY = get_secret("NVIDIA_API_KEY")
EXCHANGE_API_KEY = get_secret("EXCHANGE_API_KEY")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_country_for_comparison" not in st.session_state:
    st.session_state.selected_country_for_comparison = []

if "travel_plan_data" not in st.session_state:
    st.session_state.travel_plan_data = None

if "country_info_data" not in st.session_state:
    st.session_state.country_info_data = None

if "conversion_info_data" not in st.session_state:
    st.session_state.conversion_info_data = None

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Initialize NVIDIA Client
nvidia_configured = False
client = None

if NVIDIA_API_KEY:
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
        nvidia_configured = True
        logging.info("NVIDIA API configured successfully.")
    except Exception as e:
        logging.error("Error configuring NVIDIA API: %s", e)
        nvidia_configured = False
        client = None
else:
    logging.warning("No NVIDIA API key found. AI features will be disabled.")
    st.warning("⚠️ NVIDIA API key not configured. AI features will not work.")

# Simple continent list used in UI
CONTINENTS = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

# --- Health Check Functions ---

def check_nvidia_api():
    """Check if NVIDIA API is configured."""
    if NVIDIA_API_KEY:
        return {"status": "✅ OK", "details": "NVIDIA API key is configured", "working": True}
    else:
        return {"status": "❌ Error", "details": "NVIDIA API key not found", "working": False}

def check_exchange_api():
    """Check if Exchange API is accessible."""
    try:
        response = requests.get(
            "https://api.exchangerate.host/latest",
            params={"base": "USD"},
            timeout=5
        )
        if response.status_code == 200:
            return {"status": "✅ OK", "details": "Exchange API is accessible", "working": True}
        else:
            return {"status": "⚠️ Warning", "details": f"Exchange API returned status {response.status_code}", "working": False}
    except requests.exceptions.Timeout:
        return {"status": "⚠️ Warning", "details": "Exchange API request timed out", "working": False}
    except Exception:
        return {"status": "❌ Error", "details": "Exchange API is unreachable", "working": False}

def check_rest_countries_api():
    """Check if REST Countries API is accessible."""
    try:
        response = requests.get(
            "https://restcountries.com/v3.1/all",
            timeout=5
        )
        if response.status_code == 200:
            return {"status": "✅ OK", "details": "REST Countries API is accessible", "working": True}
        else:
            return {"status": "⚠️ Warning", "details": f"REST Countries API returned status {response.status_code}", "working": False}
    except requests.exceptions.Timeout:
        return {"status": "⚠️ Warning", "details": "REST Countries API request timed out", "working": False}
    except Exception:
        return {"status": "❌ Error", "details": "REST Countries API is unreachable", "working": False}

def check_nvidia_client():
    """Check if NVIDIA client is properly initialized."""
    if nvidia_configured and client:
        return {"status": "✅ OK", "details": "NVIDIA client is initialized and ready", "working": True}
    else:
        return {"status": "⚠️ Warning", "details": "NVIDIA client not initialized", "working": False}

@st.cache_data(ttl=60)
def get_all_health_checks():
    """Run all health checks and return results."""
    return {
        "nvidia_api": check_nvidia_api(),
        "exchange_api": check_exchange_api(),
        "rest_countries_api": check_rest_countries_api(),
        "nvidia_client": check_nvidia_client(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

# --- Utility Functions ---

def clean_json_response(text):
    """Clean and parse JSON from AI response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# --- Core API Functions ---

def get_country_info(country):
    """Fetches the capital, primary currency code, and currency name for a country."""
    try:
        rc = requests.get(f"https://restcountries.com/v3.1/name/{country}", params={"fullText": "true"})
        if rc.status_code != 200:
            return {"error": "Failed to fetch country data from external API."}
        
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
            "error": None
        }
    except Exception as e:
        logging.error("Error in get_country_info for %s: %s", country, e)
        return {"error": "An error occurred while processing country data."}

def get_currency_conversion_to_inr(currency_code):
    """Fetches the live exchange rate from the specified currency to INR."""
    conversion_info = {"from": currency_code, "to": "INR", "rate": None, "error": None}
    if not currency_code:
        conversion_info["error"] = "No currency code provided for conversion."
        return conversion_info

    try:
        params = {"from": currency_code, "to": "INR", "amount": 1}
        if EXCHANGE_API_KEY:
            params["access_key"] = EXCHANGE_API_KEY

        resp = requests.get("https://api.exchangerate.host/convert", params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get("success") is False:
            err = data.get("error", {})
            conversion_info["error"] = err.get("info") if isinstance(err, dict) else str(err)
            return conversion_info

        rate = None
        if isinstance(data, dict):
            if data.get("result") is not None:
                rate = data.get("result")
            elif data.get("quotes"):
                quotes = data.get("quotes", {})
                for k, v in quotes.items():
                    if k.endswith("INR"):
                        rate = v
                        break
            elif data.get("rates"):
                rate = data.get("rates", {}).get("INR")

        conversion_info["rate"] = rate
        if rate is None:
            conversion_info["error"] = "API returned no exchange rate."
    except Exception as e:
        conversion_info["error"] = f"Conversion failed: {str(e)}"

    return conversion_info

def get_countries_for_continent(continent):
    """Fetches countries for a given continent."""
    try:
        url = f"https://restcountries.com/v3.1/region/{continent}"
        resp = requests.get(url, params={"fields": "name"})
        if resp.status_code != 200:
            return []
        data = resp.json()
        countries = sorted([c.get("name", {}).get("common", "") for c in data])
        return countries
    except Exception as e:
        logging.error("Error fetching countries: %s", e)
        return []

# --- AI Generation Functions (NVIDIA) ---

def call_nvidia_llm(prompt, temperature=0.2, top_p=0.7, max_tokens=2048):
    """Calls the NVIDIA Llama 3.3 70B model."""
    if not client:
        raise Exception("NVIDIA API client not configured")
    
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling NVIDIA LLM: {e}")
        raise

def call_nvidia_vision_llm(prompt, image_data, temperature=0.2, top_p=0.7, max_tokens=1024):
    """Calls the NVIDIA Llama 3.2 90B Vision model for image analysis."""
    if not client:
        raise Exception("NVIDIA API client not configured")
    try:
        # Convert uploaded file/Image to base64
        if isinstance(image_data, Image.Image):
            buffered = BytesIO()
            image_data.convert('RGB').save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            image_bytes = image_data.getvalue()
            img_str = base64.b64encode(image_bytes).decode("utf-8")
            
        image_url = f"data:image/jpeg;base64,{img_str}"
        
        completion = client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling NVIDIA Vision LLM: {e}")
        raise

def generate_travel_plan(country):
    """Generates a comprehensive travel plan for a country."""
    prompt = (
        f"Generate a travel guide for {country} as a single, valid JSON object. "
        "Do not include any introductory text, closing text, or markdown formatting like ```json. "
        "The JSON object should have the following keys:\n"
        "1. 'cities': An array of 4-5 must-visit city objects. Each city object should have:\n"
        "   - 'name': The city's name (string).\n"
        "   - 'reason': A brief, compelling reason to visit (string).\n"
        "2. 'activities': An object where each key is a city name from the 'cities' list. The value for each city key should be an array of 3 activity objects. Each activity object should have:\n"
        "   - 'name': The activity name (string).\n"
        "   - 'description': A short description of the activity (string).\n"
        "   - 'price_inr': An estimated price in Indian Rupees (INR) as an integer. Use 0 for free activities.\n"
        "3. 'foods': An array of 5 must-try food objects. Each food object should have:\n"
        "   - 'name': The food's name (string).\n"
        "   - 'description': A brief description (string).\n"
        "   - 'image_query': A search query for finding an image of this food (string).\n"
        "4. 'tips': An array of 5-6 essential travel tip strings for visitors.\n\n"
        "Example structure for 'activities' for one city:\n"
        "\"activities\": {\n"
        "  \"CityName\": [\n"
        "    { \"name\": \"Explore Old Town\", \"description\": \"Wander through historic streets.\", \"price_inr\": 500 },\n"
        "    { \"name\": \"Visit National Museum\", \"description\": \"Learn about the country's history.\", \"price_inr\": 800 }\n"
        "  ]\n"
        "}"
    )

    if not nvidia_configured:
        return {"error": "NVIDIA API not configured."}

    try:
        response_text = call_nvidia_llm(prompt, max_tokens=4096)
        at = clean_json_response(response_text)
        return json.loads(at)
    except Exception as e:
        logging.error("Generation failed: %s", e)
        return {"error": str(e)}

def get_travel_chat_response(country, user_question, chat_history):
    """AI chatbot for answering travel-related questions."""
    if not nvidia_configured:
        return {"error": "NVIDIA API not configured"}
    
    history_context = "\n".join([
        f"User: {h['user']}\nAssistant: {h['assistant']}" 
        for h in chat_history[-5:]
    ])
    
    prompt = f"""You are an expert travel assistant for {country}. 
    Answer the user's question helpfully and concisely.
    
    Previous conversation:
    {history_context}
    
    User's new question: {user_question}
    
    Provide a helpful, accurate response. If you're unsure about specific current 
    information (prices, hours), mention that the user should verify locally.
    """
    
    try:
        response_text = call_nvidia_llm(prompt)
        return {"response": response_text}
    except Exception as e:
        logging.error("Chat error: %s", e)
        return {"error": str(e)}

def generate_detailed_itinerary(country, num_days, travel_style, budget_level):
    """Generates a detailed day-by-day travel itinerary."""
    prompt = f"""Create a detailed {num_days}-day travel itinerary for {country}.
    
    Travel Style: {travel_style}
    Budget Level: {budget_level}
    
    Return a JSON object with this structure:
    {{
        "itinerary": [
            {{
                "day": 1,
                "title": "Day title/theme",
                "city": "City name",
                "morning": {{
                    "activity": "Activity name",
                    "description": "What to do",
                    "duration": "2 hours",
                    "cost_inr": 500
                }},
                "afternoon": {{
                    "activity": "Activity name",
                    "description": "What to do",
                    "duration": "3 hours",
                    "cost_inr": 1000
                }},
                "evening": {{
                    "activity": "Activity name",
                    "description": "What to do",
                    "duration": "2 hours",
                    "cost_inr": 800
                }},
                "meals": {{
                    "breakfast": "Restaurant/food suggestion",
                    "lunch": "Restaurant/food suggestion",
                    "dinner": "Restaurant/food suggestion"
                }},
                "accommodation": "Hotel/area suggestion",
                "travel_tip": "Specific tip for this day"
            }}
        ],
        "total_estimated_cost_inr": 50000,
        "packing_essentials": ["item1", "item2"],
        "best_time_to_visit": "Month or season"
    }}
    
    Make it realistic and detailed. Only output valid JSON, no markdown.
    """
    
    try:
        response_text = call_nvidia_llm(prompt, max_tokens=4096)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

def generate_budget_plan(country, num_days, travel_style, num_travelers):
    """Creates a detailed budget breakdown for a trip."""
    prompt = f"""Create a detailed travel budget for {num_travelers} traveler(s) 
    visiting {country} for {num_days} days.
    Travel style: {travel_style}
    
    Return JSON with all costs in INR:
    {{
        "summary": {{
            "total_per_person": 50000,
            "total_trip_cost": 100000,
            "daily_average_per_person": 7000
        }},
        "breakdown": {{
            "accommodation": {{
                "total": 20000,
                "daily_rate": 3000,
                "hotel_type": "3-star hotel",
                "tips": "Book in advance for better rates"
            }},
            "food": {{
                "total": 15000,
                "daily_rate": 2000,
                "breakdown": {{
                    "breakfast": 300,
                    "lunch": 600,
                    "dinner": 800,
                    "snacks": 300
                }},
                "tips": "Street food is authentic and cheaper"
            }},
            "transportation": {{
                "total": 8000,
                "local_transport_daily": 500,
                "intercity_estimate": 3000,
                "tips": "Use public transport to save money"
            }},
            "activities": {{
                "total": 10000,
                "popular_activities": [
                    {{"name": "Activity", "cost": 500}}
                ],
                "tips": "Book online for discounts"
            }},
            "miscellaneous": {{
                "total": 5000,
                "includes": ["Tips", "Souvenirs", "Emergency fund"]
            }}
        }},
        "money_saving_tips": ["tip1", "tip2", "tip3"],
        "hidden_costs_warning": ["Visa fees", "Travel insurance"],
        "best_value_period": "Off-season months for better deals"
    }}
    
    Only output valid JSON.
    """
    try:
        response_text = call_nvidia_llm(prompt, max_tokens=3000)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

def get_travel_advisory(country, nationality="Indian"):
    """Gets safety information and visa requirements."""
    prompt = f"""Provide comprehensive travel advisory for {nationality} travelers 
    visiting {country}.
    
    Return JSON:
    {{
        "safety_rating": "Safe/Moderate Caution/Exercise Caution/Reconsider Travel",
        "safety_score": 8,
        "visa_requirements": {{
            "visa_required": true,
            "visa_type": "Tourist Visa / E-Visa / Visa on Arrival / Visa Free",
            "duration_allowed": "30/60/90 days",
            "processing_time": "3-5 business days",
            "approximate_cost_inr": 5000,
            "documents_required": ["Passport", "Photos", "Bank statements"],
            "apply_link": "Official visa application website"
        }},
        "health_advisories": [
            {{
                "type": "Vaccination",
                "details": "Recommended vaccines",
                "mandatory": false
            }}
        ],
        "safety_tips": [
            {{
                "category": "General Safety",
                "tips": ["tip1", "tip2"]
            }}
        ],
        "areas_to_avoid": ["Area name - reason"],
        "emergency_numbers": {{
            "police": "100",
            "ambulance": "102",
            "tourist_helpline": "number",
            "indian_embassy": "embassy contact"
        }},
        "local_laws_to_know": ["Important law 1", "law 2"],
        "scams_to_watch": ["Common scam 1", "scam 2"]
    }}
    
    Only output valid JSON.
    """
    try:
        response_text = call_nvidia_llm(prompt)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

def identify_landmark(image_data, country_hint=None):
    """Identifies landmarks from uploaded images using NVIDIA Vision model."""
    prompt = f"""Analyze this image and identify any landmarks, tourist attractions, 
    or notable locations visible.
    
    {"The image is likely from " + country_hint if country_hint else ""}
    
    Return JSON:
    {{
        "identified": true,
        "landmark_name": "Name of the landmark",
        "location": "City, Country",
        "description": "Brief history and significance",
        "visitor_info": {{
            "best_time_to_visit": "Morning/Afternoon/Evening",
            "typical_visit_duration": "2 hours",
            "entry_fee_inr": 500,
            "tips": ["tip1", "tip2"]
        }},
        "nearby_attractions": ["attraction1", "attraction2"],
        "photo_tips": "Best angles or times for photography"
    }}
    
    If no landmark is identifiable, set identified to false and provide 
    a general description of what you see.
    Only output valid JSON.
    """
    try:
        response_text = call_nvidia_vision_llm(prompt, image_data)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

def generate_packing_list(country, num_days, travel_style, travel_dates=None):
    """Generates a personalized packing list based on destination and travel details."""
    date_context = f"Travel dates: {travel_dates}" if travel_dates else "General packing advice"
    
    prompt = f"""Generate a comprehensive packing list for a {num_days}-day trip to {country}.
    Travel style: {travel_style}
    {date_context}
    
    Return JSON:
    {{
        "weather_summary": "Expected weather conditions",
        "categories": {{
            "clothing": [
                {{"item": "item name", "quantity": 2, "notes": "optional note"}}
            ],
            "toiletries": [...],
            "electronics": [...],
            "documents": [...],
            "health_safety": [...],
            "accessories": [...],
            "country_specific": [...]
        }},
        "pro_tips": ["tip1", "tip2"],
        "items_to_avoid": ["item1 - reason"]
    }}
    Only output valid JSON.
    """
    try:
        response_text = call_nvidia_llm(prompt, max_tokens=2048)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

def compare_destinations(countries_list, criteria):
    """Compares multiple destinations based on user-selected criteria."""
    criteria_str = ", ".join(criteria)
    countries_str = ", ".join(countries_list)
    
    prompt = f"""Compare these travel destinations: {countries_str}
    Compare based on these criteria: {criteria_str}
    
    Return a JSON object:
    {{
        "comparison_table": {{
            "criteria_name": {{
                "Country1": {{"score": 8, "details": "explanation"}},
                "Country2": {{"score": 7, "details": "explanation"}}
            }}
        }},
        "overall_winner": "Country name",
        "winner_reason": "Why this country wins overall",
        "best_for": {{
            "budget_travelers": "Country name",
            "families": "Country name",
            "adventure_seekers": "Country name",
            "foodies": "Country name",
            "culture_lovers": "Country name"
        }},
        "summary": "Brief overall comparison summary"
    }}
    Score each country 1-10 for each criterion. Only output valid JSON.
    """
    try:
        response_text = call_nvidia_llm(prompt, max_tokens=3000)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

def get_essential_phrases(country):
    """Generates essential travel phrases in the local language."""
    prompt = f"""For travelers visiting {country}, provide essential phrases.
    
    Return JSON:
    {{
        "primary_language": "Language name",
        "greeting_culture": "Brief note on greeting customs",
        "categories": {{
            "greetings": [
                {{"english": "Hello", "local": "translation", "pronunciation": "phonetic"}}
            ],
            "directions": [...],
            "dining": [...],
            "emergencies": [...]
        }},
        "cultural_notes": ["Important cultural tip 1", "tip 2"],
        "common_mistakes": ["Mistake tourists make with language"]
    }}
    Include 5-8 phrases per category. Only output valid JSON.
    """
    try:
        response_text = call_nvidia_llm(prompt, max_tokens=2048)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

def get_weather_activities(country, month):
    """Recommends activities based on weather conditions for a specific month."""
    prompt = f"""For {country} in {month}, provide weather-based activity recommendations.
    
    Return JSON:
    {{
        "weather_summary": {{
            "temperature_range": "15-25°C",
            "rainfall": "Low/Moderate/High",
            "humidity": "Low/Moderate/High",
            "general_conditions": "Warm and dry"
        }},
        "is_peak_season": true,
        "tourist_crowd_level": "High/Medium/Low",
        "recommended_activities": [
            {{
                "activity": "Beach hopping",
                "why_this_month": "Perfect weather for swimming",
                "best_locations": ["Location 1"],
                "what_to_pack": ["Sunscreen"]
            }}
        ],
        "activities_to_avoid": [
            {{"activity": "Trekking", "reason": "Heavy monsoon rains"}}
        ],
        "regional_differences": [
            {{"region": "Northern region", "weather": "Cooler temperatures", "best_activities": ["Activity 1"]}}
        ],
        "festivals_events": [
            {{"name": "Festival", "date": "Date", "location": "Location", "description": "Description"}}
        ],
        "packing_for_weather": ["Item 1", "Item 2"]
    }}
    Only output valid JSON.
    """
    try:
        response_text = call_nvidia_llm(prompt, max_tokens=2048)
        return json.loads(clean_json_response(response_text))
    except Exception as e:
        return {"error": str(e)}

# --- UI Render Functions ---
# (Keeping UI components mostly identical, adapting minor naming from Gemini to NVIDIA AI)

def render_travel_chatbot(country):
    st.subheader("💬 Ask AI About Your Destination")
    st.write(f"Ask me anything about traveling to **{country}**!")
    
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
    
    user_input = st.chat_input(f"Ask anything about {country}...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_travel_chat_response(
                    country, 
                    user_input, 
                    st.session_state.chat_history
                )
                
                if response.get("error"):
                    st.error(response["error"])
                else:
                    st.write(response["response"])
                    st.session_state.chat_history.append({
                        "user": user_input,
                        "assistant": response["response"]
                    })
                    st.rerun()
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

def render_itinerary_planner(country):
    st.subheader("📅 AI Itinerary Planner")
    st.write(f"Create a personalized day-by-day itinerary for **{country}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_days = st.slider("Trip Duration (days)", 1, 30, 7, key="itinerary_days")
    with col2:
        travel_style = st.selectbox("Travel Style", ["Adventure", "Relaxation", "Cultural", "Family", "Romantic", "Solo Backpacking"], key="itinerary_style")
    with col3:
        budget_level = st.selectbox("Budget Level", ["Budget", "Moderate", "Luxury"], key="itinerary_budget")
    
    if st.button("🗓️ Generate Itinerary", key="gen_itinerary"):
        with st.spinner(f"Creating your {num_days}-day adventure..."):
            itinerary = generate_detailed_itinerary(country, num_days, travel_style.lower(), budget_level.lower())
            
            if itinerary.get("error"):
                st.error(f"Error: {itinerary['error']}")
            else:
                st.success(f"Your {num_days}-day itinerary is ready!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Days", num_days)
                with col2:
                    total_cost = itinerary.get("total_estimated_cost_inr", "N/A")
                    st.metric("Est. Budget", f"₹{total_cost:,}" if isinstance(total_cost, (int, float)) else total_cost)
                with col3:
                    st.metric("Best Time", itinerary.get("best_time_to_visit", "Any time"))
                
                if itinerary.get("packing_essentials"):
                    st.info("🎒 **Packing Essentials:** " + ", ".join(itinerary["packing_essentials"]))
                
                st.divider()
                
                for day_plan in itinerary.get("itinerary", []):
                    with st.expander(f"📍 Day {day_plan.get('day', '?')}: {day_plan.get('title', '')} - {day_plan.get('city', '')}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🌅 Morning**")
                            morning = day_plan.get("morning", {})
                            if isinstance(morning, dict):
                                st.write(f"• {morning.get('activity', 'Free time')}")
                                st.write(f"  _{morning.get('description', '')}_")
                            
                            st.markdown("**☀️ Afternoon**")
                            afternoon = day_plan.get("afternoon", {})
                            if isinstance(afternoon, dict):
                                st.write(f"• {afternoon.get('activity', 'Free time')}")
                                st.write(f"  _{afternoon.get('description', '')}_")
                            
                            st.markdown("**🌙 Evening**")
                            evening = day_plan.get("evening", {})
                            if isinstance(evening, dict):
                                st.write(f"• {evening.get('activity', 'Free time')}")
                                st.write(f"  _{evening.get('description', '')}_")
                        
                        with col2:
                            st.markdown("**🍽️ Meals**")
                            meals = day_plan.get("meals", {})
                            if isinstance(meals, dict):
                                st.write(f"🥐 Breakfast: {meals.get('breakfast', 'Local options')}")
                                st.write(f"🍛 Lunch: {meals.get('lunch', 'Local options')}")
                                st.write(f"🍽️ Dinner: {meals.get('dinner', 'Local options')}")
                            
                            st.markdown("**🏨 Accommodation**")
                            st.write(day_plan.get("accommodation", "Various options available"))
                            
                            if day_plan.get("travel_tip"):
                                st.info(f"💡 Tip: {day_plan['travel_tip']}")

def render_budget_planner(country):
    st.subheader("💰 AI Budget Planner")
    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.number_input("Trip Duration (days)", 1, 60, 7, key="budget_days")
    with col2:
        travelers = st.number_input("Number of Travelers", 1, 10, 2, key="budget_travelers")
    with col3:
        style = st.selectbox("Travel Style", ["Budget", "Mid-Range", "Luxury"], key="budget_style")
    
    if st.button("💵 Calculate Budget", key="calc_budget"):
        with st.spinner("Calculating your travel budget..."):
            budget = generate_budget_plan(country, days, style, travelers)
            
            if budget.get("error"):
                st.error(f"Error: {budget['error']}")
            else:
                summary = budget.get("summary", {})
                st.subheader("📊 Budget Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    per_person = summary.get('total_per_person', 0)
                    st.metric("Per Person", f"₹{per_person:,}" if isinstance(per_person, (int, float)) else per_person)
                with col2:
                    total = summary.get('total_trip_cost', 0)
                    st.metric("Total Trip", f"₹{total:,}" if isinstance(total, (int, float)) else total)
                with col3:
                    daily = summary.get('daily_average_per_person', 0)
                    st.metric("Daily Average", f"₹{daily:,}" if isinstance(daily, (int, float)) else daily)
                
                st.divider()
                breakdown = budget.get("breakdown", {})
                pie_data = {"Category": [], "Amount": []}
                for category, details in breakdown.items():
                    if isinstance(details, dict) and "total" in details:
                        pie_data["Category"].append(category.replace("_", " ").title())
                        pie_data["Amount"].append(details["total"])
                
                if pie_data["Category"]:
                    fig = px.pie(pie_data, values="Amount", names="Category", title="Budget Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 Detailed Breakdown")
                for category, details in breakdown.items():
                    if isinstance(details, dict):
                        total_val = details.get('total', 0)
                        total_str = f"₹{total_val:,}" if isinstance(total_val, (int, float)) else total_val
                        with st.expander(f"{category.replace('_', ' ').title()} - {total_str}"):
                            for key, value in details.items():
                                if key not in ["total", "tips"]:
                                    if isinstance(value, dict):
                                        st.write(f"**{key.replace('_', ' ').title()}:**")
                                        for k, v in value.items():
                                            st.write(f"  • {k.title()}: ₹{v}" if isinstance(v, (int, float)) else f"  • {k.title()}: {v}")
                                    else:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            if details.get("tips"):
                                st.info(f"💡 {details['tips']}")

def render_safety_advisor(country):
    st.subheader("🛡️ Safety & Visa Information")
    nationality = st.selectbox("Your Nationality", ["Indian", "American", "British", "Canadian", "Australian", "German", "French", "Other"], key="nationality_select")
    
    if st.button("🔍 Check Requirements", key="check_safety"):
        with st.spinner("Fetching travel advisory..."):
            advisory = get_travel_advisory(country, nationality)
            
            if advisory.get("error"):
                st.error(f"Error: {advisory['error']}")
            else:
                rating = advisory.get("safety_rating", "Unknown")
                score = advisory.get("safety_score", 5)
                if isinstance(score, (int, float)):
                    if score >= 7: st.success(f"### Safety Rating: {rating} ({score}/10) ✅")
                    elif score >= 4: st.warning(f"### Safety Rating: {rating} ({score}/10) ⚠️")
                    else: st.error(f"### Safety Rating: {rating} ({score}/10) 🚨")
                else:
                    st.info(f"### Safety Rating: {rating}")
                
                st.divider()
                st.subheader("🛂 Visa Requirements")
                visa = advisory.get("visa_requirements", {})
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Visa Type", visa.get("visa_type", "Check embassy"))
                with col2: st.markdown(f"<div style='font-size:14px; line-height:1.1'><b>Duration Allowed</b><br>{visa.get('duration_allowed', 'Varies')}</div>", unsafe_allow_html=True)
                with col3: 
                    cost = visa.get("approximate_cost_inr", "Varies")
                    st.metric("Approx. Cost", f"₹{cost}" if isinstance(cost, (int, float)) else cost)
                
                if visa.get("documents_required"):
                    st.write("**📄 Documents Required:**")
                    for doc in visa["documents_required"]:
                        st.write(f"  ✅ {doc}")
                
                st.divider()
                if advisory.get("health_advisories"):
                    st.subheader("🏥 Health Advisories")
                    for health in advisory["health_advisories"]:
                        if isinstance(health, dict):
                            st.write(f"{'🔴 Required' if health.get('mandatory') else '🟡 Recommended'} **{health.get('type', 'Health')}**: {health.get('details', '')}")
                        else:
                            st.write(f"• {health}")

def render_landmark_recognition(country=None):
    st.subheader("📸 AI Landmark Recognition")
    st.write("Upload a photo to identify landmarks and get travel information!")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="landmark_upload")
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Identify Landmark", key="identify_btn"):
                with st.spinner("Analyzing image..."):
                    result = identify_landmark(uploaded_file, country)
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    elif result.get("identified"):
                        st.success(f"**🏛️ {result.get('landmark_name', 'Unknown')}**")
                        st.write(f"📍 {result.get('location', 'Unknown location')}")
                        st.divider()
                        st.write("**📖 About:**")
                        st.write(result.get("description", "No description available."))
                        
                        visitor = result.get("visitor_info", {})
                        if visitor:
                            st.divider()
                            st.write("**ℹ️ Visitor Information:**")
                            col_a, col_b = st.columns(2)
                            with col_a: st.metric("Best Time", visitor.get("best_time_to_visit", "Anytime"))
                            with col_b: st.metric("Duration", visitor.get("typical_visit_duration", "1-2 hours"))
                    else:
                        st.warning("Could not identify a specific landmark in this image.")
                        if result.get("description"):
                            st.write(f"What I see: {result['description']}")

def render_packing_list(country):
    st.subheader("🎒 AI Packing List Generator")
    col1, col2 = st.columns(2)
    with col1: days = st.number_input("Trip Duration (days)", 1, 60, 7, key="packing_days")
    with col2: style = st.selectbox("Trip Type", ["Leisure", "Business", "Adventure", "Beach", "Winter Sports", "Backpacking"], key="packing_style")
    
    travel_dates = st.date_input("Travel Start Date (optional)", value=None, key="packing_dates")
    
    if st.button("📦 Generate Packing List", key="gen_packing"):
        with st.spinner("Creating your personalized packing list..."):
            packing = generate_packing_list(country, days, style, str(travel_dates) if travel_dates else None)
            
            if packing.get("error"):
                st.error(f"Error: {packing['error']}")
            else:
                st.info(f"🌤️ **Weather:** {packing.get('weather_summary', 'Check local forecasts')}")
                categories = packing.get("categories", {})
                for category, items in categories.items():
                    with st.expander(f"📂 {category.replace('_', ' ').title()}", expanded=True):
                        for item in items:
                            if isinstance(item, dict):
                                qty = item.get("quantity", 1)
                                notes = f" - _{item.get('notes')}_" if item.get("notes") else ""
                                st.checkbox(f"{item.get('item', 'Item')} (x{qty}){notes}", key=f"pack_{category}_{item.get('item')}")
                            else:
                                st.checkbox(str(item), key=f"pack_{category}_{item}")

def render_destination_comparison():
    st.subheader("🌍 Compare Destinations")
    
    all_countries = []
    for continent in CONTINENTS:
        all_countries.extend(get_countries_for_continent(continent))
    all_countries = sorted(set(all_countries))
    
    selected_countries = st.multiselect("Select 2-4 countries to compare", all_countries, max_selections=4, key="compare_countries")
    criteria = st.multiselect("Comparison Criteria", ["Cost of Living", "Safety", "Weather", "Food Scene", "Nightlife", "Cultural Attractions"], default=["Cost of Living", "Safety", "Food Scene"], key="compare_criteria")
    
    if len(selected_countries) >= 2 and criteria:
        if st.button("🔍 Compare Destinations", key="compare_btn"):
            with st.spinner("Analyzing destinations..."):
                comparison = compare_destinations(selected_countries, criteria)
                
                if comparison.get("error"):
                    st.error(f"Error: {comparison['error']}")
                else:
                    st.success(f"🏆 **Overall Winner: {comparison.get('overall_winner', 'N/A')}**")
                    st.write(comparison.get('winner_reason', ''))
                    st.divider()
                    
                    st.subheader("📊 Detailed Comparison")
                    table_data = comparison.get("comparison_table", {})
                    for criterion, country_scores in table_data.items():
                        st.markdown(f"**{criterion}**")
                        cols = st.columns(len(selected_countries))
                        for idx, country in enumerate(selected_countries):
                            with cols[idx]:
                                data = country_scores.get(country, {})
                                if isinstance(data, dict):
                                    st.metric(country, f"{data.get('score', 'N/A')}/10")
                                    st.caption(data.get("details", ""))
                                else:
                                    st.metric(country, f"{data}/10" if isinstance(data, (int, float)) else data)
                        st.divider()
    elif len(selected_countries) < 2:
        st.info("Please select at least 2 countries to compare.")

def render_language_helper(country):
    st.subheader("🗣️ Language & Phrase Guide")
    if st.button("📚 Load Essential Phrases", key="load_phrases"):
        with st.spinner(f"Loading phrases for {country}..."):
            phrases = get_essential_phrases(country)
            if phrases.get("error"):
                st.error(f"Error: {phrases['error']}")
            else:
                st.info(f"**🌐 Primary Language:** {phrases.get('primary_language', 'Unknown')}")
                if phrases.get("greeting_culture"):
                    st.write(f"**🤝 Greeting Culture:** {phrases['greeting_culture']}")
                
                st.divider()
                categories = phrases.get("categories", {})
                if categories:
                    tabs = st.tabs([cat.replace("_", " ").title() for cat in categories.keys()])
                    for tab, (category, phrase_list) in zip(tabs, categories.items()):
                        with tab:
                            if isinstance(phrase_list, list):
                                for phrase in phrase_list:
                                    if isinstance(phrase, dict):
                                        col1, col2 = st.columns([1, 2])
                                        with col1: st.write(f"**{phrase.get('english', '')}**")
                                        with col2:
                                            st.write(f"🗣️ {phrase.get('local', '')}")
                                            if phrase.get('pronunciation'): st.caption(f"_{phrase.get('pronunciation', '')}_")
                                        st.divider()
                                    else:
                                        st.write(f"• {phrase}")

def render_weather_activities(country):
    st.subheader("🌦️ Weather-Based Activities")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    selected_month = st.selectbox("When are you planning to visit?", months, key="weather_month")
    
    if st.button("🌤️ Get Recommendations", key="get_weather"):
        with st.spinner(f"Analyzing {country} weather for {selected_month}..."):
            weather_data = get_weather_activities(country, selected_month)
            if weather_data.get("error"):
                st.error(f"Error: {weather_data['error']}")
            else:
                weather = weather_data.get("weather_summary", {})
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("🌡️ Temperature", weather.get("temperature_range", "N/A"))
                with col2: st.metric("🌧️ Rainfall", weather.get("rainfall", "N/A"))
                with col3: st.metric("Peak Season", "Yes 🔥" if weather_data.get("is_peak_season") else "No")
                with col4: st.metric("👥 Crowds", weather_data.get("tourist_crowd_level", "N/A"))
                
                st.divider()
                if weather_data.get("recommended_activities"):
                    st.subheader("✅ Recommended Activities")
                    for activity in weather_data["recommended_activities"]:
                        if isinstance(activity, dict):
                            with st.expander(f"🎯 {activity.get('activity', 'Activity')}", expanded=True):
                                st.write(f"**Why this month:** {activity.get('why_this_month', '')}")
                        else:
                            st.write(f"• {activity}")

# --- Main Streamlit App ---

st.set_page_config(page_title="RoamWise - Travel Guide", layout="wide")

st.title("🌍 RoamWise - Your Travel Companion")
st.markdown("Discover amazing destinations and plan your next adventure!")

page = st.sidebar.radio("📍 Navigation", ["Travel Planner", "🏥 Health Check"], index=0)

if page == "🏥 Health Check":
    st.subheader("🏥 System Health Check")
    col1, col2 = st.columns([3, 1])
    with col1: st.write("Checking system health...")
    with col2:
        if st.button("🔄 Refresh", key="refresh_health"):
            st.cache_data.clear()
            st.rerun()
    
    st.divider()
    health_results = get_all_health_checks()
    all_working = all(check["working"] for check in [
        health_results["nvidia_api"], health_results["exchange_api"], health_results["rest_countries_api"], health_results["nvidia_client"]
    ])
    
    if all_working: st.success("### ✅ All Systems Operational")
    else: st.warning("### ⚠️ Some Components May Have Issues")
    
    st.caption(f"Last checked: {health_results['timestamp']}")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**🤖 NVIDIA API**")
        nv_check = health_results["nvidia_api"]
        st.success(nv_check["status"]) if "✅" in nv_check["status"] else st.error(nv_check["status"])
        st.caption(nv_check["details"])
        
        st.write("**🔄 NVIDIA Client**")
        client_check = health_results["nvidia_client"]
        st.success(client_check["status"]) if "✅" in client_check["status"] else st.error(client_check["status"])
        st.caption(client_check["details"])
    
    with col2:
        st.write("**💱 Exchange Rate API**")
        ex_check = health_results["exchange_api"]
        st.success(ex_check["status"]) if "✅" in ex_check["status"] else st.error(ex_check["status"])
        st.caption(ex_check["details"])
        
        st.write("**🌍 Countries Data API**")
        c_check = health_results["rest_countries_api"]
        st.success(c_check["status"]) if "✅" in c_check["status"] else st.error(c_check["status"])
        st.caption(c_check["details"])

else:
    st.sidebar.header("🗺️ Select Your Destination")
    selected_continent = st.sidebar.selectbox("Choose a Continent:", CONTINENTS)
    countries = get_countries_for_continent(selected_continent)

    if countries:
        selected_country = st.sidebar.selectbox("Choose a Country:", countries)
        
        if "last_country" not in st.session_state:
            st.session_state.last_country = selected_country
        elif st.session_state.last_country != selected_country:
            st.session_state.chat_history = []
            st.session_state.last_country = selected_country
        
        st.subheader(f"✨ Exploring {selected_country}")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Travel Plan", "📅 Itinerary", "💬 AI Chat", "💰 Budget", "🛡️ Safety", "📸 Landmark ID"
        ])
        
        with tab1:
            st.write("Get a comprehensive travel guide for your destination")
            if st.button("📋 Get Travel Plan", key="fetch_details"):
                with st.spinner(f"Generating travel plan for {selected_country}..."):
                    country_info = get_country_info(selected_country)
                    if not country_info.get("error"):
                        st.session_state.country_info_data = country_info
                        st.session_state.conversion_info_data = get_currency_conversion_to_inr(country_info["currency_code"])
                        st.session_state.travel_plan_data = generate_travel_plan(selected_country)
            
            travel_plan = st.session_state.get("travel_plan_data")
            country_info = st.session_state.get("country_info_data", {})
            conversion_info = st.session_state.get("conversion_info_data", {})
            
            if travel_plan:
                st.divider()
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1: st.metric("🏛️ Capital", country_info.get("capital", "N/A"))
                with col_info2: st.metric("💱 Currency", f"{country_info.get('currency_code', 'N/A')}")
                with col_info3:
                    if conversion_info.get("rate"): st.metric("📈 Exchange Rate", f"1 {country_info.get('currency_code', '')} = ₹{conversion_info['rate']:.2f}")
                    else: st.warning("Conversion rate unavailable")
                
                st.divider()
                if isinstance(travel_plan, dict) and not travel_plan.get("error"):
                    if "cities" in travel_plan:
                        st.subheader("🏙️ Must-Visit Cities")
                        for city in travel_plan["cities"]:
                            with st.expander(f"📍 {city.get('name', 'Unknown')}"):
                                st.write(city.get("reason", ""))
                    if "foods" in travel_plan:
                        st.subheader("🍽️ Must-Try Foods")
                        for food in travel_plan["foods"]:
                            st.write(f"**{food.get('name', 'Food')}** - {food.get('description', '')}")
                else:
                    st.error("Travel plan data incomplete.")
            else:
                st.info("No travel plan data available. Click 'Get Travel Plan' to generate.")
        
        with tab2: render_itinerary_planner(selected_country)
        with tab3: render_travel_chatbot(selected_country)
        with tab4: render_budget_planner(selected_country)
        with tab5: render_safety_advisor(selected_country)
        with tab6: render_landmark_recognition(selected_country)
        
        st.sidebar.divider()
        st.sidebar.subheader("🛠️ More Tools")
        tool_selection = st.sidebar.radio("Select a tool:", ["None", "🎒 Packing List", "🌍 Compare Destinations", "🗣️ Language Helper", "🌦️ Weather Activities"], key="tool_selection")
        
        if tool_selection != "None":
            st.divider()
            if tool_selection == "🎒 Packing List": render_packing_list(selected_country)
            elif tool_selection == "🌍 Compare Destinations": render_destination_comparison()
            elif tool_selection == "🗣️ Language Helper": render_language_helper(selected_country)
            elif tool_selection == "🌦️ Weather Activities": render_weather_activities(selected_country)

    else:
        st.warning(f"No countries found for {selected_continent}")

st.sidebar.divider()
st.sidebar.info("**🌍 RoamWise**\n_Your AI-powered travel companion_")
st.sidebar.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')}")