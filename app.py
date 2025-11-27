import streamlit as st
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai
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
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    # Fall back to environment variables (for local development)
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    return default

# Get API Keys using the unified function
GEMINI_API_KEY = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
EXCHANGE_API_KEY = get_secret("EXCHANGE_API_KEY")
MODEL_NAME = get_secret("GEMINI_MODEL", "gemini-2.5-pro")

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

# Initialize Gemini
gemini_configured = False
model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        logging.info("Gemini API configured successfully.")
        
        try:
            ModelClass = getattr(genai, "GenerativeModel", None) or \
                        getattr(genai, "Model", None) or \
                        getattr(genai, "Client", None)
            if callable(ModelClass):
                try:
                    model = ModelClass(MODEL_NAME)
                except Exception:
                    try:
                        model = ModelClass()
                    except Exception:
                        model = None
            logging.info("Model instance created: %s", bool(model))
        except Exception as e:
            logging.debug("Model instantiation skipped: %s", e)
    except Exception as e:
        logging.error("Error configuring Gemini API: %s", e)
        gemini_configured = False
        model = None
else:
    logging.warning("No Gemini API key found. AI features will be disabled.")
    st.warning("⚠️ Gemini API key not configured. AI features will not work.")
# Simple continent list used in UI
CONTINENTS = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

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
    """
    Fetches the capital, primary currency code, and currency name for a country.
    """
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
    """
    Fetches the live exchange rate from the specified currency to INR.
    """
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
    except requests.exceptions.Timeout:
        conversion_info["error"] = "Conversion API request timed out."
    except requests.exceptions.HTTPError as he:
        conversion_info["error"] = f"HTTP error when calling conversion API: {he}"
    except Exception as e:
        conversion_info["error"] = f"Conversion failed: {str(e)}"

    return conversion_info

def get_countries_for_continent(continent):
    """
    Fetches countries for a given continent.
    """
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

# --- AI Generation Functions ---

def generate_gemini_travel_plan(country):
    """
    Generates a comprehensive travel plan for a country using the Gemini model.
    """
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

    if not gemini_configured:
        logging.error("Gemini client is not configured.")
        return {"error": "Gemini model not configured."}

    try:
        logging.info("Calling Gemini model for %s using model %s", country, MODEL_NAME)
        
        analysis_text = None

        if model is not None and hasattr(model, "generate_content"):
            try:
                response = model.generate_content(prompt)
                analysis_text = getattr(response, "text", None) or getattr(response, "content", None) or str(response)
            except Exception as e:
                logging.debug("model.generate_content failed: %s", e)
                analysis_text = None

        if not analysis_text and hasattr(genai, "generate_text"):
            try:
                resp = genai.generate_text(model=MODEL_NAME, prompt=prompt, temperature=0.1, max_output_tokens=4096)
                if isinstance(resp, str):
                    analysis_text = resp
                else:
                    if hasattr(resp, "text") and getattr(resp, "text"):
                        analysis_text = getattr(resp, "text")
                    elif hasattr(resp, "candidates") and getattr(resp, "candidates"):
                        cand0 = getattr(resp, "candidates")[0]
                        if isinstance(cand0, dict):
                            analysis_text = cand0.get("content") or cand0.get("text")
                        else:
                            analysis_text = getattr(cand0, "content", None) or getattr(cand0, "text", None)
                    if not analysis_text:
                        analysis_text = str(resp)
            except Exception as e:
                logging.error("genai.generate_text failed: %s", e)

        if not analysis_text:
            logging.error("No usable text returned from Gemini API.")
            return {"error": "No usable text returned from Gemini."}

        at = clean_json_response(analysis_text)

        try:
            parsed = json.loads(at)
            return parsed
        except Exception as e:
            logging.debug("JSON parse failed: %s", e)
            return {"raw": analysis_text}

    except Exception as e:
        logging.error("Gemini model call failed: %s", e)
        return {"error": str(e)}

def get_travel_chat_response(country, user_question, chat_history):
    """
    AI chatbot for answering travel-related questions about a specific country.
    """
    if not gemini_configured:
        return {"error": "Gemini not configured"}
    
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
        if model is not None and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            return {"response": response.text}
    except Exception as e:
        logging.error("Chat error: %s", e)
        return {"error": str(e)}
    
    return {"error": "Could not generate response"}

def generate_detailed_itinerary(country, num_days, travel_style, budget_level):
    """
    Generates a detailed day-by-day travel itinerary.
    """
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
        if model is not None and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        logging.error("Itinerary generation failed: %s", e)
        return {"error": str(e)}
    
    return {"error": "Could not generate itinerary"}

def generate_budget_plan(country, num_days, travel_style, num_travelers):
    """
    Creates a detailed budget breakdown for a trip.
    """
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
        if model and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not generate budget plan"}

def get_travel_advisory(country, nationality="Indian"):
    """
    Gets safety information and visa requirements.
    """
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
        if model and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not fetch travel advisory"}

def identify_landmark(image_data, country_hint=None):
    """
    Identifies landmarks from uploaded images using Gemini Vision.
    """
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
        image = Image.open(image_data)
        
        if model and hasattr(model, "generate_content"):
            response = model.generate_content([prompt, image])
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not identify landmark"}

def generate_packing_list(country, num_days, travel_style, travel_dates=None):
    """
    Generates a personalized packing list based on destination and travel details.
    """
    date_context = f"Travel dates: {travel_dates}" if travel_dates else "General packing advice"
    
    prompt = f"""Generate a comprehensive packing list for a {num_days}-day trip to {country}.
    Travel style: {travel_style}
    {date_context}
    
    Consider:
    - Local weather and climate
    - Cultural requirements (dress codes, religious sites)
    - Activities typical for this travel style
    - Essential documents and tech
    
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
        if model and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not generate packing list"}

def compare_destinations(countries_list, criteria):
    """
    Compares multiple destinations based on user-selected criteria.
    """
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
    
    Score each country 1-10 for each criterion.
    Only output valid JSON.
    """
    
    try:
        if model and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not compare destinations"}

def get_essential_phrases(country):
    """
    Generates essential travel phrases in the local language.
    """
    prompt = f"""For travelers visiting {country}, provide essential phrases.
    
    Return JSON:
    {{
        "primary_language": "Language name",
        "greeting_culture": "Brief note on greeting customs",
        "phrases": [
            {{
                "english": "Hello",
                "local": "Local translation",
                "pronunciation": "Phonetic pronunciation",
                "context": "When to use"
            }}
        ],
        "categories": {{
            "greetings": [
                {{"english": "Hello", "local": "translation", "pronunciation": "phonetic"}}
            ],
            "directions": [...],
            "dining": [...],
            "shopping": [...],
            "emergencies": [...],
            "polite_expressions": [...]
        }},
        "cultural_notes": ["Important cultural tip 1", "tip 2"],
        "common_mistakes": ["Mistake tourists make with language"]
    }}
    
    Include 5-8 phrases per category.
    Only output valid JSON.
    """
    
    try:
        if model and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not generate phrases"}

def get_weather_activities(country, month):
    """
    Recommends activities based on weather conditions for a specific month.
    """
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
                "best_locations": ["Location 1", "Location 2"],
                "what_to_pack": ["Sunscreen", "Swimwear"]
            }}
        ],
        "activities_to_avoid": [
            {{
                "activity": "Trekking",
                "reason": "Heavy monsoon rains make trails dangerous"
            }}
        ],
        "regional_differences": [
            {{
                "region": "Northern region",
                "weather": "Cooler temperatures",
                "best_activities": ["Activity 1"]
            }}
        ],
        "festivals_events": [
            {{
                "name": "Festival name",
                "date": "Approximate date",
                "location": "Where it's celebrated",
                "description": "Brief description"
            }}
        ],
        "packing_for_weather": ["Item 1", "Item 2"]
    }}
    
    Only output valid JSON.
    """
    
    try:
        if model and hasattr(model, "generate_content"):
            response = model.generate_content(prompt)
            text = clean_json_response(response.text)
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not get weather activities"}

# --- UI Render Functions ---

def render_travel_chatbot(country):
    """Render the AI travel chatbot interface."""
    st.subheader("💬 Ask AI About Your Destination")
    st.write(f"Ask me anything about traveling to **{country}**!")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
    
    # Chat input
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
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

def render_itinerary_planner(country):
    """Render the AI itinerary planner interface."""
    st.subheader("📅 AI Itinerary Planner")
    st.write(f"Create a personalized day-by-day itinerary for **{country}**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_days = st.slider("Trip Duration (days)", 1, 30, 7, key="itinerary_days")
    
    with col2:
        travel_style = st.selectbox(
            "Travel Style",
            ["Adventure", "Relaxation", "Cultural", "Family", "Romantic", "Solo Backpacking"],
            key="itinerary_style"
        )
    
    with col3:
        budget_level = st.selectbox(
            "Budget Level",
            ["Budget", "Moderate", "Luxury"],
            key="itinerary_budget"
        )
    
    if st.button("🗓️ Generate Itinerary", key="gen_itinerary"):
        with st.spinner(f"Creating your {num_days}-day adventure..."):
            itinerary = generate_detailed_itinerary(
                country, num_days, travel_style.lower(), budget_level.lower()
            )
            
            if itinerary.get("error"):
                st.error(f"Error: {itinerary['error']}")
            else:
                st.success(f"Your {num_days}-day itinerary is ready!")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Days", num_days)
                with col2:
                    total_cost = itinerary.get("total_estimated_cost_inr", "N/A")
                    if isinstance(total_cost, (int, float)):
                        st.metric("Est. Budget", f"₹{total_cost:,}")
                    else:
                        st.metric("Est. Budget", total_cost)
                with col3:
                    st.metric("Best Time", itinerary.get("best_time_to_visit", "Any time"))
                
                # Packing essentials
                if itinerary.get("packing_essentials"):
                    st.info("🎒 **Packing Essentials:** " + ", ".join(itinerary["packing_essentials"]))
                
                st.divider()
                
                # Day-by-day breakdown
                for day_plan in itinerary.get("itinerary", []):
                    with st.expander(f"📍 Day {day_plan.get('day', '?')}: {day_plan.get('title', '')} - {day_plan.get('city', '')}", expanded=False):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🌅 Morning**")
                            morning = day_plan.get("morning", {})
                            if isinstance(morning, dict):
                                st.write(f"• {morning.get('activity', 'Free time')}")
                                st.write(f"  _{morning.get('description', '')}_")
                                st.caption(f"Duration: {morning.get('duration', 'N/A')} | Cost: ₹{morning.get('cost_inr', 0)}")
                            
                            st.markdown("**☀️ Afternoon**")
                            afternoon = day_plan.get("afternoon", {})
                            if isinstance(afternoon, dict):
                                st.write(f"• {afternoon.get('activity', 'Free time')}")
                                st.write(f"  _{afternoon.get('description', '')}_")
                                st.caption(f"Duration: {afternoon.get('duration', 'N/A')} | Cost: ₹{afternoon.get('cost_inr', 0)}")
                            
                            st.markdown("**🌙 Evening**")
                            evening = day_plan.get("evening", {})
                            if isinstance(evening, dict):
                                st.write(f"• {evening.get('activity', 'Free time')}")
                                st.write(f"  _{evening.get('description', '')}_")
                                st.caption(f"Duration: {evening.get('duration', 'N/A')} | Cost: ₹{evening.get('cost_inr', 0)}")
                        
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
    """Render the AI budget planner interface."""
    st.subheader("💰 AI Budget Planner")
    st.write(f"Get a detailed budget breakdown for your trip to **{country}**")
    
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
                
                # Summary metrics
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
                
                # Breakdown
                breakdown = budget.get("breakdown", {})
                
                # Create pie chart
                pie_data = {"Category": [], "Amount": []}
                
                for category, details in breakdown.items():
                    if isinstance(details, dict) and "total" in details:
                        pie_data["Category"].append(category.replace("_", " ").title())
                        pie_data["Amount"].append(details["total"])
                
                if pie_data["Category"]:
                    fig = px.pie(pie_data, values="Amount", names="Category", 
                                title="Budget Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
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
                                    elif isinstance(value, list):
                                        st.write(f"**{key.replace('_', ' ').title()}:**")
                                        for item in value:
                                            if isinstance(item, dict):
                                                st.write(f"  • {item.get('name', 'Item')}: ₹{item.get('cost', 0)}")
                                            else:
                                                st.write(f"  • {item}")
                                    else:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            if details.get("tips"):
                                st.info(f"💡 {details['tips']}")
                
                # Money saving tips
                if budget.get("money_saving_tips"):
                    st.subheader("💡 Money-Saving Tips")
                    for tip in budget["money_saving_tips"]:
                        st.success(f"✓ {tip}")
                
                # Hidden costs warning
                if budget.get("hidden_costs_warning"):
                    st.subheader("⚠️ Don't Forget")
                    for warning in budget["hidden_costs_warning"]:
                        st.warning(warning)

def render_safety_advisor(country):
    """Render the safety and visa advisor interface."""
    st.subheader("🛡️ Safety & Visa Information")
    st.write(f"Get important safety information and visa requirements for **{country}**")
    
    nationality = st.selectbox(
        "Your Nationality",
        ["Indian", "American", "British", "Canadian", "Australian", "German", "French", "Other"],
        key="nationality_select"
    )
    
    if st.button("🔍 Check Requirements", key="check_safety"):
        with st.spinner("Fetching travel advisory..."):
            advisory = get_travel_advisory(country, nationality)
            
            if advisory.get("error"):
                st.error(f"Error: {advisory['error']}")
            else:
                # Safety rating
                rating = advisory.get("safety_rating", "Unknown")
                score = advisory.get("safety_score", 5)
                
                if isinstance(score, (int, float)):
                    if score >= 7:
                        st.success(f"### Safety Rating: {rating} ({score}/10) ✅")
                    elif score >= 4:
                        st.warning(f"### Safety Rating: {rating} ({score}/10) ⚠️")
                    else:
                        st.error(f"### Safety Rating: {rating} ({score}/10) 🚨")
                else:
                    st.info(f"### Safety Rating: {rating}")
                
                st.divider()
                
                # Visa requirements
                st.subheader("🛂 Visa Requirements")
                visa = advisory.get("visa_requirements", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Visa Type", visa.get("visa_type", "Check embassy"))
                with col2:
                    # Use a compact markdown block with smaller font so long durations don't get truncated
                    duration = visa.get("duration_allowed", "Varies")
                    st.markdown(
                        f"<div style='font-size:14px; line-height:1.1'><b>Duration Allowed</b><br>{duration}</div>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    cost = visa.get("approximate_cost_inr", "Varies")
                    st.metric("Approx. Cost", f"₹{cost}" if isinstance(cost, (int, float)) else cost)
                
                if visa.get("processing_time"):
                    st.info(f"⏱️ Processing Time: {visa['processing_time']}")
                
                if visa.get("documents_required"):
                    st.write("**📄 Documents Required:**")
                    for doc in visa["documents_required"]:
                        st.write(f"  ✅ {doc}")
                
                st.divider()
                
                # Health advisories
                if advisory.get("health_advisories"):
                    st.subheader("🏥 Health Advisories")
                    for health in advisory["health_advisories"]:
                        if isinstance(health, dict):
                            mandatory = "🔴 Required" if health.get("mandatory") else "🟡 Recommended"
                            st.write(f"{mandatory} **{health.get('type', 'Health')}**: {health.get('details', '')}")
                        else:
                            st.write(f"• {health}")
                
                # Emergency numbers
                st.subheader("📞 Emergency Contacts")
                emergency = advisory.get("emergency_numbers", {})
                if emergency:
                    cols = st.columns(min(4, len(emergency)))
                    for idx, (service, number) in enumerate(emergency.items()):
                        with cols[idx % len(cols)]:
                            # Render emergency contact with smaller font to avoid truncation in the metric UI
                            service_title = service.replace("_", " ").title()
                            number_text = number
                            st.markdown(
                                f"<div style='font-size:14px; line-height:1.1'><b>{service_title}</b><br>{number_text}</div>",
                                unsafe_allow_html=True,
                            )
                
                # Safety tips
                if advisory.get("safety_tips"):
                    st.subheader("🔒 Safety Tips")
                    for tip_group in advisory["safety_tips"]:
                        if isinstance(tip_group, dict):
                            st.write(f"**{tip_group.get('category', 'General')}:**")
                            for tip in tip_group.get("tips", []):
                                st.write(f"  • {tip}")
                        else:
                            st.write(f"• {tip_group}")
                
                # Areas to avoid
                if advisory.get("areas_to_avoid"):
                    st.subheader("🚫 Areas to Avoid")
                    for area in advisory["areas_to_avoid"]:
                        st.error(f"⚠️ {area}")
                
                # Scams to watch
                if advisory.get("scams_to_watch"):
                    st.subheader("🎭 Common Scams")
                    for scam in advisory["scams_to_watch"]:
                        st.warning(f"👁️ {scam}")
                
                # Local laws
                if advisory.get("local_laws_to_know"):
                    st.subheader("⚖️ Important Local Laws")
                    for law in advisory["local_laws_to_know"]:
                        st.info(f"📜 {law}")

def render_landmark_recognition(country=None):
    """Render the landmark recognition interface."""
    st.subheader("📸 AI Landmark Recognition")
    st.write("Upload a photo to identify landmarks and get travel information!")
    
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png", "webp"],
        key="landmark_upload"
    )
    
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
                        
                        # Visitor info
                        visitor = result.get("visitor_info", {})
                        if visitor:
                            st.divider()
                            st.write("**ℹ️ Visitor Information:**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Best Time", visitor.get("best_time_to_visit", "Anytime"))
                            with col_b:
                                st.metric("Duration", visitor.get("typical_visit_duration", "1-2 hours"))
                            
                            fee = visitor.get("entry_fee_inr", 0)
                            if fee:
                                st.metric("Entry Fee", f"₹{fee}")
                            
                            if visitor.get("tips"):
                                st.write("**💡 Tips:**")
                                for tip in visitor["tips"]:
                                    st.write(f"  • {tip}")
                        
                        # Nearby attractions
                        if result.get("nearby_attractions"):
                            st.divider()
                            st.write("**🗺️ Nearby Attractions:**")
                            for attr in result["nearby_attractions"]:
                                st.write(f"  📍 {attr}")
                        
                        # Photo tips
                        if result.get("photo_tips"):
                            st.info(f"📷 Photo Tip: {result['photo_tips']}")
                    else:
                        st.warning("Could not identify a specific landmark in this image.")
                        if result.get("description"):
                            st.write(f"What I see: {result['description']}")

def render_packing_list(country):
    """Render the packing list generator interface."""
    st.subheader("🎒 AI Packing List Generator")
    st.write(f"Get a personalized packing list for **{country}**")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("Trip Duration (days)", 1, 60, 7, key="packing_days")
    with col2:
        style = st.selectbox(
            "Trip Type", 
            ["Leisure", "Business", "Adventure", "Beach", "Winter Sports", "Backpacking"],
            key="packing_style"
        )
    
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
                                item_name = item.get('item', 'Item')
                                st.checkbox(f"{item_name} (x{qty}){notes}", key=f"pack_{category}_{item_name}")
                            else:
                                st.checkbox(str(item), key=f"pack_{category}_{item}")
                
                # Pro tips
                if packing.get("pro_tips"):
                    st.divider()
                    st.subheader("💡 Pro Tips")
                    for tip in packing["pro_tips"]:
                        st.success(f"✓ {tip}")
                
                # Items to avoid
                if packing.get("items_to_avoid"):
                    st.divider()
                    st.subheader("🚫 Items to Avoid")
                    for item in packing["items_to_avoid"]:
                        st.warning(item)

def render_destination_comparison():
    """Render the destination comparison interface."""
    st.subheader("🌍 Compare Destinations")
    st.write("Compare multiple countries to find your perfect destination!")
    
    # Get all countries for selection
    all_countries = []
    for continent in CONTINENTS:
        all_countries.extend(get_countries_for_continent(continent))
    all_countries = sorted(set(all_countries))
    
    selected_countries = st.multiselect(
        "Select 2-4 countries to compare",
        all_countries,
        max_selections=4,
        key="compare_countries"
    )
    
    criteria = st.multiselect(
        "Comparison Criteria",
        ["Cost of Living", "Safety", "Weather", "Food Scene", "Nightlife", 
         "Cultural Attractions", "Natural Beauty", "Adventure Activities",
         "Ease of Travel", "English Friendliness", "Visa Requirements for Indians"],
        default=["Cost of Living", "Safety", "Food Scene"],
        key="compare_criteria"
    )
    
    if len(selected_countries) >= 2 and criteria:
        if st.button("🔍 Compare Destinations", key="compare_btn"):
            with st.spinner("Analyzing destinations..."):
                comparison = compare_destinations(selected_countries, criteria)
                
                if comparison.get("error"):
                    st.error(f"Error: {comparison['error']}")
                else:
                    # Winner announcement
                    st.success(f"🏆 **Overall Winner: {comparison.get('overall_winner', 'N/A')}**")
                    st.write(comparison.get('winner_reason', ''))
                    
                    st.divider()
                    
                    # Comparison table
                    st.subheader("📊 Detailed Comparison")
                    
                    table_data = comparison.get("comparison_table", {})
                    for criterion, country_scores in table_data.items():
                        st.markdown(f"**{criterion}**")
                        cols = st.columns(len(selected_countries))
                        for idx, country in enumerate(selected_countries):
                            with cols[idx]:
                                data = country_scores.get(country, {})
                                if isinstance(data, dict):
                                    score = data.get("score", "N/A")
                                    st.metric(country, f"{score}/10")
                                    st.caption(data.get("details", ""))
                                else:
                                    st.metric(country, f"{data}/10" if isinstance(data, (int, float)) else data)
                        st.divider()
                    
                    # Best for categories
                    best_for = comparison.get("best_for", {})
                    if best_for:
                        st.subheader("🎯 Best For...")
                        icons = {
                            "budget_travelers": "💰", 
                            "families": "👨‍👩‍👧‍👦", 
                            "adventure_seekers": "🏔️", 
                            "foodies": "🍜", 
                            "culture_lovers": "🏛️"
                        }
                        cols = st.columns(min(3, len(best_for)))
                        for idx, (category, country) in enumerate(best_for.items()):
                            with cols[idx % 3]:
                                icon = icons.get(category, "✨")
                                st.info(f"{icon} **{category.replace('_', ' ').title()}:** {country}")
                    
                    # Summary
                    if comparison.get("summary"):
                        st.divider()
                        st.write("**📝 Summary:**")
                        st.write(comparison["summary"])
    elif len(selected_countries) < 2:
        st.info("Please select at least 2 countries to compare.")

def render_language_helper(country):
    """Render the language helper interface."""
    st.subheader("🗣️ Language & Phrase Guide")
    st.write(f"Learn essential phrases for your trip to **{country}**")
    
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
                                        with col1:
                                            st.write(f"**{phrase.get('english', '')}**")
                                        with col2:
                                            st.write(f"🗣️ {phrase.get('local', '')}")
                                            if phrase.get('pronunciation'):
                                                st.caption(f"_{phrase.get('pronunciation', '')}_")
                                        st.divider()
                                    else:
                                        st.write(f"• {phrase}")
                
                # Cultural notes
                if phrases.get("cultural_notes"):
                    st.divider()
                    st.subheader("🎭 Cultural Notes")
                    for note in phrases["cultural_notes"]:
                        st.info(note)
                
                # Common mistakes
                if phrases.get("common_mistakes"):
                    st.subheader("⚠️ Common Mistakes to Avoid")
                    for mistake in phrases["common_mistakes"]:
                        st.warning(mistake)

def render_weather_activities(country):
    """Render the weather-based activities interface."""
    st.subheader("🌦️ Weather-Based Activities")
    st.write(f"Find the best activities for **{country}** based on when you're traveling")
    
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    
    selected_month = st.selectbox("When are you planning to visit?", months, key="weather_month")
    
    if st.button("🌤️ Get Recommendations", key="get_weather"):
        with st.spinner(f"Analyzing {country} weather for {selected_month}..."):
            weather_data = get_weather_activities(country, selected_month)
            
            if weather_data.get("error"):
                st.error(f"Error: {weather_data['error']}")
            else:
                # Weather summary
                weather = weather_data.get("weather_summary", {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🌡️ Temperature", weather.get("temperature_range", "N/A"))
                with col2:
                    st.metric("🌧️ Rainfall", weather.get("rainfall", "N/A"))
                with col3:
                    peak = "Yes 🔥" if weather_data.get("is_peak_season") else "No"
                    st.metric("Peak Season", peak)
                with col4:
                    st.metric("👥 Crowds", weather_data.get("tourist_crowd_level", "N/A"))
                
                if weather.get("general_conditions"):
                    st.info(f"☀️ {weather['general_conditions']}")
                
                st.divider()
                
                # Recommended activities
                if weather_data.get("recommended_activities"):
                    st.subheader("✅ Recommended Activities")
                    for activity in weather_data["recommended_activities"]:
                        if isinstance(activity, dict):
                            with st.expander(f"🎯 {activity.get('activity', 'Activity')}", expanded=True):
                                st.write(f"**Why this month:** {activity.get('why_this_month', '')}")
                                
                                if activity.get("best_locations"):
                                    st.write("**📍 Best locations:**")
                                    for loc in activity["best_locations"]:
                                        st.write(f"  • {loc}")
                                
                                if activity.get("what_to_pack"):
                                    st.write(f"**🎒 Pack:** {', '.join(activity['what_to_pack'])}")
                        else:
                            st.write(f"• {activity}")
                
                # Activities to avoid
                if weather_data.get("activities_to_avoid"):
                    st.subheader("❌ Activities to Avoid")
                    for avoid in weather_data["activities_to_avoid"]:
                        if isinstance(avoid, dict):
                            st.warning(f"**{avoid.get('activity', '')}**: {avoid.get('reason', '')}")
                        else:
                            st.warning(avoid)
                
                # Regional differences
                if weather_data.get("regional_differences"):
                    st.subheader("🗺️ Regional Differences")
                    for region in weather_data["regional_differences"]:
                        if isinstance(region, dict):
                            with st.expander(f"📍 {region.get('region', 'Region')}"):
                                st.write(f"**Weather:** {region.get('weather', '')}")
                                if region.get("best_activities"):
                                    st.write("**Best Activities:**")
                                    for act in region["best_activities"]:
                                        st.write(f"  • {act}")
                
                # Festivals and events
                if weather_data.get("festivals_events"):
                    st.subheader("🎉 Festivals & Events")
                    for fest in weather_data["festivals_events"]:
                        if isinstance(fest, dict):
                            st.success(f"**{fest.get('name', '')}** - {fest.get('date', '')}")
                            st.write(f"📍 {fest.get('location', '')} - {fest.get('description', '')}")
                        else:
                            st.write(f"• {fest}")
                
                # Packing for weather
                if weather_data.get("packing_for_weather"):
                    st.divider()
                    st.write("**🎒 Pack for the Weather:**")
                    st.write(", ".join(weather_data["packing_for_weather"]))

# --- Main Streamlit App ---

st.set_page_config(page_title="RoamWise - Travel Guide", layout="wide")

# Logo
try:
    logo = Image.open("Logo.png")
    st.image(logo, width=200)
except Exception:
    pass

st.title("🌍 RoamWise - Your Travel Companion")
st.markdown("Discover amazing destinations and plan your next adventure!")

# Sidebar for destination selection
st.sidebar.header("🗺️ Select Your Destination")
selected_continent = st.sidebar.selectbox("Choose a Continent:", CONTINENTS)

# Get countries for selected continent
countries = get_countries_for_continent(selected_continent)

if countries:
    selected_country = st.sidebar.selectbox("Choose a Country:", countries)
    
    # Clear chat history when country changes
    if "last_country" not in st.session_state:
        st.session_state.last_country = selected_country
    elif st.session_state.last_country != selected_country:
        st.session_state.chat_history = []
        st.session_state.last_country = selected_country
    
    # Main content area with tabs
    st.subheader(f"✨ Exploring {selected_country}")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Travel Plan", 
        "📅 Itinerary", 
        "💬 AI Chat",
        "💰 Budget",
        "🛡️ Safety",
        "📸 Landmark ID"
    ])
    
    with tab1:
        # Travel Plan Tab
        st.write("Get a comprehensive travel guide for your destination")
        
        if st.button("📋 Get Travel Plan", key="fetch_details"):
            with st.spinner(f"Generating travel plan for {selected_country}..."):
                # Fetch country info
                country_info = get_country_info(selected_country)
                
                if country_info.get("error"):
                    st.error(f"Error: {country_info['error']}")
                else:
                    capital = country_info["capital"]
                    currency_code = country_info["currency_code"]
                    currency_name = country_info["currency_name"]
                    
                    # Fetch currency conversion
                    conversion_info = get_currency_conversion_to_inr(currency_code)
                    
                    # Generate travel plan
                    travel_plan = generate_gemini_travel_plan(selected_country)
                    
                    # Store in session state
                    st.session_state.country_info_data = country_info
                    st.session_state.conversion_info_data = conversion_info
                    st.session_state.travel_plan_data = travel_plan
        
        # Display stored travel plan data
        if st.session_state.get("travel_plan_data"):
            country_info = st.session_state.country_info_data
            conversion_info = st.session_state.conversion_info_data
            travel_plan = st.session_state.travel_plan_data
            
            # Display country information
            st.divider()
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("🏛️ Capital", country_info["capital"])
            
            with col_info2:
                st.metric("💱 Currency", f"{country_info['currency_code']} - {country_info['currency_name']}")
            
            with col_info3:
                if conversion_info.get("rate"):
                    st.metric("📈 Exchange Rate", f"1 {country_info['currency_code']} = ₹{conversion_info['rate']:.2f}")
                else:
                    st.warning("Conversion rate unavailable")
            
            st.divider()
            
            # Display travel plan
            if isinstance(travel_plan, dict) and not travel_plan.get("error"):
                # Cities
                if "cities" in travel_plan:
                    st.subheader("🏙️ Must-Visit Cities")
                    for city in travel_plan["cities"]:
                        with st.expander(f"📍 {city.get('name', 'Unknown')}"):
                            st.write(city.get("reason", ""))
                            
                            # Activities for this city
                            if "activities" in travel_plan and city.get("name") in travel_plan["activities"]:
                                st.write("**🎯 Activities:**")
                                for activity in travel_plan["activities"][city.get("name")]:
                                    price = activity.get("price_inr", 0)
                                    price_str = f"₹{price}" if price > 0 else "Free"
                                    st.write(f"• **{activity.get('name', 'Activity')}** ({price_str})")
                                    st.write(f"  _{activity.get('description', '')}_")
                
                # Foods
                if "foods" in travel_plan:
                    st.subheader("🍽️ Must-Try Foods")
                    food_list = travel_plan["foods"]
                    cols = st.columns(min(3, len(food_list)))
                    for idx, food in enumerate(food_list):
                        with cols[idx % len(cols)]:
                            st.write(f"**{food.get('name', 'Food')}**")
                            st.write(food.get("description", ""))
                
                # Tips
                if "tips" in travel_plan:
                    st.subheader("💡 Travel Tips")
                    for tip in travel_plan["tips"]:
                        st.info(tip)
            else:
                error_msg = travel_plan.get("error") if isinstance(travel_plan, dict) else str(travel_plan)
                st.error(f"Could not generate travel plan: {error_msg}")
    
    with tab2:
        render_itinerary_planner(selected_country)
    
    with tab3:
        render_travel_chatbot(selected_country)
    
    with tab4:
        render_budget_planner(selected_country)
    
    with tab5:
        render_safety_advisor(selected_country)
    
    with tab6:
        render_landmark_recognition(selected_country)
    
    # Sidebar additional tools
    st.sidebar.divider()
    st.sidebar.subheader("🛠️ More Tools")
    
    # Tool selection in sidebar
    tool_selection = st.sidebar.radio(
        "Select a tool:",
        ["None", "🎒 Packing List", "🌍 Compare Destinations", "🗣️ Language Helper", "🌦️ Weather Activities"],
        key="tool_selection"
    )
    
    # Display selected tool in main area below tabs
    if tool_selection != "None":
        st.divider()
        
        if tool_selection == "🎒 Packing List":
            render_packing_list(selected_country)
        
        elif tool_selection == "🌍 Compare Destinations":
            render_destination_comparison()
        
        elif tool_selection == "🗣️ Language Helper":
            render_language_helper(selected_country)
        
        elif tool_selection == "🌦️ Weather Activities":
            render_weather_activities(selected_country)

else:
    st.warning(f"No countries found for {selected_continent}")

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.markdown("**🌍 RoamWise**")
st.sidebar.markdown("_Your AI-powered travel companion_")
st.sidebar.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')}")