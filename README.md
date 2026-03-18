# 🌍 RoamWise - Your AI-Powered Travel Companion

<p align="center">
  <img src="Logo.png" alt="RoamWise Logo" width="200"/>
</p>

<p align="center">
  <strong>Discover amazing destinations and plan your next adventure with AI!</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">APIs</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/AI-Gemini%202.0-green.svg" alt="Gemini"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

---

## 📖 About

**RoamWise** is a comprehensive AI-powered travel planning application built with Streamlit and Google's Gemini AI. It helps travelers explore destinations, plan itineraries, manage budgets, learn local phrases, and get real-time travel advisories—all in one place.

Whether you're a solo backpacker, a family on vacation, or a business traveler, RoamWise provides personalized recommendations tailored to your travel style and preferences.

---

## ✨ Features

### 🗺️ Core Features

| Feature | Description |
|---------|-------------|
| **🗺️ Travel Plan Generator** | Get comprehensive travel guides with must-visit cities, activities, local foods, and insider tips |
| **📅 AI Itinerary Planner** | Create detailed day-by-day itineraries based on trip duration, travel style, and budget |
| **💬 AI Travel Chatbot** | Ask questions about your destination and get instant, contextual answers |
| **💰 Budget Planner** | Get detailed cost breakdowns with interactive pie charts and money-saving tips |
| **🛡️ Safety & Visa Advisor** | Access safety ratings, visa requirements, health advisories, and emergency contacts |
| **📸 Landmark Recognition** | Upload photos to identify landmarks and get visitor information |

### 🛠️ Additional Tools

---

## 🏥 Health Check Endpoint

The application includes a dedicated health check server to monitor system status and component availability.

### Running the Health Check Server

```bash
python health_check.py
```

The server will start on `http://localhost:5001`

### Available Endpoints

#### `/health` - Detailed Health Check
Returns comprehensive status of all components:
```bash
curl http://localhost:5001/health
```

**Response Example:**
```json
{
  "status": "ok",
  "timestamp": "2026-03-19T10:30:00.000000",
  "checks": {
    "gemini_api": {
      "status": "ok",
      "message": "Gemini API key is configured"
    },
    "exchange_api": {
      "status": "ok",
      "message": "Exchange API is accessible"
    },
    "rest_countries_api": {
      "status": "ok",
      "message": "REST Countries API is accessible"
    },
    "dependencies": {
      "status": "ok",
      "message": "All dependencies are installed"
    }
  }
}
```

**Status Codes:**
- `200 OK` - All checks passed
- `503 Service Unavailable` - One or more checks failed

#### `/health/quick` - Quick Health Check
Returns only the overall status (useful for load balancers):
```bash
curl http://localhost:5001/health/quick
```

**Response Example:**
```json
{
  "status": "ok",
  "timestamp": "2026-03-19T10:30:00.000000"
}
```

#### `/` - Service Information
Returns available endpoints:
```bash
curl http://localhost:5001/
```

### Checks Performed

| Component | Description |
|-----------|-------------|
| **Gemini API** | Verifies API key configuration |
| **Exchange API** | Tests currency conversion service accessibility |
| **REST Countries API** | Tests country data API accessibility |
| **Dependencies** | Verifies all required Python packages are installed |

### Integration with Monitoring Tools

You can integrate the health check endpoint with monitoring solutions like:
- **Docker/Kubernetes**: Use `/health/quick` as a readiness/liveness probe
- **Monitoring Systems**: Schedule periodic checks to `/health`
- **Load Balancers**: Configure to use `/health/quick` for health status

### Example Kubernetes Probe Configuration

```yaml
livenessProbe:
  httpGet:
    path: /health/quick
    port: 5001
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/quick
    port: 5001
  initialDelaySeconds: 5
  periodSeconds: 10
```

### 🌐 Health Check on Streamlit Cloud

When deployed to Streamlit Cloud, the app includes a built-in **Health Check page** that works without needing a separate server:

1. **Access the Health Check Page**:
   - Go to your Streamlit app on Streamlit Cloud
   - Click the radio button in the left sidebar: **"🏥 Health Check"**

2. **View System Status**:
   - 🟢 **All Systems Operational** - Perfect! Everything is working
   - 🟡 **Some Components May Have Issues** - Some features might be limited
   - 🔴 **Some Systems Have Critical Issues** - Major functionality is affected

3. **Monitor Individual Components**:
   - **🤖 Gemini API** - AI features (travel plans, recommendations)
   - **🔄 Gemini Model** - AI model initialization
   - **💱 Exchange Rate API** - Currency conversion
   - **🌍 Countries Data API** - Destination information

4. **Refresh Status** - Click the "🔄 Refresh" button for real-time updates

**Example**: If "Exchange Rate API" shows ⚠️ Warning, currency conversions may be unavailable, but other features will still work.

---

### 🛠️ Additional Tools

| Tool | Description |
|------|-------------|
| **🎒 Packing List Generator** | Get personalized packing checklists based on destination, weather, and trip type |
| **🌍 Destination Comparison** | Compare multiple countries side-by-side on various criteria |
| **🗣️ Language Helper** | Learn essential phrases, pronunciation, and cultural notes |
| **🌦️ Weather-Based Activities** | Get activity recommendations based on weather for your travel month |

### 📊 Data Features

- **Real-time Currency Conversion** - Live exchange rates to INR
- **Country Information** - Capital cities, currencies, and more
- **Interactive Visualizations** - Budget pie charts using Plotly

---

## 🎬 Demo

### Main Interface
Link: https://roamwise-nsw4bp3vqxf8ggqprqgdey.streamlit.app/
