---
title: RoamWise
emoji: 🌍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
# 🌍 RoamWise - AI-Powered Travel Companion

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-1.0.0-brightgreen?style=flat&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/NVIDIA-AI-blue?style=flat&logo=nvidia" alt="NVIDIA AI"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/Docker-ready-blue.svg" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

<h2 align="center">🌍 AI Travel Companion</h2>

<p align="center">
  <strong>Your ultimate AI travel assistant with 15+ smart features: personalized itineraries, budgets, safety info, landmark recognition, chat, and more!</strong>
</p>

<p align="center">
  <a href="#features">✨ Features</a> &nbsp;•&nbsp;
  <a href="#quickstart">🚀 Quick Start</a> &nbsp;•&nbsp;
  <a href="#setup">⚙️ Setup</a> &nbsp;•&nbsp;
  <a href="#api">🌐 API Reference</a> &nbsp;•&nbsp;
  <a href="#demo">🎬 Demo</a> &nbsp;•&nbsp;
  <a href="#deployment">🚀 Deployment</a>
</p>

---
## ✨ Features

| Feature | Description | AI-Powered |
|---------|-------------|------------|
| **🗺️ Travel Plans** | Complete guides: cities, activities, foods, insider tips | ✅ |
| **📅 Smart Itineraries** | Day-by-day plans by style/budget/duration | ✅ |
| **💬 AI Chatbot** | Ask anything about your destination | ✅ |
| **💰 Budget Calculator** | Detailed breakdowns with money-saving tips | ✅ |
| **🛡️ Safety Advisor** | Visa, health, emergency contacts, scam alerts | ✅ |
| **📸 Landmark Recognition** | Upload photos → instant identification + info | ✅ Vision |
| **🎒 Packing Lists** | Personalized checklists for any trip type | ✅ |
| **⚖️ Compare Countries** | Side-by-side analysis across any criteria | ✅ |
| **🗣️ Language Guide** | Essential phrases + pronunciation | ✅ |
| **🌦️ Weather Activities** | Month-specific recommendations | ✅ |
| **💱 Live Currency** | Real-time INR conversions | 📊 |
| **🌍 Country Data** | Capitals, currencies, continents | 📊 |
| **🏥 Health Check** | Monitor all services (sidebar) | ✅ |

**Single-Page App (SPA)** - No build step needed. Works offline for UI.

## 🇮🇳 Indian Travel Features

RoamWise delivers specialized AI support for India trips:

| Feature | Examples | AI-Powered |
|---------|----------|------------|
| **🏛️ Iconic Landmarks** | Taj Mahal, Golden Temple, Kerala backwaters houseboats, Hampi ruins | ✅ Vision AI |
| **🎉 Cultural Itineraries** | Diwali in Varanasi, Holi in Mathura, Rajasthan forts trail, Onam in Kerala | ✅ Personalized |
| **🍛 Street Food Guide** | Safe chaat spots, masala dosa variations, thali breakdowns, spice safety | ✅ Chatbot |
| **🚂 Travel Logistics** | IRCTC train tips, monsoon road prep, auto bargaining, VISA/ATMs | ✅ Practical |
| **📸 Photo Analysis** | Upload Taj photo → best time/crowd/cost; Qutub Minar → history facts | ✅ Landmark ID |

**India-Ready Endpoints:**
```bash
curl http://localhost:7860/travel/full/India
curl -X POST http://localhost:7860/travel/chat -d '{"country":"India","message":"Monsoon Kerala itinerary?"}'
curl -X POST http://localhost:7860/landmark/identify -F "image=@taj.jpg"  # → Taj Mahal info + tips
```

**Pro Tips from AI:**
- Bargain 30-50% on markets
- Carry cash + UPI
- Monsoon: South India beaches
- Carry mosquito repellent + antidiarrheal

---



---

## 🚀 Quick Start

```bash
# 1. Clone & Install
git clone <repo> && cd RoamWise
pip install -r requirements.txt

# 2. Copy env (get free NVIDIA key!)
cp .env.example .env
# Edit .env with your NVIDIA_API_KEY

# 3. Run → Open http://localhost:7860
uvicorn app:app --reload
```

**That's it!** SPA loads automatically. Select country → explore 15+ AI tools.

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- [NVIDIA API Key](https://build.nvidia.com) (free tier available)
- Optional: [ExchangeRate API](https://exchangerate.host) key

### Local Development
```bash
pip install -r requirements.txt
cp .env.example .env  # Add NVIDIA_API_KEY
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```
- Frontend: `http://localhost:7860` (SPA)
- API Docs: `http://localhost:7860/docs`
- Health: `http://localhost:7860/health`

### Docker
```bash
docker build -t roamwise .
docker run -p 7860:7860 -e NVIDIA_API_KEY=your_key roamwise
```

### Environment Variables
```env
NVIDIA_API_KEY=your_key_here  # Required for AI features
EXCHANGE_API_KEY=optional_key  # Live currency rates
```

---

## 🌐 API Reference

**OpenAPI Docs**: `/docs` (Swagger) | `/redoc`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | ✅ System status |
| `/travel/full/{country}` | GET | 🗺️ Complete travel info |
| `/travel/plan` | POST | AI travel guide |
| `/travel/itinerary` | POST | 📅 Day-by-day plan |
| `/travel/budget` | POST | 💰 Cost breakdown |
| `/travel/chat` | POST | 💬 Conversational AI |
| `/landmark/identify` | POST | 📸 Image → landmark |
| `/travel/compare` | POST | ⚖️ Multi-country analysis |
| `/continents` `/countries/{continent}` | GET | 🌍 Geography data |

**Example**:
```bash
curl http://localhost:7860/travel/full/Japan
curl -X POST http://localhost:7860/travel/chat \
  -H \"Content-Type: application/json\" \
  -d '{\"country\":\"Japan\",\"message\":\"Best time to visit?\"}'
```

---

## 🎬 Demo & Screenshots

**[Live Demo](https://shouvik99-roamwise.hf.space)**

---

## 🏥 Health Check

**Integrated** - Click **🏥 Health Check** in sidebar.

**API**:
```bash
curl http://localhost:7860/health
```
```json
{
  \"all_ok\": true,
  \"nvidia_api\": {\"status\":\"✅ OK\", \"working\":true},
  \"exchange_api\": {\"status\":\"✅ OK\", \"working\":true}
}
```

Checks: NVIDIA AI, Exchange API, Countries API, Dependencies.

---

## 🚀 Deployment

| Platform | Guide |
|----------|-------|
| **Railway/Render** | `git push` + `Dockerfile` |
| **Streamlit Cloud** | `app.py` + requirements |
| **Docker Hub** | `docker push` |
| **Kubernetes** | `/health/quick` liveness probe |

**Production**:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
```

---

## 🛠️ Development

```bash
# Linting & Formatting
pip install ruff black
ruff check . && black .

# Tests (add pytest later)
pytest

# Build frontend (none needed - pure HTML/JS)
```

**File Structure**:
```
├── main.py          # FastAPI backend (20+ endpoints)
├── static/          # SPA frontend (index.html)
├── requirements.txt # Dependencies
├── Dockerfile       # Production-ready
├── .env.example     # Config template
└── README.md        # 📖 You're reading it!
```

---

## 🤝 Contributing

1. Fork → Clone → Create branch (`git checkout -b feature/xyz`)
2. Commit (`git commit -m 'feat: add xyz'`)
3. Push → PR

**Issues?** [Create Issue](https://github.com/yourusername/roamwise/issues/new)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) © 2024

---

<div align=\"center\">
  <sub>Built with ❤️ for travelers worldwide. ✈️🌍</sub>
</div>

