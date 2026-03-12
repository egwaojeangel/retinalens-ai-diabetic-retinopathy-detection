# ⚡ SkillBridge — AI-Powered Career Gap Analyser

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-00a67e?style=flat-square&logo=meta)
![Industry](https://img.shields.io/badge/Works%20For-Any%20Industry-2dd4bf?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Core Stack:** Python · Flask · Groq API · Llama 3.3 70B · PyMuPDF · HTML · Tailwind CSS

An end-to-end AI web application that analyses your CV against any job description and delivers a **comprehensive, personalised career gap report in under 20 seconds** — for any role in any industry. Upload your CV, paste the job description, and get an instant skill gap breakdown, learning roadmap, salary insights, and actionable CV improvement tips.

---

## 📋 Table of Contents
- [Features](#features)
- [How to Run](#how-to-run)
- [Overview](#overview)
- [How the AI Works](#how-the-ai-works)
- [Analysis Output](#analysis-output)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting a Free Groq API Key](#getting-a-free-groq-api-key)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)

---

## Features

- **Instant Skill Gap Analysis** — See exactly which skills you have and which you're missing for the target role
- **Overall Match Score** — A 0–100% compatibility score with animated visual breakdown
- **Personalised Learning Roadmap** — Step-by-step plan with free and paid resources, real links, and time estimates
- **CV Improvement Tips** — Specific, actionable advice to strengthen your CV for the exact role you're targeting
- **Salary Insights** — Realistic salary ranges from entry through senior level
- **Experience Gap Analysis** — Honest assessment of your experience vs what the role requires
- **Career Strategy Advice** — 5 tailored tips from the AI career coach
- **Alternative Role Suggestions** — Similar roles you might be a stronger match for right now
- **Works for ANY industry** — Tech, medicine, law, finance, engineering, design, education, arts, hospitality and more
- **PDF & DOCX Support** — Upload your CV in any format or paste the text directly
- **User Authentication** — Register and sign in with email and password (localStorage)
- **Dark / Light Mode** — Sleek dark UI by default with light mode toggle
- **Fully Responsive** — Works on desktop and mobile

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- A free [Groq API key](https://console.groq.com) — takes 2 minutes to get

### 1. Clone the repository

```bash
git clone https://github.com/egwaojeangel/skillbridge-ai.git
cd skillbridge-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Open `run.py` and replace the placeholder:

```python
os.environ['GROQ_API_KEY'] = 'your-groq-api-key-here'
```

### 4. Run the application

```bash
python run.py
```

Open your browser at: **http://localhost:5000**

---

## Overview

Millions of people apply to jobs every year without knowing exactly why they keep getting rejected. The problem is rarely that they're unqualified — it's that they don't know which specific skills are missing, how to close those gaps, or how to present what they already have more effectively.

SkillBridge solves this by acting as a personal AI career coach that:

- Reads your CV and the job description simultaneously
- Identifies **specific skill gaps** with honest importance ratings (critical / important / nice-to-have)
- Builds a **step-by-step learning roadmap** with real resources tailored to the industry
- Gives **salary context** so you know what the role is worth before you apply
- Tells you **exactly how to rewrite your CV** to improve your chances for that specific role
- Suggests **alternative roles** you might already be ready for

The system is industry-agnostic by design. The same pipeline that analyses a Software Engineer's CV against a job description works equally well for a Nurse, a Tax Lawyer, a UX Designer, or a Civil Engineer — the AI adapts its advice to the specific field.

---

## How the AI Works

When a CV and job description are submitted, the backend sends both to **Llama 3.3 70B** — Meta's most capable open-source model, served via Groq's ultra-fast inference API — with a detailed system prompt instructing it to act as a world-class career coach.

The model is prompted to return a **structured JSON object** containing all analysis fields, which the frontend parses and renders into the results dashboard.

The AI is explicitly instructed to:
- Tailor all advice to the **specific industry** — never give generic tech advice for a healthcare or legal role
- Be **honest** about gaps while remaining constructive and encouraging
- Suggest **real, named resources** that are actually relevant to the field
- Provide **concrete, actionable** next steps the user can take this week
- Grade skill importance as **critical, important, or nice-to-have** — not just a flat list

### Why Groq + Llama 3.3 70B

| Factor | Detail |
|---|---|
| Model | Llama 3.3 70B (Meta) |
| Inference Speed | ~500 tokens/second via Groq |
| Cost | Free tier — no credit card required |
| Quality | Competitive with GPT-4 class models on structured tasks |
| Response Time | 10–20 seconds per full analysis |

---

## Analysis Output

Every analysis returns the following sections:

| Section | What It Contains |
|---|---|
| **Overall Match Score** | 0–100% compatibility with animated ring |
| **Summary** | 2–3 sentence honest assessment of candidacy |
| **Strengths** | Skills you already have with proficiency bars |
| **Skill Gaps** | Missing skills rated critical / important / nice-to-have |
| **Experience Analysis** | Required vs estimated years with honest advice |
| **Learning Roadmap** | Step-by-step plan with resources, duration, and free/paid tags |
| **Career Advice** | 5 strategic tips tailored to the role and industry |
| **Do This Week** | 3 concrete immediate actions |
| **CV Improvement Tips** | 6 specific suggestions to strengthen the CV for this role |
| **Salary Insight** | Entry, mid, and senior salary ranges |
| **Alternative Roles** | 3–6 similar roles with match scores |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Tailwind CSS (CDN), Vanilla JavaScript |
| Backend | Python 3.8+, Flask, Flask-CORS |
| AI Model | Llama 3.3 70B via Groq API |
| PDF Parsing | PyMuPDF (fitz) |
| DOCX Parsing | python-docx |
| Auth | Browser localStorage (client-side) |
| Fonts | Plus Jakarta Sans (Google Fonts) |
| Icons | Font Awesome 6 |

---

## Project Structure

```
skillbridge-ai/
├── app.py              # Flask backend — routes, file parsing, Groq API call
├── run.py              # Entry point — sets API key, starts server
├── index.html          # Complete frontend — single-file UI
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account (no credit card needed)
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key and paste it into `run.py`

Groq's free tier provides generous rate limits — more than enough for personal use, demos, and portfolio showcasing.

---

## Deployment

This app requires a running Python backend and **cannot be hosted on GitHub Pages** (which serves static files only).

### Free deployment on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repository
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python run.py`
5. Add an **Environment Variable:** `GROQ_API_KEY = your-key-here`
6. Click **Deploy**

You will get a live public URL (e.g. `skillbridge-ai.onrender.com`) in approximately 5 minutes. The free tier is sufficient for demo and portfolio use.

---

## Requirements

```
flask
flask-cors
groq
pymupdf
python-docx
python-dotenv
```

```bash
pip install -r requirements.txt
```

---

## Limitations

- CV parsing quality depends on formatting — heavily stylised CVs (columns, tables, graphics) may not parse cleanly from PDF
- No persistent database — user accounts and analysis history are stored in browser localStorage only and will not persist across devices
- Llama 3.3 70B may occasionally produce less nuanced advice for very niche or highly specialised roles compared to larger proprietary models
- The app requires a live backend — it cannot be hosted as a static site
- No rate limiting implemented — not suitable for high-traffic public deployment without adding request throttling
- Salary data is AI-estimated and may not reflect current market rates in all regions

---

## Future Work

- [ ] Export full analysis as a PDF report
- [ ] Save and compare multiple analyses per user
- [ ] LinkedIn profile URL input — auto-parse CV from profile
- [ ] Job board integration — auto-fetch job descriptions from URL
- [ ] Backend database for persistent user accounts
- [ ] Rate limiting and usage quotas for public deployment
- [ ] Email delivery of analysis results
- [ ] CV rewrite assistant — full AI-powered CV editor

---

## Disclaimer

> ⚠️ SkillBridge is an AI-assisted career tool intended to **support and inform** job seekers. It does not guarantee employment outcomes. All AI-generated advice should be critically reviewed and adapted to your personal circumstances. Salary figures are estimates based on AI knowledge and may not reflect current market rates in your region.

---

## Author

**Angel Egwaoje**

Machine Learning Engineer | AI Applications & Full-Stack Development

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/angel-egwaoje-416927280)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/egwaojeangel)
