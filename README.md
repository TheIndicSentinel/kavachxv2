# KavachX Governance Engine (v2.0) 🛡️

**KavachX** is a production-ready **Governance Framework** platform designed to monitor, audit, and regulate AI models in real-time. It provides a robust middle-layer between your LLMs/ML models and your users, ensuring every interaction is filtered for risk, fairness, and compliance.

![Main Dashboard](https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1200)

## 🚀 Vision
In the era of rapid AI adoption, trust is non-negotiable. KavachX provides a "Digital Armor" for enterprises, automating compliance with global standards like the **India DPDPA (2023)**, **EU AI Act**, and **RBI Fairness Guidelines**.

## ✨ Key Features
*   **Real-time Risk Engine:** Composite risk scoring for every inference.
*   **Fairness Monitor:** Automated detection of demographic disparities (e.g., Bias in credit scoring).
*   **Kavach Governance Layer:** Protect any external AI model via a single API Key.
*   **Executive Dashboard:** High-level metrics for non-technical stakeholders via WebSockets.
*   **Compliance Audit:** Immutable logging of violations and adherence logs.
*   **Dynamic Guardrails:** Real-time adjustment of thresholds without system downtime.

## 🏗️ Technical Stack
*   **Backend:** FastAPI (Python 3.10+), SQLAlchemy (Async), WebSockets.
*   **Frontend:** React (Vite), CSS3 (Modern/Glassmorphic), Lucide-Icons.
*   **Architecture:** Monolithic for easy deployment with support for headless API usage.

## 📦 Deployment (Single-Platform Render)
KavachX is optimized for **Render.com**.

1.  **Build Command:**
    `cd frontend && npm install && npm run build && cd ../backend && pip install -r requirements.txt`
2.  **Start Command:**
    `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3.  **Env Vars:** `SECRET_KEY`, `ENVIRONMENT=production`.

## 🛡️ License
Proprietary / Enterprise Kavach Framework. See **compliance_officer** role documentation for usage policies.

---
*Built with precision for the next generation of Responsible AI.*
