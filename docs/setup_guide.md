# 🚀 KavachX Setup Guide

Follow these steps to set up your local development environment for KavachX.

## 🛠️ Prerequisites
*   Python 3.10+
*   Node.js 18+ (for dashboard)
*   PostgreSQL (optional for local, defaults to SQLite if not configured)
*   Redis (optional, for performance caching)

## 📥 1. Clone the Repository
```bash
git clone https://github.com/TheIndicSentinel/kavachxv2.git
cd kavachxv2
```

## 🐍 2. Backend Setup (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure Environment Variables
Create a `.env` file in the `backend/` directory:
```env
SECRET_KEY=yoursecretkey
POSTGRES_URL=postgresql://user:pass@localhost:5432/kavachx
OPENAI_API_KEY=your_key_for_testing
ENVIRONMENT=development
```

## ⚛️ 3. Frontend Setup (React)
```bash
cd ../frontend
npm install
npm run dev
```

## 🛡️ 4. Running the Engine
From the `backend/` directory:
```bash
uvicorn app.main:app --reload
```

## 🧪 5. Testing the Safety Gates
You can verify the governance logic using the provided test scripts:
```bash
python tests/test_moderation.py
```

## 🌐 6. Deployment (Render/Docker)
KavachX includes a `Dockerfile` for containerized deployment. For Render users, the `render.yaml` configuration is located in the root.
