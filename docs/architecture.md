# 🏛️ KavachX Architecture

KavachX is designed as a **high-throughput asynchronous middleware layer** that sits between AI model providers and the end application. It operates on the principle of **In-Flight Governance**.

## 🔄 Data Flow Path

```text
[ Client App ] --(1) Request)--> [ KavachX Middleware ] --(2) Cleaned Request)--> [ AI Model ]
                                          |                                           |
                                 (3) Risk Scoring                            (4) Raw Response
                                          |                                           |
[ User View ] <--(6) Filtered)--- [ Policy Engine ] <--(5) Response Interception)-----┘
```

## 🧩 Core Components

### 1. The Interceptor (FastAPI)
A modular entry-point that can be integrated via:
*   **Direct API Proxies:** Point your OpenAI/Anthropic base URL to KavachX.
*   **SDK Wrappers:** Import the KavachX client into your Python application.
*   **Browser Extension:** Client-side interception of web-based AI tools.

### 2. The Multi-Gate Enforcement Engine
Every interaction passes through four specialized safety gates:
*   **Gate 1: 🔐 Security:** Detects prompt injections, NAEL attacks, and adversarial noise.
*   **Gate 3: 🧬 Safety:** ML-native classifiers for Hate, Harassment, and Harmful Intent.
*   **Gate 3: 📋 Compliance:** Regex and ML patterns for DPDP (PII) and regulatory adherence.
*   **Gate 4: 📊 Audit:** Real-time logging of the full interaction lifecycle.

### 3. The Evidence Chain (PostgreSQL)
All governance decisions are cryptographically hashed and logged to an immutable audit trail, providing auditable proof of compliance for regulatory inquiries.

## ⚡ Performance Optimization
*   **Async Processing:** Gates run in parallel where possible.
*   **Edge Classification:** Lightweight models for early-exit on high-risk prompts.
*   **Caching:** Risk telemetry for recurring prompt patterns is cached in Redis.
