from fastapi import APIRouter, Request, HTTPException, Depends
from typing import Dict, Any, List
import httpx
import uuid
import time
from app.services.governance_service import governance_service
from app.models.schemas import InferenceRequest
from app.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.orm_models import AIModel
from sqlalchemy import select

router = APIRouter()

# UNIVERSAL OPENAI PROXY ENDPOINT
# This allows any app (LangChain, Python, etc.) to be "governed" by just changing the base URL.
@router.post("/openai/chat/completions")
async def proxy_openai_chat(request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json()
    prompt = ""
    messages = body.get("messages", [])
    if messages:
        # Get the last message as the primary prompt
        prompt = messages[-1].get("content", "")

    # 1. Evaluate via KavachX Governance Engine
    # Find or create a proxy model record
    result_model = await db.execute(select(AIModel).filter(AIModel.name == "Kavach Universal Proxy"))
    model = result_model.scalars().first()
    if not model:
        model = AIModel(name="Kavach Universal Proxy", version="1.0", model_type="proxy")
        db.add(model)
        await db.commit()
        await db.refresh(model)

    gov_request = InferenceRequest(
        model_id=model.id,
        session_id=str(uuid.uuid4()),
        input_data={"prompt": prompt, "full_payload": body},
        prediction={"text": "Intercepted by Kavach Proxy"},
        confidence=1.0,
        context={"source": "api_proxy", "platform": "OpenAI-Compatible"}
    )
    
    gov_result = await governance_service.evaluate_inference(gov_request, db, model)
    
    if gov_result.enforcement_decision == "BLOCK":
        # Return OpenAI-formatted error
        return {
            "error": {
                "message": f"KavachX Governance Block: {gov_result.explanation.reason}",
                "type": "governance_policy_violation",
                "param": None,
                "code": "policy_blocked"
            }
        }

    # 2. Return a simulated response (In a real setup, this would forward to OpenAI)
    # The 'Universal' goal here is capturing and regulating.
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-3.5-turbo"),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "KavachX Universal Proxy: Your request was analyzed and allowed."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": 10, "total_tokens": (len(prompt)//4)+10}
    }
