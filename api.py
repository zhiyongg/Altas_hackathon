from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
from agent import run_itinerary_agent
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Itinerary Planner API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # This perfectly mimics Flask's request logging!
    logger.info(f"{request.client.host} - \"{request.method} {request.url.path}\" {response.status_code} - {process_time:.2f}ms")
    return response

# Allow CORS for the Vite frontend (usually runs on port 3000 or 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TripRequest(BaseModel):
    user_request: str
    custom_messages: str = ""

@app.post("/api/generate")
async def generate_trip(req: TripRequest):
    logger.info(f"Received request: {req.user_request}")
    try:
        result_json = run_itinerary_agent(req.user_request, req.custom_messages)
        
        # Safely parse the returned JSON string back into a Python dict so FastAPI can serialize it properly
        try:
            return json.loads(result_json)
        except json.JSONDecodeError:
            logger.error(f"Agent did not return valid JSON: {result_json}")
            raise HTTPException(status_code=500, detail="Agent did not return valid JSON")
            
    except Exception as e:
        logger.error(f"Error generating trip: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
