import os

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(find_dotenv(usecwd=True))  # reads a .env file in the current working directory (if present)

from classify import router as classify_router
from analyze import router as analyze_router
from champion_detection import router as champion_detection_router

app = FastAPI()

ALLOWED_ORIGINS = [
    "https://bettergameplay.com",
    "https://www.bettergameplay.com",
]

if os.getenv("RUNNING_LOCALLY", "false").lower() == "true":
    ALLOWED_ORIGINS.append("http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(classify_router)
app.include_router(analyze_router)
app.include_router(champion_detection_router)