import os

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(find_dotenv(usecwd=True))  # reads a .env file in the current working directory (if present)

from classify import router as classify_router
from analyze import router as analyze_router
from champion_detection import router as champion_detection_router

app = FastAPI()


@app.middleware("http")
async def log_http_requests(request, call_next):
    print(f"--> {request.method} {request.url.path}", flush=True)
    original_receive = request.receive
    body_started = False
    body_bytes = 0

    async def receive_with_diagnostics():
        nonlocal body_started, body_bytes
        message = await original_receive()
        if message["type"] == "http.request":
            chunk = message.get("body", b"")
            body_bytes += len(chunk)
            if not body_started and chunk:
                body_started = True
                print(
                    f"HTTP BODY 1: first request body chunk received ({len(chunk)} bytes)",
                    flush=True,
                )
            if not message.get("more_body", False):
                print(f"HTTP BODY 2: request body complete ({body_bytes} bytes)", flush=True)
        return message

    request._receive = receive_with_diagnostics
    try:
        response = await call_next(request)
    except Exception:
        import traceback

        print(f"HTTP request failed: {request.method} {request.url.path}", flush=True)
        traceback.print_exc()
        raise
    print(f"<-- {request.method} {request.url.path} {response.status_code}", flush=True)
    return response

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


@app.get("/api/debug/ping")
def debug_ping():
    print("DEBUG PING: endpoint reached", flush=True)
    return {"status": "ok", "endpoint": "debug_ping"}


@app.post("/api/debug/upload")
async def debug_upload(request: Request):
    print("DEBUG UPLOAD 1: endpoint reached; reading raw request body", flush=True)
    content_length = request.headers.get("content-length", "unknown")
    content_type = request.headers.get("content-type", "unknown").split(";", 1)[0]
    print(
        f"DEBUG UPLOAD 2: headers received content_length={content_length} content_type={content_type}",
        flush=True,
    )

    total_bytes = 0
    chunks = 0
    async for chunk in request.stream():
        total_bytes += len(chunk)
        chunks += 1
        if chunks == 1:
            print(f"DEBUG UPLOAD 3: first body chunk received ({len(chunk)} bytes)", flush=True)

    print(
        f"DEBUG UPLOAD 4: raw request body complete ({total_bytes} bytes, {chunks} chunks)",
        flush=True,
    )
    return {
        "status": "ok",
        "endpoint": "debug_upload",
        "bytes_received": total_bytes,
        "chunks_received": chunks,
    }


@app.post("/api/debug/file-upload")
async def debug_file_upload(video: UploadFile = File(...)):
    print("DEBUG FILE UPLOAD 1: UploadFile parsed; endpoint reached", flush=True)
    total_bytes = 0
    chunks = 0
    while True:
        chunk = await video.read(1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        chunks += 1
        if chunks == 1:
            print(f"DEBUG FILE UPLOAD 2: first file chunk read ({len(chunk)} bytes)", flush=True)

    print(
        f"DEBUG FILE UPLOAD 3: file read complete ({total_bytes} bytes, {chunks} chunks)",
        flush=True,
    )
    return {
        "status": "ok",
        "endpoint": "debug_file_upload",
        "bytes_received": total_bytes,
        "chunks_received": chunks,
    }


app.include_router(classify_router)
app.include_router(analyze_router)
app.include_router(champion_detection_router)