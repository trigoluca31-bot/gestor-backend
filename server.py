import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Carrega variáveis locais (.env) se existirem
load_dotenv()

# =========================
# CONFIGURAÇÕES
# =========================

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = os.environ.get("DB_NAME", "gestor_db")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI não definida nas variáveis de ambiente")

# =========================
# APP
# =========================

app = FastAPI(title="Gestor Planner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGINS] if CORS_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# BANCO DE DADOS
# =========================

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# =========================
# ROTAS BÁSICAS (TESTE)
# =========================

@app.get("/")
async def root():
    return {"status": "ok", "message": "API online 🚀"}

@app.get("/health")
async def health():
    return {"database": "connected", "app": "running"}

# =========================
# EXEMPLO DE ROTA CRUD SIMPLES
# =========================

@app.post("/events")
async def create_event(event: dict):
    result = await db.events.insert_one(event)
    return {"id": str(result.inserted_id)}

@app.get("/events")
async def list_events():
    events = []
    async for event in db.events.find():
        event["_id"] = str(event["_id"])
        events.append(event)
    return events

