uvicorn server:app --host 0.0.0.0 --port 10000

DB_NAME=gestor

from dotenv import load_dotenv
import os

# Carrega as variáveis de ambiente
load_dotenv()

# Pega o nome do banco de dados
DB_NAME = os.environ.get("DB_NAME")

# Verifica se o nome foi carregado corretamente
if not DB_NAME:
    print("Erro: A variável DB_NAME não está configurada!")
else:
    print(f"Banco de dados: {DB_NAME}")

# Agora, você pode continuar com a conexão ao banco
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.environ.get("MONGO_URI")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
import os

app = FastAPI()

# Pegando as variáveis de ambiente
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = os.environ.get("DB_NAME")

# Conectando ao MongoDB de forma assíncrona
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

@app.get("/health")
def health_check():
    return {"status": "ok"}


fastapi
uvicorn[standard]
motor
python-dotenv




from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import bcrypt
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env', override=False)

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = "HS256"

# ============= MODELS =============

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Goal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    description: str
    target_date: str
    progress: int = 0
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GoalCreate(BaseModel):
    title: str
    description: str
    target_date: str

class GoalUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    target_date: Optional[str] = None
    progress: Optional[int] = None
    status: Optional[str] = None

class Routine(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    description: str
    time: str
    days: List[str]
    completed_today: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RoutineCreate(BaseModel):
    title: str
    description: str
    time: str
    days: List[str]

class RoutineUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    time: Optional[str] = None
    days: Optional[List[str]] = None
    completed_today: Optional[bool] = None

class Note(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    date: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NoteCreate(BaseModel):
    date: str
    content: str

class NoteUpdate(BaseModel):
    content: str

class Meal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    meal_type: str
    name: str
    calories: int
    protein: int
    carbs: int
    fats: int
    date: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MealCreate(BaseModel):
    meal_type: str
    name: str
    calories: int
    protein: int
    carbs: int
    fats: int
    date: str

class MealStatusUpdate(BaseModel):
    status: str

class CalendarEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    description: str
    date: str
    time: str
    type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CalendarEventCreate(BaseModel):
    title: str
    description: str
    date: str
    time: str
    type: str

class CalendarEventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    type: Optional[str] = None

class ShoppingItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    category: str
    completed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ShoppingItemCreate(BaseModel):
    name: str
    category: str

class ShoppingItemUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    completed: Optional[bool] = None

class ChatMessageCreate(BaseModel):
    message: str

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatResponse(BaseModel):
    response: str

class PerformanceStats(BaseModel):
    goals_completion: float
    routine_completion: float
    diet_adherence: float
    weekly_performance: List[dict]

# ============= AUTH UTILITIES =============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(days=7)
    return jwt.encode(
        {"user_id": user_id, "exp": expiration},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["user_id"]
    except:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# ============= ROOT =============

@api_router.get("/")
async def root():
    return {"message": "Gestor API"}

# ============= AUTH ROUTES =============

@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    existing = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(email=user_data.email, name=user_data.name)
    user_dict = user.model_dump()
    user_dict['created_at'] = user_dict['created_at'].isoformat()
    user_dict['password'] = hash_password(user_data.password)
    
    await db.users.insert_one(user_dict)
    token = create_token(user.id)
    
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.name}}

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user or not verify_password(credentials.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user['id'])
    return {"token": token, "user": {"id": user['id'], "email": user['email'], "name": user['name']}}

# ============= GOALS ROUTES =============

@api_router.post("/goals", response_model=Goal)
async def create_goal(goal_data: GoalCreate, user_id: str = Depends(get_current_user)):
    goal = Goal(user_id=user_id, **goal_data.model_dump())
    goal_dict = goal.model_dump()
    goal_dict['created_at'] = goal_dict['created_at'].isoformat()
    await db.goals.insert_one(goal_dict)
    return goal

@api_router.get("/goals", response_model=List[Goal])
async def get_goals(user_id: str = Depends(get_current_user)):
    goals = await db.goals.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    for goal in goals:
        if isinstance(goal['created_at'], str):
            goal['created_at'] = datetime.fromisoformat(goal['created_at'])
    return goals

@api_router.put("/goals/{goal_id}", response_model=Goal)
async def update_goal(goal_id: str, goal_data: GoalUpdate, user_id: str = Depends(get_current_user)):
    update_data = {k: v for k, v in goal_data.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")
    
    result = await db.goals.update_one(
        {"id": goal_id, "user_id": user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    goal = await db.goals.find_one({"id": goal_id}, {"_id": 0})
    if isinstance(goal['created_at'], str):
        goal['created_at'] = datetime.fromisoformat(goal['created_at'])
    return Goal(**goal)

@api_router.delete("/goals/{goal_id}")
async def delete_goal(goal_id: str, user_id: str = Depends(get_current_user)):
    result = await db.goals.delete_one({"id": goal_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Goal not found")
    return {"message": "Goal deleted"}

# ============= ROUTINES ROUTES =============

@api_router.post("/routines", response_model=Routine)
async def create_routine(routine_data: RoutineCreate, user_id: str = Depends(get_current_user)):
    routine = Routine(user_id=user_id, **routine_data.model_dump())
    routine_dict = routine.model_dump()
    routine_dict['created_at'] = routine_dict['created_at'].isoformat()
    await db.routines.insert_one(routine_dict)
    return routine

@api_router.get("/routines", response_model=List[Routine])
async def get_routines(user_id: str = Depends(get_current_user)):
    routines = await db.routines.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    for routine in routines:
        if isinstance(routine['created_at'], str):
            routine['created_at'] = datetime.fromisoformat(routine['created_at'])
    return routines

@api_router.put("/routines/{routine_id}", response_model=Routine)
async def update_routine(routine_id: str, routine_data: RoutineUpdate, user_id: str = Depends(get_current_user)):
    update_data = {k: v for k, v in routine_data.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")
    
    result = await db.routines.update_one(
        {"id": routine_id, "user_id": user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Routine not found")
    
    routine = await db.routines.find_one({"id": routine_id}, {"_id": 0})
    if isinstance(routine['created_at'], str):
        routine['created_at'] = datetime.fromisoformat(routine['created_at'])
    return Routine(**routine)

@api_router.delete("/routines/{routine_id}")
async def delete_routine(routine_id: str, user_id: str = Depends(get_current_user)):
    result = await db.routines.delete_one({"id": routine_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Routine not found")
    return {"message": "Routine deleted"}

# ============= NOTES ROUTES =============

@api_router.post("/notes", response_model=Note)
async def create_note(note_data: NoteCreate, user_id: str = Depends(get_current_user)):
    note = Note(user_id=user_id, **note_data.model_dump())
    note_dict = note.model_dump()
    note_dict['created_at'] = note_dict['created_at'].isoformat()
    await db.notes.insert_one(note_dict)
    return note

@api_router.get("/notes", response_model=List[Note])
async def get_notes(user_id: str = Depends(get_current_user)):
    notes = await db.notes.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    for note in notes:
        if isinstance(note['created_at'], str):
            note['created_at'] = datetime.fromisoformat(note['created_at'])
    return notes

@api_router.get("/notes/{date}")
async def get_note_by_date(date: str, user_id: str = Depends(get_current_user)):
    note = await db.notes.find_one({"user_id": user_id, "date": date}, {"_id": 0})
    if not note:
        return None
    if isinstance(note['created_at'], str):
        note['created_at'] = datetime.fromisoformat(note['created_at'])
    return Note(**note)

@api_router.put("/notes/{note_id}", response_model=Note)
async def update_note(note_id: str, note_data: NoteUpdate, user_id: str = Depends(get_current_user)):
    result = await db.notes.update_one(
        {"id": note_id, "user_id": user_id},
        {"$set": note_data.model_dump()}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    
    note = await db.notes.find_one({"id": note_id}, {"_id": 0})
    if isinstance(note['created_at'], str):
        note['created_at'] = datetime.fromisoformat(note['created_at'])
    return Note(**note)

@api_router.delete("/notes/{note_id}")
async def delete_note(note_id: str, user_id: str = Depends(get_current_user)):
    result = await db.notes.delete_one({"id": note_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"message": "Note deleted"}

# ============= MEALS ROUTES =============

@api_router.post("/meals", response_model=Meal)
async def create_meal(meal_data: MealCreate, user_id: str = Depends(get_current_user)):
    meal = Meal(user_id=user_id, **meal_data.model_dump())
    meal_dict = meal.model_dump()
    meal_dict['created_at'] = meal_dict['created_at'].isoformat()
    await db.meals.insert_one(meal_dict)
    return meal

@api_router.get("/meals", response_model=List[Meal])
async def get_meals(date: Optional[str] = None, user_id: str = Depends(get_current_user)):
    query = {"user_id": user_id}
    if date:
        query["date"] = date
    meals = await db.meals.find(query, {"_id": 0}).to_list(100)
    for meal in meals:
        if isinstance(meal['created_at'], str):
            meal['created_at'] = datetime.fromisoformat(meal['created_at'])
    return meals

@api_router.put("/meals/{meal_id}/status")
async def update_meal_status(meal_id: str, status_data: MealStatusUpdate, user_id: str = Depends(get_current_user)):
    result = await db.meals.update_one(
        {"id": meal_id, "user_id": user_id},
        {"$set": {"status": status_data.status}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Meal not found")
    
    return {"message": "Meal status updated"}

@api_router.delete("/meals/{meal_id}")
async def delete_meal(meal_id: str, user_id: str = Depends(get_current_user)):
    result = await db.meals.delete_one({"id": meal_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Meal not found")
    return {"message": "Meal deleted"}

# ============= CALENDAR ROUTES =============

@api_router.post("/calendar", response_model=CalendarEvent)
async def create_event(event_data: CalendarEventCreate, user_id: str = Depends(get_current_user)):
    event = CalendarEvent(user_id=user_id, **event_data.model_dump())
    event_dict = event.model_dump()
    event_dict['created_at'] = event_dict['created_at'].isoformat()
    await db.calendar_events.insert_one(event_dict)
    return event

@api_router.get("/calendar", response_model=List[CalendarEvent])
async def get_events(user_id: str = Depends(get_current_user)):
    events = await db.calendar_events.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    for event in events:
        if isinstance(event['created_at'], str):
            event['created_at'] = datetime.fromisoformat(event['created_at'])
    return events

@api_router.put("/calendar/{event_id}", response_model=CalendarEvent)
async def update_event(event_id: str, event_data: CalendarEventUpdate, user_id: str = Depends(get_current_user)):
    update_data = {k: v for k, v in event_data.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")
    
    result = await db.calendar_events.update_one(
        {"id": event_id, "user_id": user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    
    event = await db.calendar_events.find_one({"id": event_id}, {"_id": 0})
    if isinstance(event['created_at'], str):
        event['created_at'] = datetime.fromisoformat(event['created_at'])
    return CalendarEvent(**event)

@api_router.delete("/calendar/{event_id}")
async def delete_event(event_id: str, user_id: str = Depends(get_current_user)):
    result = await db.calendar_events.delete_one({"id": event_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted"}

# ============= AI CHAT ROUTES =============

@api_router.post("/ai/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessageCreate, user_id: str = Depends(get_current_user)):
    user_msg = ChatMessage(user_id=user_id, role="user", content=message.message)
    user_msg_dict = user_msg.model_dump()
    user_msg_dict['created_at'] = user_msg_dict['created_at'].isoformat()
    await db.chat_messages.insert_one(user_msg_dict)
    
    goals = await db.goals.find({"user_id": user_id}, {"_id": 0}).to_list(10)
    routines = await db.routines.find({"user_id": user_id}, {"_id": 0}).to_list(10)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    meals = await db.meals.find({"user_id": user_id, "date": today}, {"_id": 0}).to_list(10)
    
    context = f"""Você é um assistente pessoal de planejamento e saúde. Ajude o usuário com suas metas, rotinas e dieta.
    
Contexto do usuário:
- Metas ativas: {len([g for g in goals if g.get('status') == 'active'])}
- Rotinas do dia: {len(routines)}
- Refeições hoje: {len(meals)}

Seja motivador, realista e útil. Forneça conselhos práticos e personalizados."""
    
    try:
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=user_id,
            system_message=context
        ).with_model("openai", "gpt-5.2")
        
        user_message = UserMessage(text=message.message)
        ai_response = await chat.send_message(user_message)
        
        ai_msg = ChatMessage(user_id=user_id, role="assistant", content=ai_response)
        ai_msg_dict = ai_msg.model_dump()
        ai_msg_dict['created_at'] = ai_msg_dict['created_at'].isoformat()
        await db.chat_messages.insert_one(ai_msg_dict)
        
        return ChatResponse(response=ai_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

@api_router.get("/ai/chat/history", response_model=List[ChatMessage])
async def get_chat_history(user_id: str = Depends(get_current_user)):
    messages = await db.chat_messages.find(
        {"user_id": user_id},
        {"_id": 0}
    ).sort("created_at", 1).to_list(100)
    
    for msg in messages:
        if isinstance(msg['created_at'], str):
            msg['created_at'] = datetime.fromisoformat(msg['created_at'])
    return messages

# ============= SHOPPING LIST ROUTES =============

@api_router.post("/shopping", response_model=ShoppingItem)
async def create_shopping_item(item_data: ShoppingItemCreate, user_id: str = Depends(get_current_user)):
    item = ShoppingItem(user_id=user_id, **item_data.model_dump())
    item_dict = item.model_dump()
    item_dict['created_at'] = item_dict['created_at'].isoformat()
    await db.shopping_items.insert_one(item_dict)
    return item

@api_router.get("/shopping", response_model=List[ShoppingItem])
async def get_shopping_items(user_id: str = Depends(get_current_user)):
    items = await db.shopping_items.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    for item in items:
        if isinstance(item['created_at'], str):
            item['created_at'] = datetime.fromisoformat(item['created_at'])
    return items

@api_router.put("/shopping/{item_id}", response_model=ShoppingItem)
async def update_shopping_item(item_id: str, item_data: ShoppingItemUpdate, user_id: str = Depends(get_current_user)):
    update_data = {k: v for k, v in item_data.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")
    
    result = await db.shopping_items.update_one(
        {"id": item_id, "user_id": user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    
    item = await db.shopping_items.find_one({"id": item_id}, {"_id": 0})
    if isinstance(item['created_at'], str):
        item['created_at'] = datetime.fromisoformat(item['created_at'])
    return ShoppingItem(**item)

@api_router.delete("/shopping/{item_id}")
async def delete_shopping_item(item_id: str, user_id: str = Depends(get_current_user)):
    result = await db.shopping_items.delete_one({"id": item_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Item deleted"}

# ============= PERFORMANCE ROUTES =============

@api_router.get("/performance/stats", response_model=PerformanceStats)
async def get_performance_stats(user_id: str = Depends(get_current_user)):
    goals = await db.goals.find({"user_id": user_id, "status": "active"}, {"_id": 0}).to_list(100)
    avg_goal_progress = sum(g.get('progress', 0) for g in goals) / len(goals) if goals else 0
    
    routines = await db.routines.find({"user_id": user_id}, {"_id": 0}).to_list(100)
    completed_routines = sum(1 for r in routines if r.get('completed_today', False))
    routine_completion = (completed_routines / len(routines) * 100) if routines else 0
    
    date_range = [(datetime.now(timezone.utc) - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    all_meals = await db.meals.find(
        {"user_id": user_id, "date": {"$in": date_range}}, 
        {"_id": 0}
    ).to_list(1000)
    
    meals_by_date = {}
    for meal in all_meals:
        date = meal.get('date')
        if date not in meals_by_date:
            meals_by_date[date] = []
        meals_by_date[date].append(meal)
    
    diet_adherence = 0
    for date in date_range:
        meals = meals_by_date.get(date, [])
        if meals:
            completed_meals = sum(1 for m in meals if m.get('status') == 'completed')
            diet_adherence += (completed_meals / len(meals) * 100)
    diet_adherence = diet_adherence / 7 if diet_adherence > 0 else 0
    
    weekly_performance = []
    for i in range(7):
        date = (datetime.now(timezone.utc) - timedelta(days=6-i)).strftime('%Y-%m-%d')
        day_meals = meals_by_date.get(date, [])
        completed = sum(1 for m in day_meals if m.get('status') == 'completed')
        performance = (completed / len(day_meals) * 100) if day_meals else 0
        
        weekly_performance.append({
            "date": date,
            "performance": round(performance, 1)
        })
    
    return PerformanceStats(
        goals_completion=round(avg_goal_progress, 1),
        routine_completion=round(routine_completion, 1),
        diet_adherence=round(diet_adherence, 1),
        weekly_performance=weekly_performance
    )

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
