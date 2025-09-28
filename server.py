from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import hashlib

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()

# Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    twitter_id: Optional[str] = None
    twitter_username: Optional[str] = None
    twitter_display_name: Optional[str] = None
    wallet_address: Optional[str] = None
    total_points: int = 0
    tasks_completed: List[str] = Field(default_factory=list)
    last_checkin: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    task_type: str  # "payment" or "twitter_follow"
    points: int = 25
    target_address: Optional[str] = None
    payment_amount: Optional[float] = None
    twitter_username: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    task_id: str
    completed: bool = False
    transaction_hash: Optional[str] = None
    completed_at: Optional[datetime] = None
    points_awarded: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TwitterAuthRequest(BaseModel):
    twitter_id: str
    username: str
    display_name: str

class WalletConnectRequest(BaseModel):
    wallet_address: str
    signature: str
    message: str

class PaymentVerificationRequest(BaseModel):
    transaction_hash: str
    wallet_address: str

class CheckinRequest(BaseModel):
    user_id: str

# Helper functions
def create_jwt_token(user_data: dict):
    """Create JWT token for user authentication"""
    payload = {
        "user_id": user_data["id"],
        "twitter_id": user_data.get("twitter_id"),
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, "your-secret-key", algorithm="HS256")

def prepare_for_mongo(data):
    """Convert datetime objects to ISO format for MongoDB storage"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
    return data

def parse_from_mongo(item):
    """Parse datetime fields from MongoDB"""
    if isinstance(item, dict):
        for key, value in item.items():
            if key.endswith('_at') or key == 'last_checkin':
                if isinstance(value, str):
                    try:
                        item[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except:
                        pass
    return item

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, "your-secret-key", algorithms=["HS256"])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_data = await db.users.find_one({"id": user_id}, {"_id": 0})
        if user_data is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return parse_from_mongo(user_data)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize default tasks
async def init_default_tasks():
    """Initialize default tasks if they don't exist"""
    existing_tasks = await db.tasks.count_documents({})
    if existing_tasks == 0:
        default_tasks = [
            Task(
                title="Early Adopter Payment",
                description="Send 0.0005 BNB to support the platform",
                task_type="payment",
                points=25,
                target_address="0x191cDc16FD67F22ADfb86E364a74073DfBD6634D",
                payment_amount=0.0005
            ),
        ]
        
        # Add 9 Twitter follow tasks
        for i in range(9):
            default_tasks.append(Task(
                title=f"Follow @khadijarab2517 {i+1}",
                description="Follow @khadijarab2517 on Twitter",
                task_type="twitter_follow",
                points=25,
                twitter_username="khadijarab2517"
            ))
        
        for task in default_tasks:
            await db.tasks.insert_one(prepare_for_mongo(task.dict()))

# Routes
@api_router.get("/")
async def root():
    return {"message": "Crypto Social Tasks Platform API"}

@api_router.post("/auth/twitter")
async def twitter_auth(auth_data: TwitterAuthRequest):
    """Authenticate user with Twitter credentials"""
    # Check if user exists
    existing_user = await db.users.find_one({"twitter_id": auth_data.twitter_id}, {"_id": 0})
    
    if existing_user:
        user_data = parse_from_mongo(existing_user)
        # Update last login
        await db.users.update_one(
            {"twitter_id": auth_data.twitter_id},
            {"$set": {"updated_at": datetime.now(timezone.utc).isoformat()}}
        )
    else:
        # Create new user
        user_data = User(
            twitter_id=auth_data.twitter_id,
            twitter_username=auth_data.username,
            twitter_display_name=auth_data.display_name
        ).dict()
        
        await db.users.insert_one(prepare_for_mongo(user_data.copy()))
    
    # Create JWT token
    token = create_jwt_token(user_data)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_data
    }

@api_router.post("/wallet/connect")
async def connect_wallet(wallet_data: WalletConnectRequest, current_user: dict = Depends(get_current_user)):
    """Connect MetaMask wallet to user account"""
    # In production, verify the signature here
    # For now, we'll accept the wallet address
    
    await db.users.update_one(
        {"id": current_user["id"]},
        {"$set": {
            "wallet_address": wallet_data.wallet_address,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {"message": "Wallet connected successfully", "wallet_address": wallet_data.wallet_address}

@api_router.get("/tasks")
async def get_tasks():
    """Get all available tasks"""
    tasks = await db.tasks.find({"is_active": True}, {"_id": 0}).to_list(1000)
    return [parse_from_mongo(task) for task in tasks]

@api_router.get("/user/tasks")
async def get_user_tasks(current_user: dict = Depends(get_current_user)):
    """Get user's task completion status"""
    user_tasks = await db.user_tasks.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    return [parse_from_mongo(task) for task in user_tasks]

@api_router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, current_user: dict = Depends(get_current_user)):
    """Mark a task as completed (for Twitter follow tasks)"""
    # Get task details
    task = await db.tasks.find_one({"id": task_id}, {"_id": 0})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if already completed
    existing_completion = await db.user_tasks.find_one({
        "user_id": current_user["id"],
        "task_id": task_id,
        "completed": True
    })
    
    if existing_completion:
        raise HTTPException(status_code=400, detail="Task already completed")
    
    # For Twitter follow tasks, we'll mock the verification for now
    if task["task_type"] == "twitter_follow":
        # Create task completion record
        user_task = UserTask(
            user_id=current_user["id"],
            task_id=task_id,
            completed=True,
            completed_at=datetime.now(timezone.utc),
            points_awarded=task["points"]
        )
        
        await db.user_tasks.insert_one(prepare_for_mongo(user_task.dict()))
        
        # Update user points
        await db.users.update_one(
            {"id": current_user["id"]},
            {"$inc": {"total_points": task["points"]}}
        )
        
        return {"message": "Task completed successfully", "points_awarded": task["points"]}
    
    raise HTTPException(status_code=400, detail="Invalid task type for this endpoint")

@api_router.post("/payment/verify")
async def verify_payment(payment_data: PaymentVerificationRequest, current_user: dict = Depends(get_current_user)):
    """Verify BNB payment for early adopter task"""
    # For now, we'll mock the payment verification
    # In production, this would verify the transaction on BSC
    
    # Find the payment task
    payment_task = await db.tasks.find_one({"task_type": "payment"}, {"_id": 0})
    if not payment_task:
        raise HTTPException(status_code=404, detail="Payment task not found")
    
    # Check if already completed
    existing_completion = await db.user_tasks.find_one({
        "user_id": current_user["id"],
        "task_id": payment_task["id"],
        "completed": True
    })
    
    if existing_completion:
        raise HTTPException(status_code=400, detail="Payment already verified")
    
    # Mock payment verification (in production, verify on blockchain)
    if len(payment_data.transaction_hash) > 10:  # Basic validation
        user_task = UserTask(
            user_id=current_user["id"],
            task_id=payment_task["id"],
            completed=True,
            transaction_hash=payment_data.transaction_hash,
            completed_at=datetime.now(timezone.utc),
            points_awarded=payment_task["points"]
        )
        
        await db.user_tasks.insert_one(prepare_for_mongo(user_task.dict()))
        
        # Update user points
        await db.users.update_one(
            {"id": current_user["id"]},
            {"$inc": {"total_points": payment_task["points"]}}
        )
        
        return {"message": "Payment verified successfully", "points_awarded": payment_task["points"]}
    
    raise HTTPException(status_code=400, detail="Invalid transaction hash")

@api_router.post("/checkin")
async def daily_checkin(current_user: dict = Depends(get_current_user)):
    """Daily check-in for bonus points"""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Check if already checked in today
    user_data = await db.users.find_one({"id": current_user["id"]}, {"_id": 0})
    user_data = parse_from_mongo(user_data)
    
    if user_data.get("last_checkin"):
        last_checkin = user_data["last_checkin"]
        if isinstance(last_checkin, str):
            last_checkin = datetime.fromisoformat(last_checkin.replace('Z', '+00:00'))
        
        if last_checkin >= today_start:
            raise HTTPException(status_code=400, detail="Already checked in today")
    
    # Award check-in points
    checkin_points = 5
    await db.users.update_one(
        {"id": current_user["id"]},
        {
            "$set": {"last_checkin": now.isoformat()},
            "$inc": {"total_points": checkin_points}
        }
    )
    
    return {"message": "Check-in successful", "points_awarded": checkin_points}

@api_router.get("/leaderboard")
async def get_leaderboard(limit: int = 50):
    """Get global leaderboard"""
    pipeline = [
        {"$match": {"total_points": {"$gt": 0}}},
        {"$sort": {"total_points": -1}},
        {"$limit": limit},
        {"$project": {
            "_id": 0,
            "id": 1,
            "twitter_username": 1,
            "twitter_display_name": 1,
            "total_points": 1,
            "wallet_address": 1
        }}
    ]
    
    leaderboard = await db.users.aggregate(pipeline).to_list(limit)
    
    # Add rank
    for i, user in enumerate(leaderboard):
        user["rank"] = i + 1
    
    return leaderboard

@api_router.get("/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return parse_from_mongo(current_user)

@api_router.get("/user/stats")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    """Get user statistics"""
    total_tasks = await db.tasks.count_documents({"is_active": True})
    completed_tasks = await db.user_tasks.count_documents({
        "user_id": current_user["id"],
        "completed": True
    })
    
    # Get user rank
    users_above = await db.users.count_documents({
        "total_points": {"$gt": current_user["total_points"]}
    })
    rank = users_above + 1
    
    return {
        "total_points": current_user["total_points"],
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "rank": rank,
        "last_checkin": current_user.get("last_checkin")
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize default tasks on startup"""
    await init_default_tasks()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()