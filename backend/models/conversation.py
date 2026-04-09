from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Message(BaseModel):
    sender: str  # "user" or "ai"
    content: str
    timestamp: datetime

class ConversationBase(BaseModel):
    user_id: str
    mode: str  # "training" or "application"
    messages: List[Message] = []

class ConversationCreate(ConversationBase):
    pass

class ConversationInDB(ConversationBase):
    id: str
    created_at: datetime
    updated_at: datetime

class ConversationResponse(ConversationBase):
    id: str
    created_at: datetime</content>
<parameter name="filePath">c:\Users\moham\Desktop\alia-web-main\backend\models\conversation.py