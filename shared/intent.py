# shared_intent.py  (llm_server.py와 main.py 둘 다 import)
from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List, Any

class Intent(str, Enum):
    ROUTE_REQUEST = "ROUTE_REQUEST"
    PLACE_INFO    = "PLACE_INFO"
    CALL_STAFF    = "CALL_STAFF"
    TRANSLATE     = "TRANSLATE"
    SET_LANGUAGE  = "SET_LANGUAGE"
    SMALL_TALK    = "SMALL_TALK"
    ALARM_EMERGENCY = "ALARM_EMERGENCY"
    UNKNOWN       = "UNKNOWN"

class Action(BaseModel):
    service: Literal["NAV","INFO","CALL","NLP","UI","SAFETY"]
    name: str
    params: Optional[Dict[str, Any]] = None

class IntentResult(BaseModel):
    intent: Intent = Intent.UNKNOWN
    slots: Dict[str, Any] = Field(default_factory=dict)
    actions: List[Action] = Field(default_factory=list)
    speak: str = ""
    ui: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    need_confirmation: Optional[bool] = None
    ask_user: Optional[str] = None
    end_conversation: Optional[bool] = None
