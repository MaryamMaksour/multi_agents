from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class AgentType(Enum):
    SUB_AGENT = "sub_agent"
    ORCHESTRATOR = "orchestrator"

@dataclass
class ProviderSpec:
    name: str
    type: AgentType
    system_prompt: str
    history_table: str
    tools: List[str] 
    tables: Optional[List[str]] = None


