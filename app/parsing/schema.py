from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    class BaseModelEx(BaseModel):
        model_config = ConfigDict(extra="allow")

except ImportError:  # pydantic v1 fallback

    class BaseModelEx(BaseModel):
        class Config:
            extra = "allow"


class Node(BaseModelEx):
    id: str
    type: str
    txt: str = ""
    cnt: Tuple[float, float]
    wh: Tuple[float, float]
    actor: Optional[str] = None


class Label(BaseModelEx):
    txt: str = ""
    cnt: Tuple[float, float]
    wh: Tuple[float, float]


class Arrow(BaseModelEx):
    id: str
    tip: Tuple[float, float]
    starts: List[Tuple[float, float]]


class Diagram(BaseModelEx):
    source_file: Optional[str] = None
    nodes: List[Node]
    labels: List[Label] = Field(default_factory=list)
    arrows: List[Arrow] = Field(default_factory=list)


def parse_diagram(data: dict) -> Diagram:
    if hasattr(Diagram, "model_validate"):
        return Diagram.model_validate(data)
    return Diagram.parse_obj(data)
