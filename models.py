from typing import Any, Optional
from pydantic import BaseModel, Field


class ParameterDefinition(BaseModel):

    type: str
    description: Optional[str] = None


class FunctionDefinition(BaseModel):

    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):

    prompt: str


class FunctionCallResult(BaseModel):

    prompt: str
    name: str
    parameters: dict[str, Any]