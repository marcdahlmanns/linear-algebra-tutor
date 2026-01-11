"""Simple solution structure for CLI usage (bypasses Pydantic validation)."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SimpleSolutionStep:
    """A single step in a solution (simple version for CLI)."""
    description: str
    mathematical_expression: str = ""
    explanation: str = ""
    intermediate_result: Optional[Any] = None


@dataclass
class SimpleSolution:
    """Complete solution with steps (simple version for CLI)."""
    operation: str
    final_answer: Any
    steps: List[SimpleSolutionStep] = field(default_factory=list)
