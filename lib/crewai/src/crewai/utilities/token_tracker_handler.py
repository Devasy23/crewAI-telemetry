from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Dict, List, Optional

@dataclass
class LLMCallMetrics:
    """Represents a single LLM call."""
    timestamp: datetime
    agent_name: str
    task_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    call_type: str  # "generation", "tool_use", "delegation"

@dataclass
class AgentTokenMetrics:
    """Token usage for a specific agent."""
    agent_name: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    calls: List[LLMCallMetrics] = field(default_factory=list)

@dataclass
class TaskTokenMetrics:
    """Token usage for a specific task."""
    task_name: str
    agent_name: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    calls: List[LLMCallMetrics] = field(default_factory=list)

@dataclass
class WorkflowTokenMetrics:
    """Overall token usage for entire crew workflow."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    per_agent: Dict[str, AgentTokenMetrics] = field(default_factory=dict)
    per_task: Dict[str, TaskTokenMetrics] = field(default_factory=dict)
    all_calls: List[LLMCallMetrics] = field(default_factory=list)

class CrewTokenTracker:
    """Thread-safe registry for tracking LLM token usage across agents and tasks."""

    def __init__(self):
        self._lock = RLock()
        self._metrics = WorkflowTokenMetrics()

    def record_llm_call(self,
                        agent_name: str,
                        task_name: str,
                        model: str,
                        prompt_tokens: int,
                        completion_tokens: int,
                        call_type: str = "generation"):
        """Record an LLM call with explicit context."""
        with self._lock:
            total = prompt_tokens + completion_tokens

            # Create call record
            call_record = LLMCallMetrics(
                timestamp=datetime.now(),
                agent_name=agent_name,
                task_name=task_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total,
                call_type=call_type
            )

            # Update workflow totals
            self._metrics.total_tokens += total
            self._metrics.prompt_tokens += prompt_tokens
            self._metrics.completion_tokens += completion_tokens
            self._metrics.all_calls.append(call_record)

            # Update per-agent metrics
            if agent_name not in self._metrics.per_agent:
                self._metrics.per_agent[agent_name] = AgentTokenMetrics(
                    agent_name=agent_name
                )
            agent_metrics = self._metrics.per_agent[agent_name]
            agent_metrics.total_tokens += total
            agent_metrics.prompt_tokens += prompt_tokens
            agent_metrics.completion_tokens += completion_tokens
            agent_metrics.call_count += 1
            agent_metrics.calls.append(call_record)

            # Update per-task metrics
            # Task name might not be unique globally, so we can key by task_name + agent_name or just task_name if sufficient
            # But the user asked for "Per-Task Token Tracking". A task is usually specific to an agent in CrewAI (unless delegated).
            # The user plan suggested keying by "task_name_agent_name".
            # For now I will use the task description as task_name, or title if available.
            # Using a composite key seems safer.
            task_key = f"{task_name}_{agent_name}"
            if task_key not in self._metrics.per_task:
                self._metrics.per_task[task_key] = TaskTokenMetrics(
                    task_name=task_name,
                    agent_name=agent_name
                )
            task_metrics = self._metrics.per_task[task_key]
            task_metrics.total_tokens += total
            task_metrics.prompt_tokens += prompt_tokens
            task_metrics.completion_tokens += completion_tokens
            task_metrics.call_count += 1
            task_metrics.calls.append(call_record)

    def get_metrics(self) -> WorkflowTokenMetrics:
        """Get current workflow metrics."""
        with self._lock:
            # Return a copy or the object itself.
            # Since we are mostly appending, returning the object is risky if mutated outside lock,
            # but for reporting it might be fine.
            return self._metrics

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics = WorkflowTokenMetrics()
