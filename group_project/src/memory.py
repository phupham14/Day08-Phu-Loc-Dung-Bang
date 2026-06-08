"""
Conversation memory for the group RAG chatbot.

The generation module expects history in OpenAI chat format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

Role = Literal["user", "assistant", "system"]


@dataclass
class ConversationMemory:
    """Keep the last N chat turns and expose them as LLM-ready messages."""

    max_turns: int = 6
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, user_message: str, assistant_message: str) -> None:
        """Append one user/assistant turn."""
        self.add_message("user", user_message)
        self.add_message("assistant", assistant_message)

    def add_message(self, role: Role, content: str) -> None:
        """Append one message and trim old turns."""
        content = (content or "").strip()
        if not content:
            return
        self.messages.append({"role": role, "content": content})
        self._trim()

    def extend(self, messages: Iterable[dict[str, str]]) -> None:
        """Append existing chat messages."""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role in {"user", "assistant", "system"}:
                self.add_message(role, content)  # type: ignore[arg-type]

    def get(self) -> list[dict[str, str]]:
        """Return a copy of history safe to pass into generation.generate()."""
        return [message.copy() for message in self.messages]

    def clear(self) -> None:
        """Remove all saved conversation messages."""
        self.messages.clear()

    def last_user_question(self) -> str | None:
        """Return the latest user message, if any."""
        for message in reversed(self.messages):
            if message["role"] == "user":
                return message["content"]
        return None

    def to_text(self) -> str:
        """Compact readable history for logs, debug, or prompts."""
        labels = {"user": "User", "assistant": "Assistant", "system": "System"}
        return "\n".join(
            f"{labels.get(message['role'], message['role'])}: {message['content']}"
            for message in self.messages
        )

    def _trim(self) -> None:
        max_messages = max(1, self.max_turns) * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        turns = len([m for m in self.messages if m["role"] == "user"])
        return f"{turns}/{self.max_turns} turns"
