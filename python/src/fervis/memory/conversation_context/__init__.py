"""Conversation context-source and memory activation boundary."""

from fervis.memory.conversation_context.activation import (
    ExpandedActivatedMemory,
    expand_activated_memory_cards,
)
from fervis.memory.conversation_context.model import (
    ConversationCallableParameter,
    ConversationCallableSignature,
    ConversationContextFrame,
    ConversationContextSource,
    ConversationFramePart,
    ConversationFramePartKind,
    ConversationMeaningAnchor,
    ConversationMemoryActivation,
    ConversationMemoryActivationKind,
    ConversationMemoryCard,
    ConversationMemoryCardProjection,
)

__all__ = [
    "ConversationCallableParameter",
    "ConversationCallableSignature",
    "ConversationContextFrame",
    "ConversationContextSource",
    "ConversationFramePart",
    "ConversationFramePartKind",
    "ConversationMeaningAnchor",
    "ConversationMemoryActivation",
    "ConversationMemoryActivationKind",
    "ConversationMemoryCard",
    "ConversationMemoryCardProjection",
    "ExpandedActivatedMemory",
    "expand_activated_memory_cards",
]
