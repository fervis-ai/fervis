"""Conversation context-source and memory activation boundary."""

from fervis.memory.conversation_context.activation import (
    ExpandedActivatedMemory,
    expand_activated_memory_cards,
)
from fervis.memory.conversation_context.model import (
    ConversationAnswerShape,
    ConversationCallableSignature,
    ConversationContextFrame,
    ConversationContextSource,
    ConversationFrameParameter,
    ConversationFramePart,
    ConversationFramePartKind,
    ConversationMeaningAnchor,
    ConversationMemoryActivation,
    ConversationMemoryActivationKind,
    ConversationMemoryCard,
    ConversationMemoryCardProjection,
)

__all__ = [
    "ConversationAnswerShape",
    "ConversationCallableSignature",
    "ConversationContextFrame",
    "ConversationContextSource",
    "ConversationFrameParameter",
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
