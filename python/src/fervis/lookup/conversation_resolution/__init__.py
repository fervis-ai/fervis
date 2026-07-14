"""Conversation resolution public boundary."""

from .compilation import (
    CompiledConversationResolution,
    ResolvedCanonicalIdentity,
    ResolvedLiteralQuestionInput,
    ResolvedQuestionInput,
    ResolvedRowSetQuestionInput,
    compile_conversation_resolution,
)
from .model import (
    CandidateInterpretation,
    CarriedFrameArgument,
    ContextAnchorSource,
    ConversationFrameCall,
    ConversationResolution,
    ConversationResolutionRequest,
    ConversationResolutionResult,
    CurrentSpanSource,
    FrameArgument,
    FrameArgumentKind,
    FrameParameterRef,
    FramePartSource,
    ResolutionSource,
    ResolutionSourceKind,
    ResolvedConversationClause,
    ResolvedConversationValue,
    ResolvedValueFrameArgument,
    SourceEvidence,
    UnresolvedResolution,
)
from .parser import parse_conversation_resolution
from .prompt import (
    ConversationResolutionTurnPrompt,
    conversation_resolution_context_sources,
)
from .schema import build_conversation_resolution_tool_schemas
from .tools import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    CONVERSATION_RESOLUTION_TOOL_NAMES,
)
from .turn import (
    ConversationResolutionGenerationError,
    ConversationResolutionTurnResult,
    generate_conversation_resolution,
)

__all__ = [
    "CONVERSATION_RESOLUTION_TOOL_NAME",
    "CONVERSATION_RESOLUTION_TOOL_NAMES",
    "CandidateInterpretation",
    "CarriedFrameArgument",
    "CompiledConversationResolution",
    "ContextAnchorSource",
    "ConversationFrameCall",
    "ConversationResolution",
    "ConversationResolutionGenerationError",
    "ConversationResolutionRequest",
    "ConversationResolutionResult",
    "ConversationResolutionTurnPrompt",
    "ConversationResolutionTurnResult",
    "CurrentSpanSource",
    "FrameArgument",
    "FrameArgumentKind",
    "FrameParameterRef",
    "FramePartSource",
    "ResolutionSource",
    "ResolutionSourceKind",
    "ResolvedCanonicalIdentity",
    "ResolvedConversationClause",
    "ResolvedConversationValue",
    "ResolvedLiteralQuestionInput",
    "ResolvedQuestionInput",
    "ResolvedRowSetQuestionInput",
    "ResolvedValueFrameArgument",
    "SourceEvidence",
    "UnresolvedResolution",
    "build_conversation_resolution_tool_schemas",
    "compile_conversation_resolution",
    "conversation_resolution_context_sources",
    "generate_conversation_resolution",
    "parse_conversation_resolution",
]
