"""Catalog selection scoring constants and stopwords."""

from __future__ import annotations

import re

from fervis.lookup.question_contract import KnownInputKind

DEFAULT_MAX_CATALOG_READS_PER_FACT = 5
MIN_CATALOG_READS_PER_FACT = 3
_CATALOG_TERM_SCORE = 1
_RESOURCE_TERM_SCORE = 2

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_SYNTHETIC_IDENTIFIER_RE = re.compile(
    r"^(?:rf|fact|ao|answer|output|input|ki)[_-]?\d+$",
    re.IGNORECASE,
)
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "at",
        "be",
        "being",
        "between",
        "both",
        "by",
        "can",
        "did",
        "do",
        "does",
        "during",
        "each",
        "for",
        "from",
        "give",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "list",
        "make",
        "me",
        "no",
        "of",
        "on",
        "or",
        "our",
        "should",
        "show",
        "so",
        "that",
        "the",
        "there",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "where",
        "which",
        "who",
        "with",
        "would",
        "yes",
    }
)
_QUESTION_CONTRACT_BOILERPLATE_TERMS = frozenset(
    {
        "answer",
        "answering",
        "answers",
        "ao",
        "ask",
        "asked",
        "asks",
        "fact",
        "facts",
        "factual",
        "input",
        "ki",
        "output",
        "outputs",
        "question",
        "queried",
        "query",
        "requested",
        "rf",
        "scope",
        "user",
    }
)
_REST_API_ENDPOINT_GENERIC_TERMS = frozenset(
    {
        "api",
        "data",
        "dataset",
        "detail",
        "details",
        "endpoint",
        "entities",
        "entity",
        "field",
        "fields",
        "read",
        "record",
        "records",
        "reference",
        "result",
        "results",
        "value",
        "values",
    }
)
_REST_ENDPOINT_VERBS = frozenset(
    {
        "create",
        "delete",
        "get",
        "list",
        "patch",
        "post",
        "put",
        "retrieve",
        "update",
    }
)
_RESOLVER_ENDPOINT_STOPWORDS = frozenset(
    _ENGLISH_STOPWORDS | _REST_API_ENDPOINT_GENERIC_TERMS | _REST_ENDPOINT_VERBS
)
_ENDPOINT_SELECTION_NON_IDENTITY_TERMS = frozenset(
    {
        "active",
        "allow",
        "associated",
        "available",
        "availability",
        "calculated",
        "calculate",
        "calculating",
        "calculation",
        "capturing",
        "compared",
        "comparison",
        "complete",
        "contain",
        "contains",
        "count",
        "counted",
        "counting",
        "counts",
        "criterion",
        "current",
        "date",
        "dates",
        "day",
        "days",
        "determine",
        "determining",
        "determination",
        "evaluated",
        "expose",
        "exposed",
        "exposes",
        "exist",
        "exists",
        "find",
        "frame",
        "generated",
        "highest",
        "identification",
        "identify",
        "identifier",
        "include",
        "included",
        "includes",
        "including",
        "includ",
        "made",
        "maximum",
        "mentioned",
        "most",
        "month",
        "months",
        "needed",
        "presence",
        "present",
        "provide",
        "reason",
        "reasons",
        "relevant",
        "reported",
        "separately",
        "specified",
        "subject",
        "support",
        "supporting",
        "tell",
        "thi",
        "time",
        "two",
        "used",
        "whether",
    }
)
_ENDPOINT_SELECTION_STOPWORDS = (
    _ENGLISH_STOPWORDS
    | _QUESTION_CONTRACT_BOILERPLATE_TERMS
    | _REST_API_ENDPOINT_GENERIC_TERMS
    | _ENDPOINT_SELECTION_NON_IDENTITY_TERMS
)
_KNOWN_TEXT_QUERY_KINDS = frozenset(
    {
        KnownInputKind.REFERENCE,
    }
)
