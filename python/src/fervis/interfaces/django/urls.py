"""Fervis Django interface URL routing."""

from django.urls import path

from .views import (
    ConversationListView,
    FervisRuntimeStatusView,
    QuestionCreateView,
    QuestionDetailView,
    QuestionRunDetailView,
    QuestionRunAnswerQuestionView,
    QuestionRunListView,
)

app_name = "fervis_django_interface"

urlpatterns = [
    path("", FervisRuntimeStatusView.as_view(), name="fervis-runtime-status"),
    path(
        "conversations/",
        ConversationListView.as_view(),
        name="fervis-conversation-list",
    ),
    path(
        "questions/",
        QuestionCreateView.as_view(),
        name="fervis-question-create",
    ),
    path(
        "questions/<str:question_id>/",
        QuestionDetailView.as_view(),
        name="fervis-question-detail",
    ),
    path(
        "questions/<str:question_id>/runs/",
        QuestionRunListView.as_view(),
        name="fervis-question-run-list",
    ),
    path(
        "questions/<str:question_id>/runs/<uuid:run_id>/",
        QuestionRunDetailView.as_view(),
        name="fervis-question-run-detail",
    ),
    path(
        "questions/<str:question_id>/runs/<uuid:run_id>/ask/",
        QuestionRunAnswerQuestionView.as_view(),
        name="fervis-question-run-ask",
    ),
]
