"""Public Flask integration surface."""

from fervis.interfaces.flask import fervis_flask_blueprint

from .integration import FlaskIntegration

__all__ = [
    "FlaskIntegration",
    "fervis_flask_blueprint",
]
