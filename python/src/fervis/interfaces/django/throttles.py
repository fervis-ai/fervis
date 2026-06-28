"""Fervis API throttle classes.

Dedicated scopes prevent Fervis routes from sharing the global `user`
throttle bucket, which can be exhausted by unrelated API activity.
"""

from rest_framework.throttling import UserRateThrottle


class FervisQuestionThrottle(UserRateThrottle):
    scope = "fervis_question"
