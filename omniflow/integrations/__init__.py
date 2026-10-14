"""External integrations (Gmail, Google Calendar, image generation).

Every integration reports an honest status (connected, needs setup, disabled, mock) and is
safe when unconfigured: nothing pretends to be connected, mock data is labelled MOCK, and
any action that changes external data or sends data to a new provider goes through the
agent's per-action approval.
"""
