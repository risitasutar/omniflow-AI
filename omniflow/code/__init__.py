"""Coding assistant: repository indexing, search, security review, patches and checks.

Repository content is untrusted input throughout: it is never executed by indexing or
review, secret-looking values are redacted before anything reaches the model, and only
files inside the selected project root that pass `files.classify` are ever read.
"""
