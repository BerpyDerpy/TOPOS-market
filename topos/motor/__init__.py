"""Deterministic intent -> message compiler (P10).

A pure function of its declared inputs: no randomness, no hidden state
(INV-8). Intent and compiled messages are logged side by side in the
`WorkspaceRecord`.
"""
