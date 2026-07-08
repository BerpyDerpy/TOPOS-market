"""Blackboard, salience competition, arbiter, broadcast (P9).

The bounded, typed workspace. Salience consequence-weights derive from the
module dependency graph (registry centrality), computed once at startup —
never from outcome statistics of any kind (INV-7). The `WorkspaceRecord`
logged each cycle IS the interpretability story. Arbitration receives
`SelfStateCognitive` only (INV-5).
"""
