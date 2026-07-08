"""Experiment/probe generation and EIG scoring (P8).

Generates candidate probes (including the null action, a first-class
candidate with its own EIG in an active market) and scores them by
MARGINAL expected information gain over null (INV-4). Receives
`SelfStateCognitive` only: no PnL fields exist on anything this package
sees (INV-5).
"""
