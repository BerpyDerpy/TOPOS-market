"""Homeostat (P7).

Set-point bands with superlinear drives and hard vetoes. Homeostat
variables are set-points, not maximands: drive is exactly zero inside the
soft band, grows superlinearly between soft and hard bounds, and a hard
veto fires at the hard bound (INV-6). This is the ONLY agent package that
may consume `SelfStateFull` (PnL as drawdown distance-to-bound, INV-5).
"""
