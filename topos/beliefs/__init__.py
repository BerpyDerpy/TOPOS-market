"""World predictors (P4, P5, P11).

Kalman fair-value and Poisson-Gamma flow-intensity models, the shared
`BeliefModule` protocol and EIG machinery (P4), the queue-position filter
(P5), and the regime tracker with regime-gated forgetting (P11).

All adaptation is closed-form posterior updates plus forgetting (INV-2).
Curiosity quantities are mutual information on parameter posteriors,
never predictive variance or predictive entropy alone (INV-3).
"""
