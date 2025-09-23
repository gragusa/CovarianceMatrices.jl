"""
Variance estimator forms for different model types and assumptions.

This module defines functions for working with variance estimation forms.
The type definitions are in model_interface.jl to avoid circular dependencies.

Mathematical framework:
- Information:
  * MLE: V = H⁻¹ (Fisher Information Matrix)
  * GMM: V = (G'Ω⁻¹G)⁻¹ (efficient GMM under correct specification)
- Misspecified:
  * MLE: V = G⁻¹ΩG⁻ᵀ (robust sandwich for misspecified M-like models)
  * GMM: V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹ (robust GMM under misspecification)
"""
