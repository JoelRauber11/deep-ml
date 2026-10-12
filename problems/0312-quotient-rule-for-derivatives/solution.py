import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    # Höhen an der Stelle x
    g = np.polyval(g_coeffs, x)
    h = np.polyval(h_coeffs, x)

    # Ableitung bilden, dann x einsetzen --> Steigung
    g_abl = np.polyval(np.polyder(g_coeffs), x)
    h_abl = np.polyval(np.polyder(h_coeffs), x)

    # Quiotientenregel
    return float((g_abl * h - g * h_abl) / h ** 2)