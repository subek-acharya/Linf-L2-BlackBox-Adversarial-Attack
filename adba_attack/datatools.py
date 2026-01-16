# coding:utf-8
"""
DataTools for ADBA Attack
Contains only the helper functions used in the attack.
Adapted from: https://github.com/BUPTAIOC/ADBA
"""

from scipy.integrate import quad


# ==================== Distribution Parameters ====================
# Fitted parameters for probability density function ρ(r)
# From paper Appendix D

a_hat = 0.03133292769944518
b_hat = 3.0659694403842903
c_hat = 0.16755646970211466
d_hat = 0.13403850261898806


# ==================== Helper Functions ====================

def func_rho(r, a, b, c, d):
    """
    Probability density function for decision boundary distribution.
    ρ(r) = a / ((r + d)^b) + c
    """
    return a / ((r + d) ** b) + c


def find_midK_of_k1k2(k1, k2):
    """
    Find median point of func_rho between k1 and k2.
    Uses numerical integration to find the point where the integral
    from k1 to mid equals half the integral from k1 to k2.
    """
    Sk1k2, error = quad(func_rho, k1, k2, args=(a_hat, b_hat, c_hat, d_hat))
    low, high, mid = k1, k2, 0
    Sk1mid = None
    while high - low > 1.0 / 600:
        mid = (low + high) / 2
        Sk1mid, error = quad(func_rho, k1, mid, args=(a_hat, b_hat, c_hat, d_hat))
        if Sk1mid < Sk1k2 / 2:
            low = mid
        else:
            high = mid
    return mid


def next_ADB(r1, r2, aim_r, max_r, mod):
    """
    Decide next ADB (Approximate Decision Boundary) using func_rho().
    
    Args:
        r1: Lower bound of search range
        r2: Upper bound of search range
        aim_r: Target epsilon
        max_r: Maximum radius (Rmax)
        mod: Search mode
            - 0: ADBA (simple midpoint)
            - 1: ADBA-md (median-based)
    
    Returns:
        Next ADB value to test
    """
    if mod == 0:  # ADBA
        return (r1 + r2) / 2.0
    if mod == 1:  # ADBA-md
        k1, k2 = r1 / max_r, r2 / max_r
        kmid = find_midK_of_k1k2(1 - k2, 1 - k1)
        median = (1 - kmid) * max_r
        return median