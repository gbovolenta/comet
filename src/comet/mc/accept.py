"""Metropolis acceptance criteria."""

from __future__ import annotations

import logging
import random
from typing import Union

import numpy as np
from ase.units import kB

from comet.physics.thermo import lambda_wl

logger = logging.getLogger(__name__)


def metropolis_probability(
    E_current: float,
    E_new: float,
    T: float,
    m_amu: float,
    mu: float,
    ins: bool,
    V: float,
    n: int,
    gas_en: float,
) -> tuple[float, float]:
    """
    Return (acceptance_probability, delta_E) for a GCMC move.
    """
    beta = 1.0 / (kB * T)
    lam = lambda_wl(T, m_amu)
    if ins:
        delta_E = E_new - (E_current + gas_en)
        prob = min(1.0, (V / (lam**3 * (n + 1))) * np.exp(beta * (mu - delta_E)))
    else:
        delta_E = E_new - (E_current - gas_en)
        prob = min(1.0, ((lam**3 * n) / V) * np.exp(-beta * (mu + delta_E)))
    return float(prob), float(delta_E)


def metropolis_criteria(
    E_current: float,
    E_new: float,
    T: float,
    m_amu: float,
    mu: float,
    ins: bool,
    V: float,
    n: int,
    gas_en: float,
) -> bool:
    """
    Decide acceptance of a GCMC move via the Metropolis criterion.

    Args:
        E_current (float): Current potential energy (eV).
        E_new (float): Proposed new energy (eV).
        T (float): Temperature (K).
        m_amu (float): Mass in amu.
        mu (float): Chemical potential (eV).
        ins (bool): True if insertion move, False if deletion.
        V (float): System volume (Å³).
        n (int): Current number of gas molecules.
        gas_en (float): Reference energy of gas (eV).

    Returns:
        bool: True if move is accepted, False otherwise.
    """
    prob, delta_E = metropolis_probability(
        E_current=E_current,
        E_new=E_new,
        T=T,
        m_amu=m_amu,
        mu=mu,
        ins=ins,
        V=V,
        n=n,
        gas_en=gas_en,
    )
    move = "insertion" if ins else "deletion"
    accept = random.random() < prob
    logger.debug(
        "Metropolis %s: ΔE=%.6f, prob=%.6f, accepted=%s", move, delta_E, prob, accept
    )
    return accept
