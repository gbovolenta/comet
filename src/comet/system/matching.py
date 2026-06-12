"""Helpers for matching gas coordinates."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from ase import Atoms


def get_matching_gas_mols(xyz_arr1, xyz_arr2):
    """Split coordinates into exact matches and non-matches.

    Args:
        xyz_arr1: Coordinate arrays to classify.
        xyz_arr2: Reference coordinate arrays to match against.

    Returns:
        tuple: `(common_gas_mols, not_common_gas_mols)` preserving the order of
        `xyz_arr1`.
    """
    common_gas_mols = []
    not_common_gas_mols = []
    for xyz1 in xyz_arr1:
        matched = False
        for xyz2 in xyz_arr2:
            if np.array_equal(xyz1, xyz2):
                common_gas_mols.append(xyz1)
                matched = True
                break
        if not matched:
            not_common_gas_mols.append(xyz1)
    return common_gas_mols, not_common_gas_mols


def get_matching_gas_mols_with_symbols(
    box_gas: Atoms,
    old_gas_pos: np.ndarray,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compare current box_gas positions to old_gas_pos (from initial idx_in),
    return (common_positions, new_positions, new_symbols).

    Matching is done with an absolute tolerance tol per Cartesian component.
    """
    pos1 = np.asarray(box_gas.get_positions())
    syms1 = box_gas.get_chemical_symbols()
    pos2 = np.asarray(old_gas_pos)

    common = []
    new_pos = []
    new_syms = []

    for p1, s1 in zip(pos1, syms1):
        if pos2.size and np.any(np.all(np.abs(pos2 - p1) <= tol, axis=1)):
            common.append(p1)
        else:
            new_pos.append(p1)
            new_syms.append(s1)

    return np.array(common), np.array(new_pos), new_syms


def filter_dict(dict_org, common_gas_mols):
    """Return dictionary keys whose array values match selected coordinates.

    Args:
        dict_org: Mapping from keys to coordinate arrays.
        common_gas_mols: Coordinate arrays that should be retained.

    Returns:
        dict_keys: Keys from `dict_org` whose values match one of the selected
        coordinate arrays.
    """
    # Initialize an empty dictionary to store the filtered key-value pairs
    filtered_dict = {}

    # Loop through the dictionary items
    for key, value in dict_org.items():
        # Check if the value is in the common_gas_mols list using numpy's array_equal
        for common_array in common_gas_mols:
            if np.array_equal(value, common_array):
                # If a match is found, add the key-value pair to the filtered_dict
                filtered_dict[key] = value
                break  # Exit the inner loop once a match is found
    return list(filtered_dict)
