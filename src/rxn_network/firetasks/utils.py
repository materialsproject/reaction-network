"""
Utility Fireworks functions. Some of these functions are borrowed from the atomate package.
"""
from typing import Optional

import numpy as np
from fireworks import FireTaskBase
from monty.serialization import loadfn

from rxn_network.costs.calculators import (
    PrimarySelectivityCalculator,
    SecondarySelectivityCalculator,
)
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.hull import InterfaceReactionHull
from rxn_network.utils import limited_powerset


def get_decorated_rxn(rxn, competing_rxns, precursors_list, temp):
    if len(precursors_list) == 1:
        other_energies = np.array(
            [r.energy_per_atom for r in competing_rxns if r != rxn]
        )
        primary_selectivity = InterfaceReactionHull._primary_selectivity_from_energies(
            rxn.energy_per_atom, other_energies, temp=temp
        )
        energy_diffs = rxn.energy_per_atom - other_energies
        max_diff = energy_diffs.max()
        secondary_selectivity = max_diff if max_diff > 0 else 0.0
        rxn.data["primary_selectivity"] = primary_selectivity
        rxn.data["secondary_selectivity"] = secondary_selectivity
        decorated_rxn = rxn
    else:
        competing_rxns = competing_rxns.get_rxns()

        if rxn not in competing_rxns:
            competing_rxns.append(rxn)

        irh = InterfaceReactionHull(
            precursors_list[0],
            precursors_list[1],
            competing_rxns,
        )

        calc_1 = PrimarySelectivityCalculator(irh=irh, temp=temp)
        calc_2 = SecondarySelectivityCalculator(irh=irh)

        decorated_rxn = calc_1.decorate(rxn)
        decorated_rxn = calc_2.decorate(decorated_rxn)

    return decorated_rxn


def env_chk(
    val: str,
    fw_spec: dict,
    strict: Optional[bool] = True,
    default: Optional[str] = None,
):
    """
    Code borrowed from the atomate package.

    env_chk() is a way to set different values for a property depending
    on the worker machine. For example, you might have slightly different
    executable names or scratch directories on different machines.

    Args:
        val: any value, with ">><<" notation reserved for special env lookup values
        fw_spec: fw_spec where one can find the _fw_env keys
        strict: if True, errors if env format (>><<) specified but cannot be found in fw_spec
        default: if val is None or env cannot be found in non-strict mode,
                 return default
    """
    if val is None:
        return default

    if isinstance(val, str) and val.startswith(">>") and val.endswith("<<"):
        if strict:
            return fw_spec["_fw_env"][val[2:-2]]
        return fw_spec.get("_fw_env", {}).get(val[2:-2], default)
    return val


def load_json(firetask: FireTaskBase, param: str, fw_spec: dict) -> dict:
    """
    Utility function for loading json file related to a parameter of a FireTask. This first looks
    within the task to see if the object is already serialized; if not, it looks for a
    file with the filename stored under the {param}_fn attribute either within the
    FireTask or the fw_spec.

    Args:
        firetask: FireTask object
        param: parmeter name
        fw_spec: Firework spec.

    Returns:
        A loaded object (dict)
    """
    obj = firetask.get(param)

    if not obj:
        param_fn = param + "_fn"
        obj_fn = firetask.get(param_fn)

        if not obj_fn:
            obj_fn = fw_spec[param_fn]

        obj = loadfn(obj_fn)

    return obj


def load_entry_set(firetask, fw_spec):
    """
    Loads a GibbsEntrySet, either from the firetask itself (or its fw_spec), or from a
    file given the entries_fn attribute.
    """
    entries = firetask["entries"]

    if not entries:
        entries_fn = firetask.get("entries_fn")
        entries_fn = entries_fn if entries_fn else fw_spec["entries_fn"]
        entries = loadfn(entries_fn)

    entries = GibbsEntrySet(entries)
    return entries


def get_all_precursor_strs(precursors):
    formulas = [comp.reduced_formula for comp in precursors]
    combos = limited_powerset(formulas, len(formulas))
    return ["-".join(sorted(c)) for c in combos]
