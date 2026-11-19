# The full row for a magnetic structure: spin_symmetry.py's groups, order,
# allowed responses and spin texture, with irrep_labels.py's labels of the parts
# of Γ and their wave vectors.
#
#     python find_gamma.py 0.15.mcif [more ...] [--tolerance 0.01] [--operations]
#                          [--table isoir.json.gz]
#
# A file is a magnetic CIF, or read_mcif.py's JSON. The labels are in spglib's
# standard setting of G; they need ISO-IR's tables, which extract_isoir.py
# writes, and without them Γ and k are written as a dash. With --operations,
# G's operations are listed in that same setting.

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

import irrep_labels
from read_mcif import read_mcif
from spin_symmetry import analyse


# The structure in a file: a magnetic CIF, or read_mcif.py's JSON. A
# left-handed cell is made right-handed (c to -c), or spglib would find the
# mirror image of a chiral group.
def read(path):
    if path.endswith(".json"):
        with open(path) as file:
            cell = json.load(file)
        if np.linalg.det(cell["lattice"]) < 0:
            cell["lattice"][2] = [-x for x in cell["lattice"][2]]
            for site in cell["sites"]:
                site["position"][2] = -site["position"][2]
        return cell
    return read_mcif(Path(path).read_text(errors="replace"), Path(path).stem)


# Argparse interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G0, G, Γ with its labels and wave vectors, and the allowed responses "
                                                 "and spin texture of magnetic structures.")
    parser.add_argument("files", nargs="+", help="magnetic CIFs, or read_mcif.py's JSON")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Å for positions, μB for moments (default 0.01)")
    parser.add_argument("--operations", action="store_true",
                        help="list each operation {R|t} of G with its Γ, in the setting the labels are in")
    parser.add_argument("--table", type=Path, default=Path(__file__).with_name("isoir.json.gz"),
                        help="extract_isoir.py's output")
    args = parser.parse_args()

    # ISO-IR's tables, if they have been made
    table = None
    if args.table.exists():
        with gzip.open(args.table, "rt") as file:
            table = json.load(file)
    else:
        print(f"{args.table} not found (python extract_isoir.py CIR_data.txt PIR_data.txt): "
              "Γ and k are not labelled", file=sys.stderr)

    # one row for each file
    print("name", "formula", "MSG", "G0", "order", "G", "Γ", "k", "allowed", "texture", sep="\t")
    for path in args.files:
        cell = read(path)
        found = analyse(cell, args.tolerance)
        lattice, R, t, g = (found[key] for key in ("lattice", "rotations", "translations", "gamma"))
        gamma, k = "-", "-"
        if table is not None:
            gamma, k = irrep_labels.name(lattice, R, t, g, table, np.array(cell["lattice"]))
        print(cell["name"], cell["formula"], cell["magnetic_space_group"], found["G0"], found["order"], found["G"],
              gamma, k, " ".join(found["allowed"]) or "none", found["texture"], sep="\t")

        # each operation in the setting the labels are in: x -> P x + p takes
        # {R|t} to {P R P^-1 | P t + p - P R P^-1 p}. Γ acts on the moments, and
        # is the same in any setting.
        if args.operations:
            _, P, p = irrep_labels.standard_setting(lattice, R, t, np.array(cell["lattice"]))
            for rotation, translation, matrix in zip(R, t, g):
                S = np.rint(P @ rotation @ np.linalg.inv(P)).astype(int)
                shift = np.round(P @ translation + p - S @ p, 6) + 0.0
                print(f"  R = {S.tolist()}  t = {shift.tolist()}  Γ = {np.round(matrix, 6).tolist()}")
