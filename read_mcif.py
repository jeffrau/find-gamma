# Read a magnetic CIF (MCIF) into a plain form: the whole magnetic cell, every
# atom with its moment.
#
#     python read_mcif.py MnF2.mcif > MnF2.json
#
# An MCIF gives a cell, a few atoms with their moments, and the operations of a
# magnetic space group that make the others from them. Here every operation is
# applied to every atom, so that what comes out has no symmetry left to apply:
#
#     lattice    the cell's edges a, b, c as rows, Cartesian, in Angstroms
#     sites      each with its kind (its element, with its occupancy where the
#                site is not full), its position (in fractions of the cell)
#                and its moment (Cartesian, in Bohr magnetons)
#
# Three rules, for the files that need them:
#  1. An occupancy above 1 is 1.
#  2. Atoms of one kind within 0.01 Angstroms of each other are one atom (an
#     atom's own images where it lies on a symmetry element, or an atom the file
#     lists twice).
#  3. Atoms of different kinds there share one site, which is a kind of its own
#     (Bi0.78+Mn0.22). A site's moment is its average moment: each of its atoms'
#     moments times that atom's occupancy.
#
# pymatgen reads the CIF syntax, the cell and the operations as they are written
# (x,y,z,+1). The rest is below.

import json
import re
import sys
from pathlib import Path

import numpy as np
from pymatgen.core.operations import MagSymmOp
from pymatgen.io.cif import CifFile, CifParser, str2float

# Atoms closer than this (in Angstroms) are at the same place
CLOSE = 0.01


# Read an MCIF's text into the plain form above (as a dictionary), under a name.
def read_mcif(text, name=""):

    # the file's first data block with a cell (FullProf writes an empty
    # data_global block before it), with its tags in one form: lower case,
    # without dots and underscores (files write both _a_b.c and _a.b_c)
    block = next(block for block in CifFile.from_str(text).data.values() if "_cell_length_a" in block.data)
    tag = {re.sub("[._]", "", name.lower()): value for name, value in block.data.items()}

    # an incommensurate structure is not treated
    if "cellmodulationdimension" in tag:
        raise ValueError("an incommensurate structure: not treated")

    # The cell, from its lengths and angles, and unit vectors along its edges (a
    # moment is written in components along these)
    lattice = CifParser.get_lattice_no_exception(block).matrix
    axes = lattice / np.linalg.norm(lattice, axis=1)[:, None]

    # The operations: each one listed with each centring. (R, t, θ) takes x to
    # R x + t, and θ = -1 reverses time. (pymatgen's CifParser.get_magsymops
    # does the same, but it knows one spelling of these tags. Where it finds
    # none, as in 0.189, it applies the identity alone.)
    listed = [MagSymmOp.from_xyzt_str(text) for text in tag["spacegroupsymopmagnoperationxyz"]]
    centrings = [MagSymmOp.from_xyzt_str(text) for text in tag.get("spacegroupsymopmagncenteringxyz", ["x,y,z,+1"])]
    operations = [(op.rotation_matrix, op.translation_vector + c.translation_vector, op.time_reversal * c.time_reversal)
                  for op in listed for c in centrings]

    # the moments as the file writes them, by label, and the occupancies
    written = {label: [str2float(tag["atomsitemomentcrystalaxis" + i][n]) for i in "xyz"]
               for n, label in enumerate(tag.get("atomsitemomentlabel", []))}
    occupancies = tag.get("atomsiteoccupancy", ["1"] * len(tag["atomsitelabel"]))

    # Apply every operation to every atom of the file. Each site found is kept
    # as its position and what is there: {kind: moment}.
    sites = []
    for n, label in enumerate(tag["atomsitelabel"]):

        # the atom's kind (its element, and its occupancy if not full), its
        # position, and its moment (Cartesian, times its occupancy)
        element = re.match("[A-Z][a-z]?", tag["atomsitetypesymbol"][n]).group()
        occupancy = 1.0 if occupancies[n] in (".", "?") else min(str2float(occupancies[n]), 1.0)
        kind = element if occupancy == 1 else f"{element}{occupancy:g}"
        x = np.array([str2float(tag["atomsitefract" + i][n]) for i in "xyz"])
        moment = occupancy * np.array(written.get(label, [0, 0, 0])) @ axes

        for R, t, theta in operations:

            # the image's position, and its moment: R as it acts on Cartesian
            # vectors, and a moment is an axial vector, odd under time reversal
            position = (R @ x + t) % 1
            turned = lattice.T @ R @ np.linalg.inv(lattice.T)
            image = theta * np.linalg.det(turned) * turned @ moment

            # if a site is already there, add this kind to it (a kind counts
            # once at a place); if not, the image is a new site
            for other, atoms in sites:
                d = position - other
                if np.linalg.norm((d - np.rint(d)) @ lattice) < CLOSE:
                    atoms.setdefault(kind, image)
                    break
            else:
                sites.append((position, {kind: image}))

    # Return the plain form: each site's kinds joined (Bi0.78+Mn0.22), and its
    # moment the sum of theirs
    return dict(
        name=name,
        formula=tag.get("chemicalformulasum", ""),
        magnetic_space_group=tag.get("spacegroupmagnnamebns", "").replace(" ", ""),
        lattice=np.round(lattice, 8).tolist(),
        sites=[dict(kind="+".join(sorted(atoms)), position=(np.round(position, 8) + 0.0).tolist(),
                    moment=(np.round(sum(atoms.values()), 8) + 0.0).tolist()) for position, atoms in sites],
    )


# The plain form as JSON, a site to a line
def write(cell):
    head = [f' "{key}": {json.dumps(cell[key], ensure_ascii=False)}' for key in cell if key != "sites"]
    sites = ",\n  ".join(json.dumps(site) for site in cell["sites"])
    return "{\n" + ",\n".join(head) + ',\n "sites": [\n  ' + sites + "\n ]\n}"


# Command-line interface: print the plain form of the file given
if __name__ == "__main__":
    path = Path(sys.argv[1])
    print(write(read_mcif(path.read_text(errors="replace"), path.stem)))
