# Make isoir.json.gz, the tables irrep_labels.py reads, from ISO-IR's two data
# files.
#
#     python extract_isoir.py CIR_data.txt PIR_data.txt
#
# This writes isoir.json.gz beside this script, in a few seconds.
#
# ISO-IR (Stokes, Campbell and Cordes, Acta Cryst. A 69, 388 (2013); version
# 1.1.1 of 2011) gives the irreducible representations of every space group at
# every rational wavevector, in ITA's standard settings: CIR_data.txt the
# complex ones, PIR_data.txt the physically irreducible ones. Both are at
# https://iso.byu.edu (CIR_data.zip, PIR_data.zip). Each record is one
# representation of the whole group: a header, the star of k, and for each
# listed operation {R|t} (one for each rotation of the group) its matrix.
#
# What is kept, for each group and each k-point (by its name: GM, X, DT, ...):
#
#     k           the star's first arm, as fractions: its constant part and, on
#                 a line or plane, the parts that multiply its free parameters
#                 (alpha, beta, gamma)
#     operations  the listed operations whose R leaves that arm as it is, up to
#                 a reciprocal lattice vector: the little group, in the
#                 conventional cell
#     irreps      for each complex representation there, on each of those
#                 operations: the trace of the block of its matrix at the first
#                 arm, and a translation v with which the character at k is that
#                 trace times exp(2πi k·v) (zero at a point, where the matrix is
#                 the representation's own)
#
# and for each physically irreducible representation: the complex ones it is
# made of (its name is theirs, written one after the other: M3+M4+ is M3+ and
# M4+) and its dimension.

import gzip
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

# Every matrix element in the files is one of these numbers, written to 5
# decimals
ROOTS = (math.sqrt(3) / 2, math.sqrt(2) / 2, math.sqrt(3) / 4, math.cos(math.pi / 12) / math.sqrt(2),
         math.sin(math.pi / 12) / math.sqrt(2), math.cos(math.pi / 12), math.sin(math.pi / 12), math.sqrt(3 / 8),
         math.sqrt(1 / 8))
CONSTANTS = np.array([0, 1, -1, 0.5, -0.5, 0.25, -0.25] + [sign * x for x in ROOTS for sign in (1, -1)])

# A record's header line (the only lines with quotation marks), and a complex
# matrix element as the files write it, (re,im)
HEADER = re.compile(r'^\s*(\d+)\s+(\d+) "([^"]*)" "([^"]*)"' + r"\s+(\d+)" * 5 + r"\s*$")
COMPLEX = re.compile(r"\(([^,()]+),([^,()]+)\)")

# The tabulated number found for each written one, as they are met
SNAPPED = {}


# The tabulated number a matrix element is written for
def snap(text):
    if text not in SNAPPED:
        nearest = CONSTANTS[np.argmin(np.abs(CONSTANTS - float(text)))]
        assert abs(nearest - float(text)) < 1e-4, f"{text} is not one of the tabulated numbers"
        SNAPPED[text] = float(nearest)
    return SNAPPED[text]


# The records of CIR_data.txt, one at a time, read as the authors' Fortran reads
# them: each read begins on a new line and takes as many lines as it needs. Each
# record gives its group, label, the size of its block at one arm, the star's
# first arm, whether that is a point, and its operations.
def records(path):

    # the lines after the three lines of title
    lines = iter(Path(path).read_text().splitlines()[3:])

    # read `count` values, from as many lines as they take
    def read(count, split):
        values = []
        while len(values) < count:
            values += split(next(lines))
        assert len(values) == count
        return values

    # the values of a line: integers, or complex matrix elements
    def integers(line):
        return [int(x) for x in line.split()]

    def elements(line):
        return [complex(snap(a), snap(b)) for a, b in COMPLEX.findall(line)]

    for line in lines:

        # the header, then the star of k (a point has no free parameters)
        _, group, _, label, dimension, _, arms, _, count = HEADER.match(line).groups()
        dimension, arms = int(dimension), int(arms)
        star = np.array(read(16 * arms, integers)).reshape(arms, 4, 4)
        special = not star[0, 1:, :3].any()

        # each operation: {R|t}, its translation v (off a point), and its matrix
        operations = []
        for _ in range(int(count)):
            operation = np.array(read(16, integers)).reshape(4, 4)
            v = None if special else read(4, integers)
            matrix = np.array(read(dimension * dimension, elements)).reshape(dimension, dimension)
            operations.append((operation, v, matrix))
        yield int(group), label.strip(), dimension // arms, star[0], special, operations


# A number for the output, rounded (and + 0.0, so that there is no -0.0)
def number(x):
    return round(float(x), 12) + 0.0


# The tables, from the two files
def extract(cir, pir):

    # 1. Each k-point of each group, from the complex representations
    # ---------------------------------------------------------------
    groups = {}
    for group, label, block, k, special, operations in records(cir):

        # the star's first arm as fractions, and the operations that leave it
        # as it is: those whose matrix has a block at that arm
        terms = [tuple(Fraction(int(n), int(row[3])) if row[3] else Fraction(0) for n in row[:3]) for row in k]
        little = [operation for operation in operations if operation[2][:block, :block].any()]

        # the k-point: its arm and its little group, the same for every
        # representation there
        point_k = {"constant": [str(x) for x in terms[0]]}
        point_k.update({p: [str(x) for x in term] for p, term in zip(("alpha", "beta", "gamma"), terms[1:]) if any(term)})
        listed = [{"R": (op[:3, :3] // op[3, 3]).tolist(), "t": [str(Fraction(int(n), int(op[3, 3]))) for n in op[:3, 3]]}
                  for op, _, _ in little]
        point = groups.setdefault(str(group), {}).setdefault(
            re.sub(r"\d+[+-]?$", "", label), {"k": point_k, "operations": listed, "irreps": {}})
        assert (point["k"], point["operations"]) == (point_k, listed), (group, label)

        # the representation: on each operation of the little group, the trace
        # of its block at the first arm, and the translation v of its phase
        point["irreps"][label] = {
            "characters": [[number(x.real), number(x.imag)] for x in (np.trace(m[:block, :block]) for _, _, m in little)],
            "phases": [["0", "0", "0"]] * len(little) if special else
                      [[str(Fraction(n, v[3])) for n in v[:3]] for _, v, _ in little],
        }

    # 2. The physically irreducible representations, from their headers alone
    # -----------------------------------------------------------------------
    physical = {}
    for line in Path(pir).read_text().splitlines():
        if header := HEADER.match(line):
            _, group, _, label, dimension, *_ = header.groups()
            physical.setdefault(str(int(group)), {})[label.strip()] = {
                "complex": re.findall(r"[A-Z]+\d+[+-]?", label), "dimension": int(dimension)}
    return {"source": "ISO-IR", "groups": groups, "physical": physical}


# Command-line interface: the two files, CIR_data.txt then PIR_data.txt
if __name__ == "__main__":
    tables = extract(*sys.argv[1:3])
    output = Path(__file__).with_name("isoir.json.gz")
    with gzip.open(output, "wt") as file:
        json.dump(tables, file, separators=(",", ":"))
    print(f"wrote {output}: {len(tables['groups'])} groups, "
          f"{sum(len(points) for points in tables['groups'].values())} k-points")
