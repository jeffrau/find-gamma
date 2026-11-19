# Labels for the parts of Γ: Mulliken's at k = 0 (B2g), Miller and Love's
# elsewhere (M3+M4+), from ISO-IR's tables.
#
# Γ's operations and matrices are spin_symmetry.py's. Nothing is decomposed: each
# irreducible representation is counted in Γ by its characters.
#
#  1. G is put in spglib's standard setting of it (ITA's, with origin choice 2,
#     and right-handed), found for a crystal with exactly G's symmetry, built in
#     the file's cell. The names away from k = 0 depend on that choice of origin
#     and axes.
#  2. The wavevectors in Γ are read off its characters on G's translations, and
#     gathered into stars. They are written in that setting's conventional
#     reciprocal axes (the second thing returned).
#  3. At one wavevector of each star, ISO-IR's table gives the little group's
#     irreducible representations, and each is counted in Γ by the average of
#     the product of the two characters over the little group.
#  4. ISO-IR's table of physically irreducible representations puts them
#     together into Γ's real parts, with their names. A part at k = 0 is renamed
#     by Mulliken's rules, from its characters on the point group.
#
# The tables are ISO-IR's (Stokes, Campbell and Cordes, Acta Cryst. A 69, 388
# (2013)), as extract_isoir.py writes them.

import functools
import itertools
import re
from fractions import Fraction

import numpy as np
import spglib

# The axes x, y, z
X, Y, Z = np.eye(3, dtype=int)

# ISO-IR's names of three lines of symmetry, written as Greek letters
GREEK = {"DT": "Δ", "LD": "Λ", "SM": "Σ"}

# The letters of a few points of symmetry (CDML's), by lattice: the group whose
# rotations make each one's star, and one wavevector of it in the conventional
# reciprocal axes
POINTS = {
    "cP": (221, {"X": (0, 1 / 2, 0), "M": (1 / 2, 1 / 2, 0), "R": (1 / 2, 1 / 2, 1 / 2)}),
    "cF": (225, {"X": (0, 1, 0), "L": (1 / 2, 1 / 2, 1 / 2), "W": (1 / 2, 1, 0)}),
    "cI": (229, {"H": (0, 0, 1), "N": (1 / 2, 1 / 2, 0), "P": (1 / 2, 1 / 2, 1 / 2)}),
    "tP": (123, {"Z": (0, 0, 1 / 2), "X": (0, 1 / 2, 0), "M": (1 / 2, 1 / 2, 0), "R": (0, 1 / 2, 1 / 2), "A": (1 / 2, 1 / 2, 1 / 2)}),
    "hP": (191, {"A": (0, 0, 1 / 2), "M": (1 / 2, 0, 0), "L": (1 / 2, 0, 1 / 2), "K": (1 / 3, 1 / 3, 0), "H": (1 / 3, 1 / 3, 1 / 2)}),
    "hR": (166, {"T": (0, 0, 3 / 2)}),
    "oP": (47, {"X": (1 / 2, 0, 0), "Y": (0, 1 / 2, 0), "Z": (0, 0, 1 / 2), "S": (1 / 2, 1 / 2, 0), "U": (1 / 2, 0, 1 / 2),
                "T": (0, 1 / 2, 1 / 2), "R": (1 / 2, 1 / 2, 1 / 2)}),
}


# Two points of no symmetry (numpy's default_rng(0).random((2, 3)), written out
# so that they do not turn on numpy's generator). Confirmed as a general Wyckoff
# position for all 230 space groups at 1e-5 tolerance.
GENERIC = ((0.6369616873214543, 0.2697867137638703, 0.04097352393619469),
           (0.016527635528529094, 0.8132702392002724, 0.9127555772777217))


# Rows of fractions, as the tables write them ("1/2"), as an array of numbers
def numbers(rows):
    return np.array([[float(Fraction(x)) for x in row] for row in rows])


# spglib's number for a space group's standard setting, as ISO-IR's: origin
# choice 2 where there are two
@functools.cache
def standard_hall_number(number):
    halls = [hall for hall in range(1, 531) if spglib.get_spacegroup_type(hall).number == number]
    return next((hall for hall in halls if spglib.get_spacegroup_type(hall).choice == "2"), halls[0])


# The operations' translations made exact: each to the nearest fraction with a
# denominator up to 48, where every one is within 10⁻³ of one (all of them or
# none: snapping some and not others, where the structure's origin lies a little
# off a special point, would no longer describe a group). A structure a little
# off its ideal positions gives them only to its own precision, and spglib, asked
# for 10⁻⁵, would read the difference as a different crystal. It matters where
# G lies in a more symmetric lattice, so that several standard settings tie
# exactly and spglib's pick among them turns on the last digits it is given.
# With the rounding of the cell in standard_setting, it decides the setting,
# and so the labels, of 17 MAGNDATA entries (P-1, C2/m, P3_121 and P-6m2 groups:
# 1.91-1.93, 1.210, 1.257, 1.304, 1.348, 1.578, 1.627, 1.658, 1.741, 1.755,
# 1.786, 1.808, 1.826, 1.860 and 1.861). On its own it decides the six with
# G = C2/m (1.348, 1.578, 1.627, 1.658, 1.755 and 1.826).
def exact(rotations, translations):
    translations = np.array(translations, dtype=float)
    snapped = np.array([float(Fraction(float(x)).limit_denominator(48)) for x in translations.flat]).reshape(translations.shape)
    if not np.all(np.abs(snapped - translations) < 1e-3):
        return translations

    # and only if they are still a group: the product {R|t}{S|s} = {RS|t + Rs}
    # of every two is one of them, up to a translation of the cell. (Almost any
    # number is within 10⁻³ of such a fraction, so the translations of an origin
    # at no special position pass the test above, but, each snapped on its own,
    # they no longer multiply.)
    def key(R, t):
        return R.tobytes(), tuple(np.round(t % 1, 9) % 1)
    listed = {key(R, t) for R, t in zip(rotations, snapped)}
    closed = all(key(R @ S, t + R @ s) in listed for R, t in zip(rotations, snapped) for S, s in zip(rotations, snapped))
    return snapped if closed else translations


# G's number, and the standard setting the labels are in: x goes to P x + p
# there. spglib finds it for a crystal made of two points of no symmetry, a kind
# each, and all their images under G, which has exactly G's symmetry. Among
# equivalent settings its pick depends on what it is given, so it is always
# given them the same way: in the file's cell, in order.
def standard_setting(lattice, rotations, translations, cell):
    translations = exact(rotations, translations)

    # the metric made to have G's lengths and angles exactly (averaged over G)
    metric = np.mean(rotations.transpose(0, 2, 1) @ (lattice @ lattice.T) @ rotations, axis=0)

    # the file's cell in the operations' cell (as rows), and a lattice point of
    # the operations' cell for each place in it
    N = np.rint(cell @ np.linalg.inv(lattice)).astype(int)
    reach = int(np.abs(N).sum(axis=0).max())
    steps = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)))
    steps = steps[np.unique(np.round(steps @ np.linalg.inv(N) % 1, 6) % 1, axis=0, return_index=True)[1]]

    # the two generic points and all their images under G, in the file's cell,
    # a kind each, sorted
    points = np.array(GENERIC) @ N
    images = [(x @ rotations.transpose(0, 2, 1) + translations)[:, None] + steps[None] for x in points]
    positions = np.concatenate(images).reshape(-1, 3) @ np.linalg.inv(N) % 1
    kinds = np.repeat([1, 2], len(rotations) * len(steps))
    order = np.lexsort(np.vstack([np.round(positions, 6).T[::-1], kinds]))

    # spglib's standard setting of that crystal, the one ISO-IR uses
    # (the file's cell from its metric, in one orientation whatever basis the
    # operations came in, rounded, as the positions are, so that two ways of
    # reaching the same G give spglib the same numbers)
    crystal = (np.round(np.linalg.cholesky(N @ metric @ N.T), 6), np.round(positions[order], 8) % 1, kinds[order])
    number = spglib.get_symmetry_dataset(crystal, symprec=1e-5).number
    found = spglib.get_symmetry_dataset(crystal, symprec=1e-5, hall_number=standard_hall_number(number))
    return number, found.transformation_matrix @ np.linalg.inv(N).T, found.origin_shift


# A wavevector as the table writes it, in the conventional reciprocal axes of
# G's standard setting (the one the names are in): 0, its letter where it is one
# of POINTS, or the coordinates of its shortest equal (of several as short, the
# one with the largest first component, then second, then third).
def wavevector(k, same, number, metric):
    if same(k, np.zeros(3)):
        return "0"

    # a letter, if k is one of POINTS for G's lattice
    family = next(letter for last, letter in ((2, "a"), (15, "m"), (74, "o"), (142, "t"), (194, "h"), (230, "c")) if number <= last)
    centring = spglib.get_spacegroup_type(standard_hall_number(number)).international_short[0]
    holohedry, points = POINTS.get(family + ("S" if centring in "ABC" else centring), (1, {}))
    for letter, point in points.items():
        for rotation in spglib.get_symmetry_from_database(standard_hall_number(holohedry))["rotations"]:
            if same(k, np.linalg.inv(rotation).T @ point):
                return letter

    # else the coordinates of its shortest equal, as fractions where they are
    equals = [k - whole for shift in itertools.product(range(-2, 3), repeat=3)
              if same(whole := np.rint(k) + shift, np.zeros(3))]
    k = min(equals, key=lambda q: (round(q @ metric @ q, 8), tuple(-np.round(q, 8))))
    fractions = [Fraction(x).limit_denominator(48) for x in k]
    return "(" + ",".join(str(f) if abs(f - x) < 1e-4 else f"{x:.4f}" for f, x in zip(fractions, k)) + ")"


# Mulliken's name of a real irreducible representation of a point group, from
# its characters on the group's operations (matrices of whole numbers, in the
# standard setting's axes: the main axis along z, or b if twofold). The names
# agree with the Point Group Tables of the Bilbao Crystallographic Server.
def mulliken(rotations, chi):

    # each operation: its rotation with any inversion taken out, that rotation's
    # order, whether it had an inversion, and its character
    operations = []
    for R, c in zip(rotations, chi):
        improper = np.linalg.det(R) < 0
        rotation = -R if improper else R
        operations.append((rotation, {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}[np.trace(rotation)], improper, c))
    orders = [order for _, order, _, _ in operations]


    # the character on the first operation of a kind (or None if there is none)
    def first(order, improper, test=lambda rotation: True):
        return next((c for rotation, n, i, c in operations if n == order and i == improper and test(rotation)), None)

    # is the rotation about this axis?
    def about(axis):
        return lambda rotation: (rotation @ axis == axis).all()

    # the subscript 1 or 2, by the sign of a character
    def index(c):
        return "" if c is None else "1" if c > 0 else "2"

    # the characters that decide the name
    dimension = round(chi[0])
    inversion, mirror = first(1, True), first(2, True)
    C6, C4, S4 = first(6, False), first(4, False), first(4, True)
    twofolds = [first(2, False, about(axis)) for axis in (Z, Y, X)]
    # the letter and subscript, by the kind of point group
    if sum(order == 3 and not improper for _, order, improper, _ in operations) == 8:  # cubic
        name = "AET"[dimension - 1] + ("" if dimension == 2 else index(C4 if C4 is not None else S4))
    elif dimension == 2:
        name = "E" + index(C6)
    elif max(orders) <= 2 and None not in twofolds:  # 222, mmm: B1, B2, B3 are even under the twofold axis along z, y, x
        even = [n for n, c in zip("123", twofolds) if c > 0]
        name = "A" if len(even) == 3 else "B" + even[0]
    elif max(orders) <= 2 and twofolds[0] is not None and mirror is not None and inversion is None:  # mm2
        name = "A" + index(mirror) if twofolds[0] > 0 else "B" + index(first(2, True, about(Y)))
    elif max(orders) <= 2:  # 2, m, 2/m, 1, -1
        name = "B" if (first(2, False) or 1) < 0 else "A"
    else:  # one axis of order 3, 4 or 6, along z
        main = C6 if C6 is not None else C4 if C4 is not None else S4
        # the axes a of the cell or, where the main axis is sixfold, the axes 30°
        # from them, as Bilbao has it
        a = [X - Y, X + 2 * Y, 2 * X + Y] if C6 is not None else [X, Y] + ([X + Y] if 3 in orders or 6 in orders else [])
        # the subscript: by a twofold axis across z (along a, where some are),
        # or else by a mirror through z (holding a)
        tests = [(False, lambda rotation: any(about(axis)(rotation) for axis in a)),
                 (False, lambda rotation: not about(Z)(rotation)),
                 (True, lambda rotation: not about(Z)(rotation) and any(about(axis)(-rotation) for axis in a)),
                 (True, lambda rotation: not about(Z)(rotation))]
        found = (first(2, improper, test) for improper, test in tests)
        name = ("B" if main is not None and main < 0 else "A") + index(next((c for c in found if c is not None), None))
    # Return the name, with g or u where there is an inversion, or else with '
    # or '' by a mirror across the main axis (or, with nothing else, the mirror)
    if inversion is not None:
        return name + ("g" if inversion > 0 else "u")
    across = first(2, True, about(Z))
    across = across if across is not None else mirror if len(operations) == 2 else None
    return name + ("" if across is None else "'" if across > 0 else "''")


# Γ as a sum of named real irreducible representations of G, and the wavevectors
# in it. The operations and Γ are spin_symmetry.py's, in its `lattice`; `cell`
# is the file's cell.
def name(lattice, rotations, translations, gamma, table, cell):

    # 1. G in its standard setting, and its operations
    # -------------------------------------------------
    number, P, p = standard_setting(lattice, rotations, translations, cell)
    R = np.rint(P @ rotations @ np.linalg.inv(P)).astype(int)
    t = translations @ P.T + p - R @ p

    # ISO-IR's tables for G, the character of Γ, and which operations are G's
    # translations (up to the magnetic lattice's)
    entries, physical = table["groups"][str(number)], table["physical"][str(number)]
    chi = np.trace(gamma, axis1=1, axis2=2)
    pure = (R == np.eye(3, dtype=int)).all(axis=(1, 2))

    # in a setting: are two wavevectors the same, up to a vector of G's
    # reciprocal lattice? (G's translations all come of `whole`)
    def modulo(setting):
        whole = np.concatenate([translations[pure] @ setting.T, setting.T])
        return lambda k, q: np.allclose(((k - q) @ whole.T + 0.5) % 1, 0.5, atol=1e-6)

    # 2. The wavevectors in Γ: those of the magnetic lattice's reciprocal
    #    lattice at which its translations have a part
    # --------------------------------------------------------------------
    same = modulo(np.eye(3))
    sizes = [round(1 / min([x for x in translations[pure][:, i] if x > 1e-6], default=1)) for i in range(3)]
    present = []  # (in the operations' cell, whole numbers)
    for k in itertools.product(*map(range, sizes)):
        if abs(np.mean(chi[pure] * np.exp(-2j * np.pi * translations[pure] @ k))) > 1e-6 and not any(same(np.array(k), q) for q in present):
            present.append(np.array(k))

    # the same in the standard setting, and as the table writes them
    same, present = modulo(P), [k @ np.linalg.inv(P) for k in present]
    edges = lattice.T @ np.linalg.inv(P)
    written = [wavevector(k, same, number, np.linalg.inv(edges.T @ edges)) for k in present]

    # 3. Each irreducible representation of a little group in Γ, and how many
    #    times, star by star
    # -------------------------------------------------------------------------
    found = {}
    while present:
        # the star of the first wavevector left, and the others it takes away
        # (with their -k's, which are the k's conjugated)
        star = [present[0]]
        for rotation in R:
            arm = np.linalg.inv(rotation).T @ present[0]
            if not any(same(arm, q) for q in star):
                star.append(arm)
        present = [q for q in present if not any(same(q, arm) or same(q, -arm) for arm in star)]

        # the table's wavevector this star lies on: a point before a line before
        # a plane, and where it has parameters, all positive and the shortest (a
        # parameter of 0 can come out of the least squares as -10⁻¹⁷, so -10⁻⁹
        # counts as 0: 1.0.46's k = (2/3,0,0) on ISO-IR's plane B)
        matches = []
        for entry in sorted(entries.values(), key=lambda entry: len(entry["k"])):
            base, *free = numbers(entry["k"].values())
            for arm in star + [-arm for arm in star]:
                for shift in itertools.product((-1, 0, 1), repeat=3):
                    if same(np.array(shift), np.zeros(3)):
                        offset = arm + shift - base
                        values = np.linalg.lstsq(np.array(free).T, offset, rcond=None)[0] if free else np.zeros(0)
                        if np.allclose(values @ np.array(free).reshape(-1, 3), offset, atol=1e-6):
                            matches.append((not (values > -1e-9).all(), np.linalg.norm(arm + shift), len(matches), entry, arm + shift))
            if matches:
                break
        if not matches:
            continue

        # the table's operation with each of G's rotations, the translation from
        # the table's to ours, and each irreducible representation counted in Γ
        # by its characters on the little group
        *_, entry, k = min(matches, key=lambda match: match[:3])
        index = {np.array(op["R"]).tobytes(): n for n, op in enumerate(entry["operations"])}
        which = np.array([index.get(rotation.tobytes(), -1) for rotation in R])
        little = which >= 0
        beyond = t[little] - numbers(op["t"] for op in entry["operations"])[which[little]]
        for label, irrep in entry["irreps"].items():
            characters = np.array([complex(*c) for c in irrep["characters"]])[which[little]]
            phases = np.exp(2j * np.pi * (numbers(irrep["phases"])[which[little]] + beyond) @ k)
            count = round(abs(np.mean(chi[little] * np.conj(characters * phases))))
            if count:
                found[label] = count

    # 4. The real parts these make, by ISO-IR's physically irreducible
    #    representations, renamed by Mulliken's rules at k = 0
    # -------------------------------------------------------------------------
    parts, named = [], 0
    for label, part in physical.items():
        count = found.get(part["complex"][0], 0) // part["complex"].count(part["complex"][0])
        named += count * part["dimension"]
        if count and label.startswith("GM"):
            real = sum(np.array([complex(*c) for c in entries["GM"]["irreps"][c]["characters"]]) for c in part["complex"]).real
            label = mulliken([np.array(op["R"]) for op in entries["GM"]["operations"]], real)
        if count:
            parts.append(("" if count == 1 else str(count)) + re.sub(r"(?<![A-Z])(DT|LD|SM)(?=\d)", lambda found: GREEK[found.group()], label))
    # Return the names (with ? for a part the table has no name for) and the
    # wavevectors
    return " ⊕ ".join(parts + ["?"] * (named != gamma.shape[1])), ", ".join(dict.fromkeys(written))

