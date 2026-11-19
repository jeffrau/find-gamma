# The table of the paper's Supplemental Material, from a folder of MAGNDATA's
# files: for each commensurate structure, spin_symmetry.py's groups, order and
# responses and irrep_labels.py's Γ and k, as a LaTeX longtable to \input (and,
# with --tsv, as magndata_table.tsv's columns).
#
#     python build_magndata_table.py ../magndata --latex magndata-index.tex
#                                    [--tsv magndata_table.tsv] [--jobs 28]
#                                    [--table isoir.json.gz]
#     python build_magndata_table.py ../magndata --copies corrected
#
# The second writes only the files as they are read, corrected, so that one
# entry can be run through find_gamma.py by hand.
#
# Each file is read as MAGNDATA gives it, except the 50 with a correction in
# magndata_corrections.json. Each correction comes with its reason and the
# SHA-256 of the file it was made for, and a file of another version is read as
# it is, with a warning. The tolerance is 0.01, or the one given in TOLERANCES.
# A compound is named as MAGNDATA's list names it (magndata_compounds.tsv,
# corrected where that name is wrong), or else by the file's formula. The rows
# are in MAGNDATA's order, with a small space between its series.

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import spglib

import irrep_labels
from read_mcif import read_mcif
from spin_symmetry import analyse

# The folder of this script, and of magndata_corrections.json and
# magndata_compounds.tsv
HERE = Path(__file__).parent

# Where the default, 0.01 (Angstroms and Bohr magnetons), would find a symmetry
# that the file resolves, or that its moments, a calculation's, do not have
TOLERANCES = {"0.226": 0.005, "0.989": 0.005, "1.791": 0.005, "1.800": 0.005, "3.28": 0.005, "0.1067": 0.002}

# The corrections: for an entry, the SHA-256 of the file it was made for, its
# reason, and each text to replace with its replacement
CORRECTIONS = json.loads((HERE / "magndata_corrections.json").read_text(encoding="utf-8"))

# MAGNDATA's order of its series
SERIES = ("0", "1.0", "1", "2", "3")

# The LaTeX for each response of spin_symmetry.py, each Greek point of
# irrep_labels.py, and each character LaTeX reserves
SYMBOLS = {"M": r"$\vec{M}$", "C": r"$\vec{C}_{\mu\nu}$", "Σ-": r"$\Sigma^{-}$", "Σ+": r"$\Sigma^{+}$",
           "σ-": r"$\sigma^{-}$", "α-": r"$\alpha^{-}$", "α+": r"$\alpha^{+}$"}
GREEK = {"Γ": r"\Gamma", "Δ": r"\Delta", "Λ": r"\Lambda", "Σ": r"\Sigma"}
ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

# A chemical formula, read as an element (two letters only when they are one: in
# KyFe2-xSe2, K has y) or a closing bracket, and its count
ELEMENTS = ("He Li Be Ne Na Mg Al Si Cl Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Zr Nb Mo Tc Ru Rh Pd "
            "Ag Cd In Sn Sb Te Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po "
            "At Rn Fr Ra Ac Th Pa Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
FORMULA = re.compile(rf"(?P<head>{'|'.join(ELEMENTS)}|[A-Z]|[)\]])"
                     r"(?P<count>\d+(?:\.\d+)?(?:[-+][xyzδ](?![a-z]))?|[xyzδ](?![a-z])|(?<=[)\]])n(?![a-z]))?|(?P<other>.)", re.S)

# The full stop MAGNDATA writes for a hydrate's dot (Fe2(C2O4)3.4H2O,
# Cs2FeCl5.D2O): one before the water and its count
HYDRATE = re.compile(r"\.(?=\d*[HD]2O)")

# The table's header row
HEAD = (r"ID & Compound & PM SG & MSG (BNS) & Order & Parent SG & $\Gamma$ & $\vec{k}$ & Allowed responses"
        r" & $\vec{s}(\vec{q})$ \\" "\n" r"\midrule")


# Text with the characters LaTeX reserves escaped
def escape(text):
    return "".join(ESCAPES.get(c, c) for c in text)


# A space group's symbol (P4_2/mnm, Fd-3m', P_c2_1/c) as LaTeX
def space_group(symbol):
    return "$" + re.sub(r"_(\d)", r"_{\1}", re.sub(r"-(\d)", r"\\bar{\1}", re.sub(r"[^A-Za-z0-9_/'.:-]", "", symbol))) + "$"


# A chemical formula (CoTa3S6) as LaTeX
# (CoTa\textsubscript{3}S\textsubscript{6}), breakable after a count or a
# bracket
def chemical(name):

    # a hydrate: the two formulas with a raised dot between them, and the
    # water's count before it, not as a subscript
    if len(pieces := HYDRATE.split(name, maxsplit=1)) == 2:
        count, water = re.fullmatch(r"(\d*)(.*)", pieces[1], re.S).groups()
        return chemical(pieces[0]) + r"$\cdot$" + count + chemical(water)

    # split it into its parts, each with its count: anything else is joined to
    # the part before it, where that has no count
    parts = []
    for match in FORMULA.finditer(name.replace(" ", "")):
        if match["other"] is None:
            parts.append((match["head"], "" if match["count"] in (None, "1") else match["count"]))
        elif parts and not parts[-1][1]:
            parts[-1] = (parts[-1][0] + match["other"], "")
        else:
            parts.append((match["other"], ""))

    # write the counts as subscripts, and allow a line break after each
    # subscript or bracket (but not before a closing bracket)
    text = "".join(escape(t) + (rf"\textsubscript{{{escape(n)}}}" if n else "") for t, n in parts)
    return re.sub(r"(\\textsubscript\{[^}]*\}|[\])](?!\\textsubscript))(?![\])])", r"\1\\allowbreak{}", text)


# A Mulliken name (B2g, A2''): a letter, an index, a parity and primes
MULLIKEN = re.compile(r"([ABET])(\d?)([gu]?)('*)")


# A name of one of Γ's parts as LaTeX: Mulliken's (B2g, A2'') or Miller and
# Love's (M3+M4+, Λ1)
def label(text):

    # Mulliken's: the index and parity as a subscript
    if mulliken := MULLIKEN.fullmatch(text):
        letter, index, parity, primes = mulliken.groups()
        return letter + (f"_{{{index + parity}}}" if index + parity else "") + primes

    # Miller and Love's: each point with its index as a subscript and its parity
    # as a superscript (anything else is written as it is)
    parts = re.findall(r"([A-Z]+|[ΓΔΛΣ])(\d+)([+-]?)", text)
    if "".join(map("".join, parts)) != text:
        return escape(text)
    return "".join(GREEK.get(point, point) + f"_{{{index}}}" + (f"^{{{sign}}}" if sign else "") for point, index, sign in parts)


# irrep_labels.py's Γ (2A2- ⊕ Bg) as LaTeX, its parts at k = 0 first, then the
# others, each by name
def representation(text):
    parts = sorted(re.fullmatch(r"(\d*)(.+)", part).groups() for part in text.split(" ⊕ "))
    parts.sort(key=lambda part: (MULLIKEN.fullmatch(part[1]) is None, part[1]))
    return " \\oplus ".join(count + label(name) for count, name in parts)


# The BNS symbol of the file's magnetic space group, as MAGNDATA's list writes
# it: curly primes as ', the newer notation's ".1" of a type-I group dropped
# (Pnma.1), and a screw axis written without its underscore (P 21/c) as one (P
# 2_1/c) where the space group's own symbol has it
def bns(text):

    # a tag's value (_space_group_magn.name_BNS or, in older files,
    # _space_group.magn_name_BNS), without every pair of quotes (1.23 writes
    # "'P_A c c n'") and without invisible formatting characters
    def tag(name):
        found = re.search(rf"^\s*_space_group[._]magn[._]{name}_BNS\s+(.+?)\s*$", text, re.M | re.I)
        value = found[1].strip() if found else ""
        while len(value) > 1 and {"'": "'", '"': '"', "\u2018": "\u2019", "\u201c": "\u201d"}.get(value[0]) == value[-1]:
            value = value[1:-1].strip()
        return " ".join("".join(c for c in value if unicodedata.category(c) != "Cf").split())

    # the symbol, made regular; the space group's number (from the BNS number)
    # gives its own symbol, to tell a screw axis from two separate axes
    symbol = re.sub(r"\s*\.\s*1\s*$", "", re.sub("[\u2019\u2032\u2018]", "'", tag("name"))).strip()
    group = re.match(r"\D*(\d+)\.", tag("number"))
    standard = spglib.get_spacegroup_type(irrep_labels.standard_hall_number(int(group[1]))).international_short if group else ""
    return re.sub(r"(?<![\d_\-])([2346])([1-5])(?!\d)",
                  lambda m: f"{m[1]}_{m[2]}" if int(m[2]) < int(m[1]) and f"{m[1]}_{m[2]}" in standard else m[0], symbol)


# irrep_labels.py's k (0, M, (1/2,0,-1/3)) as LaTeX, each wave vector its own
# formula
def wavevectors(text):

    # one wave vector: 0 as a bold zero, a letter as it is, and coordinates with
    # 1/2 as \tfrac{1}{2}
    def one(k):
        if k == "0":
            return r"\vec{0}"
        if k.startswith("("):
            return re.sub(r"(-?)(\d+)/(\d+)", r"\1\\tfrac{\2}{\3}", k)
        return k
    return ", ".join(f"${one(k)}$" for k in text.split(", "))


# One file's row: spin_symmetry.py's and irrep_labels.py's results (None for an
# incommensurate structure)
def row(path):

    # the entry's tolerance
    identifier = path.stem
    tolerance = TOLERANCES.get(identifier, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # read the file (an incommensurate structure has no row)
        text = corrected(path)
        try:
            cell = read_mcif(text, identifier)
        except ValueError as error:
            if "incommensurate" in str(error):
                return None
            raise

        # analyse it, then name Γ's parts from its operations
        found = analyse(cell, tolerance)
        gamma, k = irrep_labels.name(found["lattice"], found["rotations"], found["translations"], found["gamma"], TABLE,
                                     np.array(cell["lattice"]))

    # Return the row (the space groups without their numbers, as magndata_table.tsv has
    # them)
    return dict(id=identifier, tolerance=tolerance, G0=found["G0"].split()[0], order=found["order"], G=found["G"].split()[0],
                allowed=" ".join(found["allowed"]) or "none", texture=found["texture"], gamma=gamma, k=k,
                msg=bns(text), formula=cell["formula"])


# One file's row, or its error if it cannot be read or analysed
def attempt(path):
    try:
        return row(path)
    except Exception as error:
        return dict(id=path.stem, error=f"{type(error).__name__}: {error}"[:120])


# A file's text, with its correction if it has one and is the version the
# correction was made for
def corrected(path):
    text = path.read_text(errors="replace")
    if correction := CORRECTIONS.get(path.stem):
        if hashlib.sha256(path.read_bytes()).hexdigest() != correction["sha256"]:
            print(f"warning: {path.name} is not the file its correction was made for; read as it is", file=sys.stderr)
        else:
            for old, new in correction["replace"]:
                assert text.count(old) == 1, f"{path.stem}: the correction does not fit the file"
                text = text.replace(old, new)
    return text


# An entry's place in MAGNDATA's order: its series, then its number
def order(identifier):
    series, number = identifier.rsplit(".", 1)
    return (SERIES.index(series) if series in SERIES else len(SERIES), series, int(number))


# The table as LaTeX: a longtable, a row for each entry, a small space between
# MAGNDATA's series
def latex(rows, compounds):

    # a cell that wraps, in a column of the given width (under revtex, a
    # longtable's p{} columns fail)
    def wrap(width, text):
        return rf"\parbox[t]{{{width}}}{{\raggedright\strut {text}\strut}}"

    # the header, repeated on every page
    lines = [r"\begingroup\footnotesize\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\setlength{\tabcolsep}{2pt}",
             r"\begin{longtable}{@{\extracolsep{\fill}}llllllllll@{}}",
             r"\toprule", HEAD, r"\endfirsthead", r"\toprule", HEAD, r"\endhead", r"\bottomrule", r"\endlastfoot"]

    # each row: the entry (linked to MAGNDATA), the compound, the groups, the
    # order, Γ, k, the allowed responses and the texture's lowest ranks
    for n, r in enumerate(rows):
        if n and order(r["id"])[:2] != order(rows[n - 1]["id"])[:2]:
            lines.append(r"\addlinespace[6pt]")
        cells = [
            rf"\href{{https://cryst.ehu.es/magndata/index.php?this_label={r['id']}}}{{{r['id']}}}",
            wrap("1.05in", chemical(compounds.get(r["id"], r["formula"]))),
            space_group(r["G0"]), space_group(r["msg"]) if r["msg"] else "", r["order"].capitalize(), space_group(r["G"]),
            f"${representation(r['gamma'])}$", wrap("0.55in", wavevectors(r["k"])),
            wrap("1.17in", ", ".join(SYMBOLS[s] for s in r["allowed"].split() if s in SYMBOLS)),
            "/".join("--" if letter == "-" else f"${letter}$" for letter in r["texture"].split("/")),
        ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\end{longtable}", r"\endgroup"]) + "\n"


# Argparse interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The table of the paper's Supplemental Material, from a folder of MAGNDATA's files.")
    parser.add_argument("folder")
    parser.add_argument("--latex", type=Path, help="the table to \\input")
    parser.add_argument("--tsv", type=Path, help="the same as magndata_table.tsv's columns")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--table", default=HERE / "isoir.json.gz", help="extract_isoir.py's output")
    parser.add_argument("--copies", type=Path, help="write the files read, corrected, to this folder")
    args = parser.parse_args()

    # MAGNDATA's files, named by their entry (0.15.mcif); with --copies, each
    # is written as it is read
    files = [path for path in Path(args.folder).glob("*.mcif") if re.fullmatch(r"\d+(\.\d+)+", path.stem)]
    if not files:
        parser.error(f"no MAGNDATA files (<id>.mcif) in {args.folder}")
    if args.copies:
        args.copies.mkdir(parents=True, exist_ok=True)
        for path in files:
            (args.copies / path.name).write_text(corrected(path), encoding="utf-8")

    # read ISO-IR's tables and the compounds' names
    with gzip.open(args.table, "rt") as file:
        TABLE = json.load(file)
    with open(HERE / "magndata_compounds.tsv", newline="", encoding="utf-8") as file:
        compounds = {r["id"]: r["compound"] for r in csv.DictReader(file, delimiter="\t")}

    # every file's row, in parallel (forked, so that the workers have TABLE)
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = [r for r in pool.map(attempt, files, chunksize=8) if r is not None]

    # the rows in MAGNDATA's order, and the files that could not be read
    failed = [r for r in results if "error" in r]
    rows = sorted((r for r in results if "error" not in r), key=lambda r: order(r["id"]))
    for r in failed:
        print(f"not in the table: {r['id']}: {r['error']}", file=sys.stderr)

    # write the table, as LaTeX and as magndata_table.tsv's columns
    if args.latex:
        args.latex.write_text(latex(rows, compounds), encoding="utf-8")
    if args.tsv:
        columns = ("id", "tolerance", "G0", "order", "G", "allowed", "texture", "gamma", "k")
        with open(args.tsv, "w", newline="", encoding="utf-8") as file:
            file.write("\t".join(columns) + "\n" + "".join("\t".join(str(r[c]) for c in columns) + "\n" for r in rows))
    print(f"{len(rows)} structures in the table" + (f"; {len(failed)} could not be read" if failed else ""))
