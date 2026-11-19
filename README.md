# find-gamma

Code accompanying J. G. Rau, "Spin Space Symmetry Without Spin Space Groups", arXiv:2610.37770.

For a commensurate magnetic structure, it determines the paramagnetic space group G₀, the parent space group G of
the spin space group, the defining representation Γ of G, and the non-relativistic responses and spin texture that Γ
allows. Applied to the MAGNDATA database, it produces the table in the paper's Supplemental Material, which
also describes the method.

## Requirements

Python 3, with numpy, spglib and pymatgen. `requirements.txt` gives the versions used for the table.

## Usage

```bash
python find_gamma.py 0.15.mcif
```

```
name   formula   MSG          G0               order       G                Γ     k   allowed   texture
0.15   Mn F2     P4_2'/mnm'   P4_2/mnm (136)   collinear   P4_2/mnm (136)   B2g   0   C Σ-      d/-
```

`0.15.mcif` is MAGNDATA entry 0.15 (MnF₂), from https://cryst.ehu.es/magndata.

`find_gamma.py` takes one or more magnetic CIFs, or structures in the input format below. `--operations` also lists
each operation {R|t} of G with its matrix Γ(g), and `--tolerance` sets the tolerance for positions (Å) and moments
(μB), 0.01 by default.

The allowed responses are given by the symbols of Table I of the paper:

| Symbol | Response |
|---|---|
| M | Net spin magnetization |
| C | Piezomagnetism |
| Σ- | Time-odd spin conductivity |
| Σ+ | Time-even spin conductivity |
| σ- | Anomalous Hall conductivity |
| α- | Time-odd Edelstein effect |
| α+ | Time-even Edelstein effect |

`none` is printed where no response is allowed.

The texture is given as the lowest even and the lowest odd multipole rank of the momentum-space spin texture
(s, p, d, f, g, h, i, k for ℓ = 0, …, 7), with a dash where the texture has no part of that parity.

The columns Γ and k give the irreducible components of Γ and their wave vectors, in spglib's standard setting of G
(that of the International Tables, with origin choice 2). Their labels need the ISO-IR tables
of irreducible representations [H. T. Stokes, B. J. Campbell and R. Cordes, Acta Cryst. A 69, 388 (2013)], which
`extract_isoir.py` converts to `isoir.json.gz` from ISO-IR's data files `CIR_data.txt` and `PIR_data.txt`
(in `CIR_data.zip` and `PIR_data.zip` at https://iso.byu.edu). It writes `isoir.json.gz` beside the scripts, where they
look for it. Without them, Γ and k are written as a dash.

A `?` at the end of Γ means that something has gone wrong: the parts named do not add up to the dimension of Γ, and
the labels of that structure should not be trusted. It does not occur in the MAGNDATA table.

```bash
python extract_isoir.py CIR_data.txt PIR_data.txt
```

## Input format

```json
{
 "name": "0.15",
 "formula": "Mn F2",
 "magnetic_space_group": "P4_2'/mnm'",
 "lattice": [[4.8736, 0.0, 0.0], [0.0, 4.8736, 0.0], [0.0, 0.0, 3.3]],
 "sites": [
  {"kind": "Mn", "position": [0.0, 0.0, 0.0], "moment": [0.0, 0.0, 4.6]},
  {"kind": "Mn", "position": [0.5, 0.5, 0.5], "moment": [0.0, 0.0, -4.6]},
  {"kind": "F", "position": [0.30464, 0.30464, 0.0], "moment": [0.0, 0.0, 0.0]},
  …
 ]
}
```

The rows of `lattice` are the cell vectors in Cartesian coordinates (Å). Positions are fractional and moments are
Cartesian (μB). Every atom of the cell is listed. `read_mcif.py` writes this form for a magnetic CIF:

```bash
python read_mcif.py 0.15.mcif > 0.15.json
```

A structure from any other source can be given in this form. The analysis uses only `lattice` and `sites`, but
`find_gamma.py` also prints `name`, `formula` and `magnetic_space_group`, so these must be present (they can be empty).

## The MAGNDATA table

With `isoir.json.gz` in place:

```bash
python build_magndata_table.py <MAGNDATA folder> --latex magndata-index.tex --tsv magndata_table.tsv --jobs 8
```

This analyzes every commensurate structure in a folder of MAGNDATA magCIF files (`<id>.mcif`) and writes the
Supplemental Material's table as a LaTeX longtable and as `magndata_table.tsv`. The MAGNDATA files are not included here.
They are available from the Bilbao Crystallographic Server (https://cryst.ehu.es/magndata).

| File | Contents |
|---|---|
| `find_gamma.py` | The full row for one or more structures. |
| `build_magndata_table.py` | Runs the analysis over the MAGNDATA files and writes the table. |
| `extract_isoir.py` | Converts ISO-IR's data files to the tables used by `irrep_labels.py`. |
| `read_mcif.py` | Reads a magnetic CIF and writes the full magnetic unit cell in the input format above. |
| `spin_symmetry.py` | Computes G₀, G, the type of order, Γ, the allowed responses and the spin texture. |
| `irrep_labels.py` | Labels the irreducible components of Γ and gives their wave vectors. |
| `magndata_corrections.json` | Corrections applied to 50 MAGNDATA files, each with its reason and the SHA-256 checksum of the file it applies to. |
| `magndata_compounds.tsv` | The compound name of each entry as MAGNDATA lists it, with the 17 names corrected where they are in error and the reason for each. |
| `magndata_table.tsv` | The results for the 2318 commensurate entries: ID, tolerance, G₀, order, G, allowed responses, texture, Γ and k. |

Each file is analyzed with a tolerance of 0.01, except for the six entries listed in `TOLERANCES` in
`build_magndata_table.py`. A corrected file is read with its correction only if its checksum matches.

Running `build_magndata_table.py` on the same MAGNDATA files reproduces `magndata_table.tsv`:

```bash
python build_magndata_table.py <MAGNDATA folder> --tsv rerun.tsv --jobs 8
diff rerun.tsv magndata_table.tsv
```

## Citation

If you use this code, please cite the paper, J. G. Rau, "Spin Space Symmetry Without Spin Space Groups",
arXiv:2610.37770, and the code, J. G. Rau, Code for "Spin Space Symmetry Without Spin Space Groups", Zenodo
(2026), by its DOI there.

## License

Copyright © 2026 Jeffrey G. Rau. Licensed under the Creative Commons Attribution 4.0 International License
(CC BY 4.0); see `LICENSE`.
