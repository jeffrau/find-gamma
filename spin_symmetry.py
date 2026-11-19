# The parent group G of a magnetic structure's spin space group, its
# representation Γ, and the responses Γ allows.
#
# analyse() takes a structure as read_mcif.py gives it: a cell and its sites,
# each with a kind, a position and a moment. The steps, in the notation of
# "Spin Space Symmetry Without Spin Space Groups" (arXiv:2610.37770):
#
#  1. G0, the space group of the atoms with the moments left out (spglib).
#  2. The operations [U || R | t] of the spin space group: a rotation U of
#     every moment, with an operation {R|t} of space, that together leave the
#     structure as it is. Each {R|t} is a symmetry of the atoms (spglib), and U
#     is the rotation that best carries every moment onto the moment at its
#     image. Their spatial parts {R|t} make the group G.
#  3. Γ(g) = Q' U Q, with the columns of Q an orthonormal basis of the space the
#     moments span. Γ is a representation of G by N by N orthogonal matrices: N
#     = 1 for collinear, 2 for coplanar and 3 for non-coplanar order.
#  4. A response is allowed where an exterior power of Γ and a multipole of the
#     point group share an irreducible part: where the product of their
#     characters, averaged over G, is not zero.
# 
# Lengths are in Angstroms and moments in Bohr magnetons, and one tolerance
# serves for both (default is 0.01).

import numpy as np
import spglib

# Each response: the exterior power of Γ that carries it, and the multipoles it
# needs to overlap, specified by rank (l = 0, 1, 2) and parity (+1 or -1) as
# (rank, parity).
RESPONSES = {
    "M": (1, [(0, +1)]),                     # net spin magnetization
    "C": (1, [(2, +1)]),                     # piezomagnetism
    "Σ-": (1, [(0, +1), (1, +1), (2, +1)]),  # time-odd spin conductivity
    "Σ+": (2, [(0, +1), (1, +1), (2, +1)]),  # time-even spin conductivity
    "σ-": (3, [(1, +1)]),                    # anomalous Hall conductivity
    "α-": (1, [(1, -1)]),                    # time-odd Edelstein effect
    "α+": (2, [(1, -1)]),                    # time-even Edelstein effect
}

# The three classes of magnetic order (spin-only groups)
ORDERS = {1: "collinear", 2: "coplanar", 3: "non-coplanar"}

# the letter of each rank = 0, 1, 2, ..., 12 that is searched
WAVES = "spdfghiklmnoq" # highest rank in MAGNDATA is k-wave  


# For the point given for each site, the first site of that site's kind found
# there, or -1.
def site_at(lattice, positions, kinds, points, tolerance):
    d = points[:, None] - positions[None]
    near = np.linalg.norm((d - np.rint(d)) @ lattice, axis=2) < tolerance
    near &= kinds[:, None] == kinds[None]
    return np.where(near.any(axis=1), near.argmax(axis=1), -1)


# The structure in the smallest cell that repeats it, moments and all. spglib
# finds the operations whose R is a matrix of whole numbers in the cell it is
# given, and in this cell that is every operation of G.
def magnetic_primitive_cell(lattice, positions, kinds, moments, tolerance):

    # find all translations that preserve the sites and moments
    translations = []
    # translation candidates that connect same kind to same kind
    t_candidates = positions[kinds == kinds[0]] - positions[0]
    for t in t_candidates:  # for each candidate
        
        # check whether the translated image has sites in the right place
        image = site_at(lattice, positions, kinds, positions + t, tolerance)
        cond_sites = (image >= 0).all() 

        # check if all the moments in the image match
        cond_moments = np.abs(moments[image] - moments).max() < tolerance

        if cond_sites and cond_moments: # if both match append
            translations.append(t % 1)  

    # if only zero translation, return original cell
    if len(translations) == 1:
        return lattice, positions, kinds, moments
    
    # The lattice these translations make: get from spglib using the primitive
    # cell of a crystal with one atom at each of them. 
    points = (lattice, translations, [1] * len(translations))
    primitive = spglib.get_symmetry_dataset(points, symprec=tolerance).primitive_lattice

    # Sanity check: number of translations matches reduction in cell volume
    assert round(np.linalg.det(lattice) / np.linalg.det(primitive)) == len(translations)

    # Convert the positions into the new cell's basis
    positions = positions @ lattice @ np.linalg.inv(primitive) % 1
    # Keep only one of the copies of each site
    kept = site_at(primitive, positions, kinds, positions, tolerance) == np.arange(len(kinds)) 

    # Return the reduced cell
    return primitive, positions[kept], kinds[kept], moments[kept]


# The operations [U || R | t] that leave the structure as it is, up to the
# translations of the cell given. For each operation {R|t} of the atoms
# (spglib), U is the orthogonal matrix that best carries every moment onto the
# moment at the site {R|t} takes it to. 
#
# This is the orthogonal Procrustes problem: U = u v' from the singular value
# decomposition u s v' of their product. It is kept if every moment then lands
# within the tolerance. From the appendix we then have
#
#     Γ(g) = S P_g S' (S S')^-1
#
# when using exact arithmetic.
def spin_operations(lattice, positions, kinds, moments, tolerance):
    
    # find all crystal symmetries
    found = spglib.get_symmetry((lattice, positions, kinds), symprec=tolerance)

    kept = []
    for R, t in zip(found["rotations"], found["translations"]):
        
        # find if there is the right kind of atom at the image site for {R|t}
        image = site_at(lattice, positions, kinds, positions @ R.T + t, tolerance) 
        # if any atoms missing, not a symmetry (a safety check: spglib accepts
        # {R|t} with its own tolerance test, not quite site_at's, and a missing
        # image, -1, would quietly index the last site's moment)
        if (image < 0).any():
            continue
        
        # Propose the transformation from the SVD of the 3 x 3 matrix
        # sum_i m_g(i) m_i' of the image moments times the original moments
        # (the appendix's S P_g S')
        u, _, vt = np.linalg.svd(moments[image].T @ moments)
        U = u @ vt

        # Check if the proposed U is really a symmetry, if so keep it
        if np.linalg.norm(moments @ U.T - moments[image], axis=1).max() < tolerance:
            kept.append((R, t, U))

    # Return all kept operations
    return [np.array(x) for x in zip(*kept)]


# For a set of rotation matrices determine the character of the multipole of a
# given rank and parity on each. From each rotation determine the angle using
# that
#
#   Tr(R) = 1 + 2 cos(theta)
#
# and invert to give theta. Improper vs proper rotations via det(R): an improper
# R is -1 times a proper rotation, whose angle comes from -Tr(R). In O(3) the
# character of a rank-l multipole is
#
#    sign * sum_{m=-l}^l cos(m*theta)
#
# where theta is the rotation angle and sign is +1 for proper rotations
# and equal to the multipole's parity for improper rotations.
def multipole(rotations, rank, parity):
    proper = np.linalg.det(rotations) > 0
    cosine = (np.where(proper, 1, -1) * np.trace(rotations, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(cosine, -1, 1))
    character = sum(np.cos(m * theta) for m in range(-rank, rank + 1))
    return np.where(proper, character, parity * character)


# Find the Γ(g) matrices and all allowed responses
def analyse(cell, tolerance=0.01):
    # Unpack the crystal from the input dictionary
    lattice = np.array(cell["lattice"])
    positions = np.array([site["position"] for site in cell["sites"]])
    moments = np.array([site["moment"] for site in cell["sites"]])
    kinds = np.unique([site["kind"] for site in cell["sites"]], return_inverse=True)[1]  # a number for each kind

    # 1. Symmetries of the atoms alone
    # --------------------------------

    # Paramagnetic space group of the crystal from spglib
    G0 = spglib.get_symmetry_dataset((lattice, positions, kinds), symprec=tolerance)

    # 2. The spin space group, and G: its operations' spatial parts, counted up
    #    to the magnetic lattice's translations.
    # -------------------------------------------------------------------------

    # Get the (magnetic) primitive cell
    lattice, positions, kinds, moments = magnetic_primitive_cell(lattice, positions, kinds, moments, tolerance)
    # Extract the spin-space symmetries
    rotations, translations, spin_rotations = spin_operations(lattice, positions, kinds, moments, tolerance)
    # Parent space group formed by the spin-space symmetries with U discarded
    G = spglib.get_spacegroup_type_from_symmetry(rotations, translations, lattice=lattice, symprec=tolerance)

    # 3. Γ. The moments span a direction where they reach along it by more than
    #    the tolerance (root mean square).
    # -------------------------------------------------------------------------

    # Determine if the order is collinear, coplanar or noncoplanar
    directions, reach, _ = np.linalg.svd(moments.T)
    N = int(np.sum(reach > tolerance * np.sqrt(np.sum(moments.any(axis=1)))))
    if N == 0:
        raise ValueError("every moment is zero, to within the tolerance")

    # Determine the Γ matrices using the reduced axes 
    Q = directions[:, :N]
    gamma = np.array([Q.T @ U @ Q for U in spin_rotations])

    # 4. The characters of exterior powers of Γ, and what they share with the
    #    multipoles.
    # -----------------------------------------------------------------------

    # Compute the character of Γ
    trace = np.trace(gamma, axis1=1, axis2=2)
    # Compute the characters of the three exterior powers
    character = {
        # Original representation, Γ
        1: trace, 
        # Anti-symmetric square
        2: (trace**2 - np.trace(gamma @ gamma, axis1=1, axis2=2)) / 2, 
        # Determinant of Γ
        3: np.linalg.det(gamma) if N == 3 else 0 * trace,
    }
    # Note: conditional to zero out third power for collinear, coplanar

    # Overlap between exterior power and some multipole; only looking for *any*
    # nonzero overlap
    def shares(power, rank, parity):         
        return round(np.mean(character[power] * multipole(rotations, rank, parity))) > 0
    # Note: rotations is the full group, so the conjugacy class factors are
    # included automatically

    # Compute the required overlaps following the RESPONSES table
    allowed = [name for name, (power, multipoles) in RESPONSES.items()
               if any(shares(power, rank, parity) for rank, parity in multipoles)]
    # Compute the required overlaps for even/odd spin textures (up to rank 12)
    even = next((WAVES[rank] for rank in range(0, 13, 2) if shares(1, rank, +1)), "-") 
    odd = next((WAVES[rank] for rank in range(1, 13, 2) if shares(2, rank, -1)), "-")

    # Output: - Paramagnetic and parent space group names/numbers
    #         - Collinear, coplanar or noncoplanar
    #         - Allowed response list, lowest allowed even/odd textures
    #         - G's operations {R|t} with their Γ, and the cell they are written
    #           in: the magnetic lattice's primitive cell
    return dict(
        G0=f"{G0.international} ({G0.number})", G=f"{G.international_short} ({G.number})", order=ORDERS[N],
        allowed=allowed, texture=f"{even}/{odd}",
        rotations=rotations, translations=translations, gamma=gamma, lattice=lattice,
    )
