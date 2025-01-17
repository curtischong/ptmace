import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Tuple
from pynanoflann import KDTree as NanoKDTree

# Optional: If you rely on matscipy for neighbor lists
import matscipy.neighbours

########################
# Helper Functions
########################


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
):
    """
    Mimics the original JAX-based get_neighborhood, but returns NumPy arrays.
    """
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or np.all(cell == 0.0):
        cell = np.identity(3, dtype=float)

    # Using matscipy neighbour_list
    receivers, senders, senders_unit_shifts = matscipy.neighbours.neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )
    return senders, receivers, senders_unit_shifts


def _message_passing(
    ext_senders_in_cell0,
    ext_receivers_in_cell0,
    ext_senders_unit_shifts_from_cell0,
    ext_receivers_unit_shifts_from_cell0,
    senders,
    receivers,
    senders_unit_shifts,
):
    """
    Same as the original _message_passing, but remains in NumPy.
    """
    x = np.unique(
        np.concatenate(
            [
                np.concatenate(
                    [
                        ext_senders_in_cell0[:, None],
                        ext_senders_unit_shifts_from_cell0,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        ext_receivers_in_cell0[:, None],
                        ext_receivers_unit_shifts_from_cell0,
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        ),
        axis=0,
    )
    node_index_in_cell0, node_shifts_from_cell0 = x[:, 0], x[:, 1:4]

    new_edges = []
    for i, us in zip(node_index_in_cell0, node_shifts_from_cell0):
        mask = receivers == i
        s = senders[mask]
        r = receivers[mask]
        s_us = us + senders_unit_shifts[mask]
        r_us = us + np.zeros_like(s_us)
        new_edges += [np.concatenate([s[:, None], r[:, None], s_us, r_us], axis=1)]

    new_edges = np.concatenate(new_edges, axis=0)
    new_edges = np.unique(new_edges, axis=0)

    ext_senders_in_cell0 = new_edges[:, 0]
    ext_receivers_in_cell0 = new_edges[:, 1]
    ext_senders_unit_shifts_from_cell0 = new_edges[:, 2:5]
    ext_receivers_unit_shifts_from_cell0 = new_edges[:, 5:8]

    return (
        ext_senders_in_cell0,
        ext_receivers_in_cell0,
        ext_senders_unit_shifts_from_cell0,
        ext_receivers_unit_shifts_from_cell0,
    )


def into_concrete_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    ext_senders_in_cell0: np.ndarray,
    ext_receivers_in_cell0: np.ndarray,
    ext_senders_unit_shifts_from_cell0: np.ndarray,
    ext_receivers_unit_shifts_from_cell0: np.ndarray,
):
    """
    Same as the original into_concrete_graph, but remains in NumPy.
    Returns the positions, index_in_cell0, shifts, senders, receivers.
    """
    x, i = np.unique(
        np.concatenate(
            [
                np.concatenate(
                    [
                        np.sum(
                            ext_senders_unit_shifts_from_cell0**2,
                            axis=1,
                            keepdims=True,
                        ),
                        ext_senders_unit_shifts_from_cell0,
                        ext_senders_in_cell0[:, None],  # index in 0th cell
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        np.sum(
                            ext_receivers_unit_shifts_from_cell0**2,
                            axis=1,
                            keepdims=True,
                        ),
                        ext_receivers_unit_shifts_from_cell0,
                        ext_receivers_in_cell0[:, None],
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        ),
        axis=0,
        return_inverse=True,
    )

    ext_node_unit_shifts_from_cell0, ext_node_index_in_cell0 = x[:, 1:4], x[:, 4]
    ext_node_positions = (
        positions[ext_node_index_in_cell0.astype(int)]
        + ext_node_unit_shifts_from_cell0 @ cell
    )

    # Map the edges onto these unique node indices
    ext_senders = i[: len(ext_senders_in_cell0)]
    ext_receivers = i[len(ext_senders_in_cell0) :]

    # Sort edges by (sender, receiver) if desired
    j = np.lexsort((ext_senders, ext_receivers))
    ext_senders = ext_senders[j]
    ext_receivers = ext_receivers[j]

    return (
        ext_node_positions,
        ext_node_index_in_cell0.astype(int),
        ext_node_unit_shifts_from_cell0,
        ext_senders,
        ext_receivers,
    )


def pad_periodic_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    senders_unit_shifts: np.ndarray,
    num_message_passing: int,
):
    """
    Same as the original pad_periodic_graph, returning the extended graph in NumPy arrays.
    """
    ext_senders_in_cell0 = senders
    ext_receivers_in_cell0 = receivers
    ext_senders_unit_shifts_from_cell0 = senders_unit_shifts
    ext_receivers_unit_shifts_from_cell0 = np.zeros_like(senders_unit_shifts)

    for _ in range(num_message_passing - 1):
        (
            ext_senders_in_cell0,
            ext_receivers_in_cell0,
            ext_senders_unit_shifts_from_cell0,
            ext_receivers_unit_shifts_from_cell0,
        ) = _message_passing(
            ext_senders_in_cell0,
            ext_receivers_in_cell0,
            ext_senders_unit_shifts_from_cell0,
            ext_receivers_unit_shifts_from_cell0,
            senders,
            receivers,
            senders_unit_shifts,
        )

    return into_concrete_graph(
        positions,
        cell,
        ext_senders_in_cell0,
        ext_receivers_in_cell0,
        ext_senders_unit_shifts_from_cell0,
        ext_receivers_unit_shifts_from_cell0,
    )


########################
# Main Conversion
########################


def atoms_to_ext_graph_torch(atoms, cutoff: float, num_message_passing: int) -> Data:
    """
    Converts an ase.Atoms object into a torch_geometric.data.Data object,
    mimicking the original JAX/jraph version.
    """

    # 1) Build the basic neighborhood list
    senders, receivers, senders_unit_shifts = get_neighborhood(
        positions=atoms.positions,
        cutoff=cutoff,
        pbc=atoms.pbc,
        cell=atoms.cell.array,
    )
    num_atoms = len(atoms)

    # 2) Gather global info
    kpt = (
        atoms.info["k_point"] if "k_point" in atoms.info else np.array([0.0, 0.0, 0.0])
    )
    dynmat = None
    atoms_hessian = None

    # 3) Construct either molecular or crystal case
    if "ifc" in atoms.info:
        # Molecular case
        atoms_ifc = atoms.info["ifc"]  # e.g. [i1, c1, i2, c2]
        atoms_hessian = atoms.info["ifc_hessian"]  # Hessian
        # One-hot in PyTorch
        idx1 = atoms_ifc[0] * 3 + atoms_ifc[1]
        idx2 = atoms_ifc[2] * 3 + atoms_ifc[3]
        u1 = (
            F.one_hot(torch.tensor([idx1]), 3 * num_atoms).reshape(num_atoms, 3).float()
        )
        u2 = (
            F.one_hot(torch.tensor([idx2]), 3 * num_atoms).reshape(num_atoms, 3).float()
        )
    else:
        # Periodic crystal case
        u1 = atoms.arrays["eigvec0_re"] if "eigvec0_re" in atoms.arrays else None
        u2 = atoms.arrays["eigvec1_re"] if "eigvec1_re" in atoms.arrays else None
        if "dynmatR" in atoms.info and "dynmatI" in atoms.info:
            dynmat = atoms.info["dynmatR"] + 1j * atoms.info["dynmatI"]

    # 4) Extend the graph periodically
    (
        ext_node_positions,
        ext_node_index_in_cell0,
        ext_node_unit_shifts_from_cell0,
        ext_senders,
        ext_receivers,
    ) = pad_periodic_graph(
        atoms.positions,
        atoms.cell.array,
        senders,
        receivers,
        senders_unit_shifts,
        num_message_passing,
    )

    # 5) Identify which extended nodes are in the primitive cell
    mask_primitive = np.all(
        ext_node_unit_shifts_from_cell0 == 0, axis=1
    )  # [n_ext_nodes]

    # 6) Build species
    ext_node_species = atoms.numbers[ext_node_index_in_cell0]  # [n_ext_nodes]

    # 7) Optionally build v1, v2. Here we define a placeholder contraction_vector
    def contraction_vector(r_np, idx_in_cell0, kpt_np, u_np):
        """
        Example contraction, returns an array of shape [num_nodes, 3]
        or something domain-specific. This is user-defined.
        """
        # For demonstration, let us pretend we do something trivial
        # and incorporate kpt phase, for example:
        phase = np.exp(2j * np.pi * r_np @ kpt_np)
        # u_np might be either a numpy or torch array; ensure correct type
        # Typically we'd do something more domain-specific here.
        if isinstance(u_np, torch.Tensor):
            u_np = u_np.numpy()
        # shape: [num_nodes, 3]
        out = (u_np[idx_in_cell0] * phase[:, None]).real  # or .imag, domain-specific
        return out

    # v1
    if (kpt is not None) and (u1 is not None):
        v1_np = contraction_vector(ext_node_positions, ext_node_index_in_cell0, kpt, u1)
        v1 = torch.from_numpy(v1_np).float()
    else:
        v1 = None

    # v2
    if (kpt is not None) and (u2 is not None):
        v2_np = contraction_vector(ext_node_positions, ext_node_index_in_cell0, kpt, u2)
        v2 = torch.from_numpy(v2_np).float()
    else:
        v2 = None

    # 8) Build a PyTorch Geometric Data object
    data = Data()

    # ----- Node attributes -----
    data.pos = torch.from_numpy(ext_node_positions).float()  # (n_ext_nodes, 3)
    data.species = torch.from_numpy(ext_node_species).long()  # (n_ext_nodes,)
    data.mask_primitive = torch.from_numpy(mask_primitive).bool()  # (n_ext_nodes,)

    # Optionally store these in the same data object
    data.index_cell0 = torch.from_numpy(ext_node_index_in_cell0).long()
    data.unit_shifts_from_cell0 = torch.from_numpy(
        ext_node_unit_shifts_from_cell0
    ).float()

    # v1, v2 may be None
    if v1 is not None:
        data.v1 = v1
    if v2 is not None:
        data.v2 = v2

    # ----- Edge attributes / edge_index -----
    # PyTorch Geometric expects shape = [2, num_edges]
    edge_index = np.stack([ext_senders, ext_receivers], axis=0)
    data.edge_index = torch.from_numpy(edge_index).long()

    # If you have actual edge features, store them in data.edge_attr = ...
    # But here we set them to None.

    # ----- Global attributes -----
    # You can store them as top-level attributes or in a dictionary
    if kpt is not None:
        data.kpt = torch.from_numpy(kpt).float().unsqueeze(0)  # shape = (1, 3)
    if dynmat is not None:
        # If dynmat is complex: you can separate real and imag or store as is
        # PyTorch does have complex dtypes, but handle with caution
        # e.g. data.dynmat = torch.view_as_real(torch.from_numpy(dynmat))
        data.dynmat = torch.from_numpy(np.stack([dynmat.real, dynmat.imag], axis=-1))
    if atoms_hessian is not None:
        # e.g. shape = (1, x, y) or however you want it
        # original code: hessian = np.array([atoms_hessian])[None,:]
        h = np.array([atoms_hessian])[None, :]
        data.hessian = torch.from_numpy(h).float()

    # Cell
    data.cell = (
        torch.from_numpy(atoms.cell.array).float().unsqueeze(0)
    )  # shape = (1, 3, 3)

    # Optionally track how many edges/nodes
    data.n_edge = ext_senders.shape[0]
    data.n_node = ext_node_positions.shape[0]

    # new features to help sevennet
    data.edge_src_cell0 = ext_node_index_in_cell0[ext_senders]
    data.edge_dst_cell0 = ext_node_index_in_cell0[ext_receivers]
    data.edge_vec = ext_node_positions[ext_receivers] - ext_node_positions[ext_senders]

    return data


supercell = []
for i in range(-1, 2):
    for j in range(-1, 2):
        for k in range(-1, 2):
            supercell.append((i, j, k))
OFFSETS = torch.tensor(supercell)  # {27,3)


def _compute_img_positions_torch(lattice, coords):
    lattice_offset = torch.einsum(
        "ijk,ij->ijk", lattice.expand(27, 3, 3), torch.tensor(OFFSETS)
    )  # Shape: (27, 3, 3)
    coord_offsets = lattice_offset.sum(
        dim=1
    )  # Shape: (27, 3) This is called the coord_offsets because it's how much each coord should move when placed into the corresponding supercell

    # the below line is the core idea: for each of the coords, you want to add the lattice offset
    # coords is (n, 3).
    # coords.unsqueeze(1) => (n, 1, 3)
    # coord_offsets.unsqueeze(0) => (1, 27, 3)
    # broadcast add => (n, 27, 3)
    res = coords.unsqueeze(1) + coord_offsets.unsqueeze(0)
    return res  # shape (n, 27, 3)


def brute_force_knn(supercell_coords, repeated_coords, k):
    dist = torch.cdist(repeated_coords, supercell_coords)
    return torch.topk(dist, k, largest=False, sorted=True)


NUM_OFFSETS = len(OFFSETS)


def compute_pbc_radius_graph(
    *,
    lattice: torch.Tensor,
    cart_coord: torch.Tensor,
    radius: int = 5,
    max_number_neighbors: int,
    n_workers: int = 1,
):
    cart_supercell_coords = _compute_img_positions_torch(lattice, cart_coord)
    cart_supercell_coords = cart_supercell_coords.transpose(0, 1)
    cart_supercell_coords = cart_supercell_coords.reshape(-1, 3)

    num_positions = len(cart_coord)
    node_id = (
        torch.arange(num_positions)
        .unsqueeze(-1)
        .expand(num_positions, NUM_OFFSETS)
        .transpose(0, 1)
    )

    return masked_positions_to_graph(
        cart_supercell_coords.reshape(-1, 3),
        positions=cart_coord,
        node_id2=node_id,
        radius=radius,
        max_number_neighbors=max_number_neighbors,
        n_workers=n_workers,
    )


def masked_positions_to_graph(
    supercell_positions, positions, node_id2, radius, max_number_neighbors, n_workers
):
    tree_data = supercell_positions.clone().detach().cpu().numpy()
    tree_query = positions.clone().detach().cpu().numpy()
    num_positions = positions.shape[0]

    tree = NanoKDTree(
        n_neighbors=min(max_number_neighbors + 1, len(supercell_positions)),
        radius=radius,
        leaf_size=100,
        metric="l2",
    )
    tree.fit(tree_data)
    distance_values, nearest_img_neighbors = tree.kneighbors(
        tree_query, n_jobs=n_workers
    )
    nearest_img_neighbors = nearest_img_neighbors.astype(np.int32)  # type: ignore

    # remove the self node which will be closest
    index_array = nearest_img_neighbors[:, 1:]
    # remove distances greater than radius
    within_radius = distance_values[:, 1:] < (radius + 1e-6)
    receivers_imgs = index_array[within_radius]
    num_neighbors_per_position = within_radius.sum(-1)

    # We construct our senders and receiver indexes.
    senders = np.repeat(np.arange(num_positions), list(num_neighbors_per_position))  # type: ignore
    receivers_img_torch = torch.tensor(receivers_imgs, device=positions.device)
    # Map back to indexes on the central image.
    # receivers = receivers_img_torch % num_positions # this no longer works since we pruned the supercell
    # receivers = node_id2[receivers_img_torch]
    receivers = receivers_img_torch % num_positions
    senders_torch = torch.tensor(senders, device=positions.device)

    # Finally compute the vector displacements between senders and receivers.
    vectors = supercell_positions[receivers_img_torch] - positions[senders_torch]
    return torch.stack((senders_torch, receivers), dim=0), vectors
