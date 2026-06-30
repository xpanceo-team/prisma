"""Implementation based on the template of ALIGNN."""
import numpy as np
from jarvis.core.specie import get_node_attributes

from collections import defaultdict
import torch
from torch_geometric.data import Data


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def E3Graph_r(
    atoms=None,
    cutoff=6.0,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)

    edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors_now):
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        for dst, image in zip(ids, images):
            src_id, dst_id, _, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))
    return edges


def build_undirected_edgedata_r(
    atoms=None,
    edges={},
    reduce=False,
    equivalent_atoms=None,
):
    u = []
    v = []
    r = []
    cell_offsets = []
    offsets = []
    pos = []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
            cell_offsets.append(np.array(dst_image))
            cell_offsets.append(-np.array(dst_image))
            offsets.append(atoms.lattice.cart_coords(dst_image))
            offsets.append(-atoms.lattice.cart_coords(dst_image))

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    cell_offsets = torch.tensor(cell_offsets)
    offsets = torch.tensor(offsets)
    pos = torch.tensor(atoms.lattice.cart_coords(atoms.frac_coords))
    return u, v, r, cell_offsets, offsets, pos


def atoms2graphs_etgnn(
    atoms=None,
    cutoff=6.0,
    use_canonize=True,
):
    edges = E3Graph_r(
        atoms=atoms,
        cutoff=cutoff,
        use_canonize=use_canonize,
    )
    u, v, r, cell_offsets, offsets, pos = build_undirected_edgedata_r(atoms, edges)
    sps_features = []
    for ii, s in enumerate(atoms.elements):
        feat = list(get_node_attributes(s, atom_features="atomic_number"))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features)
    edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
    g = Data(
        atomic_numbers=node_features,
        edge_index=edge_index,
        edge_attr=r,
        cell_offsets=cell_offsets,
        offsets=offsets,
        pos=pos
    )

    return g
