#!/usr/bin/env python3

"""Connected-components clustering over radar detections."""

import numpy as np
from scipy.spatial import cKDTree


def cluster_points_kdtree(positions, tolerance, min_points, max_points):
    """Connected-components clustering over a distance graph of radius `tolerance`.

    A lightweight DBSCAN equivalent built on scipy's cKDTree, so sklearn is
    not needed. Returns a list of index arrays, one per cluster, keeping only
    clusters with between min_points and max_points members.
    """
    n = positions.shape[0]
    if n == 0:
        return []

    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=tolerance)

    # Union-find over point indices.
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in pairs:
        union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return [np.array(idxs) for idxs in groups.values()
            if min_points <= len(idxs) <= max_points]
