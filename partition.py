# Inspired by luis's blog post (http://louis-needless.hatenablog.com/entry/rewrite_opencv_util_partition_in_python)

import numpy as np
from itertools import combinations
from unionfind.unionfind import UnionFind

def partition(src_len, equal):
    # equal: ndarray (src_len**2 / 2,) 2つの要素が等しいかどうか
    src_list = list(range(src_len))
    uf = UnionFind(src_list)

    indice = list(combinations(src_list, 2))
    for eq_idx, idx in enumerate(indice):
        # print(idx)
        if idx[0] == idx[1] or not equal[eq_idx]:
            continue
        uf.union(idx[0], idx[1])

    roots = [uf.find(item) for item in src_list]
    return np.array(_renumber(roots))

def _renumber(id_list):
    id_set = set(id_list)
    replace_dict = dict(zip(
        sorted(list(id_set)),
        list(range(len(id_set)))
    ))
    return [replace_dict[elem] for elem in id_list]
