"""Calculate network analysis statistics.
"""

from . import utils
import numpy
from scipy.spatial import distance_matrix


def calc_sinuosity(net):
    """Calculate segment sinuosity and network-level descriptive statistics.
    See tigernet.Network.calc_net_stats() for top-level method.

    Parameters
    ----------
    net : tigernet.Network

    """

    # Calculate absolute shortest path along segments
    # Euclidean distance from vertex1 to vertex2
    net.s_data = utils.euc_calc(net, col="euclid")

    # Calculate sinuosity for segments
    # curvilinear length / Euclidean Distance
    net.s_data["sinuosity"] = net.s_data[net.len_col] / net.s_data["euclid"]

    # network sinuosity
    # set loop sinuosity (inf) to max nonloop sinuosity in dataset
    sinuosity = net.s_data["sinuosity"].copy()
    max_sin = sinuosity[sinuosity != numpy.inf].max()
    sinuosity = sinuosity.replace(to_replace=numpy.inf, value=max_sin)
    net.max_sinuosity = max_sin
    net.min_sinuosity = sinuosity.min()
    net.mean_sinuosity = sinuosity.mean()
    net.std_sinuosity = sinuosity.std()


def set_node_degree(net):
    """Set descriptive node degree statistics as attributes.
    See tigernet.Network.calc_net_stats() for top-level method.

    Parameters
    ----------
    net : tigernet.Network

    """

    # node degree stats
    net.max_node_degree = net.n_data["degree"].max()
    net.min_node_degree = net.n_data["degree"].min()
    net.mean_node_degree = net.n_data["degree"].mean()
    net.std_node_degree = net.n_data["degree"].std()


def connectivity(net, measure="alpha"):
    """Connectivity indices.

    Parameters
    ----------
    net : tigernet.Network
    measure : str
        Statistic to calculate.

    Returns
    -------
    con : float
        Connectivity measure in desired.

    Notes
    -----
    Levinson D. (2012) Network Structure and City Size.
                PLoS ONE 7(1): e29721. doi:10.1371/journal.pone.0029721

    * alpha
    The alpha index is the ratio of the actual number of circuits
    in a network to the maximum possible number of circuits on that
    planar network. Values of a range from 0 percent - no circuits -
    to 100 percent - a completely interconnected network.

        :math:`alpha = e - v + p / 2*v - 5`

    * beta
    The beta index measures the connectivity relating the number of
    edges to the number of nodes. The greater the value of beta,
    the greater the connectivity.

        :math:`beta = e / v`

    * gamma
    The gamma index measures the connectivity in a network. It is a
    measure of the ratio of the number of edges in a network to the
    maximum number possible in a planar network. Gamma ranges from 0
    (no connections between nodes) to 1.0 (the maximum number of
    connections, with direct links between all the nodes).

        :math:`gamma = e / 3(v-2)`

    * eta
    The eta index measures the length of the graph over
    the number of edges.

        :math:`eta = L(G) / e`

    e = number of edges (links)
    v = number of vertices (nodes)
    p = number of graphs or subgraphs, and for a network where every
        place is connected to the network p = 1
    L(G) = total length of the graph

    """

    e = float(net.n_segm)
    v = float(net.n_node)
    p = float(net.n_ccs)
    L = net.network_length

    if measure == "alpha":
        con = (e - v + p) / ((2 * v) - 5)

    if measure == "beta":
        con = e / v

    if measure == "gamma":
        # number of edges in a maximally connected planar network
        e_max = 3 * (v - 2)
        con = e / e_max

    if measure == "eta":
        con = L / e

    return con


def entropies(ent_col, frame, n_elems):
    """Entropy.

    Levinson D. (2012) Network Structure and City Size.
                PLoS ONE 7(1): e29721. doi:10.1371/journal.pone.0029721

    Entropy measure of heterogeneity:

    :math:`H = -sum([pi np.log2 * (pi) for pi in subset_porportions])`

    Parameters
    ----------

    ent_col : str
        Column on with to calculate entropy.
    frame : geopandas.GeoDataFrame
        Full network dataframe.
    n_elems : int
        Number of dataframe records.

    Returns
    -------
    indiv_entropies : dict
        Individual entropies of MTFCC categories.

    """

    indiv_entropies = {}

    for value in frame[ent_col].unique():
        subset = frame[frame[ent_col] == value].shape[0]
        subset_proportion = float(subset) / float(n_elems)
        entropy = subset_proportion * numpy.log2(subset_proportion)
        indiv_entropies[value] = entropy

    return indiv_entropies


def dist_metric(mtx, stat):
    """Get the diameter or radius of a network with associated nodes.

    Parameters
    ----------
    mtx : numpy.ndarray
        Cost matrix.
    stat : str
        ``'min'`` or ``'max'``. Default is ``'max'``.

    Returns
    -------
    idx_val : list
        Metric in the form ``[idx,value]``.

    """

    if stat == "max":
        value = mtx.max()
    else:
        value = mtx[mtx != 0.0].min()

    idx = tuple(numpy.where(mtx == value))

    if len(idx[0]) > 1:
        idx = tuple(idx[0])
    else:
        idx = tuple([idx[0][0], idx[1][0]])

    idx_val = [idx, value]

    return idx_val


def circuity(net):
    """Calculate network circuity.

    Parameters
    ----------
    net : tigernet.Network
    mtx : numpy.ndarray
        Cost matrix.

    """

    # all network shortest paths
    net.d_net = net.n2n_matrix.sum()
    coords = [v[0] for (k, v) in net.node2coords.items()]
    n2n_euclidean = distance_matrix(coords, coords)

    # all euclidean shortest paths
    net.d_euc = n2n_euclidean.sum()
    net.circuity = net.d_net / net.d_euc
