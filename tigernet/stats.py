"""Calculate network analysis statistics.
"""

from . import utils
import numpy


def _calc_sinuosity(net):
    """Calculate segment sinuosity and network-level descriptive statistics.
    See tigernet.TigerNet.calc_net_stats() for top-level method.

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


def _set_node_degree(net):
    """Set descriptive node degree statistics as attributes.
    See tigernet.TigerNet.calc_net_stats() for top-level method.

    """

    # node degree stats
    net.max_node_degree = net.n_data["degree"].max()
    net.min_node_degree = net.n_data["degree"].min()
    net.mean_node_degree = net.n_data["degree"].mean()
    net.std_node_degree = net.n_data["degree"].std()
