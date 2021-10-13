[![GitHub release](https://img.shields.io/github/v/tag/jGaboardi/tigernet?include_prereleases&logo=GitHub)](https://img.shields.io/github/v/tag/jGaboardi/tigernet?include_prereleases&logo=GitHub)  [![PyPI version](https://badge.fury.io/py/tigernet.svg)](https://badge.fury.io/py/tigernet) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/tigernet.svg)](https://anaconda.org/conda-forge/tigernet) [![Conda Recipe](https://img.shields.io/badge/recipe-tigernet-yellow.svg)](https://github.com/conda-forge/tigernet-feedstock)


# TigerNet
Network Topology via TIGER/Line Edges

[![unittests](https://github.com/jGaboardi/tigernet/workflows/.github/workflows/unittests.yml/badge.svg)](https://github.com/jGaboardi/tigernet/actions?query=workflow%3A.github%2Fworkflows%2Funittests.yml) [![codecov](https://codecov.io/gh/jGaboardi/tigernet/branch/main/graph/badge.svg)](https://codecov.io/gh/jGaboardi/tigernet) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## What is TigerNet and how does it work?

TigerNet is an open-source Python library that addresses concerns in topology and builds accurate spatial network representations from [TIGER/Line data](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html), specifically [TIGER/Line edges](https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2017/TGRSHP2017_TechDoc_Ch4.pdf). This is achieved through a 7-step process that roughly is as follows:

1. creation of initial TIGER/Line edges subset (features with a road-type [MTFCC](https://www.census.gov/library/reference/code-lists/mt-feature-class-codes.html))
2. creation of initial segments subset (retain only specified road-type MTFCCs)
3. welding of limited-access segments (limited-access segments — freeways, etc. — that share a non-articulation point are isolated and welded together)
4. welding of general segments (surface street segments that share a non-articulation point are isolated and welded together)
5. splitting of general segments (surface street segments that cross at known intersections are split)
6. cleansing of the segment data (steps 4 and 5 are repeated until the data is deemed "clean" enough for network instantiation)
7. building of the network (creation of network topology with the option of further simplification to eliminate all remaining non-articulation points — a pseudo graph-theoretic object — while maintaining spatial accuracy)

### Important
After some consideration, this repo will serve as a stub for the `tigernet` implementation developed for Gaboardi (2019), which can be cited in future publications through its [DOI](https://zenodo.org/record/3378057#.Xh5oli3MzVo). Currently, some of the concepts are already being incorporated into [`spaghetti`](https://github.com/pysal/spaghetti), with more of the functionality in the original `tigernet` potential (such as network measures [`pysal/spaghetti#126`](https://github.com/pysal/spaghetti/issues/126)).

* **Gaboardi, James D.** (2019). *Populated Polygons to Networks: A Population-Centric Approach to Spatial Network Allocation*. [ProQuest Dissertations Publishing]([https://search.proquest.com/openview/e928368d7bb867bbf067fcad62011de3/1?pq-origsite=gscholar&cbl=18750&diss=y).

## Examples
* Demo: [Synthetic lattice and observations](https://github.com/jGaboardi/tigernet/blob/main/examples/synthetic_network_example.ipynb)
* Applied: [Empirical susbset of roads and observations](https://github.com/jGaboardi/tigernet/blob/main/examples/empirical_network_example.ipynb)

## Installation

![Pypi python versions](https://img.shields.io/pypi/pyversions/tigernet.svg) Currently `tigernet` officially supports [3.8](https://docs.python.org/3.8/) and [3.9](https://docs.python.org/3.9/).

Install the current release from [`PyPI`](https://pypi.org/project/tigernet/) by running:

```
$ pip install tigernet
```

Install the most current development version of `tigernet` by running:

```
$ pip install git+https://github.com/jGaboardi/tigernet
```

## Support

If you are having issues, please [create an issue](https://github.com/jGaboardi/tigernet/issues).

## License

The project is licensed under the [BSD 3-Clause license](https://github.com/jGaboardi/tigernet/blob/main/LICENSE.txt).

## Citations

* **James D. Gaboardi** (2019). *[jGaboardi/tigernet](https://github.com/jGaboardi/tigernet)*. Zenodo. [![DOI](https://zenodo.org/badge/204572461.svg)](https://zenodo.org/badge/latestdoi/204572461)

```tex
@misc{tigernet_gaboardi_2019,
  author  = {James David Gaboardi},
  title   = {jGaboardi/tigernet},
  month   = {aug},
  year    = {2019},
  doi     = {10.5281/zenodo.204572461},
  url     = {https://github.com/jGaboardi/tigernet}
}
```

## Related projects
* [`osmnx`](https://osmnx.readthedocs.io/en/stable/)
* [`pandana`](http://udst.github.io/pandana/)
* [`pyrosm`](https://github.com/HTenkanen/pyrosm)
* [`sanet`](http://sanet.csis.u-tokyo.ac.jp)
* [`snkit`](https://github.com/tomalrussell/snkit)
* [`spaghetti`](https://github.com/pysal/spaghetti)
* [`momepy`](https://github.com/pysal/momepy)


## References
* The original method for `tigernet` is described in Chapter 1 of Gaboardi (2019).
  * **James D. Gaboardi** (2019). *Populated Polygons to Networks: A Population-Centric Approach to Spatial Network Allocation*. [ProQuest Dissertations Publishing]([https://search.proquest.com/openview/e928368d7bb867bbf067fcad62011de3/1?pq-origsite=gscholar&cbl=18750&diss=y).
* The results of secondary analysis (spatial representions of population) were presented in Gaboardi (2020) and can also be found in Chapter 3 of Gaboardi (2019).
  * **James D. Gaboardi** (2020, November). *Validation of Abstract Population Representations*. Presented at the 2019 Atlanta Research Data Center Annual Research Conference at Vanderbilt University (ARDC), Nashville, Tennessee: Zenodo. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4287456.svg)](https://doi.org/10.5281/zenodo.4287456)
* The [`WeightedParcels_Leon_FL_2010`](https://github.com/jGaboardi/tigernet/blob/main/test_data/) dataset is based on that used in Gaboardi (2019), which was produced in Strode et al. (2018).
  * **Georgianna Strode, Victor Mesev, and Juliana Maantay** (2018). Improving Dasymetric Population Estimates for Land Parcels: Data Pre-processing Steps. Southeastern Geographer 58 (3), 300–316. doi: [10.1353/sgo.2018.0030](https://muse.jhu.edu/article/705475).
