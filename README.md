# TigerNet
Network Topology via TIGER/Line Shapefiles

[![GitHub release](https://img.shields.io/github/v/tag/jGaboardi/tigernet?include_prereleases&logo=GitHub)](https://img.shields.io/github/v/tag/jGaboardi/tigernet?include_prereleases&logo=GitHub) 
[![unittests](https://github.com/jGaboardi/tigernet/workflows/.github/workflows/unittests.yml/badge.svg)](https://github.com/jGaboardi/tigernet/actions?query=workflow%3A.github%2Fworkflows%2Funittests.yml) [![codecov](https://codecov.io/gh/jGaboardi/tigernet/branch/master/graph/badge.svg)](https://codecov.io/gh/jGaboardi/tigernet) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

-----------------

### Important
After some consideration, this repo will serve as a stub for the `tigernet` implementation developed for Gaboardi (2019), which can be cited in future publications through its [DOI](https://zenodo.org/record/3378057#.Xh5oli3MzVo). Currently, some of the concepts are already being incorporated into [`spaghetti`](https://github.com/pysal/spaghetti), with more of the functionality in the original `tigernet` potential (such as network measures [`pysal/spaghetti#126`](https://github.com/pysal/spaghetti/issues/126)).

* **Gaboardi, James D**. 2019. *Populated Polygons to Networks: A Population-Centric Approach to Spatial Network Allocation*. [ProQuest Dissertations Publishing]([https://search.proquest.com/openview/e928368d7bb867bbf067fcad62011de3/1?pq-origsite=gscholar&cbl=18750&diss=y).

-------------

## What is TigerNet and how does it work?

TigerNet is a Python library that addresses concerns in topology and builds accurate spatial network representations from [TIGER/Line® shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html), specifically [TIGER/Line edges](https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2017/TGRSHP2017_TechDoc_Ch4.pdf). This is achieved through a 7-step process that roughly is as follows:

1. creation of initial TIGER/Line edges subset (features with a road-type [MTFCC](https://www.census.gov/library/reference/code-lists/mt-feature-class-codes.html))
2. creation of initial segments subset (retain only specified road-type MTFCCs)
3. welding of limited-access segments (limited-access segments — freeways, etc. — that share a non-articulation point are isolated and welded together)
4. welding of general segments (surface street segments that share a non-articulation point are isolated and welded together)
5. splitting of general segments (surface street segments that cross at known intersections are split)
6. cleansing of the segment data (steps 4 and 5 are repeated until the data is deemed "clean" enough for network instantiation)
7. building of the network (creation of network topology with the option of further simplification to eliminate all remaining non-articulation points — a pseudo graph-theoretic object — while maintaining spatial accuracy)

## Installation

Currently `tigernet` officially supports Python [3.6](https://docs.python.org/3.6/), [3.7](https://docs.python.org/3.7/), and [3.8](https://docs.python.org/3.8/). Please make sure that you are operating in a Python >= 3.6 environment. Install the most current development version of `tigernet` by running:

```
$ pip install git+https://github.com/jGaboardi/tigernet
```

## Support

If you are having issues, please [create an issue](https://github.com/jGaboardi/tigernet/issues).

## License

The project is licensed under the [BSD 3-Clause license](https://github.com/jGaboardi/tigernet/blob/master/LICENSE.txt).

## Citations

* **James D. Gaboardi**. *[jGaboardi/tigernet](https://github.com/jGaboardi/tigernet)*. Zenodo. 2019. [![DOI](https://zenodo.org/badge/204572461.svg)](https://zenodo.org/badge/latestdoi/204572461)

```tex
@misc{tigernet_gaboardi_2019,
  author       = {James David Gaboardi},
  title        = {jGaboardi/tigernet},
  month        = {aug},
  year         = {2019},
  doi          = {10.5281/zenodo.204572461},
  url          = {https://github.com/jGaboardi/tigernet}
  }
```
