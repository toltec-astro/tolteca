A package for analyzing TolTEC data
-----------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

TolTECA is a package for analyzing TolTEC data.

Following the development HEAD
------------------------------

TolTECA is in its alpha version and some of the APIs are not stable. Please see

https://github.com/toltec-astro/tolteca/wiki/Following-the-Development-HEAD

for more info.

Install
-------

The package depends on `tollan` and `kidsproc`, which has to be installed
separately before installing `tolteca` (the package `dasha` is also needed
to run `tolteca.web`).

The follows will install the three packages as development version. This will
allow you make local modification of the package and contribute bug fixes
easily.

.. code-block:: bash

    $ git clone https://github.com/toltec-astro/tollan
    $ git clone https://github.com/toltec-astro/kidsproc
    $ git clone https://github.com/toltec-astro/tolteca
    $ cd tollan
    $ pip install -e .
    $ cd ../kidsproc
    $ pip install -e .
    $ cd ../tolteca
    $ pip install -e .

Note that :code:`pip install` by default will install the packages to `/usr/local` which may
not be writable for users without using `sudo` (which is not recommended). In this case,
use :code:`pip install --user -e .` instead to install the packages.

The packages `tollan`, `kidsproc`, and `tolteca` are under active development.
It is recommended that you frequently do `git pull` in the repositories to
update all of them.

Alternatively, one can install them without cloning the git repositories:

.. code-block:: bash

    pip install git+https://github.com/toltec-astro/tollan
    pip install git+https://github.com/toltec-astro/kidsproc
    pip install git+https://github.com/toltec-astro/tolteca


It is recommended to use a virtualenv manager (venv, pyenv, etc.) to install
`tolteca` into its own virtualenv, which makes things super easy for
uninstall or upgrade.


Usage
-----

The `tolteca` package comes with a command-line interface, that can be used
to run sub-package specific tasks including those defined as part of
the `tolteca.simu` and `tolteca.reduce`.

To run the simulator and reduction pipeline via commandline interface, see
instructions here:
`the simulator tutorial<https://github.com/toltec-astro/tolteca/blob/main/docs/tolteca/toltec_simu_tutorial.md>`.

To see the Jupyter notebook tutorials, visit: https://toltec-astro.github.io/tolteca_tutorials/

Please see the `API documentation
<https://toltec-astro.github.io/tolteca>`_ for details.

License
-------

This project is Copyright (c) Zhiyuan Ma and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.
