A package for analyzing TolTEC data
-----------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

TolTECA is a package for analyzing TolTEC data.


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

To run the simulator, see `the simulator tutorial<https://github.com/toltec-astro/tolteca/blob/master/docs/tolteca/toltec_simu_tutorial.md>`.


License
-------

This project is Copyright (c) Zhiyuan Ma and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.


Contributing
------------

We love contributions! TolTECA is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
TolTECA based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
