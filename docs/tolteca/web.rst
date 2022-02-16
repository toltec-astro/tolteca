.. _tolteca_web:

*****************************************
Web-based Tools (`tolteca.web`)
*****************************************

Introduction
============

The `~tolteca.web` package provides a suite of web tools (refer to as ``apps``)
covering a wide range of use cases.

Web apps in this package are implemented using the
`DashA <https://github.com/toltec-astro/dasha.git>`_ framework.

The `~tolteca.web` package provides class `~tolteca.web.WebRuntime` to manage
the app configurations and the context at runtime.

The web apps can be launched with the commandline interface ``tolteca web``.

.. _tolteca_web_getting_started:

Getting Started
===============

Additional Dependencies
-----------------------

The web tools requires additional dependencies to run, which can be installed
using

.. code-block:: sh

    $ python -m pip install tolteca[web]

Individual app may require further packages/environments to run. Please
make sure to check the detailed documentations of the :ref:`_tolteca_web_apps`.

Commandline Interface
---------------------

The primary way of running the web tools is to use the commandline interface
``tolteca web``

.. code-block:: console

    $ tolteca web -h
    usage: tolteca web [-h] -a APP [--ext_proc EXT]

    optional arguments:
      -h, --help         Show help for this subcommand and exit.
      -a APP, --app APP  The name of app to run.
      --ext_proc EXT     The extension process to run. See DashA
                         doc for details.

The ``tolteca web`` CLI follows a similar semantics as other CLI subcommand
such as ``tolteca simu`` or ``tolteca reduce``, where user can customize the
behavior through the use of stand-alone YAML config files, YAML config files
installed in a :ref:`_tolteca_workdir`, environment variables (from current
shell or :ref:`_tolteca_cli_env_files`), additional commandline arguments, or a
combination of above all.

As an example, to run the ``obs_planner`` tool

.. code-block:: console

    $ tolteca web -a obs_planner

The option ``--ext_proc`` is to specify the extension process to run. It is
often the case that an app consists of components that run in different
processes. For example, a Flask server process for processing HTTP request and
a task scheduling process for distributing the computation. When omitted, the
Flask server process is run.

The app configs can be specified in a variety of ways

.. code-block:: console

    $ tolteca -c web_conf.yaml -e obs_planner_conf.env -- \
      web -a obs_planner --app.0.tilte_text="My App"

Here ``web_conf.yaml`` may contain the config dict that resembles a
`~tolteca.web.WebConfig`

.. code-block:: yaml

    # web_conf.yaml
    web:
      apps:
      - name: obs_planner
        title_text: "Obs Planner"

And this config dict is overwritten by the commandline option
``app.0.title_text`` such that ``title_text`` becomes ``My App``.


Finally, the ``web_conf.env`` may contain a list of environment variables
to load

.. code-block:: sh

    # web_conf.env
    TOLTECA_WEB_OBS_PLANNER_TITLE_TEXT=My Obs Planner

The package uses the order of precedence as ``env_file > CLI > config_file``.
As a result, the final title text appears on the web page will be ``My Obs
Planner``.


WebRuntime
----------

Under the hood, the run of the web apps is managed by the
`~tolteca.web.WebRuntime` class.

An `~tolteca.web.WebRuntime` object initializes and manages a web config
instance `~tolteca.web.WebConfig`. A typical web config dict look like

.. code-block:: yaml

    # web_conf.yaml
    web:
      apps:
      - name: app1
      - option1: value1
      - name: app2
      - option2: value2

Each entry in the ``web.apps`` list specify the config dict of an app,
identified by the ``web.apps.<i>.name`` key.

The available apps are defined in the submodule `~tolteca.web.apps`.

Each of the app module (e.g., `~tolteca.web.apps.obs_planner`) defines
a ``DASHA_SITE`` entry point that is consumed by the
`DashA <https://github.com/toltec-astro/dasha.git>`_ package.

Call to `~tolteca.web.WebRuntime.run` will be passed to the DashA
package to start the designated extension process, which by default is
to run the Flask server in local development mode.


WebConfig
---------

The `~tolteca.web.WebConfig` handles all web related configuration options.
All app config classes are registered to the `~tolteca.web.apps_registry`.

See :ref:`_tolteca_web_config_schema`` for a table of the schema.


Run Apps in Production Mode
---------------------------

The ``tolteca web`` CLI runs the Flask server locally and in development mode.

To serve the web app for production use, one needs a full HTTP server and
gunicorn.

.. todo::

    Add details.
