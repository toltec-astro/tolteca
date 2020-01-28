
TolTECA workflow
================


- Pipeline Setup

.. code-block:: bash

    $ tolteca setup DIR


This will setup `DIR` as the working directory for running TolTECA.

.. code-block:: bash

    $ tolteca setup test_workspace
    $ ls test_workspace
    bin/  cal/  config_base.yaml  log/  idx/

* `bin/`: The directory that contains external executables, in particular,
  the `citlali` pipeline executables.

* `cal/`: The directory that contains calibration objects. The naming
  conventions is `{type}_{ver}.{ext}`, where `{type}` could be `beammap`,
  `extinction`, or the packed form `calobj` (TBD).

* `log/`: The directory where log files live.

* `idx/`: The directory where databases for file index live.

* (TBA) more directories as necessary

* `config_base.yaml`: The file that contains info on the state
  of TolTECA and this working directory. An example:

    .. code-block:: bash

        # generated on xxxx-xx-xx xx:xx:xx
        namespace: null
        system_info:
            hostname: clipy

        manager:
            bin: /path/to/tolteca/bin
            version: 0.0.1

        pipeline_runtime:
            rootpath: .
            logdir: ./log
            bindir: ./bin
            caldir: ./cal
            idxdir: ./idx
            configfile:
                - config_base.yaml

        pipeline:
            bin: /bin/citlali
            version: 0.0.1

- Pipeline Prepare

Within an already setup work directory,

.. code-block:: bash

    $ tolteca prepare [-i INPUT_CONFIG] [--key0 value0 [--key1 value1 ...]] JOBDIR

This will create and setup subdirectory `JOBDIR` using information
provided in `INPUT_CONFIG`, optionally overwritten by `--key value` pairs.

There are several ways to specify what input files to use in `INPUT_CONFIG`,
among which the most fundamental one is to use an SQL query to the `idx`
database, e.g.,

.. code-block:: bash

    # job0_in.yaml

    source: idx
    sql_query: "select * from toltecdb.toltec limit 10"

This will get 10 entries from table `toltecdb.toltec`.

The created `JOBDIR` contains the follows:

.. code-block:: bash

        $ tolteca prepare -i job0_in.yaml job0
        $ ls -l job0/
        00_base.yaml*
        30_citlali.yaml
        60_inputs.yaml
        _raw/
        _cal/

* `??_*.yaml`: The configurations that will be picked up by the pipeline.

* `00_base.yaml`: This is a symbolic link to ../config_base.yaml

* `30_citlali.yaml`: This is a dump of the default pipeline program config.
  Under the hood, it runs the follows

  .. code-block:: bash

        $ pipeline_bin --dump --all > config

  where `pipeline_bin` is `pipeline.bin` in `00_base.yaml`, i.e., the
  `citlali` executable. An example file:

  .. code-block:: bash

    namespace: pipeline.config
    io:
        time_chunking:
            enabled: yes
            method:
                value: "fixed"
                choises: ["hold_signal", "fixed_length"]
            parameters:
               hold_signal:
                    value: 1
               fixed_length:
                    value: 10
                    unit: second
    reduce:
        map_making:
            coordsys:
                value: "equatorial"
                choises: ["equatorial", "horizontal", 'galactic']

* `60_inputs.yaml`: This is a file that include all found data files
  to be reduced, organized as follows:

  .. code-block:: bash

    namespace: pipeline.config.io
    inputs:
      - name: obs1
        data_items:
          - interface: toltec0
            filepath: "_raw/toltec0_obs1.nc"
          - interface: toltec1
            filepath: "_raw/toltec1_obs1.nc"
          - interface: toltec2
            filepath: "_raw/toltec2_obs1.nc"
          - interface: lmt 
            filepath: "_raw/lmt_obs1.nc"
          - interface: hwp 
            filepath: "_raw/hwp_obs1.nc"
        cal_items:
          - name: beammap
            filepath: "_cal/beammap_obs1.nc"
          - name: extinction
            filepath: "_cal/extinction_obs1.nc"
      - name: obs2
        data_items:
          - interface: toltec0
            filepath: "_raw/toltec0_obs2.nc"
          - interface: toltec1
            filepath: "_raw/toltec1_obs2.nc"
          - interface: toltec2
            filepath: "_raw/toltec2_obs2.nc"
          - interface: lmt 
            filepath: "_raw/lmt_obs2.nc"
          - interface: hwp 
            filepath: "_raw/hwp_obs2.nc"
        cal_items:
          - name: beammap
            filepath: "_cal/beammap_obs2.nc"
          - name: extinction
            filepath: "_cal/extinction_obs2.nc"

* `_raw/`: The directory that contains symbolic links to all data files,
  collected from querying the `idx` database, using information provided
  in `INPUT_CONFIG`.

* `_cal/`: The directory that contains *interpreted* calibration objects,
  from using the files in `../cal`.

- Pipeline execution

The pipeline can be executed by

.. code-block:: bash

    $ tolteca run [OPTIONS] JOBDIR

This will invoke the pipeline. A unique suffix is generated, which is used
to create a output directory `JOBDIR.suffix`. A unified config file `jobkey_somerandomhash.yaml`
is produced out of all `??_*.yaml` files and put into it as well. Finally,
the pipeline is run by effectively:

.. code-block:: bash

    $ pipeline_bin -c JOBDIR.suffix/JOBDIR_suffix.yaml -o JOBDIR.suffix
