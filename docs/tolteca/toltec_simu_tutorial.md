# `tolteca.simu` Tutorial

## Install

The `tolteca.simu` is available after the installation of the `tolteca`
package.

No extra dependency is needed to run the simulator core functionalities.
However, to invoke the built-in animation (by setting `plot: true` in
the config), one needs to install the (fork of) python package `animatplot`
using `pip install git+https://github.com/Jerry-Ma/animatplot.git`.

## Usage

The command-line interface of `tolteca.simu` is integrated as a sub-command
of the `tolteca` command-line interface. The steps described here are also
applicable to running the `citlali` reduction using `tolteca`.


### Setup

One needs to setup a working directory before they can use the simulator to
generate data.

The working directory is a self-contained directory within which we maintain a
persistent context (referred to as the "runtime context") that only references
to itself, that is, the content of a work directory is reproducible using only
the information inside the directory.

To setup working directory,

```
$ mkdir test_simu && cd test_simu
$ tolteca setup
```

This will populate the `test_simu` folder with sub-dirs like `bin`, `cal`,
`log`, and most importantly, a YAML file named `50_setup.yaml` in the
directory.

All files match the pattern `[0-9]*.yaml` in the working directory are
recognized as configuration files, that will be loaded as part of the runtime
context for the actual simulator or pipeline reduction run at later times.

To actually setup a simulator, as need to create a config file that contains
the related settings. An example file `60_simu.yaml` can be:

```
# vim: et ts=2 sts=2 sw=2
---

_:
  # this demonstrates the use of yaml anchors to make the config
  # modularized
  example_mapping_tel_nc: &example_mapping_tel_nc
    type: lmt_tcs
    filepath: example_tel.nc
  example_mapping_model_raster: &example_mapping_model_raster
    type: tolteca.simu:SkyRasterScanModel
    rot: 30. deg
    length: 4. arcmin
    space: 5. arcsec
    n_scans: 24
    speed: 30. arcsec/s
    t_turnover: 5 s
    target: 180d 0d
    t0: 2020-04-12T00:00:00
    # lst0: ...

simu:
  # this is the actual simulator
  jobkey: example_simu
  plot: true
  instrument:
    name: toltec
    calobj: cal/calobj_default/index.yaml
    select: 'array_name == "a1100"'
  obs_params:
    f_smp: 12.2 Hz  # the sample frequency
    t_exp: 2 min    # the lenth of the exposure
  sources:
    - type: point_source_catalog
      filepath: inputs/example_input.asc
    # - type: image
    #   filepath: example_input.fits
    # - type: bkg
    #   value: 1 pW
  mapping: *example_mapping_model_raster
  # mapping: *example_mapping_tel_nc
```

The example above contains entries that refer to local files which
shall be made available prior to running the simulator.

* `calobj: cal/calobj_default/index.yaml`. This is the calibration object that
  is used to initialize the simulator. This file (and the related data files)
  can be downloaded from
  [here](https://github.com/toltec-astro/toltec_calib/tree/master/prod)
  (remember to rename it to `calobj_default` as stated in the config).

* Point source catalog `inputs/example_input.asc`. This is used as an example
  of simulation input. The content of this file could be like:

  ```
  # name ra dec flux_a1100 flux_a1400 flux_a2000
    src0 180. 0. 50.  40. 30.
    src1 180. 0.008333333333333333 5. 5. 5.
  ```
  where names like `a1100`, `a1400`, and `a2000` are the standardized code
  name for the TolTEC arrays.

### Run

To run the simulator, in the `test_simu` working directory, do

```
$ tolteca simu
```

This will execute the simulator as planned in the config files in the working
directory.

The output files will be in a sub-dir named as the `jobkey` defined in the
`60_simu.yaml` file.

To turn off the plotting, set `plot: false` in the config.
