# `tolteca.simu` Tutorial

## Install

The `tolteca.simu` is available after the installation of the `tolteca`
package.  The required toltec-produced packages are:

 - tolteca (this package)
 - tollan - https://github.com/toltec-astro/tollan
 - kidsproc - https://github.com/toltec-astro/kidsproc


Additionally, one may clone the toltec_calib repo to use the calibration
objects:

 - toltec_calib - https://github.com/toltec-astro/toltec_calib

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
the related settings. An example file `60_simu.yaml` can be found here:

https://github.com/toltec-astro/tolteca/tree/main/tolteca/data/examples


To use the source catalog model, you'll need to create catalogs like this
and specify the path in the YAML config file:

```
# name ra dec flux_a1100 flux_a1400 flux_a2000
  src0 180. 0. 50.  40. 30.
  src1 180. 0.008333333333333333 5. 5. 5.
```

You can specify `calobj` as the index file of data products in the `toltec_calib`
repo, or leave it blank. If now found, the built-in calibration object is used.

```
$ cd ../cal
$ mkdir calobj_default
$ cp -r <path to toltec_calib>/prod/* calobj_default/.
```

### Run

To run the simulator, in the `test_simu` working directory, do

```
$ tolteca simu
```

This will execute the simulator as planned in the config files in the working
directory.

The output files will be in a sub-dir named as the `jobkey` defined in the
`60_simu.yaml` file.

### Run reduce

The reduce follows the same logic as running simulator.

You'll need to do create YAML config file (e.g, 80_reduce.yaml in the example folder linked above).

The actual data reduction is done by the pipeline engine, which is specified to be citlali in our case. To proceed, we need to make available the citlali executable to the tolteca, which can be done by symlinking the citlali executable to the `bin` dir:
```
$ ln -s /path/to/citlali/executable <workdir>/bin
```

And to run the reduction:

```
$ tolteca reduce
```
