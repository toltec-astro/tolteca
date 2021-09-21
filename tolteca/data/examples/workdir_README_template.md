# Contents in this TolTECA workdir

## bin/

The `bin/` folder holds the executables and scripts that are external
to the tolteca python package.

The tolteca will check both the `bin/` directory and the system `$PATH`
environment variable to locate the necessary programs to run.

However, explicitly symlinking executables to the `bin/` folder will
tie the particular program to the workdir, allowing the runtime context
to always refer to the desired executable of a particular version.


## log/

The `log/` folder holds the log files created when running the core
tasks of tolteca. This includes both the `tolteca.simu` and `tolteca.reduce`.


## cal/

The `cal/` folder holds any larger dataset for calibration purpose, either
managed by the tolteca automatically or provided by user manually.

The user typically does not need to add items to the cal folder.

## doc/

The `doc/` folder holds the following files:

* `00_config_dict_references.txt` This file contains the full description
  of objects that can be configured via the config dict in some semi-tabular
  format.

* `??_*.yaml`. These files are example YAML config files with extensive
  inline comments. The user may use these config files as a head-start for
  their usage of the package.
 
## 40_setup.yaml

The `40_setup.yaml` file contains a dump of the config dict loaded when the
`tolteca setup` command is run to create this workdir, referred to as
the "setup config". The setup config is used at later runs to check for
any change of the software versions, config settings, that may cause
compatibility issues which could lead to incorrect/changed results.

Typically, for each workdir, the setup should only be done once. This ensures
the least amount confusion in later runs.

If you would like to update the setup config, the best way is to create a
fresh new workdir, and copy over those other YAML configs that you created.

However, if insisted, you can use `tolteca setup -fo` to overwrite the setup
file. A backup file of the setup.yaml and `bin/` folder will be created by
default. Use `--no_backup` to disable the backup.

## User provided YAML configs `\d+_.+.ya?ml`

Any file with name matching the pattern `\d+_.+ya?ml` (e.g, 10_db.yaml,
60_simu.yaml, etc) are recognized as
config files and will be loaded to the config dict via the
`tolteca.utils.RuntimeContext` class. The config file has to be of YAML
format and contains dicts in its top level.

The leading number in the config file name denotes the order of precedence
when loading the config: when same entry appears, the one defined in the
file with larger number is used.

All the config dict in the found YAML config files are merged in a
recursive fashion for dicts, but not for other collections like lists.
As an example, for the following two files

```yaml
# 20_a.yaml
a:
  b:
    c: [10, 20]
  d: 1
```

```yaml
# 21_b.yaml
a:
  b:
    c: [30, ]
e: true
```

The resultant config dict will be

```yaml
a:
  b:
    c: [30, ]
  d: 1
e: true
```
