#!/usr/bin/env python

import shutil

from tollan.utils.dataclass_schema import DataclassSchema

from .misc import get_pkg_data_path
from .config_registry import ConfigRegistry


def make_config_schema_rst():
    """Generate RST doc for all config classes with schema defined."""
    from .runtime_context import rc_config_item_types
    from ..simu import simu_config_item_types
    from ..reduce import redu_config_item_types
    from ..db import db_config_item_types
    from ..web import web_config_item_types

    output = list()
    for dcls in (
            rc_config_item_types
            + simu_config_item_types
            + redu_config_item_types
            + db_config_item_types
            + web_config_item_types):
        # TODO refine this to make it in RST syntax
        if hasattr(dcls, 'pformat_schema'):
            doc = dcls.pformat_schema()
        else:
            doc = dcls.schema.pformat()
        output.append(doc)

    return '\n'.join(output)


def install_workdir_doc(rc):
    """Create and install example scripts and config reference in workdir.

    Parameters
    ----------
    rc : `tolteca.utils.RuntimeContext`
        The runtime context of the workdir to have the doc installed.
    """
    docdir = rc.docdir
    rootpath = rc.rootpath
    if docdir is None or rootpath is None:
        raise ValueError(
            f"No rootpath/docdir specified in runtime context {rc}")
    # config schema rst
    with open(
            docdir.joinpath('00_config_dict_references.txt'), 'w') as fo:
        # collect all config types from submodules
        fo.write(make_config_schema_rst())
    # copy example config files
    example_dir = get_pkg_data_path().joinpath('examples')
    for file in [
            '10_db.yaml',
            '60_simu_point_source_lissajous.yaml',
            '61_simu_blank_field_raster.yaml',
            # '62_simu_fits_input_rastajous.yaml',
            # '70_redu_simulated_data.yaml'
            '70_redu_citlali_default.yaml',
            ]:
        shutil.copyfile(example_dir.joinpath(file), docdir.joinpath(file))
    # readme file in the rootpath
    shutil.copyfile(
        example_dir.joinpath('workdir_README_template.md'),
        rootpath.joinpath('README.md'))


def collect_config_item_types(types):
    """A helper to collect config item types."""
    result = list()
    for t in types:
        if hasattr(t, 'schema') and isinstance(t.schema, DataclassSchema):
            result.append(t)
        elif isinstance(t, ConfigRegistry):
            result.append(t)
        else:
            pass
    return result
