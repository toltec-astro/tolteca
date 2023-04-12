#!/usr/bin/env python3

"""Example code to run the kids analysis."""

from pathlib import Path
import numpy as np
from tolteca.kids import ToltecaConfig
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger
import dill
import matplotlib.pyplot as plt


def collect_files(data_rootpath, pattern):
    return list(data_rootpath.glob(pattern))


def _make_obsnum_glob_patterns(obsnums):
    # this will resolve the argument to glob patterns to use
    # accepted cases:
    # "*" -- all obsnums
    # "1077*" -- all obsnum starting with 1077
    # [100, 101]  -- the list of obsnums
    # ["101*", "102*"] -- any obsnums start with 101 or 102
    def _pad_obsnum(obsnum):
        try:
            obsnum = int(obsnum)
            obsnum = f"{obsnum:06d}"
        except ValueError:
            return obsnum

    if obsnums is None:
        obsnum_patterns = ["*"]
    elif isinstance(obsnums, (str, int)):
        obsnum_patterns = [_pad_obsnum(obsnums)]
    elif isinstance(obsnums, list):
        obsnum_patterns = list(map(_pad_obsnum, obsnums))
    else:
        raise ValueError("invalid obsnums")
    return obsnum_patterns


def do_vnasweep_check(sweep_checker, data_rootpath, nw, obsnums, save_context=None):
    files = []
    for obsnum in _make_obsnum_glob_patterns(obsnums):
        files.extend(
            collect_files(
                data_rootpath,
                f"toltec/?cs/toltec*/toltec{nw}/toltec{nw}_{obsnum}_vnasweep.nc",
            )
        )
    logger.info(f"collected {len(files)} files")
    checker_contexts = []
    for f in files:
        logger.debug(f"checking {f}")
        checker_contexts.append(sweep_checker.check(f))
    # sort by obs time
    checker_contexts = sorted(checker_contexts, key=lambda c: c["swp"].meta["ut"])

    ctx = locals()
    if save_context is not None:
        dill.dump(save_context, ctx)
    return ctx


def plot_vnasweep_check(result_file):
    logger.info(f"plot vnasweep check result {result_file}")
    result_ctx = dill.load(result_file)
    nw = result_ctx["nw"]
    files = result_ctx["files"]
    n_files = len(files)
    logger.info(f"{nw=} {n_files=}")

    # access the data and plot
    checker_ctxs = result_ctx["checker_contexts"]
    n_chans = checker_ctxs[0]["n_chans"]  # 1000 for vna sweep

    all_rms_values = np.full((n_files, n_chans), np.nan)
    row_labels = []

    def _make_row_label(ctx):
        meta = ctx["swp"].meta
        return "{obsnum}-{subobsnum}-{scannum}".format(**meta)

    for i, ctx in enumerate(checker_ctxs):
        row_labels.append(_make_row_label(ctx))
        all_rms_values[i] = ctx["chan_rms_values"]

    # plot rms water fall
    fig, ax = plt.subplots(1, 1, figsize=(16, n_files / 2))
    ax.imshow(
        all_rms_values,
        aspect="auto",
        orign="upper",
        interpolation="none",
    )
    ax.set_yticks(range(n_files))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(range(n_chans))
    ax.set_xticklabels(range(n_chans))

    fig.tight_layout()
    plt.show()


def run(data_rootpath, config_file):
    """Run analysis and save result."""
    # create sweep checker from config
    cfg = ToltecaConfig.load_yaml(config_file)
    logger.info(f"loaded tolteca config:\n{pformat_yaml(cfg)}")

    sweep_checker = cfg.kids.sweep_checker()

    # run the vnasweep check for certain
    for nw in range(13):
        save_context = f"vnacheck_{nw}.pickle"
        do_vnasweep_check(
            sweep_checker,
            data_rootpath=data_rootpath,
            nw=nw,
            obsnums="*",  # see _make_obsnum_glob_patterns for permitted obsnum specifier
            save_context=save_context,
        )


def check_run(result_path):
    if result_path.startswith("vnacheck_"):
        plot_vnasweep_check(result_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="tolteca.yaml", type=Path, help="config file."
    )
    parser.add_argument(
        "--data_rootpath", "-d", default="data_lmt", type=Path, help="data rootpath."
    )
    parser.add_argument("run_result", nargs="?")

    option = parser.parse_args()

    data_rootpath = option.data_rootpath
    config_file = option.config_file

    if option.run_result:
        check_run(option.run_result)
    else:
        run(data_rootpath, config_file)
