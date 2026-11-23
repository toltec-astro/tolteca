#!/usr/bin/env python3
"""Fetch coordinated source revisions and install an isolated comparison stack.

Uses only the standard library until the pinned Python environment is installed.
Native dependencies come from an explicit CMake prefix or generated Spack env.
"""

# ruff: noqa: E501, S603, T201, TRY301 -- explicit installation commands with receipts

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def bootstrap(args):  # noqa: C901, PLR0912, PLR0915 -- ordered, recorded installation
    """Prepare fresh destinations; never change a user's existing installation."""
    lock = json.loads((HERE / "revisions.json").read_text())
    git = shutil.which("git")
    if git is None:
        msg = "Git is required to fetch the pinned sources"
        raise FileNotFoundError(msg)
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    sources = destination / "sources"
    sources.mkdir()
    receipt: dict[str, Any] = {
        "release": lock["release"],
        "commands": [],
        "sources": {},
        "status": "preparing",
        "python_installed": False,
        "native_build": "not_run",
    }

    def run(argv, name, *, env=None):
        receipt["commands"].append(argv)
        (destination / "bootstrap.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(name, flush=True)
        with (destination / "logs" / (name + ".log")).open("w") as log:
            subprocess.run(
                argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    selected = (
        ["tollan"]
        if args.python_only
        else [k for k in lock["repositories"] if k != "tolteca"]
    )
    if args.native_only:
        selected.remove("tollan")
    try:
        for name in selected:
            record = lock["repositories"][name]
            source = sources / name
            remote = (
                record["url"]
                if args.source_root is None
                else str(args.source_root.resolve() / name)
            )
            run([git, "clone", "--no-checkout", remote, str(source)], "clone-" + name)
            run(
                [git, "-C", str(source), "checkout", "--detach", record["revision"]],
                "checkout-" + name,
            )
            actual = subprocess.check_output(
                [git, "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != record["revision"]:
                msg = "Fetched revision differs from the release lock"
                raise ValueError(msg)
            receipt["sources"][name] = dict(
                record, path=str(source), fetched_revision=actual
            )
        receipt["sources"]["tolteca"] = {
            "path": str(ROOT),
            "revision": subprocess.check_output(
                [git, "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        if args.spack_env:
            native = {
                name: row
                for name, row in receipt["sources"].items()
                if "spack_spec" in row
            }
            environment = {
                "specs": ["citlali@4.1.0~openmp ^kidscpp~cli~python"],
                "concretizer": {"unify": True},
                "view": str(destination / "spack-view"),
                "repos": {
                    r["spack_namespace"]: str(
                        Path(r["path"])
                        / "spack_repo"
                        / r["spack_namespace"].replace(".", "/")
                    )
                    for r in native.values()
                },
                "develop": {
                    name.replace("_", "-"): {"spec": r["spack_spec"], "path": r["path"]}
                    for name, r in native.items()
                },
            }
            if args.spack_compiler:
                environment["packages"] = {"cxx": {"require": args.spack_compiler}}
            folder = destination / "spack"
            folder.mkdir()
            (folder / "spack.yaml").write_text(
                json.dumps({"spack": environment}, indent=2) + "\n"
            )
            receipt["native_build"] = "spack_environment_prepared"
        if not args.fetch_only and not args.native_only:
            venv = destination / "venv"
            uv = shutil.which("uv")
            install_env = dict(os.environ)
            install_env.setdefault("UV_CACHE_DIR", str(destination / "cache/uv"))
            if uv:
                run(
                    [uv, "--no-config", "venv", "--python", args.python, str(venv)],
                    "venv",
                    env=install_env,
                )
                installer = [
                    uv,
                    "--no-config",
                    "pip",
                    "install",
                    "--no-sources",
                    "--python",
                    str(venv / "bin/python"),
                ]
            else:
                run([args.python, "-m", "venv", str(venv)], "venv")
                installer = [str(venv / "bin/python"), "-m", "pip", "install"]
            run(
                [
                    *installer,
                    "--index-url",
                    "https://pypi.org/simple",
                    "--constraint",
                    str(HERE / "python-constraints.txt"),
                    str(sources / "tollan"),
                    str(ROOT),
                ],
                "install-python",
                env=install_env,
            )
            run(
                [
                    str(venv / "bin/python"),
                    "-c",
                    "import tolteca_kids, tolteca_kidsproc, tolteca_reduce.stubs, tollan; print('comparison imports passed')",
                ],
                "python-imports",
            )
            run([str(venv / "bin/tolteca_kids"), "--help"], "kids-help")
            run(
                [
                    uv,
                    "--no-config",
                    "pip",
                    "check",
                    "--python",
                    str(venv / "bin/python"),
                ]
                if uv
                else [str(venv / "bin/python"), "-m", "pip", "check"],
                "python-dependencies",
                env=install_env,
            )
            receipt["python_installed"] = True
        if not args.fetch_only and not args.python_only and not args.spack_env:
            cmake = shutil.which("cmake")
            if cmake is None:
                msg = "Install CMake >= 3.25"
                raise FileNotFoundError(msg)
            prefix = destination / "install"
            prefixes = [str(prefix), *map(str, args.deps_prefix)]
            if os.environ.get("CMAKE_PREFIX_PATH"):
                prefixes.append(os.environ["CMAKE_PREFIX_PATH"])
            projects = {
                "tula_cmake": ["-DTULA_CMAKE_BUILD_TESTING=OFF"],
                "tula": [
                    "-DTULA_BUILD_TESTS=OFF",
                    "-DTULA_ENABLE_OPENMP=OFF",
                    *[
                        f"-DTULA_ENABLE_{m}=ON"
                        for m in (
                            "LOGGING",
                            "YAML",
                            "ECSV",
                            "EIGEN",
                            "NETCDF",
                            "ENUM",
                            "CLI",
                            "PERFLIBS",
                            "GRPPI",
                            "FITTING",
                        )
                    ],
                ],
                "kidscpp": [
                    "-DKIDS_BUILD_CLI=OFF",
                    "-DKIDS_BUILD_PYTHON=OFF",
                    "-DKIDS_BUILD_DOCS=OFF",
                    "-DKIDS_BUILD_TESTS=ON",
                    "-DKIDS_ENABLE_OPENMP=OFF",
                ],
                "citlali": [
                    "-DCITLALI_BUILD_CLI=ON",
                    "-DCITLALI_BUILD_TESTS=ON",
                    "-DCITLALI_ENABLE_OPENMP=OFF",
                ],
            }
            for name, options in projects.items():
                build = destination / "build" / name
                config = [
                    cmake,
                    "-S",
                    str(sources / name),
                    "-B",
                    str(build),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
                    "-DCMAKE_INSTALL_PREFIX=" + str(prefix),
                    "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
                    *options,
                ]
                if args.cxx:
                    config.append("-DCMAKE_CXX_COMPILER=" + args.cxx)
                run(config, "configure-" + name)
                run(
                    [cmake, "--build", str(build), "--parallel", str(args.jobs)],
                    "build-" + name,
                )
                run([cmake, "--install", str(build)], "install-" + name)
            run([str(prefix / "bin/citlali"), "--version"], "citlali-version")
            receipt["native_build"] = "completed"
        receipt["status"] = "fetched" if args.fetch_only else "prepared"
    except (subprocess.CalledProcessError, OSError, ValueError):
        receipt["status"] = "failed"
        raise
    finally:
        (destination / "bootstrap.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    print(destination / "bootstrap.json")


def main():
    """Choose either existing native dependencies or a generated Spack environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New stack directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--python-only", action="store_true")
    mode.add_argument("--native-only", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument(
        "--spack-env",
        action="store_true",
        help="Write a Spack environment instead of invoking CMake; Python install still runs",
    )
    parser.add_argument(
        "--spack-compiler", help="Site compiler requirement, e.g. gcc@14.2.0"
    )
    parser.add_argument(
        "--deps-prefix",
        type=Path,
        action="append",
        default=[],
        help="Existing CMake dependency prefix; repeat as needed",
    )
    parser.add_argument("--cxx", help="C++23 compiler path")
    parser.add_argument(
        "--python", default="3.13", help="Python version for uv, or interpreter path"
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Local Git mirrors named by repository, for an offline rehearsal",
    )
    args = parser.parse_args()
    if args.python_only and args.spack_env:
        parser.error("--spack-env needs native sources")
    if not shutil.which("uv") and args.python == "3.13":
        args.python = "python3.13"
    bootstrap(args)


if __name__ == "__main__":
    main()
