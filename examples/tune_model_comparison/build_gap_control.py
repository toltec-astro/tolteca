"""Build the maintained gap/PCA correction against pinned local native sources."""

# ruff: noqa: INP001, S603, T201 -- local native build and explicit receipts

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from build_xprime_control import digest


def cache_value(path, key):
    """Read one exact value from the reference CMake cache."""
    return next(
        line.split("=", 1)[1]
        for line in path.read_text().splitlines()
        if line.startswith(key + ":")
    )


def build(native, output, *, prepare=True, patch_prefix="gap"):
    """Apply maintained patches to private sources and build/test privately."""
    native, output = native.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = native / "build/citlali/citlali"
    baseline_hash = digest(baseline)
    commands = []

    def run(args, name, cwd=None):
        commands.append({"argv": args, "cwd": str(cwd) if cwd else None})
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    patches = Path(__file__).parent / "native"
    for project in ("kidscpp", "citlali"):
        source = output / "native" / project
        if prepare:
            if source.exists():
                msg = f"Refuse to overwrite native sources: {source}"
                raise FileExistsError(msg)
            shutil.copytree(native / "native" / project, source)
            run(
                [
                    "/usr/bin/patch",
                    "-p1",
                    "-i",
                    str((patches / f"{patch_prefix}-{project}.patch").resolve()),
                ],
                f"patch-{project}",
                source,
            )
        cache = native / "build" / project / "CMakeCache.txt"
        prefix = str(output / "install") + ";" + cache_value(cache, "CMAKE_PREFIX_PATH")
        config = [
            "/opt/homebrew/bin/cmake",
            "-S",
            str(source),
            "-B",
            str(output / "build" / project),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
            "-DCMAKE_CXX_COMPILER=" + cache_value(cache, "CMAKE_CXX_COMPILER"),
            "-DCMAKE_PREFIX_PATH=" + prefix,
            "-DCMAKE_INSTALL_PREFIX=" + str(output / "install"),
            "-DBUILD_TESTING=ON",
        ]
        if project == "kidscpp":
            config += [
                "-DKIDS_BUILD_CLI=OFF",
                "-DKIDS_BUILD_TESTS=ON",
                "-DKIDS_BUILD_PYTHON=OFF",
                "-DKIDS_BUILD_DOCS=OFF",
            ]
        else:
            config += [
                "-DCITLALI_BUILD_CLI=ON",
                "-DCITLALI_BUILD_TESTS=ON",
                "-DCITLALI_ENABLE_OPENMP=OFF",
            ]
        run(config, f"configure-{project}")
        run(
            [
                "/opt/homebrew/bin/cmake",
                "--build",
                str(output / "build" / project),
                "-j",
                "2",
            ],
            f"build-{project}",
        )
        run(
            ["/opt/homebrew/bin/cmake", "--install", str(output / "build" / project)],
            f"install-{project}",
        )
    assert digest(baseline) == baseline_hash
    result = {
        "baseline": str(baseline),
        "baseline_sha256": baseline_hash,
        "baseline_unchanged": True,
        "patch_prefix": patch_prefix,
        "patch_hashes": {
            project: digest(patches / f"{patch_prefix}-{project}.patch")
            for project in ("kidscpp", "citlali")
        }
        if prepare
        else {},
        "binary": str(output / "build/citlali/citlali"),
        "binary_sha256": digest(output / "build/citlali/citlali"),
        "commands": commands,
    }
    (output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--existing-sources", dest="prepare", action="store_false")
    parser.add_argument("--patch-prefix", choices=("gap", "joint-map"), default="gap")
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
