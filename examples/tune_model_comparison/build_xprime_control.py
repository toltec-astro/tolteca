"""Build an isolated x-prime diagnostic from the pinned native comparison build.

Only the final scalar coordinate changes. Compute x_opt and all validity flags
first, then return the pre-correction x_prime for accepted physical samples.
Never modify fit parameters, source checkouts, installed libraries or baseline
executables. This is an experiment, not a production solver/config option.
"""

# ruff: noqa: INP001, S603, T201 -- explicit local native build recipe

import argparse
import difflib
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    """Identify exact source, library and executable bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patched_source(source):
    """Preserve flags and loss, changing only accepted scalar physical x output."""
    before = "      if (physical) {\n        const auto driven = x(s, ch)"
    after = """        r(s, ch) = nan;
      }
    }
  }
}

TimeStreamSolverResult solve_timestream"""
    if source.count(before) != 1 or source.count(after) != 1:
        msg = "Native scalar solving source differs from the reviewed patch anchors"
        raise ValueError(msg)
    return source.replace(
        before,
        "      const auto apparent_x = x(s, ch);\n" + before,
    ).replace(
        after,
        """        r(s, ch) = nan;
      } else if (physical) {
        // Diagnostic projection only: retain x_opt's validity decisions.
        x(s, ch) = apparent_x;
      }
    }
  }
}

TimeStreamSolverResult solve_timestream""",
    )


def ninja_command(build, target):
    """Read a pinned target command without executing or updating that build."""
    result = subprocess.check_output(
        ["/opt/homebrew/bin/ninja", "-C", str(build), "-t", "commands", target],
        text=True,
    )
    command = shlex.split(result.splitlines()[-1])
    if command[:2] == [":", "&&"]:
        command = command[2:]
    if "&&" in command:
        command = command[: command.index("&&")]
    if any(t in command for t in (";", "|", ">", "<")):
        msg = "Unexpected shell operator in native build command"
        raise ValueError(msg)
    return command


def replace_option(command, option, value):
    """Replace an explicit compiler option without shell interpolation."""
    command[command.index(option) + 1] = str(value)


FIXTURE = r"""
#include <kids/model/physical.h>
#include <kids/reduce/timestream.h>
#include <complex>
#include <iostream>
#include <iomanip>
#include <limits>
using namespace kids::reduce;
int main() {
  std::cout << std::setprecision(17);
  for (int family=0;family<3;++family) {
    ModelParamsTable table;
    table.channels={17};
    table.f_in_hz=Eigen::VectorXd::Constant(1,500e6);
    const double r0=2e-5, b=family==2 ? 0. : 2*.4*r0*r0*r0;
    table.model=family==1 ? KidsSweepWithDelayGain::schema(2)
                          : KidsSweepWithAffineGain::schema(2);
    table.parameters.resize(1,10);
    table.parameters << 500e6,500e6,r0,1e-5,2e-6,2.,1.,0.,0.,b;
    const double loss[]{r0,1.2*r0,-r0,.4*r0,r0,r0};
    const double xp[]{4*r0,-3*r0,r0,-.4*r0,r0,r0};
    Matrix I(6,1),Q(6,1),X(6,1),R(6,1); Mask flags(6,1); SolveWorkspace w;
    for(int k=0;k<6;++k) {
      auto z=std::complex<double>(2.,1.)*(1.-std::complex<double>(1e-5,2e-6)
               /std::complex<double>(loss[k],xp[k]));
      if(k==4) z=std::complex<double>(std::numeric_limits<double>::quiet_NaN(),0.);
      if(k==5) z=std::complex<double>(2.,1.);
      I(k,0)=4*z.real();Q(k,0)=4*z.imag();
    }
    solve_into(I,Q,TimeStreamSolver(table,table.channels),4.,X,R,flags,w);
    for(int k=0;k<6;++k)
      std::cout << family << ',' << k << ',' << flags(k,0) << ',' << R(k,0)
                << ',' << X(k,0) << ',' << xp[k] << ',' << b << '\n';
  }
}
"""


def build(native, output):  # noqa: PLR0915 -- captured build and numerical verification
    """Compile one changed object, copy its archive and relink private outputs."""
    native, output = native.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = native / "native/kidscpp/src/kids/reduce/timestream.cpp"
    library = native / "install/lib/libkids_reduction.a"
    baseline = native / "build/citlali/citlali"
    pinned = {str(p): digest(p) for p in (source, library, baseline)}
    text = source.read_text()
    patched = patched_source(text)
    local_source = output / "timestream.cpp"
    local_source.write_text(patched)
    (output / "xprime-output.patch").write_text(
        "".join(
            difflib.unified_diff(
                text.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="a/src/kids/reduce/timestream.cpp",
                tofile="b/src/kids/reduce/timestream.cpp",
            )
        )
    )
    commands = []

    def run(command, cwd):
        commands.append({"argv": command, "cwd": str(cwd)})
        subprocess.run(command, cwd=cwd, check=True)

    kid_build = native / "build/kidscpp"
    target = "CMakeFiles/kids_reduction.dir/src/kids/reduce/timestream.cpp.o"
    compile_command = ninja_command(kid_build, target)
    compile_command[-1] = str(local_source)
    for key, dest in (
        ("-o", "timestream.cpp.o"),
        ("-MF", "timestream.d"),
        ("-MT", "timestream.cpp.o"),
    ):
        replace_option(compile_command, key, output / dest)
    run(compile_command, kid_build)
    copied_library = output / library.name
    shutil.copy2(library, copied_library)
    archiver = str(Path(compile_command[0]).with_name("llvm-ar"))
    run([archiver, "r", str(copied_library), str(output / "timestream.cpp.o")], output)
    link = ninja_command(native / "build/citlali", "citlali")
    assert str(library) in link
    link[link.index(str(library))] = str(copied_library)
    binary = output / "citlali-xprime"
    replace_option(link, "-o", binary)
    run(link, native / "build/citlali")
    fixture = output / "coordinate_fixture.cpp"
    fixture.write_text(FIXTURE)
    cc = compile_command.copy()
    cc[-1] = str(fixture)
    for key, dest in (("-o", "fixture.o"), ("-MF", "fixture.d"), ("-MT", "fixture.o")):
        replace_option(cc, key, output / dest)
    run(cc, kid_build)
    results = []
    for name, archive in (("baseline", library), ("xprime", copied_library)):
        exe = output / f"fixture-{name}"
        run([cc[0], str(output / "fixture.o"), str(archive), "-o", str(exe)], output)
        value = subprocess.check_output([str(exe)], text=True)
        (output / f"fixture-{name}.csv").write_text(value)
        results.append(np.loadtxt(value.splitlines(), delimiter=","))
    base, prime = results
    np.testing.assert_equal(base[:, :4], prime[:, :4])
    valid = base[:, 2] == 0
    np.testing.assert_allclose(prime[valid, 4], prime[valid, 5], rtol=0, atol=1e-18)
    expected = prime[valid, 4] - prime[valid, 6] / (
        prime[valid, 3] ** 2 + prime[valid, 4] ** 2
    )
    np.testing.assert_allclose(base[valid, 4], expected, rtol=0, atol=1e-18)
    assert np.isnan(prime[~valid, 4]).all()
    assert all(digest(p) == h for p, h in pinned.items())
    receipt = {
        "scope": (
            "scalar physical x_prime, preserving x_opt validation; "
            "joint outputs unchanged"
        ),
        "baseline_files_unchanged": pinned,
        "coordinate_tests": {
            "samples": len(base),
            "valid": int(valid.sum()),
            "flags_and_r_identical": True,
            "xprime_matches_expected": True,
        },
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "commands": commands,
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "commands"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    build(**vars(parser.parse_args()))
