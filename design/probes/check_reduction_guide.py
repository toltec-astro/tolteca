"""Execute the trusted reduction-guide Python blocks with reference inputs."""

# ruff: noqa: INP001, PLR2004, T201 -- standalone verification probe

import argparse
import os
import re
import tempfile
from pathlib import Path


def main():
    """Check the examples without writing into the supplied raw-data root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    reference = args.data_root.resolve(strict=True) / "toltec/tcs/toltec0"
    target = reference / "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
    tune = reference / "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    for path in (target, tune):
        if not path.is_file():
            raise FileNotFoundError(path)
    document = Path(__file__).resolve().parents[2] / "docs/kids-reduction.md"
    blocks = re.findall(r"```python\n(.*?)```", document.read_text(), flags=re.DOTALL)
    namespace = {}
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="tolteca-reduction-guide-") as directory:
        try:
            os.chdir(directory)
            # Supply the documented local input/recipe placeholders. Only links
            # and outputs are created here; all acquisition payloads stay read-only.
            Path("sweep.nc").symlink_to(target)
            Path(target.name).symlink_to(target)
            Path(tune.name).symlink_to(tune)
            Path("recipes").mkdir()
            for index, block in enumerate(blocks, 1):
                compiled = compile(block, f"{document}:python-block-{index}", "exec")
                exec(compiled, namespace)  # noqa: S102 -- trusted guide examples
                print(f"PASS Python block {index}")
            manifest = namespace["manifest"]
            assert manifest.input.metadata.obsnum == 145606
            assert manifest.config.output.enabled
            assert namespace["context"].completed
            assert namespace["I"].shape == namespace["frequencies"].shape
            assert Path("replay.yaml").is_file()
        finally:
            os.chdir(cwd)
    print(
        f"Verified {len(blocks)} reduction-guide Python blocks; "
        "temporary outputs removed."
    )


if __name__ == "__main__":
    main()
