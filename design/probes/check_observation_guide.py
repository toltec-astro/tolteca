"""Execute the trusted observation-guide Python blocks on read-only reference data."""

# ruff: noqa: INP001, PLR2004, T201 -- standalone verification probe

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from tollan.utils.log import logger


def main():
    """Check the guide examples against the supplied raw-data root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    document = Path(__file__).resolve().parents[2] / "docs/guide.md"
    text = document.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    logger.disable("tolteca_datamodels")
    namespace = {}
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="tolteca-guide-") as directory:
        try:
            os.chdir(directory)
            for index, block in enumerate(blocks, 1):
                # Only replace the documented input-root placeholder. Catalog
                # writes use the isolated temporary working directory above.
                code = block.replace('Path("/data_lmt")', f"Path({str(data_root)!r})")
                compiled = compile(code, f"{document}:python-block-{index}", "exec")
                exec(compiled, namespace)  # noqa: S102 -- trusted guide examples
                print(f"PASS Python block {index}")
            assert namespace["meta"].uid_raw_obs_file == "tcs-145609-0-2-tel"
            assert len(namespace["guessed"]) == len(namespace["catalog"]) == 3
            assert len(namespace["telescope"]) == 1
            assert len(namespace["selected"]) == 1
            selected = namespace["by_recorded_name"]
            restored = namespace["selected_again"]
            # read_parquet(dtype_backend='pyarrow') also converts scalar string
            # columns. Compare typed source values and the native struct dtypes.
            assert (
                selected.toltec_file.to_info_list()
                == restored.toltec_file.to_info_list()
            )
            columns = ["filepath_info", "filepath_info_orig"]
            pd.testing.assert_frame_equal(selected[columns], restored[columns])
            assert namespace["record_restored"] == namespace["record"]
            assert (
                namespace["metadata"].uid_raw_obs_file
                == namespace["meta"].uid_raw_obs_file
            )
            [recipe] = re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
            Path("check.yaml").write_text(recipe.replace("/data_lmt", str(data_root)))
            commands = [
                line
                for block in re.findall(r"```console\n(.*?)```", text, flags=re.DOTALL)
                for line in block.splitlines()
                if line.startswith("tolteca_kids ")
            ]
            for index, command in enumerate(commands, 1):
                args_cli = shlex.split(command.replace("/data_lmt", str(data_root)))
                result = subprocess.run(  # noqa: S603 -- trusted CLI examples
                    [sys.executable, "-m", "tolteca_kids.cli", *args_cli[1:]],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                if "--dry_run" in args_cli:
                    assert "no-op completed" in result.stdout
                print(f"PASS KIDs CLI example {index}")
        finally:
            os.chdir(cwd)
    print(f"Verified {len(blocks)} guide Python blocks; temporary outputs removed.")


if __name__ == "__main__":
    main()
