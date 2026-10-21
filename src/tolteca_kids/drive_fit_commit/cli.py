"""Command-line drive-fit commit workflow."""

from tolteca_kids.drive_fit.cli import _make_app

from .core import DriveFitCommit, run

app = _make_app(
    key="drive_fit_commit",
    handler=DriveFitCommit,
    runner=run,
    help_text=(
        "Match a published drive fit to target sweeps and publish commit settings."
    ),
)
