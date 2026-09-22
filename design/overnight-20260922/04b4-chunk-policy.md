# 04b4 — Measure chunk-policy tradeoffs before changing defaults

## Method

`design/probes/sweep_chunk_layout.py` reads completed reference products without
modifying them. It materializes I/Q once, then writes temporary Zarr-v2 datasets
containing just those signals and their coordinates. Codec/filter/dtype/fill-value
settings match the published arrays (float64, Blosc lz4 level 5 with shuffle).
The baseline uses the exact published chunk shape, not a fresh auto-guess.
Variants use 1, 16 or 64 channels per chunk and the full sweep axis; tune keeps
one block per chunk. Stores are removed by TemporaryDirectory on exit.

Each variant is opened separately for adjacent two channels, widely separated
two channels, batches of 16/64 and the full channel set. LocalStore.get counts
only I/Q chunks; returned compressed bytes are counted. Every read is compared
exactly with the original arrays, and attrs are checked unchanged. Opening and
coordinate traffic are excluded from signal counts. These are not complete
product-size, HTTP-request, cold-cache, peak-RSS or remote-latency measurements.
Single-pass timings are diagnostic only: OS caches are warm/uncontrolled.

Executed on fresh `overnight-slice-20260922-consolidated` VNA, target and last
tune-block products. All 60 selection/layout comparisons pass. Raw JSON:
`/tmp/tolteca-20260922-chunk-layouts.json`; script and this summary are durable.

## Results

Bytes below are the combined compressed I/Q bytes, not selected in-memory size.
Full means all channels of the selected block, not all tune blocks.

| Product | Chunk shape | Two adjacent channels: chunks / bytes | Full block: chunks / bytes |
| --- | --- | ---: | ---: |
| VNA | 250 × 246 (published) | 4 / 1,292,870 | 16 / 5,177,123 |
| VNA | 1 × 491 | 4 / 12,763 | 2,000 / 6,291,109 |
| VNA | 16 × 491 | 2 / 86,494 | 126 / 5,363,863 |
| VNA | 64 × 491 | 2 / 333,700 | 32 / 5,203,937 |
| Target | 334 × 89 (published) | 4 / 645,073 | 8 / 1,300,800 |
| Target | 1 × 177 | 4 / 4,867 | 1,336 / 1,616,067 |
| Target | 16 × 177 | 2 / 33,121 | 84 / 1,393,310 |
| Target | 64 × 177 | 2 / 125,176 | 22 / 1,312,714 |
| Tune | 1 × 316 × 177 (published) | 2 / 603,159 | 4 / 1,209,462 |
| Tune | 1 × 1 × 177 | 4 / 4,862 | 1,264 / 1,530,914 |
| Tune | 1 × 16 × 177 | 2 / 33,645 | 80 / 1,321,842 |
| Tune | 1 × 64 × 177 | 2 / 126,157 | 20 / 1,243,874 |

The widely separated pair needs two channel chunks for 16/64 layouts, while
adjacent channels share one. For 64-channel batches, 64-channel layouts read two
signal chunks, versus eight for 16-channel layouts and 128 for one-channel
layouts. Whole-store signal/coordinate sizes also grow with tiny chunks: the VNA
temporary dataset is ~6.31 MB with one-channel chunks versus ~5.19 MB published.

## Decision and remaining evidence

Keep the writer default unchanged in this checkpoint. One-channel chunks favor
very sparse reads but significantly increase whole-block operations and reduce
compression. Sixteen channels is a plausible sparse-viewer layout; 64 channels
is a plausible mixed-access layout. Neither is universally best: the full-block
versus selected-channel workload and storage latency determine the tradeoff.

Before selecting a default, measure the actual serving path over controlled
HTTP/object storage with its concurrency and cache settings, and include all
four science arrays plus uncertainty consumers. If the workflow needs an output
policy knob, it belongs to configured product output, not the standalone reducer
or Web's frequency calculation. Do not introduce a duplicate viewer-only science
store merely to hide a chunk-policy choice.

This is measurement/recommendation, not a dependency upgrade, new storage API,
production service, or claim that remote acceptance is complete. Ruff (excluding
existing CPY001), formatting and scoped ty pass. Self-review confirms the probe
only writes temporary stores and preserves codecs and exact numerical contents.
