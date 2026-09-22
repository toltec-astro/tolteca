# 04b — DB/Web boundaries and measured product-read costs

Findings checkpoint: no storage policy, database schema or Web behavior changed.

## Reviewed behavior

Re-read DB registration, product catalog publication and their full focused test
file; Web product service, saved-view figure helper, catalog backend and focused
tests; prior DB/Web design records 07b/08/09/11. Registration owns no transaction,
validates source lineage, opens only product metadata, and closes the tree.
Catalog export reads manifests, not stores, and publishes atomically on a local
filesystem. Web selects a database-local product ID rather than a quartet's first
match, constrains paths to the product root, and closes on success/error. Existing
tests cover those behaviors; fresh full integration passes (03b4).

The legacy Web observation backend still assumes raw-data columns and first-row
quartet lookup. The new product service uses only its query/cache/reload portions.
Do not generalize the old interface or rewrite the portal during this slice.
Product-ID scope, ORM `obs_datetime` naming, prefix-vs-interface identity, local
roots, and immutable/rerun product discovery remain recorded design friction.

## Read probe

`design/probes/sweep_product_reads.py` instruments LocalStore `get`, partial
reads and directory listing while separately opening, validating the context,
and plotting channels [0, 1]. Original products are read-only. Consolidation is
performed only on complete temporary copies; no fixture or existing output is
changed. Results: `/tmp/tolteca-20260922-product-reads.json`, against the fresh
`overnight-slice-20260922-lifecycle` products.

| Product | Open get calls, original → consolidated | Open listings, original → consolidated | Plot I/Q chunk reads | Plot compressed I/Q bytes |
| --- | ---: | ---: | ---: | ---: |
| VNA | 285 → 6 | 4 → 0 | 4 | 1,292,870 |
| Target | 297 → 6 | 4 → 0 | 4 | 645,073 |
| Tune block 0 | 304 → 7 | 4 → 0 | 2 | 600,208 |

Open does not load signal chunks, but it does load two dimension-coordinate
chunks (three for tune) for xarray indexes. Context validation adds no store
reads. Consolidation does not change subsequent signal chunk reads or bytes.
The initial count probe checked trace counts only. A follow-up now also asserts
exact equality of every plotted x/y array before/after consolidation. The
separate four-case integration establishes actual-v2 numerical parity.

These are operations observed in a local store, including missing-key probes,
not HTTP request counts or remote wall-time measurements. OS caching is not
controlled. Opening original products reads only ~40–46 KB but makes many small
metadata operations. Consolidated copies read ~23–24 KB in a few operations.

## Focused next steps

1. **Consolidation:** the current writer and reader explicitly disable it. Use
   consolidated metadata for immutable completed Zarr-v2 products. Test writer
   completion/failure, read policy, corruption tests, native fields and exact
   parity before committing. Decide and document read behavior rather than
   accidentally making all earlier local test products unreadable. Do not change
   the Zarr format or compression in the same patch.
2. **Repeated coordinate reads:** plotting reads `f_chan` four times in all three
   cases. I/Q are already selected/batched, but their attached coordinates may be
   loaded again by separate `.compute()` calls. Trace it, then consider computing
   only the selected I/Q Dataset and shared coordinates together. Never compute
   the complete reduced Dataset (it contains unrelated metadata variables).
3. **Chunk policy:** current channel chunks (250/334/316) amplify sparse plotting.
   Retain them until measured against sparse, batch and full-array use, including
   request counts/bytes. A universal tiny chunk is not justified by this local
   probe. Record a benchmark or recommendation separately from consolidation.
4. **Evidence quality:** the DB sweep-product unit fixture still uses a positive
   synthetic acquisition name/quartet (obsnum 1). Refresh that identity from a
   recorded reference case while retaining intentionally small numerical arrays;
   distinguish numerical unit fixtures from acquisition-format evidence. Do not
   use invented names to justify registry extensions.

## Official guidance consulted

[Zarr consolidation](https://zarr.readthedocs.io/en/latest/user-guide/consolidated_metadata/)
targets read-heavy, slowly changing hierarchies and avoids repeated child-metadata
reads. Its v3 experimental warning does not require changing this v2 product.
Our write-once directories and manifest-written-last boundary fit that use case.

[Zarr performance guidance](https://zarr.readthedocs.io/en/latest/user-guide/performance/)
ties chunk shapes to access patterns; automatic shapes are heuristics, and smaller
chunks trade read amplification against overhead. This supports measurement,
not choosing a new universal chunk size from a two-channel example.

[xarray opening behavior](https://docs.xarray.dev/en/latest/generated/xarray.open_dataset.html)
distinguishes backend-lazy `chunks=None` from Dask and documents dimension-index
coordinate loads. Accordingly, DB registration remains signal-lazy, not literally
zero array reads; no switch of index semantics is proposed here.
