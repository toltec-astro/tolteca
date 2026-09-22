# 02b15a — Serializable telescope metadata with an unknown original name

## Finding and bounded behavior change

The telescope reader deliberately accepts an unrecognized Header.File.Name
unless the caller opts into `check_filepath_orig`. Its computed
`filepath_info_orig` nevertheless raised during ordinary serialization, including
the new source/catalog projection. This made a readable metadata object fail
when placed in a catalog. The standalone regression already documented that
round-trip serialization works only because it skips computed properties.

Return None for an unrecognized name, consistently with SourceInfoModel's
parsed-path property. Preserve the required raw path and every recorded header.
Recognized names and invalid dates keep their existing behavior. The explicit
reader consistency check still rejects an unrecognized name with ValueError;
the check now raises directly instead of relying on the computed property.

This is an observable error-policy correction, not an internal-only refactor.
It does not invent a telescope stream kind from a timestamp variable, relax
required headers, or make arbitrary filenames valid acquisition names.

## Verification / self-review

The existing metadata round-trip test now also checks a normal JSON-mode dump
of the intentionally unknown filename; both raw path and null parsed path survive.
The existing strict-check regression remains unchanged. All 120 LMT tests pass
in 1.66 s; eight optional v2 parity cases were not enabled in this focused run
and will run in the full checkpoint. Scoped Ruff and ty pass. Valid reference
metadata and scientific values are unchanged. Automatic source attachment is a
separate pending patch and is not included in this commit.
