# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec.

## Status — 2026-05-20

**Orphan-rebuild scaffold.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core decoder modules could not be defended against the "no external
library source as reference" rule that governs every crate in this
workspace.

Per workspace policy, the only acceptable response is a full
clean-room re-implementation against the AV1 standards documents and
black-box validator binaries. That work has not yet been scheduled.

Every public entry point currently returns `Error::NotImplemented`.

## Planned clean-room sources

The clean-room rebuild will consult only:

* AV1 Bitstream & Decoding Process Specification (AOMedia) — the
  authoritative format spec.
* Black-box invocations of `dav1d` / `aomenc` (the binaries — not
  their source) as opaque validators.

No external library source — libaom, dav1d, rav1e, libgav1, SVT-AV1,
etc. — is permitted as a reference under the workspace clean-room
policy.

## License

MIT. See `LICENSE`.
