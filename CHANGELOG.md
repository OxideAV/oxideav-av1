# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/OxideAV/oxideav-av1/compare/v0.1.0...v0.2.0) - 2026-04-24

### Other

- round 9b — unit tests for delta_q / delta_lf apply arithmetic
- round 9a — filter-intra dispatch + CurrentQIndex/DeltaLF apply
- round 8 — palette_mode_info + var-tx inter + inter-skip ordering
- round 7 — wire delta_q/lf + use_intrabc + filter_intra_mode_info (§5.11.7/.12/.13/.24)
- wire read_block_tx_size() + fix read_skip ordering (§5.11.5/.7/.16)
- document testsrc luma gap root causes (investigation notes)
- spec-correct intra neighbor gathering (§7.11.2.1)
- §7.9 reference scaling scaffolding + scaled luma MC
- retain global-motion params + compound luma MC skeleton (§7.10 / §7.11.3.9)
- round-4 inter MC — §7.11.3.3 MV clamp + §5.11.26 per-block interp filter
- spec-exact per-SB CDEF (§7.15 + §5.11.56)
- wire intra edge filter + upsample into TX-unit predictors
- add intra edge filter + upsample (§7.11.2.9-.12)
- drop goavif origin markers (self-port; no third-party attribution needed)
- add AVIF-still 128×128-SB decode fixture + test
- run intra prediction per TX unit, not per block (§7.11.2)
- split 128×N intra blocks into 64×64 TX units (§5.11.27)
- fix reduced_still_picture_header parsing (§5.9.1)
- release v0.0.4

## [0.1.0](https://github.com/OxideAV/oxideav-av1/compare/v0.0.3...v0.1.0) - 2026-04-19

### Other

- Update Cargo.toml: set release to 0.1.0
- fix debug-mode panic in decode_uniform (§5.11.41)
- rustfmt sweep across Phase 1-7 code
- finish Phase 7 — inter-intra path + gm_params + AVIS test
- land Phase 7 plumbing — interp/mv/mc/inter + DPB + inter leaf path
- add LR + film-grain integration fixtures
- wire per-unit LR signalling + finish_frame LR/grain apply
- add lr + filmgrain modules — Phase 6 primitives
- add 128×128 non-skip fixture integration test
- land Phase 5 — full intra + deblock + CDEF in one sweep
- update crate docs + codec capability tag for Phase 4
- extend TX-size coverage to every rectangular shape + add HBD test
- wire 32/64 TX sizes + tx_type decode + HBD plumbing
- port idct32/idct64/idtx/iwht4/flipadst + wire full 1D dispatch
- update crate-level docs + codec capability tag for Phase 3
- add end-to-end tile-walk decode test against /tmp/av1.ivf
- wire coefficient decoder + intra predict + reconstruct into tile walk
- add decode::coeffs + decode::reconstruct — §5.11.39 / §7.7.4
- rewrite transform as modular 4/8/16 iDCT + iADST dispatch
- add quant module — DC/AC dequantiser tables + per-plane compute
- wire decode::decode_tile_group into Av1Decoder; drop tile_decode stub
- add decode::tile + decode::superblock — partition walker + mode decoder
- add decode module helpers — block/partition/modes/coeff_ctx/tx_type_map/frame_state
- parse uncompressed-header tail — quant through film-grain
- rewrite range-coder symbol decoder against goavif wire-format
- import default CDF tables via gen_cdfs Go-to-Rust generator
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- claim AVI FourCC via oxideav-codec CodecTag registry
- drop vestigial placeholder write in SymbolDecoder::new
- parse tile_info (§5.9.15) + tile_group framing + README rewrite
