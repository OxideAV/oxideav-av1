//! Forward AV1 range coder — inverse of [`crate::symbol::SymbolDecoder`].
//!
//! Round 1 status: **API skeleton only**. The forward coder needs to
//! be a bit-exact inverse of the decoder's `od_ec_decode` so that
//! `decode(encode(symbol_stream)) == symbol_stream` holds for every
//! CDF the decoder ever uses. Implementing this correctly requires
//! careful pinning against the decoder's `SymbolDecoder` plus a
//! carry-out byte queue (libaom calls it `precarry_buf` /
//! `precarry_storage`) so that writing a `1` after a string of `0`
//! bytes can propagate the carry without a destructive in-place edit.
//! That work is deferred to round 2.
//!
//! For round 1 we expose the **type signatures** the round-2
//! implementation will fill in, plus a clear runtime panic if
//! callers actually try to encode anything. This keeps the encoder
//! entry-point compiling with the eventual API shape so tests and
//! call sites can be drafted in parallel with the coder.
//!
//! Spec references: §9.2 (arithmetic coder), §9.4 (CDF adaptation,
//! shared with the decoder via [`crate::symbol::update_cdf`]).

/// Tile-scoped forward range coder. Round 2 will land the
/// implementation; round 1 only fixes the API surface.
pub struct SymbolEncoder {
    allow_update: bool,
}

impl Default for SymbolEncoder {
    fn default() -> Self {
        Self::new(true)
    }
}

impl SymbolEncoder {
    pub fn new(allow_update: bool) -> Self {
        Self { allow_update }
    }

    /// Encode a single binary symbol with `P(bit = 0) = p` in Q15.
    /// Mirrors the decoder's
    /// [`crate::symbol::SymbolDecoder::decode_bool`] inverse.
    ///
    /// # Round 1
    ///
    /// Panics — round 1 ships only the API shape; the actual coder
    /// state machine (interval shrink + renormalise + carry-out byte
    /// queue) is round-2 work.
    pub fn encode_bool(&mut self, _bit: u32, _p: u32) {
        unimplemented!(
            "av1 SymbolEncoder::encode_bool — round 2 deliverable; \
             round 1 ships only the API surface so call sites can be \
             drafted in parallel"
        )
    }

    /// Encode a multi-symbol from a CDF, updating the CDF in place
    /// when `allow_update` was set.
    pub fn encode_symbol(&mut self, _cdf: &mut [u16], _symbol: u32) {
        unimplemented!(
            "av1 SymbolEncoder::encode_symbol — round 2 deliverable; \
             round 1 ships only the API surface so call sites can be \
             drafted in parallel"
        )
    }

    /// Emit `n` raw 50/50 bool bits — inverse of
    /// [`crate::symbol::SymbolDecoder::read_literal`].
    pub fn write_literal(&mut self, _value: u32, _n: u32) {
        unimplemented!("av1 SymbolEncoder::write_literal — round 2 deliverable")
    }

    /// Drain the encoder and return the bit-aligned tile payload.
    /// Mirrors libaom's `od_ec_enc_done`: emits the 15-bit `low` tail
    /// the decoder consumes during `init_symbol`.
    pub fn finish(self) -> Vec<u8> {
        unimplemented!(
            "av1 SymbolEncoder::finish — round 2 deliverable; the \
             stream tail must align with the decoder's init_symbol \
             15-bit consume"
        )
    }

    pub fn allow_update(&self) -> bool {
        self.allow_update
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The API surface compiles. Round 2 lands the actual encoder.
    #[test]
    fn api_surface_compiles() {
        let enc = SymbolEncoder::new(true);
        assert!(enc.allow_update());
    }
}
