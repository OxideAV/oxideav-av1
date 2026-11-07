// Load the bare-WebAssembly probe (no imports) and compare its digest
// with the native one it carries. A trap (a Rust panic lowers to
// `unreachable` on wasm32-unknown-unknown) fails the run.
//
//   node run.mjs target/wasm32-unknown-unknown/release/oxideav_av1_wasm_probe.wasm
import { readFileSync } from "node:fs";

const bytes = readFileSync(process.argv[2]);
const { instance } = await WebAssembly.instantiate(bytes, {});
const hex = (v) => "0x" + BigInt.asUintN(64, v).toString(16).padStart(16, "0");
const got = instance.exports.oxideav_av1_wasm_probe();
const want = instance.exports.oxideav_av1_wasm_probe_expected();
console.log(`wasm digest ${hex(got)}, native ${hex(want)}`);
if (BigInt.asUintN(64, got) !== BigInt.asUintN(64, want)) {
  console.error("digest mismatch");
  process.exit(1);
}
