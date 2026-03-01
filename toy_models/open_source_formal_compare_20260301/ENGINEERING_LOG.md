# Engineering Log (Open-Source Formal Compare)

## 2026-03-01

- Built and installed a fully open-source local stack under `tools/install`:
  - `yosys` (from source, plugin-capable)
  - `sby` (SymbiYosys)
  - `eqy`
- Initial Yosys build failed on missing `ffi.h`.
  - Root cause: missing `libffi-devel`.
  - Fix: installed `libffi-devel` and related dev packages (`readline-devel`, `tcl-devel`, `libedit-devel`).
- Confirmed runtime plugin loading works with local Yosys.
  - `plugin -i eqy_combine`, `eqy_partition`, `eqy_recode` all load.
- EQY failure was initially not a runtime plugin issue in the local stack.
  - Root cause: toy config top-name mismatch (`modA` vs `modB`).
  - Fix: `rename modA top` and `rename modB top` in `lec_simple.eqy`.
  - Result: EQY PASS on toy LEC.
- SymbiYosys still fails on concurrent SVA forms using `assert property (@(posedge clk) ...)` in tested toy cases.
  - Error signature: `syntax error, unexpected '@'`.
  - This remained after switching from yowasp binaries to locally built open-source Yosys/SBY.
- OpenTitan AES checker parse in Yosys remains blocked.
  - Error signature in `top_earlgrey.sv`: `unexpected TOK_PACKAGESEP`.
  - Indicates unsupported SystemVerilog/package construct in this flow configuration.
- CIRCT remains ahead on tested tasks:
  - BMC concurrent property case passes in CIRCT but fails in SBY/Yosys parser.
  - OpenTitan AES connectivity LEC has a known real-Z3 PASS in CIRCT artifacts.
