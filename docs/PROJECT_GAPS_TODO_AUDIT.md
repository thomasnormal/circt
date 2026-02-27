# Project Gaps / TODO Audit

Generated: 2026-02-27 09:48:44 UTC

Scope: `include lib tools frontends unittests test utils cmake CMakeLists.txt`

Patterns:
- TODO-like: `TODO|FIXME|XXX|TBD|NYI|WIP`
- Gap-like: `unsupported|unimplemented|not implemented|not yet implemented|cannot yet|currently unsupported`

## Summary

- Total matches (union): 2071
- TODO/FIXME/etc matches: 954
- unsupported/unimplemented/etc matches: 1121

### Matches By Top-Level Path

```text
   1158 lib
    512 test
    171 tools
     83 utils
     76 include
     36 unittests
     35 frontends
```

### Top Files By Match Count

```text
     87 lib/Conversion/ExportVerilog/ExportVerilog.cpp
     83 test/Analysis/firrtl-test-instance-info.mlir
     69 tools/circt-sim-compile/circt-sim-compile.cpp
     60 lib/Conversion/ImportVerilog/Structure.cpp
     60 lib/Conversion/ImportVerilog/AssertionExpr.cpp
     49 lib/Conversion/ImportVerilog/CrossSelect.cpp
     45 lib/Conversion/ImportVerilog/Expressions.cpp
     35 lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp
     34 lib/Conversion/MooreToCore/MooreToCore.cpp
     25 lib/Conversion/ImportVerilog/Statements.cpp
     22 unittests/Support/PrettyPrinterTest.cpp
     22 lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh
     20 lib/Conversion/VerifToSMT/VerifToSMT.cpp
     19 lib/Dialect/FIRRTL/Import/FIRParser.cpp
     18 tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp
     18 lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp
     16 lib/Runtime/uvm-core/src/reg/uvm_vreg.svh
     16 lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp
     16 lib/Conversion/FIRRTLToHW/LowerToHW.cpp
     15 lib/Dialect/FIRRTL/FIRRTLOps.cpp
     14 utils/run_yosys_sva_circt_bmc.sh
     14 test/Dialect/LLHD/Transforms/unroll-loops.mlir
     14 test/Dialect/FIRRTL/parse-errors.fir
     14 lib/Conversion/ImportVerilog/TimingControls.cpp
     12 utils/run_formal_all.sh
     12 lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp
     12 lib/Dialect/FIRRTL/FIRRTLFolds.cpp
     12 include/circt/Dialect/Sim/SimOps.td
     11 utils/refactor_continue.sh
     11 test/Tools/summarize-circt-sim-jit-reports-policy.test
     11 lib/Dialect/Synth/Transforms/CutRewriter.cpp
     11 lib/Dialect/FIRRTL/Export/FIREmitter.cpp
     10 tools/circt-test/circt-test.cpp
     10 tools/circt-bmc/circt-bmc.cpp
     10 test/Tools/summarize-circt-sim-jit-reports.test
     10 frontends/PyCDE/src/pycde/bsp/common.py
      9 test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv
      9 lib/Runtime/uvm-core/src/base/uvm_phase.svh
      9 lib/Runtime/uvm-core/src/base/uvm_component.svh
      9 lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp
      8 tools/arcilator/BehavioralLowering.cpp
      8 lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp
      8 lib/Dialect/ESI/runtime/python/esiaccel/types.py
      8 lib/Conversion/CombToSynth/CombToSynth.cpp
      8 lib/Analysis/TestPasses.cpp
      7 tools/circt-sim/LLHDProcessInterpreter.cpp
      7 test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test
      7 test/Tools/run-avip-circt-sim-jit-policy-gate.test
      7 lib/Dialect/SystemC/SystemCOps.cpp
      7 lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp
      7 lib/Dialect/FIRRTL/FIRRTLUtils.cpp
      7 lib/Dialect/Comb/CombFolds.cpp
      7 lib/Conversion/SCFToCalyx/SCFToCalyx.cpp
      6 tools/circt-sim/LLHDProcessInterpreterBytecode.cpp
      6 test/Target/ExportSystemC/basic.mlir
      6 test/Dialect/FIRRTL/errors.mlir
      6 test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv
      6 lib/Runtime/uvm-core/src/reg/uvm_vreg_field.svh
      6 lib/Firtool/Firtool.cpp
      6 lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp
      6 lib/Conversion/HandshakeToHW/HandshakeToHW.cpp
      6 lib/Analysis/FIRRTLInstanceInfo.cpp
      6 include/circt/Dialect/Arc/ArcOps.td
      5 utils/run_mutation_mcy_examples.sh
      5 unittests/Support/FVIntTest.cpp
      5 test/firtool/spec/refs/define.fir
      5 test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv
      5 test/Conversion/ExportVerilog/pretty.mlir
      5 test/Conversion/ExportVerilog/name-legalize.mlir
      5 lib/Runtime/MooreRuntime.cpp
      5 lib/Dialect/Synth/Transforms/TechMapper.cpp
      5 lib/Dialect/SSP/Transforms/Schedule.cpp
      5 lib/Dialect/LLHD/Transforms/UnrollLoops.cpp
      5 lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp
      5 lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp
      5 lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp
      5 lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp
      5 lib/Dialect/FIRRTL/FIRRTLTypes.cpp
      5 lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp
      5 lib/Dialect/Arc/Transforms/LowerState.cpp
      5 lib/Dialect/Arc/Transforms/IsolateClocks.cpp
      5 lib/Conversion/HWToSMT/HWToSMT.cpp
      5 lib/Conversion/HWToBTOR2/HWToBTOR2.cpp
      5 lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp
      5 include/circt/Analysis/FIRRTLInstanceInfo.h
      4 utils/mutation_mcy/lib/drift.sh
      4 unittests/Support/TestReportingTest.cpp
      4 tools/domaintool/domaintool.cpp
      4 tools/domaintool/ClockSpecJSONHandler.cpp
      4 tools/circt-verilog/circt-verilog.cpp
      4 tools/circt-sim/LLHDProcessInterpreterTrace.cpp
      4 tools/circt-sim/LLHDProcessInterpreterStorePatterns.h
      4 tools/circt-sim/LLHDProcessInterpreter.h
      4 tools/circt-mut/circt-mut.cpp
      4 test/firtool/dedup-modules-with-output-dirs.fir
      4 test/Dialect/Moore/canonicalizers.mlir
      4 test/Dialect/Handshake/errors.mlir
      4 test/Dialect/HW/parameters.mlir
      4 test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv
      4 lib/Tools/circt-bmc/LowerToBMC.cpp
```

## Detailed List

### TODO/FIXME/XXX/TBD/NYI/WIP

```text
test/CMakeLists.txt:53:  # TODO: circt-verilog-lsp-server is disabled due to slang API compatibility issues
unittests/Analysis/CMakeLists.txt:1:# TODO: Linting needs slang header fixes
frontends/PyRTG/src/pyrtg/control_flow.py:92:    # FIXME: this is very ugly because the MLIR python bindings do now allow us
frontends/PyRTG/src/pyrtg/control_flow.py:174:    # TODO: probably better to delete the variable
utils/run_mutation_mcy_examples.sh:1006:  tmp_values_file="$(mktemp "${OUT_DIR}/example-history-values.XXXXXX")"
utils/run_mutation_mcy_examples.sh:2379:    history_detected_file="$(mktemp "${OUT_DIR}/suite-history-detected.XXXXXX")"
utils/run_mutation_mcy_examples.sh:2380:    history_relevant_file="$(mktemp "${OUT_DIR}/suite-history-relevant.XXXXXX")"
utils/run_mutation_mcy_examples.sh:2381:    history_coverage_file="$(mktemp "${OUT_DIR}/suite-history-coverage.XXXXXX")"
utils/run_mutation_mcy_examples.sh:2382:    history_errors_file="$(mktemp "${OUT_DIR}/suite-history-errors.XXXXXX")"
include/circt/Runtime/MooreRuntime.h:4208:/// @param phase The phase being executed (for passing to super.XXX_phase())
frontends/PyRTG/src/pyrtg/contexts.py:73:    # TODO: just adding all variables in the context is not particularly nice.
frontends/PyRTG/src/pyrtg/contexts.py:87:    # TODO: we currently just assume "_context_seq" is a reserved prefix, would
lib/Runtime/MooreRuntime.cpp:2481:  // TODO: Implement proper simulation-aware waiting when a simulation
lib/Runtime/MooreRuntime.cpp:12227:      // TODO: Handle array element access via index calculation
lib/Runtime/MooreRuntime.cpp:14402:/// 1. Creates the test component using the UVM factory (TODO)
lib/Runtime/MooreRuntime.cpp:17781:    // TODO: Integrate with HDL access functions if HDL path is set
lib/Runtime/MooreRuntime.cpp:17823:    // TODO: Integrate with HDL access functions if HDL path is set
frontends/PyCDE/test/test_esi.py:170:# TODO: fixme
lib/Runtime/uvm-core/src/tlm2/uvm_tlm_time.svh:105:      // ToDo: Check resolution
test/firtool/spec/refs/read_subelement_add.fir:9:    wire x : {a: UInt<5>, b: UInt<2>} ; XXX: ADDED
test/firtool/spec/refs/read_subelement_add.fir:10:    invalidate x ; XXX: ADDED
test/firtool/spec/refs/read_subelement_add.fir:11:    define p = probe(x) ; XXX: ADDED
frontends/PyCDE/test/test_polynomial.py:120:# TODO: before generating all the modules, the IR doesn't verify since the
test/firtool/spec/refs/read_subelement.fir:10:    wire x : {a: UInt<5>, b: UInt<2>} ; XXX: ADDED
test/firtool/spec/refs/read_subelement.fir:11:    invalidate x ; XXX: ADDED
test/firtool/spec/refs/read_subelement.fir:12:    define p = probe(x) ; XXX: ADDED
test/firtool/spec/refs/read.fir:10:    wire x : UInt<4> ; XXX: ADDED
test/firtool/spec/refs/read.fir:11:    invalidate x ; XXX: ADDED
test/firtool/spec/refs/read.fir:12:    define p = probe(x) ; XXX: ADDED
test/firtool/spec/refs/probe_export_simple.fir:7:    input in: UInt<5> ; XXX: Added width.
test/firtool/spec/refs/nosubaccess.fir:8:  ; XXX: Modified to not use input probes, get probe from ext, widths.
frontends/PyCDE/test/test_instances.py:124:# TODO: Add back physical region support
frontends/PyCDE/test/test_instances.py:156:# TODO: add back anonymous reservations
tools/circt-sim/AOTProcessCompiler.cpp:1372:      // TODO: extract edge kind from wait attributes (posedge/negedge).
tools/circt-sim/LLHDProcessInterpreterStorePatterns.h:190:  auto thenYield =
tools/circt-sim/LLHDProcessInterpreterStorePatterns.h:194:  if (!thenYield || !elseYield || thenYield.getNumOperands() != 1 ||
tools/circt-sim/LLHDProcessInterpreterStorePatterns.h:199:  if (!matchFourStateStructCreateLoad(thenYield.getOperand(0), resolveAddr,
tools/circt-sim/LLHDProcessInterpreterStorePatterns.h:201:      !matchFourStateCopyStore(thenYield.getOperand(0), resolveAddr, srcAddr))
test/firtool/spec/refs/forwarding_refs_upwards.fir:6:  extmodule Foo : ; XXX: module -> extmodule
test/firtool/spec/refs/forwarding_refs_upwards.fir:7:    output p : Probe<UInt<3>> ; XXX: added width
test/firtool/spec/refs/force_initial.fir:27:    ; XXX: modified, workaround inability create non-const node w/literal initializer.
test/firtool/spec/refs/force_and_release.fir:26:    ; XXX: modified, workaround inability create non-const node w/literal initializer.
test/firtool/spec/refs/define.fir:8:    input p : {x: UInt<1>, flip y : UInt<3>} ; XXX: modified, for init
test/firtool/spec/refs/define.fir:11:    output c : Probe<UInt> ; read-only ref. to register 'r', inferred width. ; XXX: modified, needs width
test/firtool/spec/refs/define.fir:14:    connect p.y, UInt<3>(0) ; XXX: modified, for init
test/firtool/spec/refs/define.fir:16:    node q = p.x ; XXX: modified, workaround inability create non-const node w/literal initializer.
test/firtool/spec/refs/define.fir:19:    connect r, p.x ; XXX: modified, initialize register
test/firtool/spec/refs/define-subelement.fir:7:    input x : UInt<3> ; XXX: width added
test/firtool/spec/refs/define-flip-to-passive.fir:7:    input x : {a: UInt<3>, flip b: UInt} ; XXX: width on x.a
test/firtool/spec/refs/define-flip-to-passive.fir:8:    output y : {a: UInt, flip b: UInt<3>} ; XXX: width on y.b
unittests/Support/PrettyPrinterTest.cpp:75:                 "float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"};
unittests/Support/PrettyPrinterTest.cpp:204:        float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:237:              float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:247:      float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:279:             float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:285:  float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:304:        float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:317:              float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:321:      float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:335:             float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:339:  float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:350:foooooo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx);
unittests/Support/PrettyPrinterTest.cpp:356:baroo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, barooga(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx), int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx);
unittests/Support/PrettyPrinterTest.cpp:362:wahoo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, yahooooooo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx), int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx);
unittests/Support/PrettyPrinterTest.cpp:671:      StringToken("xxxxxxxxxxxxxxx"), BreakToken(),
unittests/Support/PrettyPrinterTest.cpp:690:xxxxxxxxxxxxxxx yyyyyyyyyyyyyyy
unittests/Support/PrettyPrinterTest.cpp:697:>>xxxxxxxxxxxxxxx yyyyyyyyyyyyyyy
unittests/Support/PrettyPrinterTest.cpp:702:xxxxxxxxxxxxxxx yyyyyyyyyyyyyyy
unittests/Support/PrettyPrinterTest.cpp:709:>>>>>>xxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:716:>>>>>>xxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:723:>>>   xxxxxxxxxxxxxxx
unittests/Support/PrettyPrinterTest.cpp:730:>>>   xxxxxxxxxxxxxxx
unittests/Support/FVIntTest.cpp:26:  ASSERT_EQ(FVInt::fromString("X1", 2).sext(5), FVInt::fromString("XXXX1", 2));
unittests/Support/FVIntTest.cpp:104:  auto b = FVInt::fromString("00001111XXXXZZZZ", 2);
unittests/Support/FVIntTest.cpp:108:  ASSERT_EQ(a & b, FVInt::fromString("000001XX0XXX0XXX", 2));
unittests/Support/FVIntTest.cpp:109:  ASSERT_EQ(a | b, FVInt::fromString("01XX1111X1XXX1XX", 2));
unittests/Support/FVIntTest.cpp:110:  ASSERT_EQ(a ^ b, FVInt::fromString("01XX10XXXXXXXXXX", 2));
include/circt/Support/TruthTable.h:37:/// TODO: Rename this to MultiOutputTruthTable since for single-output functions
include/circt/Support/TruthTable.h:126:  /// FIXME: Currently we are using exact canonicalization which doesn't scale
test/firtool/dedup-modules-with-output-dirs.fir:7:    "dirname": "XXX",
test/firtool/dedup-modules-with-output-dirs.fir:18:  ; CHECK: FILE "XXX{{/|\\}}A.sv"
test/firtool/dedup-modules-with-output-dirs.fir:72:    "dirname": "ZZZ/XXX",
test/firtool/dedup-modules-with-output-dirs.fir:95:  ; CHECK: FILE "ZZZ{{/|\\}}XXX{{/|\\}}A.sv"
lib/Runtime/uvm-core/src/seq/uvm_sequencer_base.svh:1149:    // TODO:
frontends/PyCDE/src/pycde/types.py:576:    # TODO: Adapt this to UInt and SInt.
frontends/PyCDE/src/pycde/system.py:122:  # TODO: Ideally, we'd be able to run the cf-to-handshake lowering passes in
frontends/PyCDE/src/pycde/system.py:249:      # TODO: handle symbolrefs pointing to potentially renamed symbols.
frontends/PyCDE/src/pycde/system.py:574:    # TODO: This was broken by MLIR. When we have a replacement, use that instead.
tools/circt-sim/LLHDProcessInterpreterTrace.cpp:3011:void LLHDProcessInterpreter::maybeTraceJoinAnyImmediate(ProcessId procId,
tools/circt-sim/LLHDProcessInterpreterTrace.cpp:3016:  static unsigned joinAnyImmDiagCount = 0;
tools/circt-sim/LLHDProcessInterpreterTrace.cpp:3017:  if (joinAnyImmDiagCount >= 30)
tools/circt-sim/LLHDProcessInterpreterTrace.cpp:3019:  ++joinAnyImmDiagCount;
frontends/PyCDE/src/pycde/ndarray.py:221:    # Todo: We should allow for 1 extra dimension in the access slice which is
frontends/PyCDE/src/pycde/instance.py:168:    # TODO: make these weak refs
test/firtool/firtool.fir:77:    ; TODO: This test should be rewritten to not be so brittle around module
unittests/Dialect/Sim/DPIRuntimeTest.cpp:84:  EXPECT_DOUBLE_EQ(intVal.toDouble(), 42.0);
unittests/Dialect/Sim/DPIRuntimeTest.cpp:87:  EXPECT_DOUBLE_EQ(realVal.toDouble(), 3.14);
include/circt/Support/LLVM.h:193:// TODO: It is better to use `NOLINTBEGIN/END` comments to disable clang-tidy
frontends/PyCDE/src/pycde/esi.py:95:    # TODO: figure out a way to verify the ports during this call.
frontends/PyCDE/src/pycde/esi.py:1141:    # TODO: implement some fairness.
tools/circt-sim/LLHDProcessInterpreter.h:2337:  void maybeTraceJoinAnyImmediate(ProcessId procId, ForkId forkId) const;
utils/run_formal_all.sh:11:    run_formal_all_snapshot="$(mktemp "${TMPDIR:-/tmp}/run_formal_all.XXXXXX.sh")"
lib/Runtime/uvm-core/src/reg/uvm_vreg_field.svh:838://TODO - add fatal messages
frontends/PyCDE/src/pycde/devicedb.py:31:    # TODO: Once we get into non-zero num primitives, this needs to be updated.
frontends/PyCDE/src/pycde/constructs.py:42:      # TODO: We assume here that names are unique within a module, which isn't
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:1267://TODO - add fatal messages
lib/Runtime/uvm-core/src/reg/uvm_reg_predictor.svh:135:  // TODO:  Is it better to replace this with:
lib/Runtime/uvm-core/src/reg/uvm_reg_predictor.svh:174:     // ToDo: Add memory look-up and call <uvm_mem::XsampleX()>
lib/Runtime/uvm-core/src/reg/uvm_reg_predictor.svh:203:         // TODO: what to do with subsequent collisions?
lib/Runtime/uvm-core/src/reg/uvm_reg_model.svh:355://|       |A|xxx|      B      |xxx|   C   |
frontends/PyCDE/src/pycde/bsp/common.py:330:  a response, the MMIO service will hang. TODO: add some kind of timeout.
frontends/PyCDE/src/pycde/bsp/common.py:356:  # TODO: make the amount of register space each client gets a parameter.
frontends/PyCDE/src/pycde/bsp/common.py:359:  # TODO: only supports one outstanding transaction at a time. This is NOT
frontends/PyCDE/src/pycde/bsp/common.py:530:    # transaction in flight at once. TODO: enforce this or make it more robust.
frontends/PyCDE/src/pycde/bsp/common.py:722:      # TODO: Implement tag-rewriting.
frontends/PyCDE/src/pycde/bsp/common.py:743:        # TODO: Should responses come back out-of-order (interleaved tags),
frontends/PyCDE/src/pycde/bsp/common.py:747:        # identifier. TODO: Implement the gating logic here.
frontends/PyCDE/src/pycde/bsp/common.py:765:                # TODO: Change this once we support tag-rewriting.
frontends/PyCDE/src/pycde/bsp/common.py:775:      # TODO: Don't release a request until the client is ready to accept
frontends/PyCDE/src/pycde/bsp/common.py:1015:      # TODO: re-write the tags and store the client and client tag.
lib/Runtime/uvm-core/src/reg/uvm_reg_map.svh:1987:        // TODO: need to test for right trans type, if not put back in q
lib/Runtime/uvm-core/src/reg/uvm_reg_map.svh:2126:      // TODO rewrite
frontends/PyCDE/src/pycde/bsp/xrt.py:351:      # TODO: Test reads > HostMemDataWidthBytes. I'm pretty sure this is wrong
frontends/PyCDE/src/pycde/bsp/xrt.py:397:      # TODO: Single writes only support HostMemDataWidthBytes bytes. Fine for
lib/Runtime/uvm-core/src/reg/uvm_reg_item.svh:104:  // TODO: parameterize
frontends/PyCDE/src/CMakeLists.txt:101:# TODO: this won't work if ESIPrimitives has multiple source files. Figure out
lib/Runtime/uvm-core/src/reg/uvm_reg_field.svh:1612:       // ToDo: Call parent.XsampleX();
lib/Runtime/uvm-core/src/reg/uvm_reg_field.svh:1737:       // ToDo: Call parent.XsampleX();
lib/Runtime/uvm-core/src/reg/uvm_reg_block.svh:1898:   /* ToDo: Call XsampleX in the parent block
lib/Runtime/uvm-core/src/reg/uvm_reg_block.svh:2158:   // TODO
lib/Runtime/uvm-core/src/reg/uvm_reg_block.svh:2165:   // TODO
lib/Runtime/uvm-core/src/reg/uvm_reg_block.svh:2631:`ifdef TODO
frontends/PyCDE/integration_test/esi_test.py:78:    # TODO: Fix snoop_xact to work with unconumed channels.
lib/Runtime/uvm-core/src/reg/uvm_reg.svh:1153:  /* ToDo: remove register from previous parent
utils/refactor_continue.sh:6:TODO_PATH="docs/WHOLE_PROJECT_REFACTOR_TODO.md"
utils/refactor_continue.sh:14:  --status           Show concise plan/todo continuation status (default)
utils/refactor_continue.sh:18:  --todo PATH        Override todo file path
utils/refactor_continue.sh:41:    --todo)
utils/refactor_continue.sh:42:      TODO_PATH="$2"
utils/refactor_continue.sh:61:if [[ ! -f "$TODO_PATH" ]]; then
utils/refactor_continue.sh:62:  echo "todo file not found: $TODO_PATH" >&2
utils/refactor_continue.sh:66:CANONICAL_PROMPT="continue according to ${PLAN_PATH}, keeping track of progress using ${TODO_PATH}"
utils/refactor_continue.sh:81:  ' "$TODO_PATH"
utils/refactor_continue.sh:92:  ' "$TODO_PATH"
utils/refactor_continue.sh:122:printf 'TODO: %s\n' "$TODO_PATH"
frontends/PyCDE/integration_test/test_software/esi_ram.py:23:# TODO: I broke this. Need to fix it.
include/circt/Dialect/Verif/VerifOps.td:213:  // TODO: initial values should eventually be handled by init region
include/circt/Dialect/Verif/VerifOpInterfaces.h:1://===- VerifOpInterfaces.h - TODO ---------------===//
include/circt/Dialect/SystemC/SystemCTypes.td:115:def AnySystemCInteger : AnyTypeOf<[AnyInteger, ValueBaseType]>;
lib/Runtime/uvm-core/src/macros/uvm_phase_defines.svh:41:// Also, they declare classes (uvm_XXXXX_phase) and singleton instances (XXXXX_ph)
lib/Runtime/uvm-core/src/macros/uvm_phase_defines.svh:47:// The uvm_user_xxx_phase() macros are provided for your convenience.
tools/circt-sim/LLHDProcessInterpreter.cpp:27568:    maybeTraceJoinAnyImmediate(procId, forkId);
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:805:      `uvm_warning("UVM/FIELDS/NO_FLAG",{"Field macro for ARG uses FLAG without or'ing any explicit UVM_xxx actions. ",behavior}) \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:882:        /* TODO if(local_success__ && printing matches) */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:935:          /* TODO if(local_success__ && printing matches) */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:979:              /* TODO if(local_success__ && printing matches) */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1026:              /* TODO if(local_success__ && printing matches) */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1068:              /* TODO if(local_success__ && printing matches) */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1161:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1235:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1316:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1387:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1484:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1577:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1688:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1775:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:1985:          /* TODO if(local_success__ && printing matches) */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:2098:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:2143:      /* TODO */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:2157:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:2372:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:2429:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:2477:          /* TODO: Non-decimal indexes */ \
lib/Runtime/uvm-core/src/macros/uvm_object_defines.svh:2539:          /* TODO: Non-decimal indexes */ \
utils/mutation_mcy/lib/drift.sh:646:    history_detected_file="$(mktemp "${OUT_DIR}/suite-history-detected.XXXXXX")"
utils/mutation_mcy/lib/drift.sh:647:    history_relevant_file="$(mktemp "${OUT_DIR}/suite-history-relevant.XXXXXX")"
utils/mutation_mcy/lib/drift.sh:648:    history_coverage_file="$(mktemp "${OUT_DIR}/suite-history-coverage.XXXXXX")"
utils/mutation_mcy/lib/drift.sh:649:    history_errors_file="$(mktemp "${OUT_DIR}/suite-history-errors.XXXXXX")"
tools/firtool/firtool.cpp:649:    // TODO: Serialize others.
include/circt/Dialect/Synth/SynthOps.td:102:  // TODO: Restrict to HWIntegerType.
utils/run_avip_circt_verilog.sh:157:        synth_filelist="$(mktemp /tmp/avip-synth-filelist.XXXXXX.f)"
utils/run_avip_circt_verilog.sh:184:    synth_filelist="$(mktemp /tmp/avip-synth-filelist.XXXXXX.f)"
lib/Runtime/uvm-core/src/dpi/uvm_hdl_xcelium.c:264:	// FIXME
lib/Runtime/uvm-core/src/dpi/uvm_hdl_vcs.c:615:    //TODO:: furture implementation for pure verilog
tools/domaintool/domaintool.cpp:92:/// TODO: Improve this format to be something less brittle.
tools/domaintool/domaintool.cpp:189:  // TODO: This is brittle and relies on the lowering of FIRRTL classes to
tools/domaintool/domaintool.cpp:327:    // TODO: Improve the structural typing here in favor of something stricter,
tools/domaintool/domaintool.cpp:429:  // TODO: Implement multi-file output.
tools/hlstool/hlstool.cpp:355:      // TODO: We assert without a canonicalizer pass here. Debug.
tools/hlstool/hlstool.cpp:395:  // XXX(rachitnigam): Duplicated from doHLSFlowDynamic. We should probably
tools/domaintool/Handler.h:47:  /// TODO: Figure out how to make this make sense for handlers which want to
tools/domaintool/ClockSpecJSONHandler.cpp:94:            // TODO: Add checks that path is empty.
tools/domaintool/ClockSpecJSONHandler.cpp:113:            // TODO: Add checks that path is empty.
tools/domaintool/ClockSpecJSONHandler.cpp:140:          // TODO: Add checks that path is empty.
tools/domaintool/ClockSpecJSONHandler.cpp:167:              // TODO: Implement this.
tools/arcilator/arcilator.cpp:906:        // TODO: This should probably be checking if DLTI is set on module.
tools/handshake-runner/Simulation.cpp:144:    out << any_cast<APFloat>(value).convertToDouble();
lib/Firtool/Firtool.cpp:59:  // TODO: Ensure instance graph and other passes can handle instance choice
lib/Firtool/Firtool.cpp:69:  // TODO: This pass should be deleted.
lib/Firtool/Firtool.cpp:183:  // TODO: This pass should be deleted along with InjectDUTHierarchy.
lib/Firtool/Firtool.cpp:250:  // TODO: Improve LowerLayers to avoid the need for canonicalization. See:
lib/Firtool/Firtool.cpp:800:  // TODO: Change this default to 'true' once this has been better tested and
lib/Transforms/FlattenMemRefs.cpp:387:// TODO: This is also possible for dynamically shaped memories.
include/circt/Dialect/Sim/SimOps.td:189:  let arguments = (ins AnyInteger:$value,
include/circt/Dialect/Sim/SimOps.td:224:  let arguments = (ins AnyInteger:$value,
include/circt/Dialect/Sim/SimOps.td:256:  let arguments = (ins AnyInteger:$value,
include/circt/Dialect/Sim/SimOps.td:382:  let arguments = (ins AnyInteger:$value,
include/circt/Dialect/Sim/SimOps.td:438:  let arguments = (ins AnyInteger:$value);
include/circt/Dialect/Sim/SimOps.td:967:  let results = (outs AnyInteger:$time);
include/circt/Dialect/Sim/SimOps.td:1439:  let arguments = (ins AnyInteger:$delay);
include/circt/Dialect/Sim/SimOps.td:1562:  let arguments = (ins AnyInteger:$keyCount);
include/circt/Dialect/Sim/SimOps.td:1589:    Optional<AnyInteger>:$keyCount
include/circt/Dialect/Sim/SimOps.td:1616:    Optional<AnyInteger>:$keyCount
include/circt/Dialect/Sim/SimOps.td:1645:    Optional<AnyInteger>:$keyCount
include/circt/Dialect/Sim/SimOps.td:1818:  let results = (outs AnyInteger:$count);
lib/Tools/circt-debug/DebugSession.cpp:1207:  // TODO: Set up Ctrl+C handling
include/circt/Dialect/Sim/DPIRuntime.h:225:  double toDouble() const {
lib/CAPI/Dialect/FIRRTL.cpp:327:  // FIXME: The `reinterpret_cast` here may voilate strict aliasing rule. Is
lib/Runtime/uvm-core/src/comps/uvm_driver.svh:62:  // TODO: Would it be useful to change this to:
lib/CAPI/Dialect/CMakeLists.txt:1:# TODO: Make the check source feature optional as an argument on *_add_library.
lib/Runtime/uvm-core/src/comps/uvm_agent.svh:54:  // TODO: Make ~is_active~ a field via field utils
lib/CAPI/Dialect/OM.cpp:261:/// TODO: This can be removed.
test/Tools/circt-sim/format-radix-fourstate-compact-rules.sv:18:    f = 7'b0zxxx0z;
test/Tools/circt-sim/format-radix-fourstate-compact-rules.sv:19:    g = 8'bxxxxxxxx;
include/circt/Dialect/Handshake/HandshakeOps.td:339:    @todo: How to support different init types? these have to be stored (and
lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogDocument.cpp:65:// TODO: Re-enable when CIRCTLinting library is available
lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogDocument.cpp:3252:        std::string moduleStub = "\n// TODO: Implement module\n"
lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/CMakeLists.txt:23:  # TODO: CIRCTLinting needs slang header fixes
lib/Dialect/LLHD/Transforms/Deseq.cpp:796:        // TODO: This should probably also check non-i1 values to see if they
lib/Dialect/LLHD/Transforms/Deseq.cpp:824:  // TODO: Reject values that depend on the triggers.
lib/Runtime/uvm-core/src/base/uvm_root.svh:1280:// TBD this looks wrong - taking advantage of uvm_root not doing anything else?
lib/Runtime/uvm-core/src/base/uvm_root.svh:1281:// TBD move to phase_started callback?
lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:86:    // TODO: This implementation does not handle expanded MACROs. Return
lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:449:  // TODO: Consider supporting other location format. This is currently
lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:472:  // TODO: Currently doesn't handle expanded macros
lib/Dialect/LLHD/Transforms/RemoveControlFlow.cpp:108:/// TODO: This eagerly aggregates all control flow decisions. It may be more
lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:45:    // TODO: instead of hardcoding the verilated module's class name, it should
lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:125:    // TODO: this has to be changed to a systemc::CallIndirectOp once the PR is
include/circt/Dialect/HWArith/HWArithOps.td:170:  let arguments = (ins AnyInteger:$in);
include/circt/Dialect/HWArith/HWArithOps.td:171:  let results = (outs AnyInteger:$out);
include/circt/Dialect/SV/SVStatements.td:68:  // TODO: ODS forces using a custom builder just to get the region terminator
include/circt/Dialect/SV/SVStatements.td:124:  // TODO: ODS forces using a custom builder just to get the region terminator
include/circt/Dialect/SV/SVStatements.td:169:  // TODO: ODS forces using a custom builder just to get the region terminator
include/circt/Dialect/SV/SVStatements.td:335:  // TODO: ODS forces using a custom builder just to get the region terminator
lib/Dialect/SystemC/SystemCOps.cpp:746:// TODO: The implementation for this operation was copy-pasted from the
lib/Dialect/SystemC/SystemCOps.cpp:751:// FIXME: This is an exact copy from upstream
lib/Dialect/SystemC/SystemCOps.cpp:816:// TODO: Most of the implementation for this operation was copy-pasted from the
lib/Dialect/SystemC/SystemCOps.cpp:878:  // FIXME: below is an exact copy of the
lib/Dialect/SystemC/SystemCOps.cpp:981:  // FIXME: inlined mlir::function_interface_impl::printFunctionOp because we
lib/Dialect/SystemC/SystemCOps.cpp:1012:// FIXME: the below clone operation are exact copies from upstream.
lib/Dialect/SystemC/SystemCOps.cpp:1123:// TODO: The implementation for this operation was copy-pasted from the
lib/Runtime/uvm-core/src/base/uvm_resource_base.svh:554:     // if dbg is null, we never called record_xxx_access and so there is nothing to print
lib/Bindings/Tcl/circt_tcl.cpp:67:    // TODO
lib/Dialect/LLHD/Transforms/InlineCalls.cpp:246:  // TODO: Compute actual offsets from class metadata or pass them as
include/circt/Dialect/SV/SVTypes.td:18:// TODO: The following should go into a `SVTypesImpl.td` if we ever actually
lib/Dialect/LLHD/Transforms/HoistSignals.cpp:652:          // TODO: This should probably create something like a `llhd.dontcare`.
lib/Bindings/Python/support.py:281:      # TODO: Make this use `UnconnectedSignalError`.
lib/Dialect/Synth/Transforms/CutRewriter.cpp:62:  // TODO: Extend this to allow comb.and/xor/or as well.
lib/Dialect/Synth/Transforms/CutRewriter.cpp:174:  // TODO: Currently numOutputs is always 1, so we can just return the first
lib/Dialect/Synth/Transforms/CutRewriter.cpp:343:  // TODO: Merge-sort `operations` and `other.operations` by operation index
lib/Dialect/Synth/Transforms/CutRewriter.cpp:402:  // TODO: Sort the inputs by their defining operation.
lib/Dialect/Synth/Transforms/CutRewriter.cpp:403:  // TODO: Update area and delay based on the merged cuts.
lib/Dialect/Synth/Transforms/CutRewriter.cpp:552:  // TODO: Use a priority queue instead of sorting for better performance.
lib/Dialect/Synth/Transforms/CutRewriter.cpp:684:  // TODO: Variadic operations and non-single-bit results can be supported
lib/Dialect/Synth/Transforms/CutRewriter.cpp:887:  // TODO: This must be removed when we support multiple outputs.
lib/Dialect/Synth/Transforms/CutRewriter.cpp:957:      // TODO: This doesn't consider a global delay. Need to capture
lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:122:    // FIXME: The following is very small compared to the default value of ABC
lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:139:  // TODO: Add balancing, rewriting, FRAIG conversion, etc.
lib/Dialect/Synth/Transforms/TechMapper.cpp:135:    // TODO: Give a better name to the instance
lib/Dialect/Synth/Transforms/TechMapper.cpp:174:    // TODO: This attribute should be replaced with a more structured
lib/Dialect/Synth/Transforms/TechMapper.cpp:183:        // TODO: Run mapping only when the module is under the specific
lib/Dialect/Synth/Transforms/TechMapper.cpp:201:      double area = areaAttr.getValue().convertToDouble();
lib/Dialect/Synth/Transforms/TechMapper.cpp:207:          // FIXME: Currently we assume delay is given as integer attributes,
lib/Dialect/Synth/Transforms/LowerWordToBits.cpp:189:    // TODO: This is not optimal as it has a depth limit and does not check
lib/Dialect/Synth/Transforms/LowerVariadic.cpp:134:  // FIXME: Currently only top-level operations are lowered due to the lack of
lib/Tools/circt-lec/ConstructLEC.cpp:54:  // FIXME: sanity check the fetched global: do all the attributes match what
lib/Tools/circt-lec/ConstructLEC.cpp:219:    // TODO: don't use LLVM here
lib/Tools/circt-lec/ConstructLEC.cpp:231:  // TODO: we should find a more elegant way of reporting the result than
lib/Analysis/FIRRTLInstanceInfo.cpp:229:bool InstanceInfo::anyInstanceUnderDut(igraph::ModuleOpInterface op) {
lib/Analysis/FIRRTLInstanceInfo.cpp:239:bool InstanceInfo::anyInstanceUnderEffectiveDut(igraph::ModuleOpInterface op) {
lib/Analysis/FIRRTLInstanceInfo.cpp:240:  return !hasDut() || anyInstanceUnderDut(op);
lib/Analysis/FIRRTLInstanceInfo.cpp:247:bool InstanceInfo::anyInstanceUnderLayer(igraph::ModuleOpInterface op) {
lib/Analysis/FIRRTLInstanceInfo.cpp:257:bool InstanceInfo::anyInstanceInDesign(igraph::ModuleOpInterface op) {
lib/Analysis/FIRRTLInstanceInfo.cpp:267:bool InstanceInfo::anyInstanceInEffectiveDesign(igraph::ModuleOpInterface op) {
lib/Bindings/Python/dialects/synth.py:57:  # TODO: Associate with an MLIR value/op
lib/Dialect/Synth/Transforms/AIGERRunner.cpp:207:      // TODO: Consider caching extract operations for efficiency
include/circt/Dialect/HW/HWStructure.td:247:    module name in Verilog we can use.  TODO: This is a hack because we don't
lib/Analysis/CMakeLists.txt:67:# TODO: Linting code needs slang header path fixes
lib/Dialect/Synth/SynthOps.cpp:68:  // TODO: Implement maj(x, 1, 1) = 1, maj(x, 0, 0) = 0
lib/Dialect/Synth/SynthOps.cpp:298:  // TODO: This is a naive implementation that creates a balanced binary tree.
lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:97:      // TODO: RewriterBase::replaceAllUsesWith is not currently supported by
lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:319:          // TODO: RewriterBase::replaceAllUsesWith is not currently supported
lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:359:        // TODO: RewriterBase::replaceAllUsesWith is not currently supported by
lib/Runtime/uvm-core/src/base/uvm_port_base.svh:528:    if (end_of_elaboration_ph.get_state() == UVM_PHASE_EXECUTING || // TBD tidy
lib/Runtime/uvm-core/src/base/uvm_port_base.svh:638:        end_of_elaboration_ph.get_state() == UVM_PHASE_DONE ) begin  // TBD tidy
include/circt/Dialect/HW/HWOps.h:42:/// TODO: Move all these functions to a hw::ModuleLike interface.
lib/Tools/circt-bmc/LowerToBMC.cpp:3312:    // TODO: don't use LLVM here
lib/Dialect/Kanagawa/Transforms/KanagawaPassPipelines.cpp:62:  // TODO @mortbopet: Add a verification pass to ensure that there are no more
lib/Analysis/TestPasses.cpp:238:               << "    anyInstanceUnderDut: " << iInfo.anyInstanceUnderDut(op)
lib/Analysis/TestPasses.cpp:242:               << "    anyInstanceUnderEffectiveDut: "
lib/Analysis/TestPasses.cpp:243:               << iInfo.anyInstanceUnderEffectiveDut(op) << "\n"
lib/Analysis/TestPasses.cpp:246:               << "    anyInstanceUnderLayer: "
lib/Analysis/TestPasses.cpp:247:               << iInfo.anyInstanceUnderLayer(op) << "\n"
lib/Analysis/TestPasses.cpp:250:               << "    anyInstanceInDesign: " << iInfo.anyInstanceInDesign(op)
lib/Analysis/TestPasses.cpp:254:               << "    anyInstanceInEffectiveDesign: "
lib/Analysis/TestPasses.cpp:255:               << iInfo.anyInstanceInEffectiveDesign(op) << "\n"
lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:211:      // TODO: Handle other operations.
lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:729:  // TODO: Make debug points optional.
lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1307:              // TODO: Add address.
lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1312:              // TODO: Add address.
lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:293:    // TODO: @mortbopet - this should be part of ModulePortInfo
lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:314:      // TODO: RewriterBase::replaceAllUsesWith is not currently supported by
lib/Analysis/DependenceAnalysis.cpp:169:  // TODO(mikeurbach): consider adding an inverted index to avoid this scan.
lib/Analysis/DebugInfo.cpp:165:        // TODO: What do we do with the port assignments? These should be
lib/Runtime/uvm-core/src/base/uvm_phase.svh:537:  // TBD add more useful debug
lib/Runtime/uvm-core/src/base/uvm_phase.svh:553:  local function string m_aa2string(edges_t aa); // TBD tidy
lib/Runtime/uvm-core/src/base/uvm_phase.svh:762:// TBD error checks if param nodes are actually in this schedule or not
lib/Runtime/uvm-core/src/base/uvm_phase.svh:1218:  // TODO: Seems like there should be an official way to set these values...
lib/Runtime/uvm-core/src/base/uvm_phase.svh:1383:  // TBD full search
lib/Runtime/uvm-core/src/base/uvm_phase.svh:1403:  // TBD full search
lib/Runtime/uvm-core/src/base/uvm_phase.svh:1432:  // TODO: add support for 'stay_in_scope=1' functionality
lib/Runtime/uvm-core/src/base/uvm_phase.svh:1442:  // TODO: add support for 'stay_in_scope=1' functionality
lib/Tools/arcilator/pipelines.cpp:98:  // TODO: maybe merge RemoveUnusedArcArguments with SinkInputs?
lib/Tools/arcilator/pipelines.cpp:111:  // TODO: LowerClocksToFuncsPass might not properly consider scf.if operations
lib/Tools/arcilator/pipelines.cpp:114:  // TODO: InlineArcs seems to not properly handle scf.if operations, thus the
lib/Dialect/Kanagawa/Transforms/KanagawaAddOperatorLibrary.cpp:46:// TODO @mortbopet: once we have c++20, change  this to a templated lambda to
include/circt/Dialect/RTG/Transforms/RTGPasses.td:208:    user mode tests) (TODO).
lib/Runtime/uvm-core/src/base/uvm_objection.svh:1359:// TODO: change to plusarg
lib/Dialect/ESI/ESIServices.cpp:98:  // TODO: The cosim op should probably be able to take a bundle type and get
lib/Target/ExportSystemC/Patterns/SystemCEmissionPatterns.cpp:140:    // TODO: the 'override' keyword is hardcoded here because the destructor can
lib/Runtime/uvm-core/src/base/uvm_misc.svh:331:    // TODO $countbits(value,'z) would be even better
lib/Target/ExportSystemC/Patterns/EmitCEmissionPatterns.cpp:72:      // TODO: template arguments not supported for now.
lib/Target/ExportSystemC/Patterns/EmitCEmissionPatterns.cpp:84:    // TODO: template arguments not supported for now.
lib/Dialect/Handshake/Transforms/Materialization.cpp:83:      // TODO: should we use other indicator for op that has been erased?
lib/Dialect/Sim/SimOps.cpp:97:                                   floatAttr.getValue().convertToDouble());
lib/Dialect/Sim/SimOps.cpp:100:             floatAttr.getValue().convertToDouble());
lib/Dialect/Handshake/Transforms/LockFunctions.cpp:61:  // TODO is this UB?
lib/Target/ExportSystemC/ExportSystemC.cpp:39:  // Remove invalid characters. TODO: a digit is not allowed as the first
include/circt/Dialect/RTG/IR/RTGInterfaces.td:42:    TODO: properly verify this; unfortunately, we don't have a 'verify' field
lib/Runtime/uvm-core/src/base/uvm_globals.svh:238:// TODO merge with uvm_enum_wrapper#(uvm_severity)
lib/Dialect/Handshake/HandshakeUtils.cpp:276:        // todo: change when handshake switches to i0
lib/Dialect/ESI/runtime/python/CMakeLists.txt:76:      # TODO: have the patience to make this work.
lib/Runtime/uvm-core/src/base/uvm_event.svh:466:         // \todo Remove this note when 6450 is resolved.
lib/Dialect/ESI/runtime/python/esiaccel/types.py:527:    # TODO: add a proper registration mechanism for service ports.
lib/Dialect/ESI/runtime/python/esiaccel/types.py:565:    # TODO: respect timeout
lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:80:    // TODO: Support a non-integer type.
lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:100:    // TODO: Check if function type matches.
lib/Support/TruthTable.cpp:217:  // FIXME: The time complexity is O(n! * 2^(n + m)) where n is the number
lib/Dialect/ESI/runtime/python/esiaccel/codegen.py:373:          # TODO: Bitfield layout is implementation-defined; consider
lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:113:  // TODO: Fix leaks! The one I know of is in the callback code -- if one
lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:209:      // TODO: "extra" field.
lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:373:        // TODO: Under certain conditions this will cause python to crash. I
include/circt/Dialect/FIRRTL/FIRRTLVisitors.h:442:  // Default to chaining visitUnhandledXXX to visitUnhandledOp.
lib/Support/JSON.cpp:64:        json.value(apfloat.convertToDouble());
test/Tools/run-formal-all-strict-gate-bmc-lec-contract-fingerprint-parity-defaults.test:8:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\n: "${OUT:?}"\n: "${LEC_RESOLVED_CONTRACTS_OUT:?}"\nmkdir -p "$(dirname "$OUT")"\nprintf "PASS\tcase_a\t/tests/case_a.sv\tsv-tests\tLEC\tEQ\n" > "$OUT"\nprintf "case_a\t/tests/case_a.sv\tmanifest\tsmtlib\t300\tlec\t0\t0\t\txxxx9999xxxx9999\n" > "$LEC_RESOLVED_CONTRACTS_OUT"\necho "sv-tests LEC summary: total=1 pass=1 fail=0 error=0 skip=0"\n' > %t/utils/run_sv_tests_circt_lec.sh
lib/Dialect/Seq/SeqOps.cpp:610:  // TODO: Once HW aggregate constant values are supported, move this
lib/Dialect/Seq/SeqOps.cpp:693:      // TODO: Support nested arrays and bundles.
lib/Dialect/Seq/SeqOps.cpp:743:  // TODO: Handle a preset value.
include/circt/Dialect/OM/OMAttributes.td:72:  // TODO: Use custom assembly format to infer an element type from elements.
lib/Runtime/uvm-core/src/base/uvm_component.svh:3337:    // TODO: Update this to use sort_by_precedence_q (Mantis 7354)
lib/Dialect/Seq/Transforms/LowerSeqHLMem.cpp:120:    // TODO: When latency > 2, and assuming that the exact read time is flexible
lib/Dialect/HW/Transforms/HWAggregateToComb.cpp:340:  // TODO: Add ArraySliceOp as well.
include/circt/Dialect/FIRRTL/FIRRTLTypes.td:248:// TODO: Migrate off of this by making the operands for `PrintfOp` use only
lib/Dialect/Seq/Transforms/HWMemSimImpl.cpp:652:      // TODO: This shares a lot of common logic with LowerToHW.  Combine
lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:55:  // TODO: Add support for: union, packed array, enum
lib/Support/PrettyPrinter.cpp:43://   (TODO)
lib/Dialect/HW/ModuleImplementation.cpp:157:    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
lib/Dialect/HW/ModuleImplementation.cpp:189:      // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
lib/Dialect/HW/ModuleImplementation.cpp:441:    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:57:  /// activeServices table. TODO: re-using this for the engines section is a
lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:178:    // TODO: Check or guide the conversion of the value to the type based on the
lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:290:  // TODO: support engines at lower levels.
lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:324:/// TODO: Hack. This method is a giant hack to reuse the getService method for
lib/Dialect/SV/Transforms/SVExtractTestCode.cpp:82:      // TODO: determine whether we want to recurse backward into the other
lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:422:  // TODO: Add a proper registration mechanism.
lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:435:  // TODO: Add a proper registration mechanism.
lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp:998:  // FIXME: hw.inout cannot be used in outputs.
include/circt/Dialect/FIRRTL/FIRParser.h:77:// TODO: This API is super wacky and should be streamlined to hide the
lib/Dialect/ESI/runtime/cpp/lib/backends/RpcClient.cpp:180:    // loop until it is. TODO: fix this with the DPI API change.
lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:167:    // TODO: consider breaking up array assignments into assignments
lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:184:  // TODO: generalise to ranges and arbitrary concatenations.
lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:351:    // TODO: We should use the condition used in ExportVerilog regarding
lib/Scheduling/SimplexSchedulers.cpp:924:  // TODO: Implement more sophisticated priority function.
lib/Scheduling/SimplexSchedulers.cpp:1149:        // TODO: Replace the last condition with a proper graph analysis.
lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:427:    // TODO: The types here are WRONG. They need to be wrapped in Channels! Fix
lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:489:  // location specified. TODO: check that the memory has been mapped.
lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:517:  // location specified. TODO: check that the memory has been mapped.
lib/Dialect/ESI/runtime/cpp/lib/backends/Trace.cpp:211:      // TODO: support other types.
lib/Dialect/ESI/runtime/cpp/lib/backends/RpcServer.cpp:153:  // TODO: use secure credentials. Not so bad for now since we only accept
lib/Dialect/ESI/runtime/cpp/lib/backends/RpcServer.cpp:310:    // TODO: adapt this to a new notification mechanism which is forthcoming.
lib/Dialect/ESI/Passes/ESIBuildManifest.cpp:32:// TODO: The code herein is a bit ugly, but it works. Consider cleaning it up.
lib/Dialect/ESI/ESIOps.cpp:478:      // TODO: other container types.
include/circt/Dialect/MSFT/MSFTConstructs.td:28:  // TODO: flesh out description once we've proved this op out.
lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:396:    // TODO: investigate better ways to do this. For now, just play nice with
test/Tools/run-formal-all-opentitan-connectivity-contract-parity-fail.test:7:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\n: "${OUT:?}"\n: "${LEC_RESOLVED_CONTRACTS_OUT:?}"\nprintf "PASS\\tconnectivity::chip.csv:RULE_CLK\\t%t/out/opentitan-connectivity-lec-work\\topentitan\\tCONNECTIVITY_LEC\\tEQ\\n" > "$OUT"\nprintf "#resolved_contract_schema_version=1\\nconnectivity::chip.csv:RULE_CLK\\t%t/ot/hw/top_earlgrey/formal/conn_csvs/chip.csv:2\\tmanifest\\tsmtlib\\t300\\tlec\\t0\\t0\\t\\txxxx9999xxxx9999\\n" > "$LEC_RESOLVED_CONTRACTS_OUT"\nprintf "rule_id\\tcase_total\\tcase_pass\\tcase_fail\\tcase_xfail\\tcase_xpass\\tcase_error\\tcase_skip\\nchip.csv:RULE_CLK\\t1\\t1\\t0\\t0\\t0\\t0\\t0\\n" > "${LEC_CONNECTIVITY_STATUS_SUMMARY_OUT:-/dev/null}"\n' > %t/utils/run_opentitan_connectivity_circt_lec.py
test/Tools/run-formal-all-opentitan-connectivity-contract-parity-fail.test:24:// PARITY: connectivity::chip.csv:RULE_CLK{{[[:space:]]+}}xxxx9999xxxx9999{{[[:space:]]+}}missing_in_bmc{{[[:space:]]+}}absent{{[[:space:]]+}}present{{[[:space:]]+}}0
lib/Dialect/Datapath/DatapathFolds.cpp:154:  // FIXME: This should be implemented as a canonicalization pattern for
lib/Dialect/Datapath/DatapathFolds.cpp:316:// TODO: use knownBits to extract all constant ones
lib/Dialect/Datapath/DatapathFolds.cpp:426:    // TODO: implement a constant multiplication for the PartialProductOp
lib/Dialect/Datapath/DatapathFolds.cpp:484:    // TODO: add support for different width inputs
include/circt/Dialect/MSFT/DeviceDB.h:50:  // TODO: Create read-only version of getLeaf.
lib/Dialect/FIRRTL/FIRRTLUtils.cpp:322:    // TODO: If needed, this could be modified to look through unary ops which
lib/Dialect/FIRRTL/FIRRTLUtils.cpp:333:  // TODO: what do we want to happen when there are flips in the type? Do we
lib/Dialect/FIRRTL/FIRRTLUtils.cpp:452:      // TODO: If needed, this could be modified to look through unary ops which
lib/Dialect/FIRRTL/FIRRTLUtils.cpp:726:        // TODO: Plumb error case out and handle in callers.
lib/Dialect/FIRRTL/FIRRTLUtils.cpp:768:          // TODO: are enums aggregates or not?  Where is walkGroundTypes called
lib/Dialect/FIRRTL/FIRRTLUtils.cpp:827:  // TODO: store/ensure always sorted, insert directly, faster search.
include/circt/Dialect/ESI/ESIStdServices.td:56:      TODO: ports for out-of-order returns
test/Tools/circt-sim/syscall-strobe.sv:2:// TODO: $strobe shows val=10 (immediate) instead of val=20 (end-of-timestep).
test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:4:// RUN: printf '#resolved_contract_schema_version=1\nconnectivity::chip.csv:RULE_CLK\t/tmp/chip.csv:2\tmanifest\tsmtlib\t300\tlec\t0\t0\t\txxxx9999xxxx9999\n' > %t/lec.tsv
test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:5:// RUN: printf 'exact:connectivity::chip.csv:RULE_CLK::aaaa1111aaaa1111\nexact:connectivity::chip.csv:RULE_CLK::xxxx9999xxxx9999\n' > %t/allow.txt
test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:12:// PARITY: connectivity::chip.csv:RULE_CLK{{[[:space:]]+}}xxxx9999xxxx9999{{[[:space:]]+}}missing_in_bmc{{[[:space:]]+}}absent{{[[:space:]]+}}present{{[[:space:]]+}}1
lib/Dialect/Datapath/DatapathOps.cpp:129:      // TODO: Fold Constant 1s
lib/Dialect/Datapath/DatapathOps.cpp:216:// TODO: Dadda's algorithm is redundant here since it assumes uniform arrival so
lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:98:/// TODO: Add a good description of direction?
test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-fail.test:4:// RUN: printf '#resolved_contract_schema_version=1\nconnectivity::chip.csv:RULE_CLK\t/tmp/chip.csv:2\tmanifest\tsmtlib\t300\tlec\t0\t0\t\txxxx9999xxxx9999\n' > %t/lec.tsv
test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-fail.test:12:// PARITY: connectivity::chip.csv:RULE_CLK{{[[:space:]]+}}xxxx9999xxxx9999{{[[:space:]]+}}missing_in_bmc{{[[:space:]]+}}absent{{[[:space:]]+}}present{{[[:space:]]+}}0
test/Tools/circt-sim/syscall-shortrealtobits.sv:2:// TODO: $shortrealtobits partial — negative shortreal values produce x instead of bits.
lib/Dialect/FIRRTL/FIRRTLReductions.cpp:1821:        // TODO: The namespace should be unnecessary. However, some FIRRTL
test/Tools/circt-sim/syscall-randomize-with.sv:2:// TODO: randomize() with inline constraints — data_constrained=0, constraint not applied.
lib/Dialect/ESI/runtime/cpp/include/esi/backends/Trace.h:44:    // TODO: Full trace mode not yet supported.
test/Tools/circt-sim/syscall-random.sv:2:// TODO: $random(seed) does not update seed variable — seed output not wired.
lib/Dialect/ESI/runtime/cpp/include/esi/backends/RpcServer.h:25:/// TODO: make this a proper backend (as much as possible).
lib/Dialect/ESI/runtime/cpp/include/esi/backends/RpcServer.h:36:  /// before the manifest is set. TODO: rework the DPI API to require that the
lib/Dialect/RTG/Transforms/ElaborationPass.cpp:1978:    // FIXME: this is not how the MLIR MemoryEffects interface intends it.
lib/Dialect/RTG/Transforms/ElaborationPass.cpp:2094:  // TODO: don't clone if this is the only remaining reference to this
lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:53:  // TODO: use a better datastructure for 'active'
lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:82:    // TODO: ideally check that the IR is already fully elaborated
lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:119:      // TODO: support labels and control-flow loops (jumps in general)
lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:131:    // TODO: don't overapproximate that much
lib/Dialect/ESI/runtime/cpp/include/esi/Utils.h:78:    // is to copy the data. TODO: Avoid copying the data.
test/Tools/circt-sim/syscall-monitor.sv:2:// TODO: $monitor fires only once (val=0) — value-change callback not re-triggering.
include/circt/Dialect/Calyx/CalyxPrimitives.td:543:  let results = (outs AnyInteger:$in, AnyInteger:$out);
test/Tools/circt-sim/syscall-isunbounded.sv:2:// TODO: $isunbounded with class type parameters — compilation or runtime failure.
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:96:    // TODO: we cannot just assume that double-slash is the way to do a line
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:120:    // TODO: don't hardcode '.word'
lib/Dialect/ESI/runtime/cpp/include/esi/Ports.h:335:  // TODO: Have the callback return something upon which the caller can check,
lib/Dialect/ESI/runtime/cpp/include/esi/Ports.h:458:  // TODO: this probably shouldn't be 'const', but bundle ports' user access are
include/circt/Dialect/Calyx/CalyxPasses.td:27:    2. TODO(Calyx): If there are multiple writes to a signal, replace the reads
test/Tools/circt-sim/syscall-generate.sv:2:// TODO: Generate-for produces data=101 instead of data=0101 — width/padding issue.
test/Tools/circt-sim/syscall-fread.sv:2:// TODO: $fread not yet implemented in ImportVerilog/interpreter.
lib/Dialect/FIRRTL/FIRRTLOps.cpp:304:// TODO: This is doing the same walk as foldFlow.  These two functions can be
lib/Dialect/FIRRTL/FIRRTLOps.cpp:1331:    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
lib/Dialect/FIRRTL/FIRRTLOps.cpp:1676:  // TODO: this should be using properties.
lib/Dialect/FIRRTL/FIRRTLOps.cpp:2100:  // TODO: this should use properties.
lib/Dialect/FIRRTL/FIRRTLOps.cpp:3461:        // FIXME error on missing bundle flip
lib/Dialect/FIRRTL/FIRRTLOps.cpp:4076:  // TODO: Relax this to allow reads from output ports,
lib/Dialect/FIRRTL/FIRRTLOps.cpp:4321:    // TODO: Make ref.sub only source flow?
lib/Dialect/FIRRTL/FIRRTLOps.cpp:4949:  // TODO: check flow
lib/Dialect/FIRRTL/FIRRTLOps.cpp:4962:  // TODO: check flow
lib/Dialect/FIRRTL/FIRRTLOps.cpp:6791:  // TODO: Determine ref.sub + rwprobe behavior, test.
lib/Dialect/FIRRTL/FIRRTLOps.cpp:6923:  // TODO: Verify that the target's type matches the type of this op.
lib/Dialect/FIRRTL/FIRRTLOps.cpp:6935:  // TODO: Verify that the target's type matches the type of this op.
lib/Dialect/FIRRTL/FIRRTLOps.cpp:7023:          // TODO: Improve this verifier.  This is intentionally _not_ verifying
test/Tools/circt-sim/syscall-feof.sv:2:// TODO: Comprehensive file I/O test — individual ops work but combined sequence fails.
include/circt/Dialect/Calyx/CalyxLoweringUtils.h:204:  /// TODO(mortbopet): Add a post-insertion check to ensure that the use-def
lib/Dialect/Comb/CombFolds.cpp:1091:    // TODO: Combine multiple constants together even if they aren't at the
lib/Dialect/Comb/CombFolds.cpp:1109:      // TODO: Generalize this for non-single-bit operands.
lib/Dialect/Comb/CombFolds.cpp:1205:  /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement
lib/Dialect/Comb/CombFolds.cpp:1421:  /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement
lib/Dialect/Comb/CombFolds.cpp:2194:    // TODO: We could handle things like "x < 2" as two entries.
lib/Dialect/Comb/CombFolds.cpp:2507:  // TODO: Generalize this to and, or, xor, icmp(!), which all occur in practice
lib/Dialect/Comb/CombFolds.cpp:3254:    // FIXME(llvm merge, cc697fc292b0): concat doesn't work with zero bit values
include/circt/Dialect/Calyx/CalyxHelpers.h:49:// TODO(github.com/llvm/circt/issues/1679): Add Invoke.
lib/Dialect/ESI/runtime/cosim_dpi_server/CMakeLists.txt:14:# the simulator, but without it we get runtime link errors. TODO: figure out how
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:541:// TODO: Move to DRR.
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:854:    /// TODO: Support SInt<1> on the LHS etc.
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:904:    /// TODO: Support SInt<1> on the LHS etc.
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:950:  // TODO: implement constant folding, etc.
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:956:  // TODO: implement constant folding, etc.
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1657:      // TODO: x ? ~0 : 0 -> sext(x)
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1658:      // TODO: "x ? c1 : c2" -> many tricks
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1660:    // TODO: "x ? a : 0" -> sext(x) & a
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1663:  // TODO: "x ? c1 : y" -> "~x ? y : c1"
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2042:  // TODO: Handle even when `index` doesn't have uint<1>.
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2155:  // TODO: Canonicalize towards explicit extensions and flips here.
lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2207:    // TODO: May need to be sensitive to "don't touch" or other
lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:227:    // TODO: containsReference().
lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:268:      // TODO: containsReference().
lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:133:// TODO: Change this by breaking it in two functions, one for read and one for
lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:357:// TODO: These had the shit broken outta them in the gRPC conversion. We're not
lib/Dialect/Comb/Transforms/IntRangeAnnotations.cpp:110:  // TODO: determine how to support subtraction
lib/Dialect/ESI/runtime/cosim_dpi_server/driver.cpp:86:  // TODO: Add max speed (cycles per second) option for small, interactive
lib/Dialect/ESI/runtime/cosim_dpi_server/driver.cpp:94:  // TODO: Support ESI reset handshake in the future.
lib/Dialect/RTG/IR/RTGAttributes.cpp:26:  // TODO: improve collision resistance
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1282:  // TODO: emitAssignLike
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1386:  // TODO: emitAssignLike ?
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1465:  // TODO: Add option to control base-2/8/10/16 output here.
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1477:  // TODO: Emit type decl for type alias.
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1743:  // TODO: Emit type decl for type alias.
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1843:  // TODO: Handle FusedLoc and uniquify locations, avoid repeated file names.
lib/Dialect/Comb/Transforms/BalanceMux.cpp:216:  // TODO: Ideally the separator index should be selected based on arrival times
lib/Dialect/FIRRTL/Import/FIRLexer.cpp:99:      // TODO: Handle the rest of the escapes (octal and unicode).
include/circt/Dialect/Arc/ArcOps.td:333:    AnyInteger:$address
include/circt/Dialect/Arc/ArcOps.td:335:  let results = (outs AnyInteger:$data);
include/circt/Dialect/Arc/ArcOps.td:411:    AnyInteger:$address
include/circt/Dialect/Arc/ArcOps.td:413:  let results = (outs AnyInteger:$data);
include/circt/Dialect/Arc/ArcOps.td:427:    AnyInteger:$address,
include/circt/Dialect/Arc/ArcOps.td:429:    AnyInteger:$data
include/circt/Dialect/Comb/CombOps.h:97:/// TODO: Support signed division and modulo.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:754:  // FIXME: Handle RelaxedId
lib/Dialect/FIRRTL/Import/FIRParser.cpp:2122:  // TODO: This is very similar to connect expansion in the LowerTypes pass
lib/Dialect/FIRRTL/Import/FIRParser.cpp:2167:/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
lib/Dialect/FIRRTL/Import/FIRParser.cpp:2426:/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
lib/Dialect/FIRRTL/Import/FIRParser.cpp:3517:  // TODO(firrtl spec): There is no reason for the 'else :' grammar to take an
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4056:/// XXX: spec says static_reference, allow ref_expr anyway for read(probe(x)).
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4097:  // TODO: Add to ref.send verifier / inferReturnTypes.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4341:  // TODO: Once support lands for agg-of-ref, add test for this check!
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4514:  // TODO: Once support lands for agg-of-ref, add test for this check!
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4722:  // TODO(firrtl spec) cmem is completely undocumented.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4754:  // TODO(firrtl spec) smem is completely undocumented.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4900:  // TODO: Move this into MemOp construction/canonicalization.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:5038:  // TODO(firrtl spec): info? should come after the clock expression before
lib/Dialect/FIRRTL/Import/FIRParser.cpp:5059:    // TODO(firrtl spec): Simplify the grammar for register reset logic.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:5082:    // TODO(firrtl scala impl): pretty print registers without resets right.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:5932:  // TODO: Remove the old `, bound = N` variant in favor of the new parameters.
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:365:// TODO: handle annotations: [[OptimizableExtModuleAnnotation]]
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:539:            // TODO: We should handle aggregate operations such as
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:861:  // TODO: Handle 'when' operations.
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1027:    // TODO: Replace entire aggregate.
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1034:    // TODO: Let materializeConstant tell us what it supports instead of this.
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1056:  // TODO: Handle WhenOps correctly.
lib/Dialect/FIRRTL/FIRRTLTypes.cpp:832:              {elt.name, false /* FIXME */, elt.type.getMaskType()});
lib/Dialect/FIRRTL/FIRRTLTypes.cpp:1698:      // TODO: Maybe just have elementType be FieldIDTypeInterface ?
lib/Dialect/FIRRTL/FIRRTLTypes.cpp:1863:  // TODO: ConstTypeInterface / Trait ?
lib/Dialect/FIRRTL/FIRRTLTypes.cpp:2137:  // TODO: ConstTypeInterface / Trait ?
lib/Dialect/FIRRTL/FIRRTLTypes.cpp:2369:    // TODO: exclude reference containing
lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:66:  // TODO: bitwidth is 1 when there are no ports, since APInt previously did not
lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:86:  // TODO: it seems weird to allow empty port annotations.
lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:89:  // TODO: Move this into an annotation verifier.
lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:207:  // TODO: Find a way to not check same things RefType::get/verify does.
lib/Dialect/FIRRTL/Transforms/AddSeqMemPorts.cpp:463:      if (!instanceInfo->anyInstanceInEffectiveDesign(op))
lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:273:    // TODO: This incongruity seems bad.  Can we instead not generate metadata
lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:275:    bool inDut = instanceInfo.anyInstanceInEffectiveDesign(mem);
lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:321:          // TODO: This unresolvable distinct seems sketchy.
lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:624:    if (!instanceInfo->anyInstanceInEffectiveDesign(mem))
lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:823:          !instanceInfo->anyInstanceInEffectiveDesign(module))
lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:909:    if (instanceInfo->anyInstanceInEffectiveDesign(extModule)) {
include/circt/Dialect/Kanagawa/KanagawaInterfaces.td:97:        // TODO: @mortbopet: fix once we have a way to do nested symbol
lib/Dialect/FIRRTL/Transforms/RemoveUnusedPorts.cpp:85:    // TODO: Handle inout ports.
lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:420:    // TODO: what to do with users that aren't local (or not mapped?).
lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:739:  // TODO: There is currently no good way to annotate an explicit parent scope
lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:1227:        // TODO: Update any symbol renames which need to be used by the next
lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:1316:        // TODO: Update any symbol renames which need to be used by the next
lib/Dialect/Pipeline/PipelineOps.cpp:719:  // TODO: This should be a pre-computed property.
lib/Dialect/Pipeline/PipelineOps.cpp:980:  // TODO: This could be optimized quite a bit if we didn't store clock
lib/Dialect/Calyx/Transforms/CompileControl.cpp:34:/// TODO(Calyx): Probably a better built-in operation?
lib/Dialect/Calyx/Transforms/CompileControl.cpp:116:    // TODO(Calyx): Eventually, we should canonicalize the GroupDoneOp's guard
lib/Dialect/FIRRTL/Transforms/MergeConnections.cpp:52:  // TODO: Add unrealized_conversion, asUInt, asSInt
lib/Dialect/FIRRTL/Transforms/InferReadWrite.cpp:57:      // TODO: This would be better handled if WhenOps were moved into the
lib/Dialect/FIRRTL/Transforms/CheckLayers.cpp:43:    if (!iInfo.anyInstanceUnderLayer(moduleOp))
lib/Dialect/FIRRTL/Transforms/CheckLayers.cpp:73:               iInfo.anyInstanceUnderLayer(parent)) {
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:248:  // TODO: This is tech debt.  This was accepted on condition that work is done
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:376:  // TODO: This defname matching is a terrible hack and should be replaced with
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:384:      if (!instanceInfo->anyInstanceInDesign(module)) {
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:391:      info.prefix = "clock_gate"; // TODO: Don't hardcode this
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:416:      if (!instanceInfo->anyInstanceInDesign(module)) {
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:423:      info.prefix = "mem_wiring"; // TODO: Don't hardcode this
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:544:        !instanceInfo->anyInstanceInDesign(parent) ||
lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:1130:             !instanceInfo->anyInstanceInDesign(cast<igraph::ModuleOpInterface>(
lib/Dialect/FIRRTL/Transforms/MemToRegOfVec.cpp:50:      if (instanceInfo.anyInstanceInEffectiveDesign(moduleOp))
lib/Dialect/FIRRTL/Transforms/IMDeadCodeElim.cpp:272:    // TODO: Handle attach etc.
lib/Dialect/FIRRTL/Transforms/IMDeadCodeElim.cpp:396:  // FIXME: We copy the list of modules into a vector first to avoid iterator
lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:1043:    // TODO: It's possible that this result is already sufficient to arrive at a
lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:2217:      // TODO: this should recurse into the element type of 0 length vectors and
lib/Dialect/FIRRTL/Transforms/CheckCombLoops.cpp:336:    // TODO: External modules not handled !!
test/Tools/circt-sim/tlul-bfm-a-ready-timeout-short-circuit.sv:2:// TODO: BFM timeout messages not printed — task-level $display in while loop not working.
lib/Dialect/FIRRTL/Transforms/LowerOpenAggs.cpp:597:  // TODO: add and erase ports without intermediate + various array attributes.
lib/Dialect/FIRRTL/Transforms/Lint.cpp:107:    if (!instanceInfo.anyInstanceInDesign(fModule))
lib/Dialect/FIRRTL/Transforms/LowerMemory.cpp:497:  // TODO: how do we lower port annotations?
lib/Dialect/FIRRTL/Transforms/LowerMemory.cpp:557:    auto shouldDedup = instanceInfo.anyInstanceInEffectiveDesign(moduleOp);
lib/Dialect/FIRRTL/Transforms/LinkCircuits.cpp:232:    // FIXME: definition and declaration may have different defname and
lib/Dialect/FIRRTL/Transforms/LinkCircuits.cpp:263:    // TODO: other circuit attributes (such as enable_layers...)
lib/Dialect/FIRRTL/Transforms/InferResets.cpp:443:///    TODO: This logic is *very* brittle and error-prone. It may make sense to
test/Tools/circt-sim/syscall-ungetc.sv:2:// TODO: $ungetc returns 66 ('B') instead of 65 ('A') — pushback not working correctly.
lib/Dialect/Calyx/Export/CalyxEmitter.cpp:974:  double doubleValue = value.convertToDouble();
lib/Dialect/OM/Transforms/FreezePaths.cpp:170:        // TODO: add support for instance choices.
lib/Dialect/FIRRTL/Transforms/LowerDPI.cpp:109:    // TODO: Support bundle or enum types.
lib/Dialect/FIRRTL/Transforms/LowerDPI.cpp:206:      // TODO: This should be implemented as a verifier once function is added
lib/Dialect/FIRRTL/Transforms/LowerCHIRRTL.cpp:308:      // should just be deleted. TODO: this is mirroring SFC, but should we be
lib/Dialect/FIRRTL/Transforms/LowerCHIRRTL.cpp:454:        // TODO: SFC does not infer any enable when using a module port as the
include/circt/Conversion/Passes.td:178:// TODO: @mortbopet: There is a possible non-neglible speedup that can be achieved
lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:446:    // TODO: Change this to recursively clone.  This will matter once FString
lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:513:        // TODO: Move to before parallel region to avoid the lock.
lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:561:  // TODO: Simplify this once wires have domain kind information [1].
lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:912:      // TODO: Stop looking through wires when wires support domain info [1].
lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:1361:  // TODO: This unnecessarily computes a new namepath for every hw::HierPathOp
lib/Dialect/FIRRTL/Transforms/SpecializeLayers.cpp:31:// TODO: this should be upstreamed.
lib/Dialect/FIRRTL/Transforms/SpecializeLayers.cpp:790:    // TODO: if both arrays are empty we could preserve specific analyses, but
lib/Dialect/FIRRTL/Transforms/Dedup.cpp:307:      // TODO: properly handle DistinctAttr, including its use in paths.
lib/Dialect/FIRRTL/Transforms/Dedup.cpp:589:        // TODO: we should print the port number if there are no port names, but
lib/Dialect/FIRRTL/Transforms/Dedup.cpp:733:        // TODO: properly handle DistinctAttr, including its use in paths.
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:305:  /// TODO: The mutation of this is _not_ thread safe.  This needs to be fixed
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:311:  llvm::MapVector<unsigned, DomainInfo> indexToDomain;
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:360:      indexToDomain[i] = port.direction == Direction::In
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:376:        indexToDomain[i].op = object;
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:377:        indexToDomain[i].temp = stubOut(body->getArgument(i));
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:441:      indexToDomain[indexAttr.getUInt()].associations.push_back({id, port.loc});
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:457:    for (auto const &[_, info] : indexToDomain) {
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:671:  // TODO: There is nothing to do unless this instance is a module or external
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:690:      indexToDomain[i].temp = stubOut(instanceOp.getResult(i));
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:696:    for (auto &[i, info] : indexToDomain) {
lib/Dialect/Calyx/CalyxOps.cpp:100:/// TODO(Calyx): This is useful to verify current MLIR can be lowered to the
lib/Dialect/Calyx/CalyxOps.cpp:1544:  // Since the guard is optional, we need to check if there is an accompanying
lib/Dialect/FIRRTL/Transforms/BlackBoxReader.cpp:223:        // TODO: Check that the new text is the _exact same_ as the prior best.
lib/Dialect/FIRRTL/Transforms/BlackBoxReader.cpp:374:           !instanceInfo->anyInstanceInEffectiveDesign(
lib/Dialect/OM/Evaluator/Evaluator.cpp:208:    // FIXME: `diag << actualParams` doesn't work for some reason.
lib/Dialect/FIRRTL/Transforms/LowerIntmodules.cpp:164:      // FIXME: Dedup group annotation could be annotated to EICG_wrapper but
lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:58:// TODO: check all argument types
lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:130:    // FIXME: Don't preserve read-only RefType for now. This is workaround for
lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:1622:  // FIXME: annotation update
lib/Dialect/FIRRTL/Transforms/AnnotateInputOnlyModules.cpp:58:    if (!instanceInfo.anyInstanceInEffectiveDesign(module) ||
lib/Dialect/FIRRTL/Transforms/InjectDUTHierarchy.cpp:13:// below, and the accompanying description of terms is provided to clarify what
lib/Dialect/FIRRTL/Transforms/InjectDUTHierarchy.cpp:340:    // TODO: It _may_ be desirable to only do this in the `moveDut=true` case.
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:91:/// TODO: This could be removed if we add `firrtl.DocStringAnnotation` support
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:106:/// TODO: Fix this once we have a solution for #1464.
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:715:  /// TODO: Investigate a way to not use a pointer here like how `getNamespace`
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:760:  // TODO: Add a comment op and lower the description to that.
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:761:  // TODO: Tracking issue: https://github.com/llvm/circt/issues/1677
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1716:    // TODO: The `append(name.getValue())` in the following should actually be
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1749:  // TODO: Require as part of attribute, structurally.
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1762:  interfaceBuilder.emplace_back(iFaceName, IntegerAttr() /* XXX */);
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1775:    // TODO: The `append(name.getValue())` in the following should actually be
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1906:  /// TODO: Handle this differently to allow construction of an options
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1998:        // TODO: Figure out what to do with this.
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:2187:                    instanceInfo->anyInstanceInEffectiveDesign(
lib/Dialect/FIRRTL/Transforms/LowerAnnotations.cpp:131:// FIXME: uniq annotation chain links
test/Tools/circt-verilog-lsp-server/interface.test:29:// TODO: Hover on interface signals not yet implemented
test/Tools/circt-verilog-lsp-server/interface.test:39:// TODO: Go-to-definition for interface members not yet implemented
lib/Dialect/MSFT/ExportQuartusTcl.cpp:80:// TODO: Currently assumes Stratix 10 and QuartusPro. Make more general.
lib/Dialect/Arc/ArcDialect.cpp:29:    // TODO
lib/Dialect/Arc/ArcDialect.cpp:34:    // TODO
lib/Dialect/MSFT/DeviceDB.cpp:357:  // TODO: Since the data structures we're using aren't sorted, the best we can
lib/Dialect/Moore/Transforms/LowerConcatRef.cpp:75:      // FIXME: Need to estimate whether the bits range is from large to
include/circt/Analysis/FIRRTLInstanceInfo.h:145:  bool anyInstanceUnderDut(igraph::ModuleOpInterface op);
include/circt/Analysis/FIRRTLInstanceInfo.h:155:  bool anyInstanceUnderEffectiveDut(igraph::ModuleOpInterface op);
include/circt/Analysis/FIRRTLInstanceInfo.h:164:  bool anyInstanceUnderLayer(igraph::ModuleOpInterface op);
include/circt/Analysis/FIRRTLInstanceInfo.h:172:  bool anyInstanceInDesign(igraph::ModuleOpInterface op);
include/circt/Analysis/FIRRTLInstanceInfo.h:180:  bool anyInstanceInEffectiveDesign(igraph::ModuleOpInterface op);
lib/Dialect/Arc/Transforms/LowerState.cpp:582:  module.builder.setInsertionPoint(ifClockOp.thenYield());
lib/Dialect/Arc/Transforms/LowerState.cpp:609:    module.builder.setInsertionPoint(ifResetOp.thenYield());
lib/Dialect/Arc/Transforms/LowerState.cpp:641:    module.builder.setInsertionPoint(ifEnableOp.thenYield());
lib/Dialect/Arc/Transforms/LowerState.cpp:731:    module.builder.setInsertionPoint(ifClockOp.thenYield());
lib/Dialect/Arc/Transforms/LowerState.cpp:749:      module.builder.setInsertionPoint(ifEnableOp.thenYield());
include/circt/Analysis/DependenceAnalysis.h:66:/// TODO(mikeurbach): consider upstreaming this to MLIR's AffineAnalysis.
lib/Dialect/Arc/Transforms/AllocateState.cpp:97:      // TODO: Can a StateWriteOp be shared across models?
lib/Dialect/Arc/Transforms/SplitLoops.cpp:240:  // TODO: This is ugly and we should only split arcs that are truly involved in
lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:249:  // TODO: we can also check for arith.select or other operations here
lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:256:/// TODO: improve and fine-tune this
lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:289:// FIXME: Assumes that the regions in which muxes exist are topologically
lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:291:// FIXME: does not consider side-effects
test/Analysis/firrtl-test-instance-info.mlir:15:  // CHECK-NEXT:   anyInstanceUnderDut: false
test/Analysis/firrtl-test-instance-info.mlir:17:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
test/Analysis/firrtl-test-instance-info.mlir:19:  // CHECK-NEXT:   anyInstanceUnderLayer: true
test/Analysis/firrtl-test-instance-info.mlir:24:  // CHECK-NEXT:   anyInstanceUnderDut: false
test/Analysis/firrtl-test-instance-info.mlir:26:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
test/Analysis/firrtl-test-instance-info.mlir:28:  // CHECK-NEXT:   anyInstanceUnderLayer: true
test/Analysis/firrtl-test-instance-info.mlir:33:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:35:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:37:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:42:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:44:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:46:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:51:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:53:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:55:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:67:  // CHECK-NEXT:   anyInstanceUnderDut: false
test/Analysis/firrtl-test-instance-info.mlir:69:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
test/Analysis/firrtl-test-instance-info.mlir:71:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:95:  // CHECK-NEXT:   anyInstanceUnderDut: false
test/Analysis/firrtl-test-instance-info.mlir:97:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:99:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:110:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:112:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:114:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:153:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:155:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:173:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:175:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:177:  // CHECK-NEXT:   anyInstanceUnderLayer: true
test/Analysis/firrtl-test-instance-info.mlir:179:  // CHECK-NEXT:   anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:181:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:186:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:188:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: true
test/Analysis/firrtl-test-instance-info.mlir:190:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:192:  // CHECK-NEXT:   anyInstanceInDesign: true
test/Analysis/firrtl-test-instance-info.mlir:194:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:209:  // CHECK-NEXT:   anyInstanceUnderDut: false
test/Analysis/firrtl-test-instance-info.mlir:211:  // CHECK-NEXT:   anyInstanceUnderEffectiveDut: false
test/Analysis/firrtl-test-instance-info.mlir:213:  // CHECK-NEXT:   anyInstanceUnderLayer: false
test/Analysis/firrtl-test-instance-info.mlir:215:  // CHECK-NEXT:   anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:217:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:231:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:233:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:237:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:239:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:247:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:249:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:262:  // CHECK:        anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:284:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:286:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:290:  // CHECK:        anyInstanceInDesign: true
test/Analysis/firrtl-test-instance-info.mlir:292:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:296:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:298:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:302:  // CHECK:        anyInstanceInDesign: true
test/Analysis/firrtl-test-instance-info.mlir:304:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:308:  // CHECK:        anyInstanceInDesign: true
test/Analysis/firrtl-test-instance-info.mlir:310:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:314:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:316:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:320:  // CHECK:        anyInstanceInDesign: true
test/Analysis/firrtl-test-instance-info.mlir:322:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:385:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:387:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:391:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:393:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:397:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:399:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:403:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:405:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:409:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:411:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:415:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:417:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:421:  // CHECK:        anyInstanceInDesign: false
test/Analysis/firrtl-test-instance-info.mlir:423:  // CHECK-NEXT:   anyInstanceInEffectiveDesign: true
test/Analysis/firrtl-test-instance-info.mlir:471:  // CHECK:        anyInstanceUnderLayer: true
test/Analysis/firrtl-test-instance-info.mlir:473:  // CHECK:        anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:478:  // CHECK:        anyInstanceUnderLayer: true
test/Analysis/firrtl-test-instance-info.mlir:480:  // CHECK:        anyInstanceInEffectiveDesign: false
test/Analysis/firrtl-test-instance-info.mlir:501:  // CHECK-NEXT:   anyInstanceUnderDut: false
test/Analysis/firrtl-test-instance-info.mlir:510:  // CHECK-NEXT:   anyInstanceUnderDut: true
test/Analysis/firrtl-test-instance-info.mlir:522:  // CHECK-NEXT:   anyInstanceUnderDut: false
lib/Dialect/Arc/Transforms/MergeIfs.cpp:340:    prevIfOp.thenYield().erase();
lib/Dialect/Arc/Transforms/LowerLUT.cpp:318:  // TODO: This class could be factored out into an analysis if there is a need
lib/Dialect/Arc/Transforms/IsolateClocks.cpp:44:  bool moveToDomain(Operation *op);
lib/Dialect/Arc/Transforms/IsolateClocks.cpp:64:bool ClockDomain::moveToDomain(Operation *op) {
lib/Dialect/Arc/Transforms/IsolateClocks.cpp:85:    // TODO: if we find a clock domain with the same clock we should merge it.
lib/Dialect/Arc/Transforms/IsolateClocks.cpp:104:    if (moveToDomain(op))
lib/Dialect/Arc/Transforms/IsolateClocks.cpp:210:      if (domain.moveToDomain(op)) {
lib/Dialect/Arc/Transforms/InlineArcs.cpp:190:        // TODO: make safe
lib/Dialect/Arc/Transforms/InlineArcs.cpp:343:    // TODO: instead of hardcoding these ops we might also be able to query the
lib/Dialect/Arc/ArcFolds.cpp:241:    // TODO: we could also push out all operations that are not clocked/don't
lib/Dialect/Arc/Transforms/InferStateProperties.cpp:113:    // TODO: split the arcs such that there is one for each reset kind, however,
lib/Dialect/Arc/Transforms/InferStateProperties.cpp:120:    // TODO: arc.state operation only supports resets to zero at the moment.
lib/Dialect/Arc/Transforms/InferStateProperties.cpp:176:    // TODO: split the arcs such that there is one for each enable kind,
lib/Dialect/Arc/ArcCostModel.cpp:19:// FIXME: May be refined and we have more accurate operation costs
lib/Dialect/Arc/Interfaces/RuntimeCostEstimateInterfaceImpl.cpp:58:        // TODO: improve this measure as it might lower to a sext or a mul
lib/Dialect/Arc/Interfaces/RuntimeCostEstimateInterfaceImpl.cpp:90:    // TODO: ArraySliceOp, ArrayConcatOp
lib/Dialect/Arc/Interfaces/RuntimeCostEstimateInterfaceImpl.cpp:104:        // TODO: this is chosen quite arbitrarily right now
lib/Dialect/Arc/Transforms/Dedup.cpp:204:    // TODO: We should probably bail out if there are any operations in the
lib/Dialect/Arc/Transforms/ArcCanonicalizer.cpp:130:    // TODO: if an operation is inserted that defines a symbol and the symbol
lib/Dialect/Arc/Transforms/ArcCanonicalizer.cpp:143:    // TODO: if an operation is inserted that defines a symbol and the symbol
lib/Conversion/ImportLiberty/ImportLiberty.cpp:705:    // TODO: Support more group types
lib/Conversion/ImportLiberty/ImportLiberty.cpp:802:      // TODO: Properly handle timing subgroups etc.
lib/Conversion/ImportLiberty/ImportLiberty.cpp:935:  // TODO: Support define.
lib/Conversion/ImportLiberty/ImportLiberty.cpp:983:  // TODO: Add array for timing attributes
test/Dialect/FIRRTL/canonicalization.mlir:2606:  // FIXME(Issue #1125): Add a test for zero width memory elimination.
test/Dialect/FIRRTL/canonicalization.mlir:2983:// TODO: Move to an appropriate place
test/Dialect/FIRRTL/canonicalization.mlir:3036://TODO: Move to an appropriate place
test/Dialect/Synth/tech-mapper.mlir:92:    // FIXME: If area-flow is implemented, this should be mapped to @area_flow with area strategy.
test/Dialect/FIRRTL/infer-domains-infer-all-errors.mlir:122:// TODO: this just relies on the op-verifier for instance choice ops.
test/Dialect/Synth/lut-mapper.mlir:1:// FIXME: max-cuts-per-root=20 is due to a lack of non-minimal cut filtering.
test/Dialect/Moore/canonicalizers.mlir:94:  // CHECK-NEXT: moore.constant bXXXXX101 : l8
test/Dialect/Moore/canonicalizers.mlir:146:  // CHECK-DAG: [[V2:%.+]] = moore.constant hXXXXXX : l24
test/Dialect/Moore/canonicalizers.mlir:164:  // CHECK-DAG: [[X:%.+]] = moore.constant hXXXXXX : l24
test/Dialect/Moore/canonicalizers.mlir:191:  // CHECK-DAG: [[X:%.+]] = moore.constant hXXXXXX : l24
test/Dialect/FIRRTL/simplify-mems.mlir:471:    // TODO: It would be good to de-duplicate these either in the pass or in a canonicalizer.
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:586:            // If the accompanying pass runs on the HW dialect, then LowerToHW
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1136:    // TODO: handle this in some sensible way based on what the SFC does with
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1151:    // TODO: handle this with annotation verification.
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1165:    // TODO: this error behavior can be relaxed to always overwrite with the
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1565:    return {}; // TODO: Emit an sv.constant here since it is unconnected.
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1858:    // TODO: we should be checking for symbol collisions here and renaming as
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:2611:          // TODO: We don't support partial connects for bundles for now.
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:2856:      // TODO: This is not printf specific anymore. Replace "Printf" with "FD"
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:3700:  // TODO: Remove this restriction and preserve aggregates in
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:4224:  // TODO: Introduce elementwise operations to HW dialect instead of abusing
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:4560:  // We lower invalid to 0.  TODO: the FIRRTL spec mentions something about
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5292:    // TODO: This should *not* be part of the op, but rather a lowering
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5444:  // TODO : Need to figure out if there is a cleaner way to get the string which
test/Dialect/FIRRTL/infer-resets.mlir:140:  // TODO: Enable the following once #1303 is fixed.
test/Dialect/FIRRTL/infer-resets.mlir:605:// TODO: Check that no extraReset port present
test/Dialect/FIRRTL/ref.mlir:211:    // TODO: infer ref result existence + type based on "forceable" or other ref-kind(s) indicator.
lib/Conversion/VerifToSMT/VerifToSMT.cpp:5049:    // TODO: the init and loop regions should be able to be concrete instead of
lib/Conversion/VerifToSMT/VerifToSMT.cpp:9607:    // TODO: swapping to a whileOp here would allow early exit once the property
lib/Conversion/VerifToSMT/VerifToSMT.cpp:10084:                  // TODO: we create a lot of ITEs here that will slow things down
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11838:          // TODO: this can be removed once we have a way to associate reg
test/Dialect/Seq/canonicalization.mlir:106:  // TODO: Use constant aggregate attribute once supported.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:726:  // TODO: We could handle concat and other operators here.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:1426:      // TODO: handle other common patterns.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:1964:    // TODO: relying on float printing to be precise is not a good idea.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:1977:    // TODO: Should we support signed parameters?
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2082:  // TODO: This could try harder to omit redundant casts like the mainline
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2111:    // TODO: Also handle (a + b + x*-1).
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2512:  // TODO: Build tree capturing precedence/fixity at same level, group those!
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2529:  // TODO: MLIR should have general "Associative" trait.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2854:// TODO: This shares a lot of code with the getNameRemotely mtehod. Combine
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3018:  // TODO: Ideally we should emit zero bit values as comments, e.g. `{/*a:
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3248:    // TODO: if any of these breaks, it'd be "nice" to break
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3370:  // TODO: For unpacked structs, once we have support for them, `printAsPattern`
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3398:  // TODO: For unpacked structs, once we have support for them, `printAsPattern`
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4639:    // TODO: if any of these breaks, it'd be "nice" to break
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4794:        // TODO: good comma/wrapping behavior as elsewhere.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4872:  // TODO: location info?
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4889:  // TODO: location info?
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4908:  // TODO: We'll probably need to store the legalized names somewhere for
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5010:    // TODO: box, break/wrap behavior!
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5411:      // TODO: group, like normal 'if'.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5484:            // TODO: We could emit in hex if/when the size is a multiple of
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5745:  // TODO: source info!
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5750:  // FIXME: Don't emit the body of this as general statements, they aren't!
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5798:  // TODO: revisit, better breaks/grouping.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5816:  // TODO: emit like emitAssignLike does, maybe refactor.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5836:  // TODO: source info!
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5989:    /// TODO: Use MLIR DominanceInfo.
lib/Conversion/ExportVerilog/ExportVerilog.cpp:6098:    // FIXME: Unpacked array is not inlined since several tools doesn't support
lib/Conversion/ExportVerilog/ExportVerilog.cpp:6123:    // FIXME: Unpacked array is not inlined since several tools doesn't support
lib/Conversion/ExportVerilog/ExportVerilog.cpp:6951:          // TODO: How do we want to handle typedefs in a split output?
lib/Conversion/SimToSV/SimToSV.cpp:240:      // TODO: If there is a return value and no output argument, use an
test/Dialect/FIRRTL/parse-basic.fir:1570:    ; TODO: This test needs to be fixed up once flow-checking for objects is implemented correctly.
lib/Conversion/ExportVerilog/PrepareForEmission.cpp:218:    // TODO: Use a result name as suffix.
lib/Conversion/ExportVerilog/PrepareForEmission.cpp:728:  // TODO: Consider using virtual functions.
lib/Conversion/ExportVerilog/PrepareForEmission.cpp:787:  // TODO: Handle procedural regions as well.
lib/Conversion/ExportVerilog/PrepareForEmission.cpp:1210:    // TODO: This is checking the Commutative property, which doesn't seem
lib/Conversion/SeqToSV/SeqToSV.cpp:763:  // TODO: We could have an operation for macros and uses of them, and
lib/Conversion/SeqToSV/FirRegLowering.cpp:466:  // TODO: Handle struct if necessary.
test/Dialect/FIRRTL/lower-domains.mlir:327:// TODO: If there was stronger verification that ExpandWhens ensures
test/Dialect/FIRRTL/lower-types.mlir:579:      // TODO: Enable this
test/Dialect/FIRRTL/lower-types.mlir:602:    // TODO: Enable this
test/Dialect/FIRRTL/lower-types.mlir:621:    // TODO: Enable this
test/Dialect/FIRRTL/errors.mlir:325:    // expected-error @+1 {{'firrtl.instance' op name for port 1 must be "arg1", but got "xxx"}}
test/Dialect/FIRRTL/errors.mlir:326:    %a:2 = firrtl.instance "" @Callee(in arg0: !firrtl.uint<1>, in xxx: !firrtl.bundle<valid: uint<1>>)
test/Dialect/FIRRTL/errors.mlir:2169:  // expected-error @below {{'firrtl.module' op port #0 has wrong name, got "xxx", expected "str"}}
test/Dialect/FIRRTL/errors.mlir:2170:  firrtl.module @ClassTypeWrongPortName(out %port: !firrtl.class<@MyClass(out xxx: !firrtl.string)>) {}
test/Dialect/FIRRTL/errors.mlir:2978:  // expected-error @below {{target #hw.innerNameRef<@XXX::@YYY> cannot be resolved}}
test/Dialect/FIRRTL/errors.mlir:2979:  firrtl.bind <@XXX::@YYY>
lib/Conversion/SeqToSV/FirMemLowering.cpp:42:    // TODO: Check if this module is in the DUT hierarchy.
lib/Conversion/SeqToSV/FirMemLowering.cpp:89:  // TODO: Handle modName (maybe not?)
lib/Conversion/SeqToSV/FirMemLowering.cpp:90:  // TODO: Handle groupID (maybe not?)
test/Dialect/FIRRTL/lower-chirrtl.mlir:53:  // TODO: How do you get FileCheck to accept "[[[DATA]]]"?
lib/Conversion/SMTToZ3LLVM/LowerSMTToZ3LLVM.cpp:1387:    // FIXME: ideally we don't want to use i1 here, since bools can sometimes be
lib/Conversion/DatapathToComb/DatapathToComb.cpp:286:    // TODO - replace with a concatenation to aid longest path analysis
lib/Conversion/DatapathToComb/DatapathToComb.cpp:293:    // TODO: sort a and b based on non-zero bits to encode the smaller input
lib/Conversion/DatapathToComb/DatapathToComb.cpp:466:    // TODO: Implement Booth lowering
lib/Conversion/DatapathToComb/DatapathToComb.cpp:571:  // TODO: Topologically sort the operations in the module to ensure that all
lib/Conversion/DCToHW/DCToHW.cpp:217:///  @todo: should be moved to support.
lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:1833:/// TODO(#1850) evaluate the usefulness of this lowering pattern.
lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2029:      /// TODO(mortbopet) can we support these? for now, do not support loops
lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2494:        /// TODO(mortbopet): we could choose to support ie. std.switch, but it
lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2861:      // TODO (https://github.com/llvm/circt/issues/7764)
test/Dialect/Kanagawa/Transforms/scoperef_tunneling.mlir:198:// TODO: support this case. Too hard to support now. Problem is that hw.module
lib/Conversion/CombToDatapath/CombToDatapath.cpp:63:    // TODO: support for variadic multipliers
lib/Conversion/CombToDatapath/CombToDatapath.cpp:107:  // TODO: determine lowering of multi-input multipliers
lib/Conversion/ConvertToArcs/ConvertToArcs.cpp:241:    // TODO: Remove the elements from the post order as we go.
lib/Conversion/CombToArith/CombToArith.cpp:290:    // TODO: building a tree would be better here
lib/Conversion/CombToArith/CombToArith.cpp:406:  // TODO: a pattern for comb.parity
lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:154:/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:188:/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:389:    // FIXME: like the rest of MLIR, this assumes sizeof(intptr_t) ==
lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:456:// @todo: should be moved to support.
lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:665:    // Todo: clean up when handshake supports i0.
lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:1667:      // Todo: Change when i0 support is added.
test/Dialect/Arc/mux-to-control-flow.mlir:40:// TODO: while above testcase should already provide high statement coverage,
lib/Conversion/CombToSynth/CombToSynth.cpp:178:  // TODO: We can handle replicate, extract, etc.
lib/Conversion/CombToSynth/CombToSynth.cpp:488:  // TODO: Perform a more thorough analysis to motivate the choices or
lib/Conversion/CombToSynth/CombToSynth.cpp:703:// TODO: Generalize to other parallel prefix trees.
lib/Conversion/CombToSynth/CombToSynth.cpp:1038:    // TODO: Implement a signed division lowering at least for power of two.
lib/Conversion/CombToSynth/CombToSynth.cpp:1056:    // TODO: Implement a signed modulus lowering at least for power of two.
lib/Conversion/CombToSynth/CombToSynth.cpp:1102:  // TODO: Lazily compute only the required prefix values. Kogge-Stone is
lib/Conversion/HWToSystemC/HWToSystemC.cpp:71:    // TODO: implement logic to extract a better name and properly unique it.
lib/Conversion/HWToSystemC/HWToSystemC.cpp:77:    // TODO: do some dominance analysis to detect use-before-def and cycles in
lib/Conversion/CalyxNative/CalyxNative.cpp:158:  // XXX(rachitnigam): This is quite baroque. We insert the new block before the
lib/Conversion/CFToHandshake/CFToHandshake.cpp:726:  // TODO how to size these?
test/Dialect/HW/parameters.mlir:33:  // CHECK-SAME: #hw.param.expr.add<#hw.param.verbatim<"xxx">, 17>>() -> ()
test/Dialect/HW/parameters.mlir:34:  hw.instance "verbatimparam" @NoArg<param: i42 = #hw.param.expr.add<#hw.param.verbatim<"xxx">, 17>>() -> ()
test/Dialect/HW/parameters.mlir:128:  // CHECK-SAME:     add<#hw.param.expr.mul<{{.*}}>, #hw.param.verbatim<"xxx">>
test/Dialect/HW/parameters.mlir:129:  %4 = hw.param.value i4 = #hw.param.expr.add<#hw.param.verbatim<"xxx">, #hw.param.expr.mul<#hw.param.decl.ref<"p1">, 4>>
lib/Conversion/ImportVerilog/ImportVerilog.cpp:1838:    // TODO: Enable the following once it not longer interferes with @(...)
lib/Conversion/HWToLLVM/HWToLLVM.cpp:704:  // TODO: Only arrays and structs supported at the moment.
lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:1316:        /// TODO(mortbopet): we could choose to support ie. std.switch, but it
lib/Conversion/ImportVerilog/HierarchicalNames.cpp:1591:  /// TODO:Skip all others.
test/Dialect/Arc/latency-retiming.mlir:13:  // COM: TODO: a canonicalization pattern could reorder the outputs such that
test/Dialect/Arc/canonicalizers.mlir:135:  // TODO: add op such that it is not folded away because it's just passthrough
test/Dialect/Arc/infer-state-properties.mlir:248:// TODO: test that patterns handle the case where the output is used for another thing as well properly
test/Dialect/Arc/infer-state-properties.mlir:249:// TODO: test that reset and enable are only added when the latency is actually 1 or higher
lib/Conversion/ImportVerilog/TimingControls.cpp:42:    // TODO: SV 16.16, what to do when no edge is specified?
lib/Conversion/ImportVerilog/Structure.cpp:1456:          // TODO: Mark Inout port as unsupported and it will be supported later.
lib/Conversion/ImportVerilog/Structure.cpp:1837:            bool hasAnyIff = false;
lib/Conversion/ImportVerilog/Structure.cpp:1841:                  hasAnyIff = true;
lib/Conversion/ImportVerilog/Structure.cpp:1858:              if (!hasAnyIff)
lib/Conversion/ImportVerilog/Structure.cpp:3075:      // TODO: Implement proper tristate modeling with high-Z support.
lib/Conversion/ImportVerilog/Structure.cpp:3132:      // TODO: Implement proper tristate/high-Z modeling.
lib/Conversion/ImportVerilog/Structure.cpp:3188:      // TODO: Implement proper complementary MOS modeling with tristate.
lib/Conversion/ImportVerilog/Structure.cpp:3228:      // TODO: Implement proper bidirectional modeling.
lib/Conversion/ImportVerilog/Structure.cpp:3289:      // TODO: Implement proper conditional bidirectional modeling.
lib/Conversion/ImportVerilog/Expressions.cpp:213:  bool hasAnyIff = false;
lib/Conversion/ImportVerilog/Expressions.cpp:217:        hasAnyIff = true;
lib/Conversion/ImportVerilog/Expressions.cpp:233:    if (!hasAnyIff)
lib/Conversion/ImportVerilog/Expressions.cpp:5072:              bool hasAnyIff = false;
lib/Conversion/ImportVerilog/Expressions.cpp:5077:                    hasAnyIff = true;
lib/Conversion/ImportVerilog/Expressions.cpp:5082:              if (!hasAnyIff)
lib/Conversion/ImportVerilog/Expressions.cpp:10809:  // TODO: Handle other conversions with dedicated ops.
test/Target/ExportSystemC/basic.mlir:159:    // TODO: add this test-case once we have support for an inlinable operation that has lower precedence
test/Target/ExportSystemC/basic.mlir:167:    // TODO: add this test-case once we have support for an inlinable operation that has lower precedence
test/Target/ExportSystemC/basic.mlir:193:    // TODO: no operation having COMMA precedence supported yet
test/Target/ExportSystemC/basic.mlir:196:    // TODO: no applicable operation having lower precedence than CAST supported yet
test/Target/ExportSystemC/basic.mlir:230:    // TODO: there is currently no appropriate inlinable operation implemented with lower precedence
test/Target/ExportSystemC/basic.mlir:237:    // TODO: there is currently no appropriate inlinable operation implemented with lower precedence
lib/Conversion/MooreToCore/MooreToCore.cpp:1969:  // TODO: Handle vtable generation over ClassMethodDeclOp here.
lib/Conversion/MooreToCore/MooreToCore.cpp:2107:      // FIXME: Once we support net<...>, ref<...> type to represent type of
lib/Conversion/MooreToCore/MooreToCore.cpp:3761:  // TODO: Once the core dialects support four-valued integers, this code
lib/Conversion/MooreToCore/MooreToCore.cpp:6031:    // TODO: Generate loop-based validation for complex element constraints.
lib/Conversion/MooreToCore/MooreToCore.cpp:6054:    // TODO: Generate weighted random number generation calls.
lib/Conversion/MooreToCore/MooreToCore.cpp:7132:    // TODO: Once the core dialects support four-valued integers, this code
lib/Conversion/MooreToCore/MooreToCore.cpp:7819:    // TODO: return X if the domain is four-valued for out-of-bounds accesses
lib/Conversion/MooreToCore/MooreToCore.cpp:8108:    // TODO: properly handle out-of-bounds accesses
lib/Conversion/MooreToCore/MooreToCore.cpp:8878:    // TODO: properly handle out-of-bounds accesses
lib/Conversion/MooreToCore/MooreToCore.cpp:11233:    // TODO: Once the core dialects support four-valued integers, we will have
lib/Conversion/MooreToCore/MooreToCore.cpp:12736:    // 1. Free functions (5 args): uvm_pkg::uvm_report_xxx(id, msg, verbosity, filename, line)
lib/Conversion/MooreToCore/MooreToCore.cpp:12737:    // 2. Class methods (6 args): uvm_pkg::uvm_report_object::uvm_report_xxx(self, id, msg, verbosity, filename, line)
lib/Conversion/MooreToCore/MooreToCore.cpp:12739:    //   __moore_uvm_report_xxx(id_ptr, id_len, msg_ptr, msg_len, verbosity,
lib/Conversion/MooreToCore/MooreToCore.cpp:12811:    // void __moore_uvm_report_xxx(
lib/Conversion/MooreToCore/MooreToCore.cpp:13009:    // void __moore_uvm_report_xxx(
lib/Conversion/MooreToCore/MooreToCore.cpp:14637:    // TODO: This lowering is only correct if the condition is two-valued. If
lib/Conversion/MooreToCore/MooreToCore.cpp:24335:  // void __moore_xxx(MooreString *message)
lib/Conversion/MooreToCore/MooreToCore.cpp:30654:  // FIXME: Unpacked arrays support more element types than their packed
lib/Conversion/MooreToCore/MooreToCore.cpp:30683:  // FIXME: Mapping unpacked struct type to struct type in hw dialect may be a
test/Conversion/ArcToLLVM/lower-arc-to-llvm.mlir:151:// FIXME: this does not really belong here, but there is no better place either.
test/Conversion/ExportVerilog/sv-dialect.mlir:1959:sv.macro.error "my message xxx yyy"
test/Conversion/ExportVerilog/sv-dialect.mlir:1961:// CHECK: `_ERROR_my_message_xxx_yyy
test/Conversion/ExportVerilog/name-legalize.mlir:84:hw.module @useParametersNameConflict(in %xxx: i8) {
test/Conversion/ExportVerilog/name-legalize.mlir:89:  // CHECK:  .p1 (xxx)
test/Conversion/ExportVerilog/name-legalize.mlir:91:  hw.instance "inst" @parametersNameConflict<p2: i42 = 27, wire: i1 = 0>(p1: %xxx: i8) -> ()
test/Conversion/ExportVerilog/name-legalize.mlir:95:    // CHECK: reg [3:0] xxx_0;
test/Conversion/ExportVerilog/name-legalize.mlir:96:    %0 = sv.reg name "xxx" : !hw.inout<i4>
test/Conversion/ExportVerilog/hw-dialect.mlir:274:/// TODO: Specify parameter declarations.
test/Conversion/ExportVerilog/hw-dialect.mlir:365:// FIXME: The MLIR parser doesn't accept an i0 even though it is valid IR,
test/Conversion/ExportVerilog/hw-dialect.mlir:1257:  // FIXME: Decl word should be localparam.
test/Conversion/ExportVerilog/pretty.mlir:243:hw.module @ForStatement(in %aaaaaaaaaaa: i5, in %xxxxxxxxxxxxxxx : i2, in %yyyyyyyyyyyyyyy : i2, in %zzzzzzzzzzzzzzz : i2) {
test/Conversion/ExportVerilog/pretty.mlir:246:    %x_and_y = comb.and %xxxxxxxxxxxxxxx, %yyyyyyyyyyyyyyy : i2
test/Conversion/ExportVerilog/pretty.mlir:247:    %x_or_y = comb.or %xxxxxxxxxxxxxxx, %yyyyyyyyyyyyyyy : i2
test/Conversion/ExportVerilog/pretty.mlir:250:    %eq = comb.icmp eq %xxxxxxxxxxxxxxx, %yyyyyyyyyyyyyyy : i2
test/Conversion/ExportVerilog/pretty.mlir:255:    // CHECK-NEXT:           xxxxxxxxxxxxxxx == yyyyyyyyyyyyyyy ? _GEN : _GEN_0) begin
test/Conversion/ImportAIGER/basic-binary.mlir:3:// TODO: After the AIGER exporter is upstreamed, generate AIG file from this MLIR.
test/Conversion/MooreToCore/interface-timing-after-inlining.sv:8:// FIXME: moore.wait_event/detect_event inside func.func are not yet lowered.
test/Conversion/MooreToCore/basic.mlir:173:    %0 = moore.constant bXXXXXX : l6
test/Conversion/MooreToCore/basic.mlir:628:  // TODO: moore.procedure always_comb
test/Conversion/MooreToCore/basic.mlir:629:  // TODO: moore.procedure always_latch
test/Conversion/ImportVerilog/basic.sv:773:    // CHECK: moore.constant hXXXXXXXX : l32
test/Conversion/ImportVerilog/four-state-constants.sv:45:  assign all_x = 4'bxxxx;
test/Conversion/SeqToSV/error.mlir:3:// TODO: Improve the error message
test/Conversion/ImportVerilog/cross-select-with-wide-auto-domain-supported.sv:3:module CrossSelectWithWideAutoDomainSupported;
```

### Unsupported/Unimplemented/Not Implemented

```text
utils/run_mutation_cover.sh:406:  echo "Unsupported mutant format: $MUTANT_FORMAT (expected il|v|sv)." >&2
utils/summarize_circt_sim_jit_reports.py:113:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
tools/circt-verilog/circt-verilog.cpp:409:  cl::opt<bool> continueOnUnsupportedSVA{
tools/circt-verilog/circt-verilog.cpp:410:      "sva-continue-on-unsupported",
tools/circt-verilog/circt-verilog.cpp:411:      cl::desc("Continue lowering when an unsupported SVA construct is "
tools/circt-verilog/circt-verilog.cpp:881:  options.continueOnUnsupportedSVA = opts.continueOnUnsupportedSVA;
tools/circt-sim-compile/circt-sim-compile.cpp:630:        *rejectionReason = "builtin.unrealized_conversion_cast:unsupported_arity";
tools/circt-sim-compile/circt-sim-compile.cpp:1417:    bool unsupported = false;
tools/circt-sim-compile/circt-sim-compile.cpp:1427:            skipReason = ("unsupported_call:" + callee->str());
tools/circt-sim-compile/circt-sim-compile.cpp:1429:            skipReason = "unsupported_call:indirect";
tools/circt-sim-compile/circt-sim-compile.cpp:1431:          skipReason = ("unsupported_op:" +
tools/circt-sim-compile/circt-sim-compile.cpp:1434:        unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1439:          skipReason = "unsupported_op:hw.struct_extract";
tools/circt-sim-compile/circt-sim-compile.cpp:1440:          unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1446:          skipReason = "unsupported_op:hw.struct_create";
tools/circt-sim-compile/circt-sim-compile.cpp:1447:          unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1453:        unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1467:          unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1480:              unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1484:              skipReason = ("operand_dep_unsupported:" +
tools/circt-sim-compile/circt-sim-compile.cpp:1486:              unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1492:      if (unsupported)
tools/circt-sim-compile/circt-sim-compile.cpp:1497:    if (unsupported || opsToClone.empty()) {
tools/circt-sim-compile/circt-sim-compile.cpp:1498:      if (unsupported) {
tools/circt-sim-compile/circt-sim-compile.cpp:1500:          skipReason = "unsupported:unknown";
tools/circt-sim-compile/circt-sim-compile.cpp:1514:    bool cloneUnsupported = false;
tools/circt-sim-compile/circt-sim-compile.cpp:1521:            cloneUnsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1550:            cloneUnsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:1575:    if (cloneUnsupported) {
tools/circt-sim-compile/circt-sim-compile.cpp:1577:        cloneSkipReason = "unsupported:unknown";
tools/circt-sim-compile/circt-sim-compile.cpp:1592:/// This keeps the module-init pipeline conservative while avoiding unsupported
tools/circt-sim-compile/circt-sim-compile.cpp:1859:/// This is intentionally conservative: unsupported fragments are rendered as a
tools/circt-sim-compile/circt-sim-compile.cpp:1872:    formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1899:      formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1911:      formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1923:      formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1936:      formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1948:      formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1961:        formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1975:        formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:1989:        formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:2009:          formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:2026:  formatStr.append("<unsupported>");
tools/circt-sim-compile/circt-sim-compile.cpp:3487:      bool unsupported = false;
tools/circt-sim-compile/circt-sim-compile.cpp:3493:          unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:3499:      if (!unsupported && calleeTy.getNumResults() == 1) {
tools/circt-sim-compile/circt-sim-compile.cpp:3504:          unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:3508:      if (unsupported)
tools/circt-sim-compile/circt-sim-compile.cpp:3526:            unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:3532:      if (unsupported)
tools/circt-sim-compile/circt-sim-compile.cpp:3825:        return 0; // nested unsupported type
tools/circt-sim-compile/circt-sim-compile.cpp:3830:  return 0; // unsupported
tools/circt-sim-compile/circt-sim-compile.cpp:4171:/// functions containing unsupported ops) without crashing at runtime.
tools/circt-sim-compile/circt-sim-compile.cpp:4215:  bool sawUnsupportedReferencedExternal = false;
tools/circt-sim-compile/circt-sim-compile.cpp:4253:      sawUnsupportedReferencedExternal = true;
tools/circt-sim-compile/circt-sim-compile.cpp:4259:    bool hasUnsupported = false;
tools/circt-sim-compile/circt-sim-compile.cpp:4260:    std::string unsupportedReason;
tools/circt-sim-compile/circt-sim-compile.cpp:4265:        hasUnsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:4266:        unsupportedReason = ("parameter " + std::to_string(i) + " has type " +
tools/circt-sim-compile/circt-sim-compile.cpp:4271:    if (!hasUnsupported && !isa<LLVM::LLVMVoidType>(funcTy.getReturnType())) {
tools/circt-sim-compile/circt-sim-compile.cpp:4275:        hasUnsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:4276:      if (hasUnsupported)
tools/circt-sim-compile/circt-sim-compile.cpp:4277:        unsupportedReason = ("return type " + formatType(retTy));
tools/circt-sim-compile/circt-sim-compile.cpp:4279:    if (hasUnsupported) {
tools/circt-sim-compile/circt-sim-compile.cpp:4282:             "function: unsupported trampoline ABI ("
tools/circt-sim-compile/circt-sim-compile.cpp:4283:          << unsupportedReason << ")";
tools/circt-sim-compile/circt-sim-compile.cpp:4284:      sawUnsupportedReferencedExternal = true;
tools/circt-sim-compile/circt-sim-compile.cpp:4289:  if (sawUnsupportedReferencedExternal)
tools/circt-sim-compile/circt-sim-compile.cpp:7498:        bool unsupported = false;
tools/circt-sim-compile/circt-sim-compile.cpp:7506:            unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:7510:        if (!unsupported) {
tools/circt-sim-compile/circt-sim-compile.cpp:7522:              unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:7525:            unsupported = true;
tools/circt-sim-compile/circt-sim-compile.cpp:7527:          if (!unsupported) {
tools/circt-sim-compile/circt-sim-compile.cpp:7536:      // If not found in original module (or unsupported type), skip.
utils/run_opentitan_fpv_circt_lec.py:414:            "case-level LEC health checks cannot yet derive native cover "
utils/formal/lib/runner_common.py:117:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
test/Runtime/uvm/uvm_phase_wait_for_state_test.sv:2:// UNSUPPORTED: true
unittests/Runtime/MooreRuntimeTest.cpp:10074:  __uvm_run_test("unimplemented_test", 18);
unittests/Runtime/MooreRuntimeTest.cpp:10081:  EXPECT_NE(output.find("unimplemented_test"), std::string::npos);
test/Runtime/uvm/uvm_phase_aliases_test.sv:2:// UNSUPPORTED: true
utils/run_sv_tests_circt_sim.sh:516:      if grep -q "unsupported\|not yet implemented\|unimplemented" "$sim_log" 2>/dev/null; then
utils/run_sv_tests_circt_sim.sh:517:        result="UNSUPPORTED"
utils/check_opentitan_target_manifest_drift.py:87:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/check_opentitan_connectivity_contract_fingerprint_parity.py:90:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
tools/circt-test/circt-test.cpp:320:  Unsupported,
tools/circt-test/circt-test.cpp:660:  case TestStatus::Unsupported:
tools/circt-test/circt-test.cpp:661:    os << ansiDim << "unsupported" << ansiReset;
tools/circt-test/circt-test.cpp:786:                 << "unsupported attributes: `" << test.attrs
tools/circt-test/circt-test.cpp:834:    return TestStatus::Unsupported;
tools/circt-test/circt-test.cpp:1099:  unsigned numUnsupported = 0;
tools/circt-test/circt-test.cpp:1111:    case TestStatus::Unsupported:
tools/circt-test/circt-test.cpp:1112:      ++numUnsupported;
tools/circt-test/circt-test.cpp:1136:  if (numUnsupported > 0)
tools/circt-test/circt-test.cpp:1137:    os << "; " << numUnsupported << " unsupported";
utils/run_opentitan_connectivity_circt_lec.py:130:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/run_opentitan_connectivity_circt_lec.py:366:                "unsupported connectivity rule type in manifest: "
utils/select_opentitan_connectivity_cfg.py:276:                    f"unsupported connectivity CSV row kind '{first}' in "
utils/run_yosys_sva_circt_bmc.sh:142:UNSUPPORTED_SVA_POLICY="${UNSUPPORTED_SVA_POLICY:-strict}"
utils/run_yosys_sva_circt_bmc.sh:298:if [[ "$UNSUPPORTED_SVA_POLICY" != "strict" && "$UNSUPPORTED_SVA_POLICY" != "lenient" ]]; then
utils/run_yosys_sva_circt_bmc.sh:299:  echo "invalid UNSUPPORTED_SVA_POLICY: $UNSUPPORTED_SVA_POLICY (expected strict or lenient)" >&2
utils/run_yosys_sva_circt_bmc.sh:2747:      echo "error: unsupported YOSYS_SVA_MODE_SUMMARY_HISTORY_JSONL_FILE line format in $file at line $lineno" >&2
utils/run_yosys_sva_circt_bmc.sh:2867:        echo "error: unsupported drop-event id hash mode: $drop_events_id_hash_mode" >&2
utils/run_yosys_sva_circt_bmc.sh:3026:        echo "error: unsupported drop-event lock backend: $selected_backend" >&2
utils/run_yosys_sva_circt_bmc.sh:3186:    echo "error: unsupported YOSYS_SVA_MODE_SUMMARY_HISTORY_TSV_FILE header in $file" >&2
utils/run_yosys_sva_circt_bmc.sh:8522:  # In smoke mode we treat sim-only tests as unsupported by BMC and skip them
utils/run_yosys_sva_circt_bmc.sh:8616:    if [[ "$UNSUPPORTED_SVA_POLICY" == "lenient" ]]; then
utils/run_yosys_sva_circt_bmc.sh:8617:      verilog_args+=("--sva-continue-on-unsupported")
utils/run_yosys_sva_circt_bmc.sh:8676:  if [[ "$UNSUPPORTED_SVA_POLICY" == "lenient" ]]; then
utils/run_yosys_sva_circt_bmc.sh:8677:    verilog_args+=("--sva-continue-on-unsupported")
utils/run_yosys_sva_circt_bmc.sh:8761:  if [[ "$UNSUPPORTED_SVA_POLICY" == "lenient" ]]; then
utils/run_yosys_sva_circt_bmc.sh:8762:    bmc_args+=("--drop-unsupported-sva")
frontends/PyRTG/src/pyrtg/support.py:60:  assert False, "Unsupported value"
frontends/PyRTG/src/pyrtg/support.py:117:  raise ValueError("unsupported type")
tools/circt-mut/circt-mut.cpp:2368:    error = (Twine("circt-mut cover: unsupported --mutant-format value for "
tools/circt-mut/circt-mut.cpp:6369:      error = (Twine("circt-mut run: unsupported TOML escape in ") + path +
tools/circt-mut/circt-mut.cpp:14642:    // Keep full compatibility by deferring unsupported/unknown options to the
tools/circt-mut/circt-mut.cpp:15299:    errs() << "circt-mut generate: unsupported design extension for "
utils/run_avip_circt_sim.sh:249:      echo "error: unsupported AVIP_SET '$AVIP_SET' (use core8 or all9)" >&2
utils/run_avip_circt_sim.sh:269:    echo "error: unsupported CIRCT_SIM_MODE '$CIRCT_SIM_MODE' (use interpret or compile)" >&2
include/circt/Transforms/Passes.td:86:    `--map-arith-to-comb`) do not encounter unsupported index-typed arithmetic.
utils/generate_mutations_yosys.sh:696:    echo "Unsupported design extension for $DESIGN (expected .il/.v/.sv)." >&2
tools/circt-sim/AOTProcessCompiler.h:211:  /// with unsupported ops are skipped.
utils/create_mutated_yosys.sh:95:    echo "Unsupported design extension in '$DESIGN_FILE' (expected .il/.v/.sv)." >&2
utils/create_mutated_yosys.sh:106:    echo "Unsupported output extension in '$OUTPUT_FILE' (expected .il/.v/.sv)." >&2
utils/run_opentitan_fpv_circt_bmc.py:231:            "unsupported stopat selector "
utils/run_opentitan_fpv_circt_bmc.py:373:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/run_opentitan_fpv_circt_bmc.py:1607:                add_contract_error(row, "unsupported_stopat_selector")
test/arcilator/apb_testbench.sv:2:// UNSUPPORTED: true
utils/check_opentitan_compile_contract_drift.py:95:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh:76:`define UVM_TLM_TASK_ERROR "TLM-2 interface task not implemented"
lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh:81:`define UVM_TLM_FUNCTION_ERROR "UVM TLM 2 interface function not implemented"
utils/run_formal_all.sh:2241:      f"refresh policy profiles manifest signer OCSP {field} parse failed for key_id '{manifest_key_id}': unsupported timestamp '{raw}'",
utils/run_formal_all.sh:2452:          f"refresh policy profiles manifest signer CRL nextUpdate parse failed for key_id '{manifest_key_id}': unsupported timestamp '{crl_next_update}'",
utils/run_formal_all.sh:7811:  print(f"unsupported refresh URI scheme '{scheme}'", file=sys.stderr)
utils/run_formal_all.sh:8513:      f"lane state Ed25519 OCSP {field} parse failed for key_id '{target_key_id}': unsupported timestamp '{raw}'",
utils/run_formal_all.sh:8716:            f"lane state Ed25519 CRL nextUpdate parse failed for key_id '{target_key_id}': unsupported timestamp '{crl_next_update}'",
utils/run_formal_all.sh:14091:      f"unsupported rule_type '{rule_type}' in connectivity rules manifest: {rules_path}"
utils/run_formal_all.sh:17626:            f"unsupported status '{status}'"
utils/run_formal_all.sh:20404:                    f"{label} unsupported entry kind '{kind}' at {allowlist_path}:{lineno}"
utils/run_formal_all.sh:20549:                    f"{label} line {lineno} unsupported selector kind '{kind}' in {budget_path}"
utils/run_formal_all.sh:20688:                    f"mutation/LEC lane-map unsupported source kind '{kind}' at {map_path}:{lineno}"
utils/run_formal_all.sh:20779:                    f"BMC/LEC contract-fingerprint case-ID map unsupported source kind '{kind}' at {map_path}:{lineno}"
tools/circt-lec/circt-lec.cpp:194:              cl::desc("Allow LEC to abstract unsupported LLHD/inout cases"),
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:707:static std::string formatUnsupportedProcessOpDetail(Operation &op) {
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:750:  auto traceUnsupported = []() {
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:752:        std::getenv("CIRCT_SIM_TRACE_JIT_UNSUPPORTED_SHAPES");
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:756:  auto dumpUnsupportedShape = [&](StringRef detail) {
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:757:    if (!traceUnsupported || !state.getProcessOp())
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:766:    llvm::errs() << "[JIT-UNSUPPORTED] proc=" << procId << " name=" << procName
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:870:    std::string detail = getUnsupportedThunkDeoptDetail(state);
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:873:    dumpUnsupportedShape(detail);
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:874:    return ProcessThunkInstallResult::UnsupportedOperation;
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:904:std::string LLHDProcessInterpreter::getUnsupportedThunkDeoptDetail(
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:917:        return (Twine("combinational_unsupported_terminator:") +
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:923:          return (Twine("combinational_unsupported:") +
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:932:    return (Twine("combinational_unsupported:first_op:") +
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:950:              return formatUnsupportedProcessOpDetail(*it);
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:964:              return (Twine("multiblock_unsupported_terminator:") +
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:972:            return (Twine("multiblock_unsupported_terminator:") +
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:979:              return formatUnsupportedProcessOpDetail(*it);
tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:1003:        return formatUnsupportedProcessOpDetail(op);
tools/circt-sim/AOTProcessCompiler.cpp:1006:                 << "[AOT] Process not compilable: unsupported sim op "
lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh:34:`define UVM_TASK_ERROR "UVM TLM interface task not implemented"
lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh:35:`define UVM_FUNCTION_ERROR "UVM TLM interface function not implemented"
utils/mutation_mcy/lib/native_mutation_plan.py:57:        raise ValueError("unsupported native ops: " + ", ".join(sorted(set(unknown))))
lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh:35:`define UVM_TLM_FIFO_TASK_ERROR "fifo channel task not implemented"
lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh:36:`define UVM_TLM_FIFO_FUNCTION_ERROR "fifo channel function not implemented"
utils/run_avip_xcelium_reference.sh:94:      echo "error: unsupported AVIP_SET '$AVIP_SET' (use core8 or all9)" >&2
lib/Runtime/uvm-core/src/tlm1/uvm_sqr_ifs.svh:38:`define UVM_SEQ_ITEM_TASK_ERROR "Sequencer interface task not implemented"
lib/Runtime/uvm-core/src/tlm1/uvm_sqr_ifs.svh:39:`define UVM_SEQ_ITEM_FUNCTION_ERROR "Sequencer interface function not implemented"
utils/check_opentitan_fpv_objective_parity_drift.py:83:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
tools/circt-bmc/circt-bmc.cpp:159:static cl::opt<bool> dropUnsupportedSVA(
tools/circt-bmc/circt-bmc.cpp:160:    "drop-unsupported-sva",
tools/circt-bmc/circt-bmc.cpp:161:    cl::desc("Drop assert-like ops tagged with circt.unsupported_sva before "
tools/circt-bmc/circt-bmc.cpp:271:static unsigned dropUnsupportedSvaOps(mlir::ModuleOp module) {
tools/circt-bmc/circt-bmc.cpp:272:  constexpr const char *kUnsupportedSvaAttr = "circt.unsupported_sva";
tools/circt-bmc/circt-bmc.cpp:275:    if (!op->hasAttr(kUnsupportedSvaAttr))
tools/circt-bmc/circt-bmc.cpp:770:    // Prune unreachable symbols early so unsupported ops in dead modules do
tools/circt-bmc/circt-bmc.cpp:1291:  if (dropUnsupportedSVA) {
tools/circt-bmc/circt-bmc.cpp:1292:    unsigned dropped = dropUnsupportedSvaOps(*module);
tools/circt-bmc/circt-bmc.cpp:1295:                   << " unsupported SVA assert-like op(s)\n";
unittests/Support/TestReportingTest.cpp:54:  tc.skip("Not implemented");
unittests/Support/TestReportingTest.cpp:58:  EXPECT_EQ(tc.message, "Not implemented");
unittests/Support/TestReportingTest.cpp:195:  tc.skip("Not implemented");
unittests/Support/TestReportingTest.cpp:203:  EXPECT_NE(output.find("Not implemented"), std::string::npos);
tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp:4695:      // (e.g., unimplemented sequencer interfaces, missing config_db entries).
utils/run_opentitan_fpv_bmc_policy_workflow.sh:323:  echo "unsupported mode: $MODE" >&2
utils/check_opentitan_fpv_bmc_evidence_parity.py:111:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/check_opentitan_connectivity_cover_parity.py:101:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/check_avip_circt_sim_mode_parity.py:94:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/check_opentitan_connectivity_status_parity.py:101:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
tools/circt-sim/circt-sim.cpp:2015:    llvm::errs() << "[circt-sim] Warning: parallel scheduler is unsupported "
tools/circt-sim/circt-sim.cpp:3277:                    "watchdog is unsupported on emscripten; using "
tools/arcilator/BehavioralLowering.cpp:197:        << "unsupported in arcilator BehavioralLowering: "
tools/arcilator/BehavioralLowering.cpp:478:        << "unsupported in arcilator BehavioralLowering: llhd.wait "
tools/arcilator/BehavioralLowering.cpp:479:           "(process suspension lowering not implemented)";
tools/arcilator/BehavioralLowering.cpp:501:/// Four-valued constants with X/Z bits remain unsupported in this path.
tools/arcilator/BehavioralLowering.cpp:516:          << "unsupported in arcilator BehavioralLowering: moore.constant "
tools/arcilator/BehavioralLowering.cpp:517:             "with X/Z bits (four-valued constant lowering not implemented)";
tools/arcilator/BehavioralLowering.cpp:547:/// semantics are not implemented in this pass yet.
tools/arcilator/BehavioralLowering.cpp:1162:        << "unsupported in arcilator BehavioralLowering: hw.bitcast "
include/circt/Support/LTLSequenceNFA.h:289:      emitError(loc, "unsupported sequence lowering for block argument");
include/circt/Support/LTLSequenceNFA.h:395:    defOp->emitError("unsupported sequence lowering");
utils/internal/checks/wasm_cxx20_contract_check.sh:45:  echo "[wasm-cxx20-contract] configure script accepted unsupported C++ standard override (17)" >&2
tools/circt-sim/LLHDProcessInterpreter.h:1518:    UnsupportedOperation,
tools/circt-sim/LLHDProcessInterpreter.h:1687:  /// Emit a concise unsupported-shape detail for deopt telemetry.
tools/circt-sim/LLHDProcessInterpreter.h:1689:  getUnsupportedThunkDeoptDetail(const ProcessExecutionState &state) const;
tools/circt-sim/LLHDProcessInterpreterBytecode.cpp:50:  /// contains unsupported operations.
tools/circt-sim/LLHDProcessInterpreterBytecode.cpp:78:  /// Try to compile a single operation. Returns false if unsupported.
tools/circt-sim/LLHDProcessInterpreterBytecode.cpp:664:      return false; // Unsupported predicate
tools/circt-sim/LLHDProcessInterpreterBytecode.cpp:783:  // Unsupported operation.
tools/circt-sim/LLHDProcessInterpreterBytecode.cpp:784:  LLVM_DEBUG(llvm::dbgs() << "[Bytecode] Unsupported op: " << op->getName()
tools/circt-sim/LLHDProcessInterpreterBytecode.cpp:817:                   << "[Bytecode] Failed to compile process: unsupported op "
test/Tools/circt-verilog/commandline.mlir:29:// CHECK-DAG: --sva-continue-on-unsupported
lib/Runtime/uvm-core/src/reg/uvm_vreg_field.svh:376:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg_field::get_rights() on unimplemented virtual field \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg_field.svh:412:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg_field::write() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg_field.svh:554:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg_field::read() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg_field.svh:651:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg_field::poke() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg_field.svh:755:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg_field::peek() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:781:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_offset_in_memory() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:793:     `uvm_error("RegModel", $sformatf("Cannot get address of of unimplemented virtual register \"%s\".", this.get_full_name()))
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:803:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_size() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:819:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_n_memlocs() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:830:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_incr() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:841:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_n_maps() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:852:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_maps() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:863:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::is_in_map() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:874:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_rights() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:885:     `uvm_error("RegModel", $sformatf("Cannot call uvm_vreg::get_rights() on unimplemented virtual register \"%s\"",
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:935:     `uvm_error("RegModel", $sformatf("Cannot write to unimplemented virtual register \"%s\".", this.get_full_name()))
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:1047:     `uvm_error("RegModel", $sformatf("Cannot read from unimplemented virtual register \"%s\".", this.get_full_name()))
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:1149:     `uvm_error("RegModel", $sformatf("Cannot poke in unimplemented virtual register \"%s\".", this.get_full_name()))
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:1196:     `uvm_error("RegModel", $sformatf("Cannot peek in from unimplemented virtual register \"%s\".", this.get_full_name()))
lib/Runtime/uvm-core/src/reg/uvm_vreg.svh:1242:     $sformat(convert2string, "%sunimplemented", convert2string);
lib/Runtime/uvm-core/src/reg/uvm_reg_sequence.svh:69:// Note- The convenience API not yet implemented.
lib/Runtime/uvm-core/src/reg/uvm_reg_model.svh:348:// if portions of the registers are not implemented.
tools/kanagawatool/kanagawatool.cpp:206:  // Legalize unsupported operations within the modules.
lib/Runtime/uvm-core/src/reg/uvm_reg_field.svh:2086:  `uvm_warning("RegModel","RegModel field copy not yet implemented")
lib/Runtime/uvm-core/src/reg/uvm_reg_field.svh:2095:  `uvm_warning("RegModel","RegModel field compare not yet implemented")
tools/hlstool/hlstool.cpp:297:  // Legalize unsupported operations within the modules.
utils/check_opentitan_connectivity_objective_parity.py:151:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
lib/Runtime/uvm-core/src/reg/uvm_reg_backdoor.svh:46:// to registers and memories that are not implemented in pure SystemVerilog
utils/run_opentitan_connectivity_circt_bmc.py:120:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/run_opentitan_connectivity_circt_bmc.py:412:                "unsupported connectivity rule type in manifest: "
test/Tools/circt-mut-forward-generate-cache-fallback.test:4:// RUN: CIRCT_MUT_SCRIPTS_DIR=%t/scripts circt-mut generate --native-unsupported foo --design %t/design.v --out %t/mutations.txt --count 2 > %t/out.txt
test/Tools/circt-mut-forward-generate-cache-fallback.test:8:// CHECK: ARG:--native-unsupported
lib/Runtime/uvm-core/src/reg/uvm_mem_mam.svh:92:   // THRIFTY  - Reused previously released memory as much as possible (not yet implemented)
utils/run_regression_unified.sh:120:        manifest_error "$row_no" "unsupported profile token '$token' (expected smoke|nightly|full|all)"
lib/Runtime/uvm-core/src/reg/uvm_mem.svh:982:   `uvm_error("RegModel", "uvm_mem::get_vreg_by_offset() not yet implemented")
utils/check_opentitan_fpv_objective_parity.py:170:                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
utils/run_avip_arcilator_sim.sh:103:      echo "error: unsupported AVIP_SET '$AVIP_SET' (use core8 or all9)" >&2
test/circt-verilog/roundtrip-register-enable.sv:4:// UNSUPPORTED: valgrind
test/circt-verilog/redundant-files.sv:4:// UNSUPPORTED: valgrind
test/circt-verilog/registers.sv:4:// UNSUPPORTED: valgrind
test/circt-verilog/library-locations.sv:4:// UNSUPPORTED: valgrind
test/circt-verilog/memories.sv:4:// UNSUPPORTED: valgrind
include/circt/Dialect/Verif/VerifOps.td:287:    elements (i.e., registers and memories) are currently unsupported.
tools/circt-sim/LLHDProcessInterpreter.cpp:26757:  return "<unsupported format>";
tools/circt-sim/LLHDProcessInterpreter.cpp:40655:  // Initialize return slots up-front so unsupported/failed paths never leak
tools/circt-sim/LLHDProcessInterpreter.cpp:40717:      static bool warnedUnsupportedTrampolineNativeAbi = false;
tools/circt-sim/LLHDProcessInterpreter.cpp:40718:      if (!warnedUnsupportedTrampolineNativeAbi) {
tools/circt-sim/LLHDProcessInterpreter.cpp:40720:            << "[circt-sim] WARNING: unsupported trampoline native fallback ABI"
tools/circt-sim/LLHDProcessInterpreter.cpp:40724:        warnedUnsupportedTrampolineNativeAbi = true;
test/circt-verilog/command-files.sv:4:// UNSUPPORTED: valgrind
lib/Target/ExportSystemC/Patterns/SystemCEmissionPatterns.cpp:311:      p.emitError(op, "member access kind not implemented");
include/circt/Dialect/SystemC/SystemCTypes.h:52:/// unsupported types result in a None return value.
lib/Target/ExportSystemC/EmissionPrinter.cpp:32:  os << "\n<<UNSUPPORTED OPERATION (" << op->getName() << ")>>\n";
lib/Target/ExportSystemC/EmissionPrinter.cpp:50:  os << "<<UNSUPPORTED TYPE (" << type << ")>>";
lib/Target/ExportSystemC/EmissionPrinter.cpp:66:  os << "<<UNSUPPORTED ATTRIBUTE (" << attr << ")>>";
include/circt/Dialect/SystemC/SystemCStructure.td:192:    (not yet implemented).
lib/Support/CoverageDatabase.cpp:617:                                   "Unsupported database version");
include/circt/Dialect/Moore/MooreOps.td:2225:    Non-integral data types define other rules which are not yet implemented.
test/circt-reduce/name-sanitizer.mlir:1:// UNSUPPORTED: system-windows
test/circt-reduce/operation-pruner.mlir:1:// UNSUPPORTED: system-windows
test/circt-reduce/must-dedup-children.mlir:1:// UNSUPPORTED: system-windows
test/circt-reduce/make-symbols-private.mlir:1:// UNSUPPORTED: system-windows
test/circt-reduce/cse-reduction.mlir:1:// UNSUPPORTED: system-windows
test/circt-reduce/canonicalize-reduction.mlir:1:// UNSUPPORTED: system-windows
lib/Runtime/uvm-core/src/comps/uvm_push_driver.svh:103:    uvm_report_fatal("UVM_PUSH_DRIVER", "Put task for push driver is not implemented", UVM_NONE);
lib/Scheduling/SimplexSchedulers.cpp:1062:    llvm_unreachable("Unsupported objective requested");
lib/Firtool/Firtool.cpp:382:  // Legalize unsupported operations within the modules.
test/Tools/run-formal-all-strict-gate-bmc-drop-remark-cases-verilator.test:4:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\n: "${OUT:?}"\nmkdir -p "$(dirname "$OUT")"\nprintf "PASS\\tcase_pass\\t/tests/pass.sv\\n" > "$OUT"\necho "warning: unsupported construct will be dropped during lowering"\necho "verilator-verification summary: total=1 pass=1 fail=0 xfail=0 xpass=0 error=0 skip=0 unknown=0 timeout=0"\n' > %t/utils/run_verilator_verification_circt_bmc.sh
test/Tools/run-formal-all-strict-gate-bmc-drop-remark-cases-verilator.test:8:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\n: "${OUT:?}"\nmkdir -p "$(dirname "$OUT")"\nprintf "PASS\\tcase_pass\\t/tests/pass.sv\\n" > "$OUT"\necho "warning: unsupported construct will be dropped during lowering"\necho "warning: unsupported construct will be dropped during lowering"\necho "verilator-verification summary: total=1 pass=1 fail=0 xfail=0 xpass=0 error=0 skip=0 unknown=0 timeout=0"\n' > %t/utils/run_verilator_verification_circt_bmc.sh
test/Tools/run-yosys-sva-circt-lec-drop-remarks.test:3:// RUN: printf '#!/usr/bin/env bash\necho \"// dummy mlir\"\necho \"warning: unsupported construct will be dropped during lowering\" >&2\n' > %t/fake-verilog.sh
test/Tools/run-yosys-sva-circt-lec-drop-remarks.test:21:// REASON: simple{{[[:space:]]+}}{{.*}}simple.sv{{[[:space:]]+}}unsupported construct will be dropped during lowering
test/Tools/run-verilator-verification-circt-lec-drop-remarks.test:4:// RUN: printf '#!/bin/sh\necho \"module @top {\"\necho \"  verif.assert %c0_i1\"\necho \"}\"\necho \"warning: unsupported construct will be dropped during lowering\" >&2\n' > %t/bin/circt-verilog
test/Tools/run-verilator-verification-circt-lec-drop-remarks.test:16:// REASON: drop_case{{[[:space:]]+}}{{.*}}drop_case.sv{{[[:space:]]+}}unsupported construct will be dropped during lowering
test/Tools/run-sv-tests-circt-lec-drop-remarks.test:4:// RUN: printf '#!/bin/sh\necho \"module @top {\"\necho \"  verif.assert %c0_i1\"\necho \"}\"\necho \"warning: unsupported construct will be dropped during lowering\" >&2\n' > %t/bin/circt-verilog
test/Tools/run-sv-tests-circt-lec-drop-remarks.test:16:// REASON: drop_case{{[[:space:]]+}}{{.*}}drop_case.sv{{[[:space:]]+}}unsupported construct will be dropped during lowering
include/circt/Dialect/Sim/DPIRuntime.h:778:    // Fall back to void return for unsupported signatures
test/Tools/run-yosys-sva-bmc-drop-remarks.test:3:// RUN: printf '#!/usr/bin/env bash\necho \"// dummy mlir\"\necho \"warning: unsupported construct will be dropped during lowering\" >&2\n' > %t/fake-verilog.sh
test/Tools/run-yosys-sva-bmc-drop-remarks.test:28:// REASON: simple{{[[:space:]]+}}{{.*}}simple.sv{{[[:space:]]+}}unsupported construct will be dropped during lowering
lib/Runtime/uvm-core/src/base/uvm_object.svh:743:  extern virtual function void m_unsupported_set_local(uvm_resource_base rsrc);
lib/Runtime/uvm-core/src/base/uvm_object.svh:826:  uvm_report_error("NOTYPID", "get_type not implemented in derived class.", UVM_NONE);
lib/Runtime/uvm-core/src/base/uvm_object.svh:948:// m_unsupported_set_local
lib/Runtime/uvm-core/src/base/uvm_object.svh:952:function void uvm_object::m_unsupported_set_local(uvm_resource_base rsrc);
lib/Runtime/uvm-core/src/base/uvm_factory.svh:739://|    // get_type_name not implemented by macro for parameterized classes
lib/Runtime/uvm-core/src/base/uvm_printer.svh:1000:     `uvm_warning("PRINTER_UNKNOWN_RADIX",$sformatf("set_radix_string called with unsupported radix %s",radix))
lib/Runtime/uvm-core/src/base/uvm_component.svh:1655:  // produce message for unsupported types from apply_config_settings
lib/Runtime/uvm-core/src/base/uvm_component.svh:1656:  uvm_resource_base m_unsupported_resource_base = null;
lib/Runtime/uvm-core/src/base/uvm_component.svh:1657:  extern function void m_unsupported_set_local(uvm_resource_base rsrc);
lib/Runtime/uvm-core/src/base/uvm_component.svh:2569:   `uvm_warning("COMP/SPND/UNIMP", "suspend() not implemented")
lib/Runtime/uvm-core/src/base/uvm_component.svh:2577:   `uvm_warning("COMP/RSUM/UNIMP", "resume() not implemented")
lib/Runtime/uvm-core/src/base/uvm_component.svh:3542:// m_unsupported_set_local (override)
lib/Runtime/uvm-core/src/base/uvm_component.svh:3545:function void uvm_component::m_unsupported_set_local(uvm_resource_base rsrc);
lib/Runtime/uvm-core/src/base/uvm_component.svh:3547:  m_unsupported_resource_base = rsrc;
lib/Runtime/uvm-core/src/base/uvm_phase.svh:2026:    `uvm_warning("NOTIMPL","uvm_phase::jump_all is not implemented and has been replaced by uvm_domain::jump_all")
include/circt/Dialect/SV/SVPasses.td:36:  let summary = "Eliminate features marked unsupported in LoweringOptions";
include/circt/Dialect/SV/SVPasses.td:39:      unsupported by some tools, e.g. multidimensional arrays.  This pass is
lib/Dialect/LLHD/Transforms/Deseq.cpp:501:                                << ": unsupported terminator " << op->getName()
include/circt/Dialect/Handshake/Visitor.h:47:    op->emitOpError("is unsupported operation");
include/circt/Dialect/Handshake/Visitor.h:118:    op->emitOpError("is unsupported operation");
lib/Dialect/Synth/Transforms/CutRewriter.cpp:87:      "Unsupported operation for simulation. isSupportedLogicOp should "
lib/Dialect/Synth/Transforms/CutRewriter.cpp:169:      return op->emitError("Unsupported operation for truth table simulation");
lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:167:    return failMatch("unsupported exit branch");
lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:181:    return failMatch("unsupported exit condition");
lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:211:      return failMatch("unsupported induction variable value");
lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:228:    return failMatch("unsupported increment");
lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:250:  return failMatch("unsupported loop bounds");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1194:          "unsupported llhd.probe on local ref without LLVM value type");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1217:          "unsupported conditional drive on local ref in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1220:          "unsupported non-constant drive time on local ref in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1224:          "unsupported llhd.drive on local ref without LLVM value type");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1244:      return mux.emitError("unsupported comb.mux on LLHD refs with non-probe "
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1304:              "unsupported predecessor for unused block argument in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1527:              "unsupported predecessor for pointer block argument collapse in "
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1834:                 << "unsupported comb.mux on divergent LLHD refs";
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1880:              "unsupported unrealized conversion on LLHD signal in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1883:              "unsupported LLHD signal cast without LLVM pointer type");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1906:                    "unsupported load from LLHD signal in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1909:                    "unsupported load conversion for LLHD signal in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1945:                  "unsupported store to LLHD signal in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1957:              "unsupported LLVM user of LLHD signal cast in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:1964:             << "unsupported LLHD signal use in LEC: " << user->getName();
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2124:              return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2337:          return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2372:        return driveOp.emitError("unsupported LLHD drive value in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2384:            return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2426:        return driveOp.emitError("unsupported LLHD register-state drive value");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2434:          return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2566:                    "unsupported LLHD drive path update in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2582:                return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2742:          return driveOp.emitError("unsupported LLHD drive value in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2750:                "unsupported LLHD drive path update in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:2769:          return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3182:        return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3209:      return probe.emitError("unsupported LLHD probe path in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3261:          return terminator->emitError("unsupported predecessor for LLHD ref");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3311:      return sigOp.emitError("unsupported LLHD signal use in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3327:        return probe.emitError("unsupported LLHD probe use in LEC");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3331:        return gep.emitError("unsupported GEP pattern for interface signal");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3349:            return cast.emitError("unsupported cast for interface signal");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3355:              return cast.emitError("unsupported cast use for interface");
lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp:3379:        return gep.emitError("unsupported interface signal use in LEC");
lib/Dialect/LLHD/Transforms/RemoveControlFlow.cpp:206:                 << "- Has unsupported terminator "
lib/Tools/circt-lec/LowerLLHDRefPorts.cpp:404:        module.emitError("unsupported ref type for driven input port");
lib/Tools/circt-lec/LowerLLHDRefPorts.cpp:522:          inst.emitError("unsupported ref type for output conversion");
lib/Tools/circt-lec/LowerLECLLVM.cpp:1717:          "unsupported LLVM aggregate conversion in LEC; add lowering");
lib/Tools/circt-lec/LowerLECLLVM.cpp:1846:  // declarations/symbols so strict unsupported-op checks only trigger on real
lib/Tools/circt-lec/LowerLECLLVM.cpp:1925:        "LEC LLVM lowering left unsupported LLVM operations");
lib/Dialect/LLHD/Transforms/InlineCalls.cpp:460:              "recursive function call cannot be inlined (unsupported in --ir-hw)");
lib/Dialect/LLHD/Transforms/InlineCalls.cpp:488:            "recursive function call cannot be inlined (unsupported in --ir-hw)");
include/circt/Dialect/RTG/Transforms/RTGPasses.td:56:    Option<"unsupportedInstructionsFile", "unsupported-instructions-file",
include/circt/Dialect/RTG/Transforms/RTGPasses.td:60:    ListOption<"unsupportedInstructions", "unsupported-instructions",
lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1383:      return Type(); // Unsupported type
lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1400:      return failure(); // Unsupported operand type
lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1409:      return failure(); // Unsupported result type
include/circt/Dialect/HW/Passes.td:162:           "false", "Ignore bitcasts involving unsupported types">
lib/Tools/circt-bmc/LowerToBMC.cpp:3126:    bool hasUnsupportedUse = false;
lib/Tools/circt-bmc/LowerToBMC.cpp:3138:        hasUnsupportedUse = true;
lib/Tools/circt-bmc/LowerToBMC.cpp:3142:    if (hasUnsupportedUse || hasEnable)
lib/Dialect/LLHD/IR/LLHDOps.cpp:917:    llvm_unreachable("Not implemented");
test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test:3:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\nif [[ " $* " != *" --sva-continue-on-unsupported "* ]]; then\n  echo "missing --sva-continue-on-unsupported" >&2\n  exit 9\nfi\necho "// dummy mlir"\n' > %t/fake-verilog.sh
test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test:4:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\nif [[ " $* " != *" --drop-unsupported-sva "* ]]; then\n  echo "missing --drop-unsupported-sva" >&2\n  exit 9\nfi\nexit 0\n' > %t/fake-bmc.sh
test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test:8:// RUN:   SKIP_VHDL=0 SKIP_FAIL_WITHOUT_MACRO=1 UNSUPPORTED_SVA_POLICY=lenient \
test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test:18:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\nif [[ " $* " == *" --sva-continue-on-unsupported "* ]]; then\n  echo "unexpected --sva-continue-on-unsupported in strict mode" >&2\n  exit 9\nfi\necho "// dummy mlir"\n' > %t.strict/fake-verilog.sh
test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test:19:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\nif [[ " $* " == *" --drop-unsupported-sva "* ]]; then\n  echo "unexpected --drop-unsupported-sva in strict mode" >&2\n  exit 9\nfi\nexit 0\n' > %t.strict/fake-bmc.sh
test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test:34:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\nif [[ " $* " != *" --sva-continue-on-unsupported "* ]]; then\n  echo "missing --sva-continue-on-unsupported (sim-only)" >&2\n  exit 9\nfi\necho "module {}"\n' > %t.sim/fake-verilog.sh
test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test:40:// RUN:   UNSUPPORTED_SVA_POLICY=lenient TEST_FILTER='.*' \
lib/Dialect/Sim/Transforms/ProceduralizeSim.cpp:106:                          "argument is unsupported.");
lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:83:             << "non-integer type argument is unsupported now";
test/Tools/circt-verilog-lsp-server/workspace-symbols.test:3:// UNSUPPORTED: valgrind
lib/Dialect/Seq/Transforms/LowerSeqHLMem.cpp:67:        return rewriter.notifyMatchFailure(user, "unsupported port type");
test/Tools/circt-verilog-lsp-server/workspace-symbol-project.test:3:// UNSUPPORTED: valgrind
include/circt/Conversion/ImportVerilog.h:231:  /// If true, continue lowering when an unsupported SVA construct is
include/circt/Conversion/ImportVerilog.h:233:  bool continueOnUnsupportedSVA = false;
test/Dialect/Emit/Reduction/pattern-registration.mlir:1:// UNSUPPORTED: system-windows
test/Tools/circt-verilog-lsp-server/uvm-diagnostics.test:3:// UNSUPPORTED: valgrind
test/Dialect/Emit/Reduction/emit-op-eraser.mlir:1:// UNSUPPORTED: system-windows
test/Tools/circt-verilog-lsp-server/types.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-sim/aot-trampoline-unsupported-abi-diagnostic.mlir:3:// Regression: referenced externs with unsupported trampoline ABI types must be
test/Tools/circt-sim/aot-trampoline-unsupported-abi-diagnostic.mlir:6:// CHECK: error: cannot generate interpreter trampoline for referenced external function: unsupported trampoline ABI (return type vector<2xi32>)
test/Tools/circt-verilog-lsp-server/signature-help.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/rename-variables.test:3:// UNSUPPORTED: valgrind
lib/Dialect/Handshake/HandshakeOps.cpp:95:    return op->emitError("unsupported type for indexing value: ") << indexType;
test/Tools/circt-verilog-lsp-server/rename-edge-cases.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/references.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/module-instantiation.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/interface.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/interface.test:29:// TODO: Hover on interface signals not yet implemented
test/Tools/circt-verilog-lsp-server/interface.test:39:// TODO: Go-to-definition for interface members not yet implemented
test/Tools/circt-verilog-lsp-server/inlay-hints.test:3:// UNSUPPORTED: valgrind
lib/Dialect/SV/SVOps.cpp:285:    return emitError("unsupported type");
lib/Dialect/SV/SVOps.cpp:300:    return emitError("unsupported type");
test/Tools/circt-verilog-lsp-server/goto-definition.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/find-references.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/document-symbols.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-sim/aot-trampoline-native-fallback-f64-incompatible-abi.mlir:10:// RUNTIME: WARNING: unsupported trampoline native fallback ABI
test/Tools/circt-verilog-lsp-server/document-links.test:3:// UNSUPPORTED: valgrind
lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp:463:             << " but the operation itself is unsupported.";
lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp:527:             << " is unsupported.";
test/Tools/circt-verilog-lsp-server/debounce.test:4:// UNSUPPORTED: valgrind
test/Tools/run-sv-tests-bmc-smtlib-no-fallback.test:2:// RUN: bash -eu -c $'cat > %t/sv-tests/tests/chapter-16/16.15--smtlib-no-fallback.sv <<\"EOF\"\n/*\n:name: smtlib_no_fallback\n:description: ensures unsupported SMT-LIB export does not retry to native backend\n:type: simulation\n:tags: 16.15\n:unsynthesizable: 1\n*/\nmodule top(input logic clk, input logic a);\n  assert property (@(posedge clk) a);\nendmodule\nEOF\n'
test/Tools/circt-verilog-lsp-server/code-actions.test:3:// UNSUPPORTED: valgrind
lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:1://===- HWLegalizeModulesPass.cpp - Lower unsupported IR features away -----===//
lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:10:// unsupported by some tools (e.g. multidimensional arrays) as specified by
lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:357:    user->emitError("unsupported packed array expression");
lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:438:          op.emitError("unsupported packed array expression");
test/Tools/circt-verilog-lsp-server/class-definition.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/workspace-symbol.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/workspace-symbol-fuzzy.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/uvm-completion.test:3:// UNSUPPORTED: valgrind
lib/Dialect/HW/Transforms/HWStopatSymbolic.cpp:68:    error = "unsupported selector form (expected 'inst[.inst...].port' or '*inst[.inst...].port')";
test/Tools/circt-verilog-lsp-server/type-hierarchy.test:3:// UNSUPPORTED: valgrind
test/Dialect/Synth/tech-mapper.mlir:88:// It produces sub-optimal mappings since currently area-flow is not implemented.
test/Tools/circt-verilog-lsp-server/semantic-tokens.test:3:// UNSUPPORTED: valgrind
test/Dialect/Synth/tech-mapper-error.mlir:59:  // expected-error@+1 {{Unsupported operation for truth table simulation}}
test/Tools/circt-verilog-lsp-server/semantic-tokens-comprehensive.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-sim/aot-native-module-init-hw-struct-create-fourstate.mlir:7:// COMPILE-NOT: unsupported_op:hw.struct_create
test/Tools/circt-verilog-lsp-server/rename.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-sim/aot-native-module-init-scf-if-struct-extract.mlir:7:// COMPILE-NOT: unsupported_op:hw.struct_extract
test/Tools/circt-verilog-lsp-server/rename-refactoring.test:3:// UNSUPPORTED: valgrind
lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:96:  assert(false && "Unsupported type");
lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:167:  assert(false && "Unsupported type");
lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:177:      bitcastOp.emitOpError("has unsupported input type");
lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:179:      bitcastOp.emitOpError("has unsupported output type");
lib/Dialect/SSP/Transforms/Schedule.cpp:73:    llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
lib/Dialect/SSP/Transforms/Schedule.cpp:163:  llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
lib/Dialect/SSP/Transforms/Schedule.cpp:201:  llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
lib/Dialect/SSP/Transforms/Schedule.cpp:223:    llvm::errs() << "ssp-schedule: Unsupported problem '" << problemName
lib/Dialect/SSP/Transforms/Schedule.cpp:254:  llvm::errs() << "ssp-schedule: Unsupported scheduler '" << scheduler
test/Tools/circt-verilog-lsp-server/rename-comprehensive.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/procedural.test:3:// UNSUPPORTED: valgrind
lib/Dialect/Calyx/Transforms/CalyxLoweringUtils.cpp:798:      assert(isa<IntegerType>(argType) && "unsupported block argument type");
lib/Dialect/Calyx/Transforms/CalyxLoweringUtils.cpp:822:           "unsupported return type");
test/Tools/circt-verilog-lsp-server/member-completion.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-sim/aot-native-module-init-skip-telemetry.mlir:6:// CHECK: 1x unsupported_call:puts
lib/Dialect/HW/HWTypes.cpp:595:    p.emitError(p.getNameLoc(), "unsupported dimension kind in hw.array");
test/Tools/run-regression-unified-manifest-profile-internal-space.test:5:// CHECK: invalid manifest row 1: unsupported profile token 's moke' (expected smoke|nightly|full|all)
test/Dialect/OM/Reduction/unused-class-remover.mlir:3:// UNSUPPORTED: system-windows
test/Tools/circt-verilog-lsp-server/inheritance-completion.test:3:// UNSUPPORTED: valgrind
test/Dialect/OM/Reduction/object-to-unknown-replacer.mlir:3:// UNSUPPORTED: system-windows
test/Tools/circt-verilog-lsp-server/include.test:5:// UNSUPPORTED: valgrind
test/Tools/run-avip-circt-sim-jit-policy-gate.test:3:// RUN: printf '#!/usr/bin/env bash\nset -euo pipefail\nout="${1:?out_dir required}"\nmkdir -p "$out/apb"\nif [[ "${CIRCT_SIM_MODE:-}" != "compile" ]]; then\n  echo "bad_mode=${CIRCT_SIM_MODE:-}" >&2\n  exit 41\nfi\nif [[ "${CIRCT_SIM_WRITE_JIT_REPORT:-0}" == "0" ]]; then\n  echo "bad_write_jit_report=${CIRCT_SIM_WRITE_JIT_REPORT:-}" >&2\n  exit 42\nfi\ncat > "$out/matrix.tsv" <<"MATRIX"\navip\tseed\tcompile_status\tcompile_sec\tsim_status\tsim_exit\tsim_sec\tsim_time_fs\tuvm_fatal\tuvm_error\tcov_1_pct\tcov_2_pct\tpeak_rss_kb\tcompile_log\tsim_log\napb\t1\tOK\t1\tOK\t0\t2\t200\t0\t0\t100\t100\t123\t/tmp/c.log\t/tmp/s.log\nMATRIX\ncat > "$out/apb/sim_seed_1.jit-report.json" <<"JSON"\n{\n  "jit": {\n    "jit_deopt_processes": [\n      {"process_id": 1, "process_name": "llhd_process_1", "reason": "unsupported_operation", "detail": "first_op:llvm.call"},\n      {"process_id": 2, "process_name": "llhd_process_2", "reason": "missing_thunk", "detail": "compile_budget_zero"}\n    ]\n  }\n}\nJSON\n' > %t/fake-run-avip-circt-sim.sh
test/Tools/run-avip-circt-sim-jit-policy-gate.test:5:// RUN: printf 'exact:reason:unsupported_operation\n' > %t/allowlist.txt
test/Tools/run-avip-circt-sim-jit-policy-gate.test:6:// RUN: env RUN_MATRIX=%t/fake-run-avip-circt-sim.sh SUMMARIZE_JIT_REPORTS=%S/../../utils/summarize_circt_sim_jit_reports.py JIT_POLICY_ALLOWLIST_FILE=%t/allowlist.txt JIT_POLICY_FAIL_ON_ANY_NON_ALLOWLISTED_DEOPT=0 JIT_POLICY_FAIL_ON_REASON=unsupported_operation %S/../../utils/run_avip_circt_sim_jit_policy_gate.sh %t/pass > %t/pass.log 2>&1
test/Tools/run-avip-circt-sim-jit-policy-gate.test:9:// RUN: not env RUN_MATRIX=%t/fake-run-avip-circt-sim.sh SUMMARIZE_JIT_REPORTS=%S/../../utils/summarize_circt_sim_jit_reports.py JIT_POLICY_FAIL_ON_ANY_NON_ALLOWLISTED_DEOPT=0 JIT_POLICY_FAIL_ON_REASON=unsupported_operation %S/../../utils/run_avip_circt_sim_jit_policy_gate.sh %t/fail > %t/fail.log 2>&1
test/Tools/run-avip-circt-sim-jit-policy-gate.test:20:// REASON: 2{{[[:space:]]+}}unsupported_operation{{[[:space:]]+}}1
test/Tools/run-avip-circt-sim-jit-policy-gate.test:23:// FAILLOG: top_non_allowlisted_reason{{\[[12]\]}} reason=unsupported_operation count=1
test/Tools/run-avip-circt-sim-jit-policy-gate.test:24:// FAILLOG: blocked reason matched: reason=unsupported_operation count=1
test/Tools/circt-verilog-lsp-server/hover.test:3:// UNSUPPORTED: valgrind
test/Dialect/OM/Reduction/list-element-pruner.mlir:6:// UNSUPPORTED: system-windows
test/Tools/summarize-circt-sim-jit-reports-policy.test:3:// RUN: python3 -c "import json,pathlib; p=pathlib.Path(r'%t/reports/a.json'); p.write_text(json.dumps({'jit':{'jit_deopt_processes':[{'process_id':1,'process_name':'llhd_process_1','reason':'unsupported_operation','detail':'prewait_impure:sim.proc.print'},{'process_id':2,'process_name':'llhd_process_2','reason':'missing_thunk','detail':'compile_budget_zero'}]}}, indent=2), encoding='utf-8')"
test/Tools/summarize-circt-sim-jit-reports-policy.test:4:// RUN: python3 -c "import json,pathlib; p=pathlib.Path(r'%t/reports/b.json'); p.write_text(json.dumps({'jit':{'jit_deopt_processes':[{'process_id':7,'process_name':'llhd_process_7','reason':'unsupported_operation','detail':'first_op:llvm.call'}]}}, indent=2), encoding='utf-8')"
test/Tools/summarize-circt-sim-jit-reports-policy.test:5:// RUN: printf 'exact:reason:missing_thunk\nexact:reason_detail:unsupported_operation:first_op:llvm.call\n' > %t/allowlist.partial
test/Tools/summarize-circt-sim-jit-reports-policy.test:6:// RUN: printf 'exact:reason:missing_thunk\nexact:reason_detail:unsupported_operation:first_op:llvm.call\nexact:reason_detail:unsupported_operation:prewait_impure:sim.proc.print\n' > %t/allowlist.full
test/Tools/summarize-circt-sim-jit-reports-policy.test:8:// RUN: not python3 %S/../../utils/summarize_circt_sim_jit_reports.py %t/reports --allowlist-file %t/allowlist.partial --fail-on-reason unsupported_operation 2>&1 | FileCheck %s --check-prefix=FAIL_REASON
test/Tools/summarize-circt-sim-jit-reports-policy.test:9:// RUN: not python3 %S/../../utils/summarize_circt_sim_jit_reports.py %t/reports --allowlist-file %t/allowlist.partial --fail-on-reason-detail unsupported_operation:prewait_impure:sim.proc.print 2>&1 | FileCheck %s --check-prefix=FAIL_DETAIL
test/Tools/summarize-circt-sim-jit-reports-policy.test:10:// RUN: python3 %S/../../utils/summarize_circt_sim_jit_reports.py %t/reports --allowlist-file %t/allowlist.full --fail-on-any-non-allowlisted-deopt --fail-on-reason unsupported_operation --fail-on-reason-detail unsupported_operation:prewait_impure:sim.proc.print > %t/log.txt 2>&1
test/Tools/summarize-circt-sim-jit-reports-policy.test:16:// FAIL_REASON: top_non_allowlisted_reason[1] reason=unsupported_operation count=1
test/Tools/summarize-circt-sim-jit-reports-policy.test:17:// FAIL_REASON: blocked reason matched: reason=unsupported_operation count=1
test/Tools/summarize-circt-sim-jit-reports-policy.test:19:// FAIL_DETAIL: top_non_allowlisted_reason_detail[1] reason=unsupported_operation detail=prewait_impure:sim.proc.print count=1
test/Tools/summarize-circt-sim-jit-reports-policy.test:20:// FAIL_DETAIL: blocked reason/detail matched: reason=unsupported_operation detail=prewait_impure:sim.proc.print count=1
test/Tools/circt-verilog-lsp-server/formatting.test:3:// UNSUPPORTED: valgrind
test/Dialect/OM/Reduction/class-parameter-pruner.mlir:3:// UNSUPPORTED: system-windows
test/Tools/summarize-circt-sim-jit-reports.test:3:// RUN: python3 -c "import json,pathlib; p=pathlib.Path(r'%t/reports/a.json'); p.write_text(json.dumps({'jit':{'jit_deopt_processes':[{'process_id':1,'process_name':'llhd_process_1','reason':'unsupported_operation','detail':'prewait_impure:sim.proc.print'},{'process_id':2,'process_name':'llhd_process_2','reason':'missing_thunk','detail':'compile_budget_zero'}]}}, indent=2), encoding='utf-8')"
test/Tools/summarize-circt-sim-jit-reports.test:4:// RUN: python3 -c "import json,pathlib; p=pathlib.Path(r'%t/reports/b.json'); p.write_text(json.dumps({'jit':{'jit_deopt_processes':[{'process_id':7,'process_name':'llhd_process_7','reason':'unsupported_operation','detail':'first_op:llvm.call'}]}}, indent=2), encoding='utf-8')"
test/Tools/summarize-circt-sim-jit-reports.test:16:// LOG: top_reason[1] reason=unsupported_operation count=2
test/Tools/summarize-circt-sim-jit-reports.test:19:// LOG: top_reason_detail[2] reason=unsupported_operation detail=first_op:llvm.call count=1
test/Tools/summarize-circt-sim-jit-reports.test:20:// LOG: top_reason_detail[3] reason=unsupported_operation detail=prewait_impure:sim.proc.print count=1
test/Tools/summarize-circt-sim-jit-reports.test:23:// REASON: 1{{[[:space:]]+}}unsupported_operation{{[[:space:]]+}}2
test/Tools/summarize-circt-sim-jit-reports.test:28:// DETAIL: 2{{[[:space:]]+}}unsupported_operation{{[[:space:]]+}}first_op:llvm.call{{[[:space:]]+}}1
test/Tools/summarize-circt-sim-jit-reports.test:29:// DETAIL: 3{{[[:space:]]+}}unsupported_operation{{[[:space:]]+}}prewait_impure:sim.proc.print{{[[:space:]]+}}1
test/Tools/summarize-circt-sim-jit-reports.test:32:// PROC: {{.*}}/reports/a.json{{[[:space:]]+}}1{{[[:space:]]+}}llhd_process_1{{[[:space:]]+}}unsupported_operation{{[[:space:]]+}}prewait_impure:sim.proc.print
test/Tools/summarize-circt-sim-jit-reports.test:34:// PROC: {{.*}}/reports/b.json{{[[:space:]]+}}7{{[[:space:]]+}}llhd_process_7{{[[:space:]]+}}unsupported_operation{{[[:space:]]+}}first_op:llvm.call
test/Dialect/OM/Reduction/class-field-pruner.mlir:3:// UNSUPPORTED: system-windows
test/Tools/run-regression-unified-manifest-profile-invalid.test:7:// CHECK: invalid manifest row 1: unsupported profile token 'broken' (expected smoke|nightly|full|all)
test/Tools/circt-verilog-lsp-server/find-references-comprehensive.test:3:// UNSUPPORTED: valgrind
test/Dialect/OM/Reduction/anycast-of-unknown-simplifier.mlir:3:// UNSUPPORTED: system-windows
test/Tools/run-verilator-verification-circt-bmc-drop-remarks.test:4:// RUN: printf '#!/bin/sh\necho \"module @top {\"\necho \"  verif.assert %c0_i1\"\necho \"}\"\necho \"warning: unsupported construct will be dropped during lowering\" >&2\n' > %t/bin/circt-verilog
test/Tools/circt-verilog-lsp-server/document-highlight.test:3:// UNSUPPORTED: valgrind
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:42:  Emitter(llvm::raw_ostream &os, const DenseSet<StringAttr> &unsupportedInstr)
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:43:      : os(os), unsupportedInstr(unsupportedInstr) {}
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:94:        unsupportedInstr.contains(instr->getName().getIdentifier());
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:191:  const DenseSet<StringAttr> &unsupportedInstr;
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:200:parseUnsupportedInstructionsFile(MLIRContext *ctxt,
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:201:                                 const std::string &unsupportedInstructionsFile,
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:202:                                 DenseSet<StringAttr> &unsupportedInstrs) {
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:203:  if (!unsupportedInstructionsFile.empty()) {
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:204:    std::ifstream input(unsupportedInstructionsFile, std::ios::in);
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:209:        unsupportedInstrs.insert(StringAttr::get(ctxt, trimmed));
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:228:  DenseSet<StringAttr> unsupportedInstr;
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:229:  for (const auto &instr : unsupportedInstructions)
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:230:    unsupportedInstr.insert(StringAttr::get(&getContext(), instr));
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:231:  parseUnsupportedInstructionsFile(
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:232:      &getContext(), unsupportedInstructionsFile.getValue(), unsupportedInstr);
lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:249:                  unsupportedInstr);
test/Tools/circt-verilog-lsp-server/diagnostic.test:4:// UNSUPPORTED: valgrind
test/Dialect/Arc/lower-clocks-to-funcs-errors.mlir:30:// expected-error @below {{op containing multiple InitialOps is currently unsupported.}}
test/Dialect/Arc/lower-clocks-to-funcs-errors.mlir:31:// expected-error @below {{op containing multiple FinalOps is currently unsupported.}}
test/Tools/circt-verilog-lsp-server/code-lens.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/class-hover.test:3:// UNSUPPORTED: valgrind
test/Dialect/SV/hw-legalize-modules-packed-arrays.mlir:14:  // expected-error @+1 {{unsupported packed array expression}}
lib/Dialect/FIRRTL/FIRRTLUtils.cpp:688:      llvm_unreachable("unsupported type");
test/Tools/circt-verilog-lsp-server/call-hierarchy.test:3:// UNSUPPORTED: valgrind
test/Dialect/Moore/vtable-partial-impl.mlir:5:// *any* unimplemented inherited method (using allHaveImpl), which caused
test/Dialect/Moore/vtable-partial-impl.mlir:8:// some implemented methods still get vtables; unimplemented slots are
test/Tools/circt-verilog-lsp-server/completion.test:3:// UNSUPPORTED: valgrind
test/Dialect/SV/errors.mlir:253:  // expected-error @+1 {{unsupported type}}
test/Dialect/SV/errors.mlir:260:  // expected-error @+1 {{unsupported type}}
lib/Dialect/Arc/Transforms/AllocateState.cpp:174:    assert("unsupported op for allocation" && false);
test/Dialect/Arc/Reduction/state-elimination.mlir:1:// UNSUPPORTED: system-windows
test/Dialect/Arc/Reduction/pattern-registration.mlir:1:// UNSUPPORTED: system-windows
lib/Dialect/FIRRTL/FIRRTLOps.cpp:211:  llvm_unreachable("Unsupported Flow type.");
lib/Dialect/FIRRTL/FIRRTLOps.cpp:226:  llvm_unreachable("Unsupported Flow type.");
lib/Dialect/Arc/Transforms/LowerClocksToFuncs.cpp:87:                << "containing multiple InitialOps is currently unsupported.";
lib/Dialect/Arc/Transforms/LowerClocksToFuncs.cpp:93:                << "containing multiple FinalOps is currently unsupported.";
test/Tools/circt-verilog-lsp-server/e2e/lit.local.cfg:2:# Mark the entire directory as unsupported to prevent lit from trying to run
test/Tools/circt-verilog-lsp-server/e2e/lit.local.cfg:5:config.unsupported = True
test/Tools/circt-verilog-lsp-server/textdocument-didclose.test:4:// UNSUPPORTED: valgrind
test/Dialect/SV/EliminateInOutPorts/hw-eliminate-inout-ports-errors.mlir:3:hw.module @unsupported(inout %a: i42) {
test/Dialect/SV/EliminateInOutPorts/hw-eliminate-inout-ports-errors.mlir:4:  // expected-error @+1 {{uses hw.inout port "a" but the operation itself is unsupported.}}
test/Dialect/SV/EliminateInOutPorts/hw-eliminate-inout-ports-errors.mlir:10:// expected-error @+1 {{multiple writers of inout port "a" is unsupported.}}
test/Tools/circt-verilog-lsp-server/textdocument-didchange.test:4:// UNSUPPORTED: valgrind
lib/Dialect/Arc/Transforms/InferMemories.cpp:95:      instOp.emitError("unsupported memory write latency ") << writeLatency;
test/Tools/circt-verilog-lsp-server/package-indexing.test:3:// UNSUPPORTED: valgrind
lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:467:        "unsupported operation, only CombMem can be used as the source of "
test/Tools/circt-sim/syscall-save-restart-warning.sv:5:// An unimplemented $save should at minimum warn the user so they know
lib/Dialect/OM/Transforms/FreezePaths.cpp:173:          return op->emitError("unsupported instance operation");
test/Conversion/VerifToSMT/bmc-for-smtlib-no-property-live-llvm-call.mlir:4:// carries unsupported LLVM ops (e.g. runtime helper calls), this should not
test/Tools/circt-verilog-lsp-server/find-package-import-def.test:4:// UNSUPPORTED: valgrind
test/Dialect/Handshake/errors.mlir:12:handshake.func @invalid_mux_unsupported_select(%arg0: tensor<i1>, %arg1: i32, %arg2: i32) {
test/Dialect/Handshake/errors.mlir:13:  // expected-error @+1 {{unsupported type for indexing value: 'tensor<i1>'}}
test/Dialect/Handshake/errors.mlir:28:handshake.func @invalid_cmerge_unsupported_index(%arg1: i32, %arg2: i32) -> tensor<i1> {
test/Dialect/Handshake/errors.mlir:29:  // expected-error @below {{unsupported type for indexing value: 'tensor<i1>'}}
test/Tools/circt-verilog-lsp-server/find-definition.test:3:// UNSUPPORTED: valgrind
test/Tools/circt-verilog-lsp-server/command-files.test:4:// UNSUPPORTED: valgrind
test/Dialect/LLHD/Transforms/unroll-loops.mlir:134:// CHECK-LABEL: @SkipLoopWithUnsupportedExitBranch
test/Dialect/LLHD/Transforms/unroll-loops.mlir:135:hw.module @SkipLoopWithUnsupportedExitBranch() {
test/Dialect/LLHD/Transforms/unroll-loops.mlir:176:// CHECK-LABEL: @SkipLoopWithUnsupportedExitCondition
test/Dialect/LLHD/Transforms/unroll-loops.mlir:177:hw.module @SkipLoopWithUnsupportedExitCondition() {
test/Dialect/LLHD/Transforms/unroll-loops.mlir:197:// CHECK-LABEL: @SkipLoopWithUnsupportedInductionVariable1
test/Dialect/LLHD/Transforms/unroll-loops.mlir:198:hw.module @SkipLoopWithUnsupportedInductionVariable1() {
test/Dialect/LLHD/Transforms/unroll-loops.mlir:244:// CHECK-LABEL: @SkipLoopWithUnsupportedInductionVariable2
test/Dialect/LLHD/Transforms/unroll-loops.mlir:245:hw.module @SkipLoopWithUnsupportedInductionVariable2(in %i: i42) {
test/Dialect/LLHD/Transforms/unroll-loops.mlir:317:// CHECK-LABEL: @SkipLoopWithUnsupportedInitialInductionVariableValue
test/Dialect/LLHD/Transforms/unroll-loops.mlir:318:hw.module @SkipLoopWithUnsupportedInitialInductionVariableValue(in %a: i42) {
test/Dialect/LLHD/Transforms/unroll-loops.mlir:360:// CHECK-LABEL: @SkipLoopWithUnsupportedIncrement
test/Dialect/LLHD/Transforms/unroll-loops.mlir:361:hw.module @SkipLoopWithUnsupportedIncrement() {
test/Dialect/LLHD/Transforms/unroll-loops.mlir:382:// CHECK-LABEL: @SkipLoopWithUnsupportedBounds
test/Dialect/LLHD/Transforms/unroll-loops.mlir:383:hw.module @SkipLoopWithUnsupportedBounds() {
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:462:  assert(intAttr && "unsupported lattice attribute kind");
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:957:      else // Treat unsupported constants as overdefined.
lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1035:    // Presently it asserts on unsupported combinations, so check this here.
test/Dialect/LLHD/Transforms/inline-calls-errors.mlir:30:  // expected-error @below {{recursive function call cannot be inlined (unsupported in --ir-hw)}}
test/Dialect/RTG/Transform/linear-scan-register-allocation.mlir:69:rtg.test @unsupportedUser() {
lib/Dialect/Moore/Transforms/CreateVTables.cpp:151:  // Last, emit any own method symbol entries (skip pure-virtual / unimplemented)
lib/Dialect/Moore/Transforms/CreateVTables.cpp:168:    // methods still get a vtable - unimplemented slots stay null and the
test/Dialect/RTG/Transform/emit-rtg-isa-assembly-errors.mlir:1:// RUN: circt-opt --rtg-emit-isa-assembly=unsupported-instructions=rtgtest.rv32i.beq %s --split-input-file --verify-diagnostics
test/Dialect/HW/svEmitErrors.mlir:3:// expected-error @+1 {{value has an unsupported verilog type 'vector<3xi1>'}}
test/Dialect/RTG/Transform/emit-rtg-isa-assembly.mlir:2:// RUN: circt-opt --rtg-emit-isa-assembly="unsupported-instructions=rtgtest.rv32i.ebreak,rtgtest.rv32i.ecall unsupported-instructions-file=%S/unsupported-instr.txt" %s 2>&1 >/dev/null | FileCheck %s --match-full-lines --strict-whitespace
test/Dialect/HW/hw-convert-bitcasts.mlir:49:// Don't crash on unsupported types
test/Dialect/HW/hw-convert-bitcasts.mlir:50:// NOCANON-LABEL: hw.module @unsupported
test/Dialect/HW/hw-convert-bitcasts.mlir:51:hw.module @unsupported(in %i: i8, out o : i8) {
test/Dialect/RTG/Reduction/virtual-register-constantifier.mlir:1:// UNSUPPORTED: system-windows
lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:644:                           "unable to partition symbol on unsupported type ")
lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:696:                   errorLoc, "unable to partition symbol on unsupported type ")
test/Dialect/HW/hw-convert-bitcasts-errors.mlir:3:hw.module @unsupported(in %i: i8, out o : i8) {
test/Dialect/HW/hw-convert-bitcasts-errors.mlir:4:  // expected-error @+1 {{has unsupported output type}}
test/Dialect/HW/hw-convert-bitcasts-errors.mlir:6:  // expected-error @+1 {{has unsupported input type}}
test/Dialect/HW/errors.mlir:418:// expected-error @+1 {{unsupported dimension kind in hw.array}}
test/Dialect/HW/Reduction/pattern-registration.mlir:1:// UNSUPPORTED: system-windows
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:358:              // This is an unsupported construct. Just drop it.
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:359:              if (tpe.getValue() == "unsupported") {
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:2477:                                  builder.getStringAttr("unsupported"));
lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:2633:                                  builder.getStringAttr("unsupported"));
test/Dialect/HW/Reduction/hw-operand-forwarder.mlir:1:// UNSUPPORTED: system-windows
test/Dialect/HW/Reduction/hw-module-output-pruner.mlir:1:// UNSUPPORTED: system-windows
test/Dialect/HW/Reduction/hw-module-input-pruner.mlir:1:// UNSUPPORTED: system-windows
lib/Conversion/VerifToSMT/VerifToSMT.cpp:4242:      op.emitError("unsupported sequence lowering for block argument");
lib/Conversion/VerifToSMT/VerifToSMT.cpp:6270:      op.emitError("unsupported boolean type in BMC conversion");
lib/Conversion/VerifToSMT/VerifToSMT.cpp:6418:          op.emitError("unsupported integer initial value in BMC conversion");
lib/Conversion/VerifToSMT/VerifToSMT.cpp:6435:          op.emitError("unsupported bool initial value in BMC conversion");
lib/Conversion/VerifToSMT/VerifToSMT.cpp:10388:  // Keep properties and enables in the live roots so unsupported ops that
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11001:    // path so they remain explicit unsupported-syntax diagnostics.
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11016:    // Keep non-scalar zeros on the generic unsupported-op diagnostics path.
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11763:          // Skip SMT-LIB unsupported-LLVM diagnostics in this case so unrelated
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11769:            Operation *unsupportedOp = nullptr;
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11780:                  unsupportedOp = &nested;
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11783:                if (unsupportedOp)
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11786:              if (unsupportedOp)
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11789:            if (unsupportedOp) {
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11790:              unsupportedOp->emitError(
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11793:                  << unsupportedOp->getName() << "'";
lib/Conversion/VerifToSMT/VerifToSMT.cpp:11828:                op->emitError("unsupported initial value for register");
test/Dialect/HW/Reduction/hw-module-externalizer.mlir:1:// UNSUPPORTED: system-windows
lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:1719:          op->emitError("target of rwprobe resolved to unsupported target");
test/Dialect/HW/Reduction/hw-constantifier.mlir:1:// UNSUPPORTED: system-windows
lib/Dialect/FIRRTL/Transforms/InferResets.cpp:746:      llvm_unreachable("unsupported type");
test/Tools/circt-sim/syscall-queue-stochastic.sv:8:    // CHECK: error: unsupported legacy stochastic queue task '$q_initialize'
test/Tools/circt-sim/syscall-q-full.sv:8:    // CHECK: error: unsupported legacy stochastic queue task '$q_initialize'
test/Tools/circt-sim/syscall-pld-sync-array.sv:14:    // CHECK: error: unsupported legacy PLD array task '$sync$and$array'
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:644:        diag.attachNote(src->getLoc()) << "unsupported source is here";
lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:683:          << "has an unimplemented lowering in LowerDomains";
test/Tools/circt-sim/syscall-pld-array.sv:14:    // CHECK: error: unsupported legacy PLD array task '$async$and$array'
test/Tools/circt-sim/syscall-getpattern.sv:2:// Test $getpattern — legacy function, returns 0 (not implemented)
lib/Conversion/SeqToSV/FirRegLowering.cpp:793:    assert(false && "unsupported type");
test/Tools/circt-sim/syscall-fread.sv:2:// TODO: $fread not yet implemented in ImportVerilog/interpreter.
lib/Dialect/FIRRTL/Transforms/ProbesToSignals.cpp:163:  // Force and release operations: reject as unsupported.
lib/Dialect/FIRRTL/Transforms/ProbesToSignals.cpp:393:  return op.emitError("memory has unsupported debug port (memtap)");
test/Tools/circt-lec/lec-prune-unreachable-before-smt.mlir:12:// `hw.type_scope` is intentionally unsupported by HWToSMT.
lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:1480:           << "Unsupported empty yieldOp outside ForOp or IfOp.";
lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:1683:  llvm_unreachable("unsupported comparison predicate");
lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2878:            llvm::report_fatal_error("Unsupported operation in TypeSwitch");
lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:979:      return source->emitError("unsupported operation '")
test/Conversion/VerifToSMT/bmc-concat-unknown-bounds.mlir:3:// CHECK: unsupported sequence lowering for block argument
lib/Dialect/FIRRTL/Transforms/InferReadWrite.cpp:62:            << "is unsupported by InferReadWrite as this pass cannot trace "
test/Tools/circt-bmc/drop-unsupported-sva.mlir:2:// RUN: circt-bmc --emit-mlir -b 1 --module top --drop-unsupported-sva %s 2>&1 | FileCheck %s --check-prefix=DROP
test/Tools/circt-bmc/drop-unsupported-sva.mlir:7:  verif.assert %true {circt.unsupported_sva} : i1
test/Tools/circt-bmc/drop-unsupported-sva.mlir:12:// DROP: circt-bmc: dropped 1 unsupported SVA assert-like op(s)
test/Dialect/FIRRTL/parse-errors.fir:848:circuit UnsupportedRadixSpecifiedIntegerLiterals:
test/Dialect/FIRRTL/parse-errors.fir:849:  module UnsupportedRadixSpecifiedIntegerLiterals:
test/Dialect/FIRRTL/parse-errors.fir:893:circuit UnsupportedStringEncodedIntegerLiterals:
test/Dialect/FIRRTL/parse-errors.fir:894:  module UnsupportedStringEncodedIntegerLiterals:
test/Dialect/FIRRTL/parse-errors.fir:896:    ; expected-error @below {{String-encoded integer literals are unsupported after FIRRTL 3.0.0}}
test/Dialect/FIRRTL/parse-errors.fir:929:circuit UnsupportedVersionDeclGroups:
test/Dialect/FIRRTL/parse-errors.fir:943:circuit UnsupportedVersionLayer:
test/Dialect/FIRRTL/parse-errors.fir:950:circuit UnsupportedVersionGroups:
test/Dialect/FIRRTL/parse-errors.fir:951:  module UnsupportedVersionGroups:
test/Dialect/FIRRTL/parse-errors.fir:966:circuit UnsupportedLayerBlock:
test/Dialect/FIRRTL/parse-errors.fir:967:  module UnsupportedLayerBlock:
test/Dialect/FIRRTL/parse-errors.fir:974:circuit UnsupportedLayerConvention:
test/Dialect/FIRRTL/parse-errors.fir:1396:circuit PublicModuleUnsupported:
test/Dialect/FIRRTL/parse-errors.fir:1398:  public module PublicModuleUnsupported:
test/Conversion/MooreToCore/errors.mlir:7:func.func @unsupportedConversion() -> !moore.string {
lib/Conversion/CombToSynth/CombToSynth.cpp:1243:            op.getLoc(), "i0 signed comparison is unsupported");
lib/Conversion/CombToSynth/CombToSynth.cpp:1361:                                         "i0 signed shift is unsupported");
lib/Dialect/FIRRTL/Import/FIRParser.cpp:578:          "String-encoded integer literals are unsupported after FIRRTL 3.0.0");
lib/Dialect/FIRRTL/Import/FIRParser.cpp:4096:  // Check for other unsupported reference sources.
lib/Dialect/FIRRTL/Import/FIRParser.cpp:5339:    auto diag = emitError(loc, "Invalid/unsupported annotation format");
lib/Dialect/FIRRTL/Import/FIRParserAsserts.cpp:13:// operations so that we may error on their unsupported use.
lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:940:          // For unsupported format operations, emit a placeholder.
lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:941:          formatStr.append("<unsupported>");
lib/Conversion/MooreToCore/MooreToCore.cpp:2100:             << port.name << "' has unsupported type " << port.type
lib/Conversion/MooreToCore/MooreToCore.cpp:7126:          loc, "unsupported net kind");
lib/Conversion/MooreToCore/MooreToCore.cpp:15148:    return rewriter.notifyMatchFailure(op, "unsupported int format");
lib/Conversion/MooreToCore/MooreToCore.cpp:16035:    // Unsupported format string type - return empty string
lib/Conversion/MooreToCore/MooreToCore.cpp:17884:      return rewriter.notifyMatchFailure(op, "unsupported queue type");
lib/Conversion/MooreToCore/MooreToCore.cpp:18190:      return rewriter.notifyMatchFailure(op, "unsupported queue type");
lib/Conversion/MooreToCore/MooreToCore.cpp:19004:      return rewriter.notifyMatchFailure(op, "unsupported input type");
lib/Conversion/MooreToCore/MooreToCore.cpp:19149:                                           "unsupported destination ref type");
lib/Conversion/MooreToCore/MooreToCore.cpp:19925:        // Unsupported type in path
lib/Conversion/MooreToCore/MooreToCore.cpp:21188:    // Unsupported operation
lib/Conversion/MooreToCore/MooreToCore.cpp:21584:      return rewriter.notifyMatchFailure(op, "unsupported array type");
lib/Conversion/MooreToCore/MooreToCore.cpp:22223:      return rewriter.notifyMatchFailure(op, "unsupported array type");
lib/Conversion/MooreToCore/MooreToCore.cpp:22610:      return rewriter.notifyMatchFailure(loc, "unsupported key type");
lib/Conversion/MooreToCore/MooreToCore.cpp:23960:            op, "sscanf: unsupported destination type for arg " +
lib/Conversion/MooreToCore/MooreToCore.cpp:24064:            op, "fscanf: unsupported destination type for arg " +
lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:304:  Operation *unsupported;
lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:365:          unsupported = op;
lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:371:    return forOp.emitError("unsupported operation ") << *unsupported;
lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:736:  llvm_unreachable("unsupported comparison predicate");
lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:994:               "unsupported pipeline result type");
lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:1256:                "Unsupported block schedulable");
test/Dialect/FIRRTL/inliner-errors.mlir:58:    // expected-error @below {{unsupported operation 'sv.ifdef' cannot be inlined}}
lib/Conversion/LTLToCore/LTLToCore.cpp:1102:        andOp.emitError("unsupported abort_on action on and");
lib/Conversion/LTLToCore/LTLToCore.cpp:1133:        orOp.emitError("unsupported abort_on action on or");
lib/Conversion/LTLToCore/LTLToCore.cpp:1265:    prop.getDefiningOp()->emitError("unsupported property lowering");
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:512:        emitOpError(op, "with unsupported parameter attribute: ") << attr;
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:513:        ps << "<unsupported-attr ";
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:956:              emitError(op, "unsupported fstring substitution type");
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1229:        emitOpError(op, "has unsupported 'debug' port");
lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1457:        ps << "<unsupported-expr-" << PPExtString(op->getName().stripDialect())
lib/Conversion/HWToSMT/HWToSMT.cpp:305:          op, "unsupported aggregate constant attribute/type combination");
lib/Conversion/HWToSMT/HWToSMT.cpp:441:                                         "unsupported bitcast result type");
lib/Conversion/HWToSMT/HWToSMT.cpp:569:      return rewriter.notifyMatchFailure(op.getLoc(), "unsupported array type");
lib/Conversion/HWToSMT/HWToSMT.cpp:601:                                         "unsupported array element type");
lib/Conversion/HWToSMT/HWToSMT.cpp:641:      return rewriter.notifyMatchFailure(op.getLoc(), "unsupported array type");
test/Dialect/FIRRTL/inferRW-errors.mlir:9:    // expected-error @below {{is unsupported by InferReadWrite}}
lib/Conversion/ImportVerilog/ImportVerilog.cpp:215:/// Rewrite unsupported width / alignment modifiers for a subset of format
lib/Conversion/ImportVerilog/CrossSelect.cpp:97:           << "unsupported non-constant intersect value range in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:104:           << "unsupported non-constant intersect value range in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:121:           << "unsupported intersect value range in cross select expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:150:           << "unsupported cross select expression: non-constant 'matches' "
lib/Conversion/ImportVerilog/CrossSelect.cpp:159:           << "unsupported cross select expression: non-constant 'matches' "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1027:             << "unsupported negation of cross identifier in cross select "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1069:           << "unsupported cross select expression with nested 'with' clause";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1072:           << "unsupported cross select expression with nested cross set "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1109:           << "unsupported cross select expression with no cross targets";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1132:           << "unsupported cross set expression element; expected tuple value";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1137:           << "unsupported cross set expression element with tuple arity "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1144:             << "unsupported non-integer tuple value in cross set expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1165:           << "unsupported non-constant cross set expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1187:           << "unsupported cross set expression; expected queue or unpacked "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1273:               << "unsupported non-constant intersect value range in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1307:             << "unsupported non-constant intersect value in cross select "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1357:             << "unsupported cross select iterator width in 'with' clause";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1457:           << "unsupported cross select expression due to large finite "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1469:           << "unsupported cross select expression due to large finite "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1482:           << "unsupported empty coverpoint bin in cross select expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1500:               << "unsupported non-constant coverpoint bin range in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1512:             << "unsupported non-constant coverpoint bin value in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1545:               << "unsupported coverpoint bin iterator width in cross select "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1585:           << "unsupported non-constant set coverpoint bin in cross select "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1592:             << "unsupported non-integer set coverpoint bin value in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1630:                   << "unsupported non-constant transition bin range in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1642:                 << "unsupported non-constant transition bin value in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1668:           << "unsupported default coverpoint bin target in cross select "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1702:           << "unsupported cross select 'with' clause over non-integral "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1709:           << "unsupported cross select 'with' clause over coverpoint '"
lib/Conversion/ImportVerilog/CrossSelect.cpp:1757:      bool unsupportedBinShape = bin->isDefaultSequence;
lib/Conversion/ImportVerilog/CrossSelect.cpp:1758:      if (unsupportedBinShape) {
lib/Conversion/ImportVerilog/CrossSelect.cpp:1830:             << "unsupported cross select expression over coverpoint '"
lib/Conversion/ImportVerilog/CrossSelect.cpp:1835:             << "unsupported cross select expression due to large finite "
lib/Conversion/ImportVerilog/CrossSelect.cpp:1898:           << "unsupported cross select condition target kind";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1904:             << "unsupported detached coverage bin in cross select expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:1908:             << "unsupported coverage bin parent in cross select expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:2013:               << "unsupported empty coverpoint bin in cross select expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:2030:             << "unsupported cross select 'with' clause due to too many "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2067:           << "unsupported non-constant cross set expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:2105:           << "unsupported cross set expression; expected queue or unpacked "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2165:                 << "unsupported cross select 'with' clause due to too many "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2356:                 << "unsupported cross select expression due to too many "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2444:               << "unsupported cross select expression due to large finite "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2457:               << "unsupported cross select expression due to invalid "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2499:                 << "unsupported non-constant intersect value range in cross "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2523:                 << "unsupported non-constant intersect value in cross select "
lib/Conversion/ImportVerilog/CrossSelect.cpp:2562:             << "unsupported negation of cross set expression";
lib/Conversion/ImportVerilog/CrossSelect.cpp:2570:             << "unsupported negation of cross select expression with 'with' "
lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:956:        .Default([&](auto expr) { visitUnsupportedOp(op); });
lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1025:  void visitUnsupportedOp(Operation *op) {
lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1026:    // Check for explicitly ignored ops vs unsupported ops (which cause a
lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1048:        // Anything else is considered unsupported and might cause a wrong
lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1051:          op->emitOpError("is an unsupported operation");
lib/Conversion/ImportVerilog/TimingControls.cpp:175:               << "unsupported global clocking symbol kind for $global_clock";
lib/Conversion/ImportVerilog/TimingControls.cpp:245:           << "unsupported event control: " << slang::ast::toString(ctrl.kind);
lib/Conversion/ImportVerilog/TimingControls.cpp:1584:             << "unsupported event kind in sequence event list";
lib/Conversion/ImportVerilog/TimingControls.cpp:1644:                   << "unsupported global clocking event kind in sequence "
lib/Conversion/ImportVerilog/TimingControls.cpp:1649:                 << "unsupported global clocking symbol kind for $global_clock";
lib/Conversion/ImportVerilog/TimingControls.cpp:1669:               << "unsupported clocking block event kind in sequence event "
lib/Conversion/ImportVerilog/TimingControls.cpp:1991:           << "unsupported delay control: " << slang::ast::toString(ctrl.kind);
lib/Conversion/ImportVerilog/TimingControls.cpp:2083:            << "unsupported global clocking symbol kind for $global_clock";
lib/Conversion/ImportVerilog/TimingControls.cpp:2274:    mlir::emitError(loc, "unsupported LTL clock control: ")
lib/Conversion/ImportVerilog/TimingControls.cpp:2407:               << "unsupported default clocking symbol kind for cycle delay";
lib/Conversion/ImportVerilog/TimingControls.cpp:2417:                 << "unsupported global clocking symbol kind for cycle delay";
lib/Conversion/ImportVerilog/TimingControls.cpp:2427:             << "unsupported clocking event kind for cycle delay";
lib/Conversion/ImportVerilog/TimingControls.cpp:2500:    return mlir::emitError(loc, "unsupported timing control: ")
lib/Conversion/ImportVerilog/Types.cpp:350:    auto d = mlir::emitError(loc, "unsupported type: ")
test/Dialect/FIRRTL/annotations-errors.fir:6:; expected-error @+2 {{Invalid/unsupported annotation format}}
test/Dialect/FIRRTL/annotations-errors.fir:17:; expected-error @+2 {{Invalid/unsupported annotation format}}
lib/Conversion/FSMToSV/FSMToSV.cpp:342:             << "is unsupported (op from the "
test/Dialect/FIRRTL/grand-central-view-errors.mlir:146:// Invalid / unsupported "class" in element.
test/Tools/select-opentitan-connectivity-cfg-invalid-row-kind.test:7:// CHECK: unsupported connectivity CSV row kind 'BROKEN'
lib/Conversion/FSMToCore/FSMToCore.cpp:428:             << "is unsupported (op from the "
lib/Dialect/ESI/runtime/python/esiaccel/types.py:73:    assert False, "unimplemented"
lib/Dialect/ESI/runtime/python/esiaccel/types.py:78:    assert False, "unimplemented"
lib/Dialect/ESI/runtime/python/esiaccel/types.py:90:    assert False, "unimplemented"
lib/Dialect/ESI/runtime/python/esiaccel/types.py:95:    assert False, "unimplemented"
lib/Dialect/ESI/runtime/python/esiaccel/types.py:464:      raise TypeError(f"unsupported type: {reason}")
lib/Dialect/ESI/runtime/python/esiaccel/types.py:574:    raise NotImplementedError("add_done_callback is not implemented")
lib/Dialect/ESI/runtime/python/esiaccel/codegen.py:302:      raise ValueError(f"Unsupported integer width: {type.bit_width}")
test/Conversion/FSMToSV/test_errors.mlir:4:  // expected-error@+1 {{'arith.constant' op is unsupported (op from the arith dialect).}}
test/Conversion/FIRRTLToHW/errors.mlir:46:// COM: Unknown widths are unsupported
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:4576:  op.emitOpError("unsupported type");
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5099:                emitError(loc, "has a substitution with an unimplemented "
lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5102:                    << "op with an unimplemented lowering is here";
lib/Conversion/ImportVerilog/Expressions.cpp:1670:      mlir::emitError(loc) << "unsupported expression: element select into "
lib/Conversion/ImportVerilog/Expressions.cpp:2480:          mlir::emitError(loc, "unsupported local assertion variable type");
lib/Conversion/ImportVerilog/Expressions.cpp:2937:    auto d = mlir::emitError(loc, "unsupported arbitrary symbol reference `")
lib/Conversion/ImportVerilog/Expressions.cpp:3132:      mlir::emitError(loc) << "unsupported lvalue type in assignment: "
lib/Conversion/ImportVerilog/Expressions.cpp:3396:    mlir::emitError(loc, "unsupported unary operator");
lib/Conversion/ImportVerilog/Expressions.cpp:3830:              << "unsupported unpacked aggregate equality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:3844:              << "unsupported dynamic unpacked equality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:3991:              << "unsupported unpacked aggregate inequality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:4005:              << "unsupported dynamic unpacked inequality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:4143:              << "unsupported dynamic unpacked case equality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:4153:              << "unsupported unpacked aggregate case equality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:4182:              << "unsupported dynamic unpacked case inequality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:4192:              << "unsupported unpacked aggregate case inequality operands";
lib/Conversion/ImportVerilog/Expressions.cpp:4261:    mlir::emitError(loc, "unsupported binary operator");
lib/Conversion/ImportVerilog/Expressions.cpp:4486:          << "unsupported conditional expression with more than one condition";
lib/Conversion/ImportVerilog/Expressions.cpp:4945:                             << "' is not yet implemented; lowering as "
lib/Conversion/ImportVerilog/Expressions.cpp:5030:                             << "' is not yet implemented; lowering as "
lib/Conversion/ImportVerilog/Expressions.cpp:5253:                             << "' is not yet implemented; lowering as "
lib/Conversion/ImportVerilog/Expressions.cpp:5450:            << "' is not yet implemented; lowering as regular function call";
lib/Conversion/ImportVerilog/Expressions.cpp:5717:                               << "' is not yet implemented";
lib/Conversion/ImportVerilog/Expressions.cpp:5839:                               << "' is not yet implemented";
lib/Conversion/ImportVerilog/Expressions.cpp:6986:        mlir::emitError(loc) << "unsupported destination type in $cast: "
lib/Conversion/ImportVerilog/Expressions.cpp:8503:    mlir::emitError(loc) << "unsupported system call `" << subroutine.name
lib/Conversion/ImportVerilog/Expressions.cpp:8762:    mlir::emitError(loc) << "unsupported assignment pattern with type " << type;
lib/Conversion/ImportVerilog/Expressions.cpp:8897:            << "unsupported streaming concat element type for queue: "
lib/Conversion/ImportVerilog/Expressions.cpp:9692:    mlir::emitError(loc, "unsupported expression: ")
lib/Conversion/ImportVerilog/Expressions.cpp:9763:          mlir::emitError(loc, "unsupported local assertion variable type")
lib/Conversion/ImportVerilog/Expressions.cpp:9789:            mlir::emitError(loc, "unsupported local assertion variable type");
lib/Conversion/ImportVerilog/Expressions.cpp:9798:          mlir::emitError(loc, "unsupported local assertion variable type")
lib/Conversion/ImportVerilog/Expressions.cpp:10449:    mlir::emitError(loc) << "unsupported conversion: " << packedType
lib/Conversion/ImportVerilog/Expressions.cpp:10491:    mlir::emitError(loc) << "unsupported conversion: " << intType
lib/Conversion/ImportVerilog/Expressions.cpp:10977:            mlir::emitError(loc) << "unsupported system call `"
lib/Conversion/ImportVerilog/Expressions.cpp:11813:            mlir::emitError(loc) << "unsupported system call `"
lib/Conversion/ImportVerilog/Expressions.cpp:11930:          // Legacy — deprecated, not implemented.
lib/Conversion/ImportVerilog/Expressions.cpp:11934:                      << "unsupported legacy stochastic queue function "
lib/Conversion/ImportVerilog/Expressions.cpp:11941:                      << "unsupported legacy stochastic queue function "
lib/Conversion/ImportVerilog/Expressions.cpp:11952:            mlir::emitError(loc) << "unsupported system call `"
lib/Conversion/ImportVerilog/Expressions.cpp:11969:            mlir::emitError(loc) << "unsupported system call `"
lib/Conversion/ImportVerilog/FormatStrings.cpp:220:             << "unsupported format specifier `" << fullSpecifier << "`";
test/Conversion/FSMToCore/errors.mlir:4:  // expected-error @below {{'arith.constant' op is unsupported (op from the arith dialect).}}
lib/Bindings/Tcl/circt_tcl.cpp:68:    return returnErrorStr(interp, "loading FIR files is unimplemented :(");
test/Conversion/ExportVerilog/verilog-errors.mlir:3:// expected-error @+1 {{value has an unsupported verilog type 'f32'}}
lib/Conversion/ImportVerilog/AssertionExpr.cpp:41:static mlir::InFlightDiagnostic emitUnsupportedSvaDiagnostic(Context &context,
lib/Conversion/ImportVerilog/AssertionExpr.cpp:43:  if (context.options.continueOnUnsupportedSVA)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:118:    emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:119:        << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:394:  emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:395:      << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:405:    emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:406:        << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:633:    emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:634:        << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:649:    emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:650:        << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:656:      emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:657:          << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:985:    emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:986:        << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1076:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1077:            << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1081:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1082:            << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1088:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1089:            << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1104:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1105:            << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1119:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1120:            << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1332:    emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1333:        << "unsupported $past value type with sampled-value controls (input "
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1336:    if (context.options.continueOnUnsupportedSVA && !context.inAssertionExpr) {
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1402:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1403:            << "unsupported $past value type with sampled-value controls "
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1411:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1412:            << "unsupported $past value type with sampled-value controls "
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1618:              emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1619:                  << "unsupported local assertion variable type";
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1631:            emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1632:                << "unsupported match item assignment type"
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1650:          emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1651:              << "unsupported match item assignment type"
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1672:          emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:1673:              << "unsupported match item unary operator";
lib/Conversion/ImportVerilog/AssertionExpr.cpp:2449:                << " (checkpoint/restart not implemented)";
lib/Conversion/ImportVerilog/AssertionExpr.cpp:2456:                << " (SDF timing annotation not implemented)";
lib/Conversion/ImportVerilog/AssertionExpr.cpp:2747:        emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:2748:            << "unsupported match item expression";
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3562:    emitUnsupportedSvaDiagnostic(context, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3563:        << "unsupported expression: "
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3713:    auto emitUnsupportedNonConcurrentSampledPlaceholder = [&]() -> Value {
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3714:      if (!(options.continueOnUnsupportedSVA && !inAssertionExpr))
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3718:          << funcName << " has unsupported sampled value type " << value.getType()
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3737:      if (auto fallback = emitUnsupportedNonConcurrentSampledPlaceholder())
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3739:      emitUnsupportedSvaDiagnostic(*this, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3740:          << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3747:      if (auto fallback = emitUnsupportedNonConcurrentSampledPlaceholder())
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3749:      emitUnsupportedSvaDiagnostic(*this, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3750:          << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3765:      emitUnsupportedSvaDiagnostic(*this, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:3766:          << "unsupported sampled value type for " << funcName;
lib/Conversion/ImportVerilog/AssertionExpr.cpp:4197:  emitUnsupportedSvaDiagnostic(*this, loc)
lib/Conversion/ImportVerilog/AssertionExpr.cpp:4198:      << "unsupported system call `" << funcName << "`";
lib/Conversion/ExportVerilog/ExportVerilog.cpp:298:  mlir::emitError(loc, "value has an unsupported verilog type ") << type;
lib/Conversion/ExportVerilog/ExportVerilog.cpp:1864:        mlir::emitError(loc, "value has an unsupported verilog type ") << type;
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2504:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2564:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2781:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2809:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2819:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2833:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2844:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2858:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2885:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2897:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2925:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2934:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2942:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2981:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:2989:    ps << "<<unsupported zero width constant: "
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3082:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3094:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3107:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3135:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3158:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3168:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3176:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3191:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3207:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3223:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3233:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3243:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3265:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3273:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3320:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3368:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3386:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3396:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3423:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3433:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3475:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3498:  ps << "<<unsupported expr: " << PPExtString(op->getName().getStringRef())
lib/Conversion/ExportVerilog/ExportVerilog.cpp:3719:  ps << "<<unsupported: " << PPExtString(op->getName().getStringRef()) << ">>";
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4300:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4307:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4325:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4351:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4456:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4606:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4625:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4658:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4700:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4732:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:4770:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5022:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5071:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5116:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:5180:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:6230:    emitError(op, "SV attributes emission is unimplemented for the op");
lib/Conversion/ExportVerilog/ExportVerilog.cpp:6339:    emitError(op, "SV attributes emission is unimplemented for the op");
test/Tools/run-opentitan-fpv-circt-bmc-stopat-selector-validation.test:11:// ROWS: ERROR{{[[:space:]]+}}foo_sec_cm{{[[:space:]]+}}hw/top_earlgrey/formal/sec_cm/foo{{[[:space:]]+}}opentitan{{[[:space:]]+}}FPV_BMC{{[[:space:]]+}}CIRCT_BMC_ERROR{{[[:space:]]+}}unsupported_stopat_selector
lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:167:    throw std::runtime_error("Unsupported ESI header version: " +
test/Dialect/FIRRTL/SFCTests/ExtractSeqMems/Simple2.fir:1:; UNSUPPORTED: system-windows
lib/Conversion/ImportVerilog/Structure.cpp:554:             << "unsupported interface port type for `" << port.name << "`";
lib/Conversion/ImportVerilog/Structure.cpp:570:             << "unsupported interface port type for `" << port.name << "`";
lib/Conversion/ImportVerilog/Structure.cpp:590:             << "unsupported interface port type for `" << port.name << "`";
lib/Conversion/ImportVerilog/Structure.cpp:641:             << "unsupported interface port `" << port->name << "` ("
lib/Conversion/ImportVerilog/Structure.cpp:671:                 << "unsupported interface port `" << port->name << "` ("
lib/Conversion/ImportVerilog/Structure.cpp:680:           << "unsupported interface port `" << portSymbol->name << "` ("
lib/Conversion/ImportVerilog/Structure.cpp:875:    mlir::emitError(loc, "unsupported construct: ")
lib/Conversion/ImportVerilog/Structure.cpp:904:    mlir::emitError(loc, "unsupported package member: ")
lib/Conversion/ImportVerilog/Structure.cpp:1269:                 << "unsupported unconnected interface port `"
lib/Conversion/ImportVerilog/Structure.cpp:1343:                   << "unsupported interface port connection for `"
lib/Conversion/ImportVerilog/Structure.cpp:1445:                     << "unsupported internal symbol for unconnected port `"
lib/Conversion/ImportVerilog/Structure.cpp:1456:          // TODO: Mark Inout port as unsupported and it will be supported later.
lib/Conversion/ImportVerilog/Structure.cpp:1459:                   << "unsupported port `" << port->name << "` ("
lib/Conversion/ImportVerilog/Structure.cpp:1465:               << "unsupported port `" << portSymbol->name << "` ("
lib/Conversion/ImportVerilog/Structure.cpp:1517:      mlir::emitError(loc) << "unsupported instance port `" << con->port.name
lib/Conversion/ImportVerilog/Structure.cpp:1901:      return mlir::emitError(loc, "unsupported net kind `")
lib/Conversion/ImportVerilog/Structure.cpp:1948:      mlir::emitError(loc) << "unsupported lvalue type in continuous assign: "
lib/Conversion/ImportVerilog/Structure.cpp:1976:            << "unsupported continuous assignment timing control: "
lib/Conversion/ImportVerilog/Structure.cpp:2441:          return emitDropped("unsupported UDP initial value");
lib/Conversion/ImportVerilog/Structure.cpp:2498:                                   bool &unsupportedSymbol, Value value,
lib/Conversion/ImportVerilog/Structure.cpp:2522:          unsupportedSymbol = true;
lib/Conversion/ImportVerilog/Structure.cpp:2530:              bool &unsupportedSymbol,
lib/Conversion/ImportVerilog/Structure.cpp:2537:          bool unsupported = false;
lib/Conversion/ImportVerilog/Structure.cpp:2538:          if (failed(addInputCharMatch(altCond, altImpossible, unsupported,
lib/Conversion/ImportVerilog/Structure.cpp:2541:          if (unsupported) {
lib/Conversion/ImportVerilog/Structure.cpp:2542:            unsupportedSymbol = true;
lib/Conversion/ImportVerilog/Structure.cpp:2546:          if (failed(addInputCharMatch(altCond, altImpossible, unsupported,
lib/Conversion/ImportVerilog/Structure.cpp:2549:          if (unsupported) {
lib/Conversion/ImportVerilog/Structure.cpp:2550:            unsupportedSymbol = true;
lib/Conversion/ImportVerilog/Structure.cpp:2611:              return emitDropped("unsupported UDP row shape");
lib/Conversion/ImportVerilog/Structure.cpp:2624:          return emitDropped("unsupported UDP row shape");
lib/Conversion/ImportVerilog/Structure.cpp:2634:            bool unsupportedSymbol = false;
lib/Conversion/ImportVerilog/Structure.cpp:2636:                                         unsupportedSymbol, rowInputValues[i],
lib/Conversion/ImportVerilog/Structure.cpp:2639:            if (unsupportedSymbol)
lib/Conversion/ImportVerilog/Structure.cpp:2640:              return emitDropped("unsupported UDP input row symbol");
lib/Conversion/ImportVerilog/Structure.cpp:2684:              return emitDropped("unsupported UDP edge transition symbol");
lib/Conversion/ImportVerilog/Structure.cpp:2688:          bool unsupportedSymbol = false;
lib/Conversion/ImportVerilog/Structure.cpp:2692:                  edgeCond, unsupportedSymbol, impossibleEdge)))
lib/Conversion/ImportVerilog/Structure.cpp:2694:          if (unsupportedSymbol)
lib/Conversion/ImportVerilog/Structure.cpp:2695:            return emitDropped("unsupported UDP edge transition symbol");
lib/Conversion/ImportVerilog/Structure.cpp:2735:            return emitDropped("unsupported UDP current-state row symbol");
lib/Conversion/ImportVerilog/Structure.cpp:2742:            return emitDropped("unsupported UDP output row symbol");
lib/Conversion/ImportVerilog/Structure.cpp:2747:            return emitDropped("unsupported UDP output row symbol");
lib/Conversion/ImportVerilog/Structure.cpp:3311:    mlir::emitError(loc) << "unsupported primitive type: " << primName;
lib/Conversion/ImportVerilog/Structure.cpp:3324:    mlir::emitError(loc, "unsupported module member: ")
lib/Conversion/ImportVerilog/Structure.cpp:3601:    mlir::emitError(loc) << "unsupported definition: "
lib/Conversion/ImportVerilog/Structure.cpp:3657:              << "unsupported generic interface port `" << port.name << "`";
lib/Conversion/ImportVerilog/Structure.cpp:3730:          << "unsupported module port `" << symbol->name << "` ("
lib/Conversion/ImportVerilog/Structure.cpp:4100:      return mlir::emitError(port.loc, "unsupported port: `")
lib/Conversion/ImportVerilog/Structure.cpp:4111:      return mlir::emitError(port.loc, "unsupported port: `")
lib/Conversion/ImportVerilog/Structure.cpp:7138:  // currently not implemented. Since forward declarations of non-interface
lib/Conversion/ImportVerilog/Structure.cpp:7328:    mlir::emitError(loc) << "unsupported construct in ClassType members: "
test/Dialect/FIRRTL/SFCTests/ExtractSeqMems/Compose.fir:1:; UNSUPPORTED: system-windows
lib/Dialect/ESI/runtime/cpp/lib/backends/Trace.cpp:59:      assert(false && "not implemented");
lib/Dialect/ESI/runtime/cpp/lib/backends/Trace.cpp:212:      throw std::runtime_error("unsupported type for read: " +
lib/Dialect/ESI/runtime/cpp/lib/Values.cpp:382:        std::format("Unsupported base '{}' for BitVector::toString", base));
test/Dialect/FIRRTL/Reduction/root-extmodule-port-pruner.mlir:1:// UNSUPPORTED: system-windows
lib/Conversion/ImportVerilog/Statements.cpp:191:constexpr const char kUnsupportedSvaAttr[] = "circt.unsupported_sva";
lib/Conversion/ImportVerilog/Statements.cpp:192:constexpr const char kUnsupportedSvaReasonAttr[] =
lib/Conversion/ImportVerilog/Statements.cpp:193:    "circt.unsupported_sva_reason";
lib/Conversion/ImportVerilog/Statements.cpp:435:  emitUnsupportedConcurrentAssertionPlaceholder(
lib/Conversion/ImportVerilog/Statements.cpp:438:    if (!context.options.continueOnUnsupportedSVA)
lib/Conversion/ImportVerilog/Statements.cpp:473:    op->setAttr(kUnsupportedSvaAttr, builder.getUnitAttr());
lib/Conversion/ImportVerilog/Statements.cpp:474:    op->setAttr(kUnsupportedSvaReasonAttr, builder.getStringAttr(reason));
lib/Conversion/ImportVerilog/Statements.cpp:476:        << "skipping unsupported SVA assertion in continue mode: " << reason;
lib/Conversion/ImportVerilog/Statements.cpp:1919:      mlir::emitError(loc) << "unsupported pattern in case statement: "
lib/Conversion/ImportVerilog/Statements.cpp:1973:      mlir::emitError(loc, "unsupported set membership pattern match");
lib/Conversion/ImportVerilog/Statements.cpp:2555:        mlir::emitError(loc) << "unsupported immediate assertion kind: "
lib/Conversion/ImportVerilog/Statements.cpp:2843:    auto tolerateUnsupportedSVA = [&](StringRef reason) -> LogicalResult {
lib/Conversion/ImportVerilog/Statements.cpp:2844:      return emitUnsupportedConcurrentAssertionPlaceholder(stmt, assertLabel,
lib/Conversion/ImportVerilog/Statements.cpp:2894:        if (succeeded(tolerateUnsupportedSVA("property lowering failed")))
lib/Conversion/ImportVerilog/Statements.cpp:3005:      if (succeeded(tolerateUnsupportedSVA("property lowering failed")))
lib/Conversion/ImportVerilog/Statements.cpp:3164:    if (succeeded(tolerateUnsupportedSVA("unsupported concurrent assertion kind")))
lib/Conversion/ImportVerilog/Statements.cpp:3166:    mlir::emitError(loc) << "unsupported concurrent assertion kind: "
lib/Conversion/ImportVerilog/Statements.cpp:3557:                             << " (checkpoint/restart not implemented)";
lib/Conversion/ImportVerilog/Statements.cpp:3632:    // Legacy gate-array modeling functions — deprecated, not implemented.
lib/Conversion/ImportVerilog/Statements.cpp:3649:      mlir::emitError(loc) << "unsupported legacy PLD array task '"
lib/Conversion/ImportVerilog/Statements.cpp:3655:    // Legacy abstract queue functions — deprecated, not implemented.
lib/Conversion/ImportVerilog/Statements.cpp:3660:      mlir::emitError(loc) << "unsupported legacy stochastic queue task '"
lib/Conversion/ImportVerilog/Statements.cpp:3669:                             << " (SDF timing annotation not implemented)";
lib/Conversion/ImportVerilog/Statements.cpp:4304:      mlir::emitError(loc) << "unsupported delayed event trigger";
lib/Conversion/ImportVerilog/Statements.cpp:5087:    mlir::emitError(loc, "unsupported statement: ")
test/Conversion/ImportVerilog/constraint-solve.sv:5:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/module-port-pruner.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ExportAIGER/errors.mlir:3:// Test unsupported variadic AND gates (should be lowered first)
test/Conversion/ExportAIGER/errors.mlir:23:// Test unsupported operation (when handleUnknownOperation is false)
test/Dialect/FIRRTL/Reduction/extmodule-port-pruner.mlir:1:// UNSUPPORTED: system-windows
lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:165:    emitError(loc) << "unsupported data type '" << type << "'";
lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:189:      oldOp->emitError("unsupported constant type");
lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:703:        emitError(loc) << "unsupported type for zero value: " << type;
test/Conversion/ImportVerilog/basic.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/list-create-element-remover.mlir:4:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/constraint-method-call.sv:5:// UNSUPPORTED: valgrind
lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:121:#eror "Unsupported platform"
lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:147:#eror "Unsupported platform"
lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:199:#eror "Unsupported platform"
lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:285:#eror "Unsupported platform"
test/Dialect/FIRRTL/Reduction/layer-disable.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/constraint-implication.sv:5:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/extmodule-instance-remover.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/procedures.sv:6:// UNSUPPORTED: valgrind
test/Conversion/ImportVerilog/continuous-assign-delay-one-step-supported.sv:11:// DIAG-NOT: unsupported continuous assignment timing control: OneStepDelay
test/Dialect/FIRRTL/Reduction/instance-stubber.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/assoc_arrays.sv:6:// UNSUPPORTED: valgrind
test/Conversion/ImportVerilog/assoc_arrays.sv:32:/// This tests the fix for "unsupported expression: element select into int$[*]"
test/Conversion/ImportVerilog/assoc_arrays.sv:345:/// This tests the fix for "unsupported expression: element select into TypedefArray"
test/Conversion/ImportVerilog/delay-cycle-supported.sv:17:// DIAG-NOT: unsupported delay control: CycleDelay
test/Conversion/ImportVerilog/global-variable-init.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/connect-forwarder.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/delay-one-step-supported.sv:8:  // DIAG-NOT: unsupported delay control: OneStepDelay
lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:50:    throw std::runtime_error("Serialization not implemented for type " + id);
lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:57:    throw std::runtime_error("Deserialization not implemented for type " + id);
lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:71:    throw std::runtime_error("Validation not implemented for type " + id);
test/Conversion/ImportVerilog/classes.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/simplify-resets.mlir:3:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/builtins.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/port-pruner.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/class-e2e.sv:5:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/pattern-registration.mlir:1:// UNSUPPORTED: system-windows
test/Dialect/FIRRTL/Reduction/object-inliner.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/errors.sv:5:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/node-symbol-remover.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/cross-select-intersect-open-range-wide-supported.sv:3:module CrossSelectIntersectOpenRangeUnsupported;
test/Dialect/FIRRTL/Reduction/module-swapper.mlir:1:// UNSUPPORTED: system-windows
test/Dialect/FIRRTL/Reduction/module-externalizer.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/types.sv:6:// UNSUPPORTED: valgrind
test/Conversion/ImportVerilog/queues.sv:6:// UNSUPPORTED: valgrind
test/Conversion/ImportVerilog/hierarchical-names.sv:6:// UNSUPPORTED: valgrind
lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:384:  assert(false && "unimplemented");
lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:396:  assert(false && "unimplemented");
test/Dialect/FIRRTL/Reduction/memory-stubber.mlir:1:// UNSUPPORTED: system-windows
test/Dialect/FIRRTL/Reduction/issue-3555.mlir:1:// UNSUPPORTED: system-windows
test/Dialect/FIRRTL/Reduction/force-dedup.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/queue-max-min.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/eager-inliner.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv:3:module CrossSelectIntersectPlusMinusUnsupported;
test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv:17:// CHECK: error: unsupported non-constant intersect value range in cross select expression
test/Conversion/ImportVerilog/inherited-virtual-methods.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/constantifier.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/queue-delete-index.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/connect-source-operand-forward.mlir:1:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/pre-post-randomize.sv:6:// UNSUPPORTED: valgrind
test/Dialect/FIRRTL/Reduction/annotation-remover.mlir:4:// UNSUPPORTED: system-windows
test/Conversion/ImportVerilog/runtime-randomization.sv:6:// UNSUPPORTED: valgrind
test/Target/ExportSystemC/errors.mlir:3:// CHECK: <<UNSUPPORTED OPERATION (hw.module)>>
test/Target/ExportSystemC/errors.mlir:9:// CHECK: <<UNSUPPORTED TYPE (!hw.inout<i2>)>>
test/Conversion/ImportVerilog/uvm_classes.sv:6:// UNSUPPORTED: valgrind
test/Conversion/ImportVerilog/static-property-fixes.sv:6:// UNSUPPORTED: valgrind
test/Conversion/ImportVerilog/randomize.sv:6:// UNSUPPORTED: valgrind
test/Tools/circt-bmc/commandline.mlir:17:// CHECK-DAG: --drop-unsupported-sva
test/Conversion/ImportVerilog/time-type-handling.sv:6:// UNSUPPORTED: valgrind
test/Conversion/ImportVerilog/randomize-inline-control.sv:6:// UNSUPPORTED: valgrind
test/Tools/circt-bmc/circt-bmc-prune-unreachable-hw-before-smt.mlir:11:  // `hw.type_scope` is intentionally unsupported by HWToSMT.
test/Conversion/ImportVerilog/system-calls-complete.sv:5:// This test verifies that none of the system calls produce "unsupported system
test/Conversion/ImportVerilog/system-calls-complete.sv:8:// CHECK-NOT: unsupported system call
test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:2:// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:3:// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR
test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:5:module immediate_past_event_continue_on_unsupported(input logic clk);
test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:12:// ERR: error: unsupported $past value type with sampled-value controls
test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:14:// WARN: warning: unsupported $past value type with sampled-value controls
test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:16:// IR-LABEL: moore.module @immediate_past_event_continue_on_unsupported
test/Conversion/ImportVerilog/sva-sequence-match-item-coverage-sdf-static-subroutine.sv:18:// DIAG: warning: $sdf_annotate is not supported in circt-sim (SDF timing annotation not implemented)
test/Conversion/ImportVerilog/sva-sequence-match-item-stacktrace-function.sv:20:// DIAG-NOT: unsupported system call `$stacktrace`
test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:2:// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:3:// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR
test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:19:// STRICT: error: unsupported sampled value type for $stable
test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:20:// WARN: warning: $stable has unsupported sampled value type
test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:21:// WARN: warning: $rose has unsupported sampled value type
test/Conversion/ImportVerilog/sva-sequence-match-item-rewind-function.sv:20:// DIAG-NOT: unsupported system call `$rewind`
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:2:// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:3:// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:6:module SvaContinueOnUnsupported(input logic clk, a);
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:10:  // `event` + sampled-value controls in `$past` is currently unsupported.
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:15:// ERR: error: unsupported $past value type with sampled-value controls
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:17:// WARN: warning: unsupported $past value type with sampled-value controls
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:18:// WARN: warning: skipping unsupported SVA assertion in continue mode: property lowering failed
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:21:// IR-SAME: {circt.unsupported_sva
test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:22:// IR-SAME: circt.unsupported_sva_reason = "property lowering failed"
test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:22:// DIAG: warning: $save is not supported in circt-sim (checkpoint/restart not implemented)
test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:23:// DIAG: warning: $restart is not supported in circt-sim (checkpoint/restart not implemented)
test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:24:// DIAG: warning: $incsave is not supported in circt-sim (checkpoint/restart not implemented)
test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:25:// DIAG: warning: $reset is not supported in circt-sim (checkpoint/restart not implemented)
```
