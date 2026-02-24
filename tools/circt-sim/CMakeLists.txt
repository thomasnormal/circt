set(LLVM_LINK_COMPONENTS Support native)

set(libs
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTEmit
  CIRCTHW
  CIRCTHWTransforms
  CIRCTLLHD
  CIRCTMoore
  CIRCTOM
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSim
  CIRCTSimTransforms
  CIRCTSV
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRMathDialect
  MLIRParser
  MLIRSCFDialect
  MLIRTransforms
)

add_circt_tool(circt-sim
  PARTIAL_SOURCES_INTENDED
  AOTProcessCompiler.cpp
  CompiledModuleLoader.cpp
  circt-sim.cpp
  LLHDProcessInterpreter.cpp
  LLHDProcessInterpreterBytecode.cpp
  LLHDProcessInterpreterCallIndirect.cpp
  LLHDProcessInterpreterDrive.cpp
  LLHDProcessInterpreterGlobals.cpp
  LLHDProcessInterpreterMemory.cpp
  LLHDProcessInterpreterModuleLevelInit.cpp
  LLHDProcessInterpreterNativeThunkExec.cpp
  LLHDProcessInterpreterNativeThunkPolicy.cpp
  LLHDProcessInterpreterTrace.cpp
  LLHDProcessInterpreterUvm.cpp
  LLHDProcessInterpreterWaitCondition.cpp
  UcontextProcess.cpp
  UVMFastPaths.cpp
  DEPENDS ${libs}
)
target_link_libraries(circt-sim
  PRIVATE
  ${libs}
  MooreRuntime
)

# For wasm/node builds, map POSIX paths directly to the host filesystem.
# This keeps CLI behavior (e.g. --vcd /tmp/test.vcd) consistent with native.
option(CIRCT_SIM_WASM_ENABLE_NODERAWFS
  "Enable Emscripten Node.js raw filesystem bridge (-sNODERAWFS=1) for circt-sim"
  ON
)
if(EMSCRIPTEN AND CIRCT_SIM_WASM_ENABLE_NODERAWFS)
  target_link_options(circt-sim PRIVATE "SHELL:-sNODERAWFS=1")
endif()

llvm_update_compile_flags(circt-sim)
mlir_check_all_link_libraries(circt-sim)

# Export symbols from the executable so that dlopen'd VPI libraries (e.g., cocotb)
# can resolve vpi_* C API functions. Without this, the VPI functions defined in
# VPIRuntime.cpp and MooreRuntime.cpp are invisible to loaded shared libraries.
set_target_properties(circt-sim PROPERTIES ENABLE_EXPORTS 1)
