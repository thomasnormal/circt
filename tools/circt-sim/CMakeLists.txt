set(LLVM_LINK_COMPONENTS Support)

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
  circt-sim.cpp
  JITCompileManager.cpp
  LLHDProcessInterpreter.cpp
  LLHDProcessInterpreterNativeThunkExec.cpp
  LLHDProcessInterpreterNativeThunkPolicy.cpp
  LLHDProcessInterpreterWaitCondition.cpp
  UVMFastPaths.cpp
  DEPENDS ${libs}
)
target_link_libraries(circt-sim
  PRIVATE
  ${libs}
  MooreRuntime
)

llvm_update_compile_flags(circt-sim)
mlir_check_all_link_libraries(circt-sim)

# Export symbols from the executable so that dlopen'd VPI libraries (e.g., cocotb)
# can resolve vpi_* C API functions. Without this, the VPI functions defined in
# VPIRuntime.cpp and MooreRuntime.cpp are invisible to loaded shared libraries.
set_target_properties(circt-sim PROPERTIES ENABLE_EXPORTS 1)
