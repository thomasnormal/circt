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
  MLIRParser
  MLIRSCFDialect
  MLIRTransforms
)

add_circt_tool(circt-sim
  circt-sim.cpp
  LLHDProcessInterpreter.cpp
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
