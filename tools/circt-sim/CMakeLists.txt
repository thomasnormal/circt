set(LLVM_LINK_COMPONENTS Support native)

# Block-level JIT: only wire ExecutionEngine/ORC when available.
if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_compile_definitions(CIRCT_SIM_JIT_ENABLED)
  set(CIRCT_SIM_JIT_DEPS
    LLVMOrcJIT
    MLIRExecutionEngine
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRSCFToControlFlow
    CIRCTCombToLLVM
    CIRCTHWToLLVM
  )
endif()

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
  JITSchedulerRuntime.cpp
  JITBlockCompiler.cpp
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
  UVMFastPaths.cpp
  DEPENDS ${libs}
)
target_link_libraries(circt-sim
  PRIVATE
  ${libs}
  ${CIRCT_SIM_JIT_DEPS}
  MooreRuntime
)

llvm_update_compile_flags(circt-sim)
mlir_check_all_link_libraries(circt-sim)

# Export symbols from the executable so that dlopen'd VPI libraries (e.g., cocotb)
# can resolve vpi_* C API functions. Without this, the VPI functions defined in
# VPIRuntime.cpp and MooreRuntime.cpp are invisible to loaded shared libraries.
set_target_properties(circt-sim PROPERTIES ENABLE_EXPORTS 1)
