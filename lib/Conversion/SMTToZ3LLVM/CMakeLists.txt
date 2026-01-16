# Set up Z3 library for JIT runtime linking
# When Z3 is found via CONFIG mode, z3::libz3 target is available
# When found via FindZ3 module, Z3_LIBRARIES is set
if(TARGET z3::libz3)
  set(CIRCT_Z3_LINK_LIBS z3::libz3)
elseif(Z3_LIBRARIES)
  set(CIRCT_Z3_LINK_LIBS ${Z3_LIBRARIES})
else()
  set(CIRCT_Z3_LINK_LIBS "")
endif()

add_circt_conversion_library(CIRCTSMTToZ3LLVM
  LowerSMTToZ3LLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRLLVMCommonConversion
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRSMT
  MLIRTransforms
  ${CIRCT_Z3_LINK_LIBS}
)
