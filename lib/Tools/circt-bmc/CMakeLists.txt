add_circt_library(CIRCTBMCTransforms
  ExternalizeRegisters.cpp
  LowerToBMC.cpp
  PruneBMCRegisters.cpp
  StripLLHDProcesses.cpp
  StripUnreachableSymbols.cpp

  DEPENDS
  CIRCTBMCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSeq
  CIRCTComb
  CIRCTVerif

  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRSCFDialect
  MLIRTransforms
)
