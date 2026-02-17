add_circt_library(CIRCTLECTransforms
  ConstructLEC.cpp
  ConstructSeqLEC.cpp
  DumpLECUnknowns.cpp
  LowerLECLLVM.cpp
  LowerLLHDRefPorts.cpp
  StripLLHDInterfaceSignals.cpp

  DEPENDS
  CIRCTLECTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLLHD
  CIRCTSeq
  CIRCTVerif

  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRTransformUtils
)
