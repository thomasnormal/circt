add_circt_library(CIRCTLECTransforms
  ConstructLEC.cpp
  LowerLLHDRefPorts.cpp
  StripLLHDInterfaceSignals.cpp

  DEPENDS
  CIRCTLECTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTLLHD
  CIRCTVerif

  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRTransformUtils
)
