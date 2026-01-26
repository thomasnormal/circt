add_circt_dialect_library(CIRCTSimTransforms
  FoldValueFormatters.cpp
  LowerDPIFunc.cpp
  ProceduralizeSim.cpp
  StripSim.cpp


  DEPENDS
  CIRCTSimTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSim
  CIRCTSeq
  CIRCTSV
  CIRCTComb
  CIRCTSupport
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTransformUtils
)
