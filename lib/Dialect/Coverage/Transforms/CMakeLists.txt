add_circt_dialect_library(CIRCTCoverageTransforms
  InstrumentCoverage.cpp
  ExportCoverageData.cpp

  DEPENDS
  CIRCTCoverageTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTCoverage
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
