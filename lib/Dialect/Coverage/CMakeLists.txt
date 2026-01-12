add_circt_dialect_library(CIRCTCoverage
  CoverageDialect.cpp
  CoverageOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Coverage

  DEPENDS
  MLIRCoverageIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)
