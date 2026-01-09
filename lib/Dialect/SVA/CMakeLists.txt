add_circt_dialect_library(CIRCTSVA
  SVADialect.cpp
  SVAOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SVA

  DEPENDS
  MLIRSVAIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTLTL
  CIRCTVerif
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
