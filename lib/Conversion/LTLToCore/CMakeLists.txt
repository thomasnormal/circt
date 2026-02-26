add_circt_conversion_library(CIRCTLTLToCore
  LTLToCore.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTLLHD
  CIRCTSeq
  CIRCTLTL
  CIRCTVerif
  MLIRTransforms
)
