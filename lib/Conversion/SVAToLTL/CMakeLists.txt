add_circt_conversion_library(CIRCTSVAToLTL
  SVAToLTL.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTLTL
  CIRCTVerif
  CIRCTHW
  CIRCTSVA
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
