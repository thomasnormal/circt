add_circt_conversion_library(CIRCTVerifToSMT
  LowerClockedAssertLike.cpp
  VerifToSMT.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTHWToSMT
  CIRCTLTL
  CIRCTVerif
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSMT
  MLIRTransforms
  MLIRTransformUtils
  MLIRReconcileUnrealizedCasts
)
