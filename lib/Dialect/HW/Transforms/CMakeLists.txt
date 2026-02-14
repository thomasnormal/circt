add_circt_dialect_library(CIRCTHWTransforms
  HWAggregateToComb.cpp
  HWExternalizeModules.cpp
  HWStopatSymbolic.cpp
  HWConvertBitcasts.cpp
  HWBypassInnerSymbols.cpp
  HWParameterizeConstantPorts.cpp
  HWPrintInstanceGraph.cpp
  HWSpecialize.cpp
  PrintHWModuleGraph.cpp
  FlattenIO.cpp
  VerifyInnerRefNamespace.cpp
  FlattenModules.cpp
  FooWires.cpp

  DEPENDS
  CIRCTHWTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTVerif
  CIRCTSeq
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
