add_circt_unittest(CIRCTSVTests
  HWEliminateInOutPortsTest.cpp
)

target_link_libraries(CIRCTSVTests
  PRIVATE
  CIRCTSVTransforms
  CIRCTSV
  CIRCTHW
  MLIRIR
  MLIRParser
  MLIRPass
)
