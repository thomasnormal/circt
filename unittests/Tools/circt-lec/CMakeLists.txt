add_circt_unittest(CIRCTLECToolTests
  LowerLECLLVMTest.cpp
  StripLLHDSignalPtrCastTest.cpp
)

target_link_libraries(CIRCTLECToolTests
  PRIVATE
  CIRCTLECTransforms
  CIRCTHW
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSupport
)
