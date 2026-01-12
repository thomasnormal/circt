add_circt_library(CIRCTDebugTool
  Debug.cpp
  DebugSession.cpp
  VCDWriter.cpp
  DAPServer.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  CIRCTSupport
)
