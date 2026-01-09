add_circt_library(CIRCTDebug
  Debug.cpp
  DebugSession.cpp
  VCDWriter.cpp
  DAPServer.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  CIRCTSupport
)
