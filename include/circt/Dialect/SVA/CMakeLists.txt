add_circt_dialect(SVA sva)
add_circt_doc(SVAOps Dialects/SVAOps -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SVA.td)
mlir_tablegen(SVAEnums.h.inc -gen-enum-decls)
mlir_tablegen(SVAEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTSVAEnumsIncGen)
add_dependencies(circt-headers CIRCTSVAEnumsIncGen)

add_dependencies(circt-headers MLIRSVAIncGen)
