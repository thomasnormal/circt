//===- MooreAttributes.td - Moore attribute definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
#define CIRCT_DIALECT_MOORE_MOOREATTRIBUTES

include "circt/Dialect/Moore/MooreDialect.td"
include "mlir/IR/AttrTypeBase.td"

def FVIntegerAttr : AttrDef<MooreDialect, "FVInteger"> {
  let mnemonic = "fvint";
  let summary = "An attribute containing a four-valued integer";
  let parameters = (ins "FVInt":$value);
  let hasCustomAssemblyFormat = 1;
  let convertFromStorage = "$_self.getValue()";
  let returnType = "circt::FVInt";
}

//===----------------------------------------------------------------------===//
// Modport Direction Enum
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

def ModportDirInput: I32EnumAttrCase<"Input", 0, "input">;
def ModportDirOutput: I32EnumAttrCase<"Output", 1, "output">;
def ModportDirInOut: I32EnumAttrCase<"InOut", 2, "inout">;
def ModportDirRef: I32EnumAttrCase<"Ref", 3, "ref">;

def ModportDirAttr: I32EnumAttr<"ModportDir", "Modport port direction",
            [ModportDirInput, ModportDirOutput, ModportDirInOut, ModportDirRef]>{
  let cppNamespace = "circt::moore";
}

//===----------------------------------------------------------------------===//
// Modport Port Attribute
//===----------------------------------------------------------------------===//

def ModportPortAttr : AttrDef<MooreDialect, "ModportPort"> {
  let mnemonic = "modport_port";
  let summary = "A port in a modport declaration";
  let description = [{
    Represents a single port within a modport declaration, combining a
    direction (input, output, inout, ref) with a signal reference.
  }];
  let parameters = (ins
    "ModportDirAttr":$direction,
    "mlir::FlatSymbolRefAttr":$signal
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def ModportPortArrayAttr : TypedArrayAttrBase<ModportPortAttr,
  "array of modport ports">;

#endif // CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
