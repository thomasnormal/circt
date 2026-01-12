//===- ArcTypes.td - Arc dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCTYPES_TD
#define CIRCT_DIALECT_ARC_ARCTYPES_TD

include "circt/Dialect/Arc/ArcDialect.td"
include "mlir/IR/AttrTypeBase.td"

class ArcTypeDef<string name> : TypeDef<ArcDialect, name> { }

def StateType : ArcTypeDef<"State"> {
  let mnemonic = "state";
  let parameters = (ins "::mlir::Type":$type);
  let assemblyFormat = "`<` $type `>`";
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let extraClassDeclaration = [{
    unsigned getBitWidth();
    unsigned getByteWidth() { return (getBitWidth() + 7) / 8; }
  }];
}

def MemoryType : ArcTypeDef<"Memory"> {
  let mnemonic = "memory";
  let parameters = (ins "unsigned":$numWords,
                        "::mlir::IntegerType":$wordType,
                        "::mlir::IntegerType":$addressType);
  let assemblyFormat = "`<` $numWords `x` $wordType `,` $addressType `>`";

  let extraClassDeclaration = [{
    unsigned getStride();
  }];
}

def StorageType : ArcTypeDef<"Storage"> {
  let mnemonic = "storage";
  let parameters = (ins OptionalParameter<"unsigned">:$size);
  let assemblyFormat = "(`<` $size^ `>`)?";
}

def SimModelInstance : ArcTypeDef<"SimModelInstance"> {
  let mnemonic = "sim.instance";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$model);
  let assemblyFormat = "`<` $model `>`";
}

//===----------------------------------------------------------------------===//
// 4-State Logic Type
//===----------------------------------------------------------------------===//

def FourStateType : ArcTypeDef<"FourState"> {
  let mnemonic = "logic";
  let summary = "4-state logic type with X/Z support";
  let description = [{
    A 4-state logic type that supports X (unknown) and Z (high-impedance) values
    in addition to 0 and 1. This type is essential for proper SystemVerilog
    simulation semantics.

    The type uses dual-rail encoding internally where each logical bit is
    represented by two physical bits:
    - value bit: the logical value when defined
    - xz_mask bit: indicates whether the bit is X or Z

    The encoding is:
    - 00 = logic 0
    - 01 = logic 1
    - 10 = X (unknown)
    - 11 = Z (high-impedance)

    Example:
    ```mlir
    %x = arc.fourstate.constant 8 : <4>  // 4-bit value = 0b1000
    %y = arc.fourstate.x : <4>           // 4-bit X value
    ```
  }];
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Get the number of logical bits in this type
    unsigned getBitWidth() const { return getWidth(); }

    /// Get the number of physical storage bits (2x logical width for dual-rail)
    unsigned getStorageBitWidth() const { return getWidth() * 2; }

    /// Get the byte width needed to store this type
    unsigned getByteWidth() const { return (getStorageBitWidth() + 7) / 8; }
  }];
}

/// Type constraint for FourState types
def AnyFourState : Type<CPred<"llvm::isa<FourStateType>($_self)">,
                         "4-state logic type">;

/// Type constraint for FourState or signless integer types
def FourStateOrInteger : AnyTypeOf<[AnyFourState, AnySignlessInteger],
                                    "4-state logic or signless integer">;

#endif // CIRCT_DIALECT_ARC_ARCTYPES_TD
