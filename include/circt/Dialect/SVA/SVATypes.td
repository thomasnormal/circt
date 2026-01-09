//===- SVATypes.td - SVA dialect types ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SVA_SVATYPES_TD
#define CIRCT_DIALECT_SVA_SVATYPES_TD

include "circt/Dialect/SVA/SVADialect.td"
include "mlir/IR/AttrTypeBase.td"

class SVATypeDef<string name, string typeMnemonic> : TypeDef<SVADialect, name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Sequence Type
//===----------------------------------------------------------------------===//

def SVASequenceType : SVATypeDef<"Sequence", "sequence"> {
  let summary = "SVA sequence type";
  let description = [{
    The `sva.sequence` type represents a temporal sequence in SystemVerilog
    Assertions. A sequence specifies a pattern of boolean conditions over time.

    Examples of sequences:
    - `a ##1 b` - "a" followed by "b" in the next cycle
    - `a ##[1:3] b` - "a" followed by "b" within 1 to 3 cycles
    - `a[*3]` - "a" repeated 3 times consecutively
    - `a ##1 b ##1 c` - "a", then "b", then "c" in consecutive cycles

    An `i1` (boolean) is implicitly convertible to a sequence, representing a
    sequence that matches in a single cycle if the boolean is true.
  }];
}

//===----------------------------------------------------------------------===//
// Property Type
//===----------------------------------------------------------------------===//

def SVAPropertyType : SVATypeDef<"Property", "property"> {
  let summary = "SVA property type";
  let description = [{
    The `sva.property` type represents a verifiable property in SystemVerilog
    Assertions. Properties are built from sequences and temporal operators.

    Examples of properties:
    - `req |-> gnt` - If "req" is true, then "gnt" must be true (overlapping)
    - `req |=> gnt` - If "req" is true, then "gnt" must be true next cycle
    - `always req |-> ##[1:3] gnt` - Whenever "req" is true, "gnt" follows

    Both `i1` (boolean) and `sva.sequence` are implicitly convertible to a
    property.
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

// A sequence or boolean (i1)
def SVAAnySequenceType : AnyTypeOf<[I1, SVASequenceType]>;

// A property, sequence, or boolean (i1)
def SVAAnyPropertyType : AnyTypeOf<[I1, SVASequenceType, SVAPropertyType]>;

// Either a sequence or property type (not i1)
def SVASequenceOrPropertyType : AnyTypeOf<[SVASequenceType, SVAPropertyType]>;

#endif // CIRCT_DIALECT_SVA_SVATYPES_TD
