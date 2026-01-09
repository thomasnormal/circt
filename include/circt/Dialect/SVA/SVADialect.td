//===- SVADialect.td - SVA dialect definition ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SVA (SystemVerilog Assertions) dialect for assertion
// checking during simulation and formal verification.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SVA_SVADIALECT_TD
#define CIRCT_DIALECT_SVA_SVADIALECT_TD

include "mlir/IR/OpBase.td"

def SVADialect : Dialect {
  let name = "sva";
  let summary = "SystemVerilog Assertions dialect for runtime assertion checking";
  let description = [{
    The SVA dialect provides operations for SystemVerilog Assertions that can be
    used for both runtime simulation checking and formal verification. This
    dialect provides a higher-level interface that maps directly to SVA syntax
    while supporting:

    - Sequence operations: Cycle delays (##n), repetition ([*n], [*m:n]),
      concatenation, and disjunction
    - Property operations: Implications (|->, |=>), assertions, assumptions,
      and coverage
    - Runtime checking: Automata-based sequence matching with failure reporting
    - Vacuity detection: Identifying vacuously passing assertions

    The SVA dialect can be lowered to LTL/Verif dialects for formal verification
    or to simulation runtime for dynamic checking.

    Example:
    ```mlir
    // ##2 req |-> ##[1:3] gnt
    %delayed_req = sva.seq.delay %req, 2 : i1
    %delayed_gnt = sva.seq.delay %gnt, 1, 3 : i1
    %prop = sva.prop.implication %delayed_req, %delayed_gnt overlapping : !sva.sequence, !sva.sequence
    sva.assert %prop label "req_gnt_check" : !sva.property
    ```
  }];

  let cppNamespace = "circt::sva";
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "ltl::LTLDialect",
    "verif::VerifDialect",
    "hw::HWDialect"
  ];

  let extraClassDeclaration = [{
    void registerOps();
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_SVA_SVADIALECT_TD
