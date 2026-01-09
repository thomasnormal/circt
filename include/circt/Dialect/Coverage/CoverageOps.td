//===- CoverageOps.td - Coverage dialect operations --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the Coverage dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COVERAGE_COVERAGEOPS_TD
#define CIRCT_DIALECT_COVERAGE_COVERAGEOPS_TD

include "circt/Dialect/Coverage/CoverageDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class CoverageOp<string mnemonic, list<Trait> traits = []> :
  Op<CoverageDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Line Coverage
//===----------------------------------------------------------------------===//

def LineCoverageOp : CoverageOp<"line"> {
  let summary = "Record line coverage information";
  let description = [{
    Records that execution has reached a particular line in the source code.
    When this operation is executed during simulation, the corresponding
    coverage counter is incremented.

    Example:
    ```mlir
    coverage.line "module.sv" line 42
    ```

    The operation takes a filename and line number as attributes. Multiple
    executions of the same coverage point will increment the same counter,
    allowing hit counts to be recorded.
  }];

  let arguments = (ins
    StrAttr:$filename,
    I32Attr:$line,
    OptionalAttr<StrAttr>:$tag
  );

  let assemblyFormat = [{
    $filename `line` $line (`tag` $tag^)? attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get a unique identifier for this coverage point
    std::string getCoveragePointId() const {
      return (getFilename() + ":" + std::to_string(getLine())).str();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Toggle Coverage
//===----------------------------------------------------------------------===//

def ToggleCoverageOp : CoverageOp<"toggle"> {
  let summary = "Record toggle coverage for a signal";
  let description = [{
    Tracks value transitions for a signal. Records both 0->1 and 1->0
    transitions separately, allowing toggle coverage metrics to be computed.

    Example:
    ```mlir
    coverage.toggle %signal name "clk_enable" : i1
    ```

    For multi-bit signals, each bit is tracked independently. The operation
    maintains state between simulation cycles to detect transitions.

    Toggle coverage is important for verifying that signals are exercised
    in both directions during simulation.
  }];

  let arguments = (ins
    AnyType:$signal,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$hierarchy
  );

  let assemblyFormat = [{
    $signal `name` $name (`hierarchy` $hierarchy^)? attr-dict `:` type($signal)
  }];

  let extraClassDeclaration = [{
    /// Get the bit width of the tracked signal
    unsigned getSignalWidth() const;
  }];
}

//===----------------------------------------------------------------------===//
// Branch Coverage
//===----------------------------------------------------------------------===//

def BranchCoverageOp : CoverageOp<"branch"> {
  let summary = "Record branch coverage information";
  let description = [{
    Records which branch of a conditional was taken. This is used to track
    coverage of if-else statements, case statements, and other control flow.

    Example:
    ```mlir
    coverage.branch %condition name "state_machine_cond" true_id 0 false_id 1 : i1
    ```

    The operation records whether the true or false branch was taken based
    on the condition value.
  }];

  let arguments = (ins
    I1:$condition,
    StrAttr:$name,
    I32Attr:$true_id,
    I32Attr:$false_id,
    OptionalAttr<StrAttr>:$filename,
    OptionalAttr<I32Attr>:$line
  );

  let assemblyFormat = [{
    $condition `name` $name `true_id` $true_id `false_id` $false_id
    (`at` $filename^ `:` $line)? attr-dict `:` type($condition)
  }];
}

//===----------------------------------------------------------------------===//
// Coverage Group
//===----------------------------------------------------------------------===//

def CoverageGroupOp : CoverageOp<"group", [SingleBlock, NoTerminator]> {
  let summary = "Group related coverage points together";
  let description = [{
    Groups related coverage points under a common name. This is useful for
    organizing coverage data and computing coverage metrics for specific
    parts of the design.

    Example:
    ```mlir
    coverage.group "fsm_coverage" {
      coverage.line "fsm.sv" line 10
      coverage.toggle %state name "current_state" : i4
    }
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<StrAttr>:$description
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $name (`description` $description^)? $body attr-dict
  }];
}

#endif // CIRCT_DIALECT_COVERAGE_COVERAGEOPS_TD
