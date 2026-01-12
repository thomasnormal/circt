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
    std::string getCoveragePointId() {
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
    unsigned getSignalWidth();
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
// FSM State Coverage
//===----------------------------------------------------------------------===//

def FSMStateCoverageOp : CoverageOp<"fsm.state"> {
  let summary = "Record FSM state coverage information";
  let description = [{
    Tracks which states of a finite state machine have been visited during
    simulation. This is essential for verifying that all FSM states are
    reachable and have been exercised.

    Example:
    ```mlir
    coverage.fsm.state %current_state name "protocol_fsm" num_states 8 : i4
    ```

    The operation records hits for each state value observed. The num_states
    attribute indicates the total number of expected states for coverage
    percentage calculation.
  }];

  let arguments = (ins
    AnySignlessInteger:$state,
    StrAttr:$name,
    I32Attr:$num_states,
    OptionalAttr<StrAttr>:$hierarchy,
    OptionalAttr<StrArrayAttr>:$state_names
  );

  let assemblyFormat = [{
    $state `name` $name `num_states` $num_states
    (`hierarchy` $hierarchy^)? (`state_names` $state_names^)?
    attr-dict `:` type($state)
  }];

  let extraClassDeclaration = [{
    /// Get the bit width of the state signal
    unsigned getStateWidth() {
      return llvm::cast<mlir::IntegerType>(getState().getType()).getWidth();
    }

    /// Get a unique identifier for this coverage point
    std::string getCoveragePointId() {
      return ("fsm_state:" + getName()).str();
    }
  }];
}

//===----------------------------------------------------------------------===//
// FSM Transition Coverage
//===----------------------------------------------------------------------===//

def FSMTransitionCoverageOp : CoverageOp<"fsm.transition"> {
  let summary = "Record FSM state transition coverage";
  let description = [{
    Tracks state transitions in a finite state machine. Records which
    state-to-state transitions have been exercised during simulation.

    Example:
    ```mlir
    coverage.fsm.transition %prev_state, %next_state name "protocol_fsm"
        num_states 8 : i4
    ```

    The operation records hits for each (from_state, to_state) pair observed.
    This helps verify that all expected state transitions are exercised.
  }];

  let arguments = (ins
    AnySignlessInteger:$prev_state,
    AnySignlessInteger:$next_state,
    StrAttr:$name,
    I32Attr:$num_states,
    OptionalAttr<StrAttr>:$hierarchy
  );

  let assemblyFormat = [{
    $prev_state `,` $next_state `name` $name `num_states` $num_states
    (`hierarchy` $hierarchy^)? attr-dict `:` type($prev_state)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Get the bit width of the state signals
    unsigned getStateWidth() {
      return llvm::cast<mlir::IntegerType>(getPrevState().getType()).getWidth();
    }

    /// Get a unique identifier for this coverage point
    std::string getCoveragePointId() {
      return ("fsm_transition:" + getName()).str();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Expression Coverage (Condition Coverage)
//===----------------------------------------------------------------------===//

def ExpressionCoverageOp : CoverageOp<"expression"> {
  let summary = "Record expression/condition coverage";
  let description = [{
    Tracks coverage for boolean expressions with multiple conditions.
    This supports condition coverage (each condition evaluated to both
    true and false) and modified condition/decision coverage (MC/DC).

    Example:
    ```mlir
    coverage.expression %a, %b, %c name "complex_cond" : i1, i1, i1
    ```

    For MC/DC coverage, each condition should independently affect the
    decision outcome. The operation tracks all combinations of condition
    values that have been observed.
  }];

  let arguments = (ins
    Variadic<I1>:$conditions,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$hierarchy,
    OptionalAttr<StrArrayAttr>:$condition_names
  );

  let assemblyFormat = [{
    $conditions `name` $name
    (`hierarchy` $hierarchy^)? (`condition_names` $condition_names^)?
    attr-dict `:` type($conditions)
  }];

  let extraClassDeclaration = [{
    /// Get the number of conditions in this expression
    unsigned getNumConditions() {
      return getConditions().size();
    }

    /// Get a unique identifier for this coverage point
    std::string getCoveragePointId() {
      return ("expression:" + getName()).str();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Assertion Coverage
//===----------------------------------------------------------------------===//

def AssertionCoverageOp : CoverageOp<"assertion"> {
  let summary = "Record assertion coverage information";
  let description = [{
    Tracks coverage of assertions. Records how many times an assertion
    was evaluated and whether it passed or failed.

    Example:
    ```mlir
    coverage.assertion %check name "data_valid_check" : i1
    ```

    This helps track which assertions have been exercised during
    simulation and identify dead or unreachable assertions.
  }];

  let arguments = (ins
    I1:$check,
    StrAttr:$name,
    OptionalAttr<StrAttr>:$hierarchy,
    OptionalAttr<StrAttr>:$filename,
    OptionalAttr<I32Attr>:$line
  );

  let assemblyFormat = [{
    $check `name` $name
    (`hierarchy` $hierarchy^)? (`at` $filename^ `:` $line)?
    attr-dict `:` type($check)
  }];

  let extraClassDeclaration = [{
    /// Get a unique identifier for this coverage point
    std::string getCoveragePointId() {
      return ("assertion:" + getName()).str();
    }
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
