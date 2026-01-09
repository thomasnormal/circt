//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPS_TD
#define CIRCT_DIALECT_SIM_SIMOPS_TD

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/Sim/SimDialect.td"
include "circt/Dialect/Sim/SimTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class SimOp<string mnemonic, list<Trait> traits = []> :
    Op<SimDialect, mnemonic, traits>;

def PlusArgsTestOp : SimOp<"plusargs.test", [Pure]> {
  let summary = "SystemVerilog `$test$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found);
  let assemblyFormat = "$formatString attr-dict";
}

def PlusArgsValueOp : SimOp<"plusargs.value", [Pure]> {
  let summary = "SystemVerilog `$value$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found, AnyType:$result);
  let assemblyFormat = "$formatString attr-dict `:` type($result)";
}

def DPIFuncOp : SimOp<"func.dpi",
     [IsolatedFromAbove, Symbol, OpAsmOpInterface,
      FunctionOpInterface]> {
  let summary = "A System Verilog function";
  let description = [{
    `sim.func.dpi` models an external function in a core dialect.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<ModuleType>:$module_type,
    OptionalAttr<DictArrayAttr>:$per_argument_attrs,
    OptionalAttr<LocationArrayAttr>:$argument_locs,
    OptionalAttr<StrAttr>:$verilogName
  );
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType() {
      return getModuleType().getFuncType();
    }

    void setFunctionTypeAttr(mlir::TypeAttr mlirType) {
      setModuleType(cast<hw::ModuleType>(mlirType.getValue()));
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    ::mlir::Region *getCallableRegion() { return nullptr; }

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

def DPICallOp : SimOp<"func.dpi.call",
     [CallOpInterface, AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "A call option for DPI function with an optional clock and enable";
  let description = [{
    `sim.func.dpi.call` represents SystemVerilog DPI function call. There are two
    optional operands `clock` and `enable`.

    If `clock` is not provided, the callee is invoked when input values are changed.
    If provided, the DPI function is called at clock's posedge. The result values behave
    like registers and the DPI function is used as a state transfer function of them.

    `enable` operand is used to conditionally call the DPI since DPI call could be quite
    more expensive than native constructs. When `enable` is low, results of unclocked
    calls are undefined and in SV results they are lowered into `X`. Users are expected
    to gate result values by another `enable` to model a default value of results.

    For clocked calls, a low enable means that its register state transfer function is
    not called. Hence their values will not be modify in that clock.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                      Optional<ClockType>:$clock,
                      Optional<I1>:$enable,
                      Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), llvm::cast<mlir::SymbolRefAttr>(callee));
    }

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];

}

//===----------------------------------------------------------------------===//
// String Formatting
//===----------------------------------------------------------------------===//

def FormatLiteralOp : SimOp<"fmt.literal", [Pure, ConstantLike]> {
  let summary = "Literal string fragment";
  let description = [{
    Creates a constant, raw ASCII string literal for formatted printing.
    The given string attribute will be outputted as is,
    including non-printable characters. The literal may be empty or contain
    null characters ('\0') which must not be interpreted as string
    terminators by backends.
  }];

  let assemblyFormat = "$literal attr-dict";

  let arguments = (ins StrAttr:$literal);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;
}

def FormatHexOp : SimOp<"fmt.hex", [Pure]> {
  let summary = "Hexadecimal format specifier";
  let description = [{
    Format the given integer value as lower-case hexadecimal string.

    The printed value will be left-padded with '0' up to the
    length required to print the maximum value of the argument's
    type. Zero width values will produce the empty string.
    No further prefix will be added.
  }];


  let arguments = (ins AnyInteger:$value, 
    BoolAttr:$isHexUppercase,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"48">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value `,`
    `isUpper`  $isHexUppercase
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatOctOp : SimOp<"fmt.oct", [Pure]> {
  let summary = "Octal format specifier";
  let description = [{
    Format the given integer value as lower-case octal (base eight) string.

    The printed value will be left-padded with '0' up to the
    length required to print the maximum value of the argument's
    type. Zero width values will produce the empty string.
    No further prefix will be added.
  }];


  let arguments = (ins AnyInteger:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"48">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth);

  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatBinOp : SimOp<"fmt.bin", [Pure]> {
  let summary = "Binary format specifier";
  let description = [{
    Format the given integer value as binary (base two) string.

    The printed value will be left-padded with '0' up to the number
    of bits of the argument's type. Zero width values will produce
    the empty string. No further prefix will be added.
  }];

  let arguments = (ins AnyInteger:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"48">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatScientificOp : SimOp<"fmt.exp", [Pure]> {
  let summary = "Real format specifier";
  let description = [{
    Format the given real value in scientific (exponential) notation
  }];

  let arguments = (ins AnyFloat:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    OptionalAttr<I32Attr>:$fieldWidth,
    DefaultValuedAttr<I32Attr,"6">:$fracDigits
  );
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatFloatOp : SimOp<"fmt.flt", [Pure]> {
  let summary = "Real format specifier";
  let description = [{
    Format the given real value in floating point (decimal) notation  
  }];

  let arguments = (ins AnyFloat:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    OptionalAttr<I32Attr>:$fieldWidth,
    DefaultValuedAttr<I32Attr,"6">:$fracDigits
  );
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatGeneralOp : SimOp<"fmt.gen", [Pure]> {
  let summary = "Real format specifier";
  let description = [{
    Format floating-point numbers using either decimal or scientific notation
    based on the exponent and precision parameter fracDigits (default: 6)

    For exp ≥ 0: 
    Uses decimal notation if exp + 1 ≤ max(1, fracDigits), otherwise switches
    to scientific notation with max(0, fracDigits - 1) fractional digits
    
    For exp < 0:
    If exp ≥ -4: Uses decimal notation with max(b - 1, 0) + |exp| fractional digits
    If exp < -4: Uses scientific notation with max(b - 1, 0) fractional digits
  }];

  let arguments = (ins AnyFloat:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    OptionalAttr<I32Attr>:$fieldWidth,
    DefaultValuedAttr<I32Attr,"6">:$fracDigits
  );
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatDecOp : SimOp<"fmt.dec", [Pure]> {
  let summary = "Decimal format specifier";
  let description = [{
    Format the given integer value as signed or unsigned decimal string.

    Leading zeros are omitted. Non-negative or unsigned values will
    _not_ be prefixed with a '+'.

    For unsigned formatting, the printed value will
    be left-padded with spaces up to _at least_ the length required to print
    the maximum unsigned value of the argument's type.
    For signed formatting, the printed value will be
    left-padded with spaces up to _at least_ the length required
    to print the minimum signed value of the argument's type
    including the '-' character.
    E.g., a zero value of type `i1` requires no padding for unsigned
    formatting and one leading space for signed formatting.
    Format specifiers of same argument type and signedness must be
    padded to the same width. Zero width values will produce
    a single '0'.

    Backends are recommended to not exceed the required amount of padding.
  }];

  let arguments = (ins AnyInteger:$value,
    DefaultValuedAttr<BoolAttr,"false">:$isLeftAligned,
    DefaultValuedAttr<I8Attr,"32">:$paddingChar,
    OptionalAttr<I32Attr>:$specifierWidth,
    UnitAttr:$isSigned);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`paddingChar` $paddingChar^)?
    (`specifierWidth` $specifierWidth^)?
    (`signed` $isSigned^)?
    attr-dict `:` qualified(type($value))
  }];

  let extraClassDeclaration = [{
      static inline unsigned getDecimalWidth(unsigned bits, bool isSigned) {
        if (bits == 0)
          return 1;
        if (bits == 1)
          return isSigned ? 2 : 1;

        if (isSigned)
          bits--;

        // Should be precise up until bits = 13301
        // log(2) / log(10) + epsilon
        const double baseConversionFactor = 0.30103;
        unsigned digits = std::ceil(bits * baseConversionFactor);

        return isSigned ? digits + 1 : digits;
      }
  }];
}

def FormatCharOp : SimOp<"fmt.char", [Pure]> {
  let summary = "Character format specifier";
  let description = [{
    Format the given integer value as a single character.

    For integer values up to 127, ASCII compatible encoding is assumed.
    For larger values, the encoding is unspecified.

    If the argument's type width is less than eight bits, the value is
    zero extended.
    If the width is greater than eight bits, the resulting formatted string
    is undefined.
  }];

  let arguments = (ins AnyInteger:$value);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
}

def FormatStringConcatOp : SimOp<"fmt.concat", [Pure]> {
  let summary = "Concatenate format strings";
  let description = [{
    Concatenates an arbitrary number of format strings from
    left to right. If the argument list is empty, the empty string
    literal is produced.

    Concatenations must not be recursive. I.e., a concatenated string should
    not contain itself directly or indirectly.
  }];

  let arguments = (ins Variadic<FormatStringType>:$inputs);
  let results = (outs FormatStringType:$result);

  let hasFolder = true;
  let hasCanonicalizeMethod = true;
  let hasVerifier = true;

  let assemblyFormat = "` ` `(` $inputs `)` attr-dict";

  let extraClassDeclaration = [{
    /// Returns true iff all of the input strings are primitive
    /// (i.e. non-concatenated) fragments or block arguments.
    bool isFlat() {
      return llvm::none_of(getInputs(), [](Value operand) {
        return !!operand.getDefiningOp<circt::sim::FormatStringConcatOp>();
      });
    };

    /// Attempts to flatten this operation's input strings as much as possible.
    ///
    /// The flattened values are pushed into the passed vector.
    /// If the concatenation is acyclic, the function will return 'success'
    /// and all the flattened values are guaranteed to _not_ be the result of
    /// a format string concatenation.
    /// If a cycle is encountered, the function will return 'failure'.
    /// On encountering a cycle, the result of the concat operation
    /// forming the cycle is pushed into the list of flattened values
    /// and flattening continues without recursing into the cycle.
    LogicalResult getFlattenedInputs(
      llvm::SmallVectorImpl<Value> &flatOperands);
  }];
}

def PrintFormattedOp : SimOp<"print"> {
  let summary = "Print a formatted string on a given clock and condition";
  let description = [{
    Evaluate a format string and print it to the simulation console on the
    rising edge of the given clock, if, and only if, the condition argument
    is 'true'.

    Multiple print operations in the same module and on the same clock edge
    are performed according to their order of occurence in the IR. The order
    of printing for operations in different modules, instances or on different
    clocks is undefined.
  }];
  let arguments = (ins
    FormatStringType:$input,
    ClockType:$clock,
    I1:$condition
  );
  let hasCanonicalizeMethod = true;
  let assemblyFormat = "$input `on` $clock `if` $condition attr-dict";
}

def PrintFormattedProcOp : SimOp<"proc.print"> {
  let summary = "Print a formatted string within a procedural region";
  let description = [{
    Evaluate a format string and print it to the simulation console.

    This operation must be within a procedural region.
  }];
  let arguments = (ins FormatStringType:$input);
  let hasVerifier = true;
  let hasCanonicalizeMethod = true;
  let assemblyFormat = "$input attr-dict";
}

//===----------------------------------------------------------------------===//
// Simulation Control
//===----------------------------------------------------------------------===//

def ClockedTerminateOp : SimOp<"clocked_terminate"> {
  let summary = "Terminate the simulation";
  let description = [{
    Implements the semantics of `sim.terminate` if the given condition is true
    on the rising edge of the clock operand.
  }];
  let arguments = (ins
    ClockType:$clock,
    I1:$condition,
    BoolAttr:$success,
    BoolAttr:$verbose
  );
  let assemblyFormat = [{
    $clock `,` $condition `,`
    custom<KeywordBool>($success, "\"success\"", "\"failure\"") `,`
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

def ClockedPauseOp : SimOp<"clocked_pause"> {
  let summary = "Pause the simulation";
  let description = [{
    Implements the semantics of `sim.pause` if the given condition is true on
    the rising edge of the clock operand.
  }];
  let arguments = (ins
    ClockType:$clock,
    I1:$condition,
    BoolAttr:$verbose
  );
  let assemblyFormat = [{
    $clock `,` $condition `,`
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

def TerminateOp : SimOp<"terminate"> {
  let summary = "Terminate the simulation";
  let description = [{
    Terminate the simulation with a success or failure exit code. Depending on
    the `verbose` operand, simulators may print additional information about the
    current simulation time and hierarchical location of the op.

    This op correpsonds to the following SystemVerilog constructs:

    | Operation                        | SystemVerilog             |
    |----------------------------------|---------------------------|
    | `sim.terminate success, quiet`   | `$finish(0)`              |
    | `sim.terminate success, verbose` | `$finish` or `$finish(1)` |
    | `sim.terminate failure, quiet`   | `$fatal(0)`               |
    | `sim.terminate failure, verbose` | `$fatal` or `$fatal(1)`   |

    Note that this op does not match the behavior of the `$exit` system task in
    SystemVerilog, which blocks execution of the calling process until all
    `program` instances have terminated, and then calls `$finish`.
  }];
  let arguments = (ins
    BoolAttr:$success,
    BoolAttr:$verbose
  );
  let assemblyFormat = [{
    custom<KeywordBool>($success, "\"success\"", "\"failure\"") `,`
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

def PauseOp : SimOp<"pause"> {
  let summary = "Pause the simulation";
  let description = [{
    Interrupt the simulation and give control back to the user in case of an
    interactive simulation. Non-interactive simulations may choose to terminate
    instead. Depending on the `verbose` operand, simulators may print additional
    information about the current simulation time and hierarchical location of
    the op.

    This op corresponds to the following SystemVerilog constructs:

    | Operation           | SystemVerilog          |
    |---------------------|------------------------|
    | `sim.pause quiet`   | `$stop(0)`             |
    | `sim.pause verbose` | `$stop` or `$stop(1)`  |
  }];
  let arguments = (ins BoolAttr:$verbose);
  let assemblyFormat = [{
    custom<KeywordBool>($verbose, "\"verbose\"" , "\"quiet\"") attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Event-Driven Simulation Operations (SimIR)
//===----------------------------------------------------------------------===//

// Trait for operations that must occur within a simulation process region.
def SimProcessRegion : NativeOpTrait<"SimProcessRegion"> {
  let cppNamespace = "::circt::sim";
}

def SimProcessOp : SimOp<"process", [
  NoRegionArguments,
  SimProcessRegion,
  RecursiveMemoryEffects,
]> {
  let summary = "A concurrent process for event-driven simulation";
  let description = [{
    `sim.process` defines a concurrent process that executes during simulation.
    The process runs from the beginning of simulation and can suspend execution
    using `sim.suspend` operations. When suspended, the process waits for
    specified events or delays before resuming.

    The process may have results that are yielded when execution suspends or
    completes. These results represent the outputs of the process and are
    updated each time the process yields.

    Unlike combinational processes, a `sim.process` maintains state across
    suspensions and can model sequential behavior with explicit timing.

    Example:
    ```mlir
    hw.module @top() {
      %out = sim.process -> i32 {
        %c0 = hw.constant 0 : i32
        sim.suspend yield %c0 : i32, ^loop
      ^loop:
        %prev = sim.process.state : i32
        %c1 = hw.constant 1 : i32
        %next = comb.add %prev, %c1 : i32
        sim.suspend delay 10ns yield %next : i32, ^loop
      }
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    (`->` type($results)^)?
    attr-dict-with-keyword $body
  }];
}

def SimCombProcessOp : SimOp<"comb_process", [
  NoRegionArguments,
  SimProcessRegion,
  RecursiveMemoryEffects,
]> {
  let summary = "A combinational process triggered by value changes";
  let description = [{
    `sim.comb_process` defines a combinational process that re-evaluates
    whenever any of the values used within its body change. This is similar
    to a Verilog `always @(*)` or `always_comb` block.

    The process executes once at the beginning of simulation to establish
    initial values, and then re-executes whenever any input signal changes.
    The body must terminate with a `sim.yield` operation.

    Example:
    ```mlir
    hw.module @alu(%a: i32, %b: i32, %op: i2) -> (result: i32) {
      %result = sim.comb_process -> i32 {
        %add = comb.add %a, %b : i32
        %sub = comb.sub %a, %b : i32
        %and = comb.and %a, %b : i32
        %or = comb.or %a, %b : i32
        %sel = hw.array_create %add, %sub, %and, %or : i32
        %out = hw.array_get %sel[%op] : !hw.array<4xi32>, i2
        sim.yield %out : i32
      }
      hw.output %result : i32
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    (`->` type($results)^)?
    attr-dict-with-keyword $body
  }];
}

def SimSuspendOp : SimOp<"suspend", [
  Terminator,
  AttrSizedOperandSegments,
]> {
  let summary = "Suspend process execution until events or delay";
  let description = [{
    `sim.suspend` suspends execution of the enclosing simulation process.
    Execution resumes at the destination block when:
    - Any of the `observed` values change, OR
    - The specified `delay` time has elapsed

    The `yield` operands provide the current values for the process results
    while suspended. The `dest` operands are passed to the destination block.

    The optional `region` attribute specifies the IEEE 1800 scheduling region
    where resumed execution should occur. Valid values are:
    - "active" (default): Normal active region execution
    - "inactive": #0 delay region
    - "nba": Non-blocking assignment update region
    - "reactive": Program block execution region

    Example:
    ```mlir
    // Wait for clock edge
    sim.suspend observed(%clk : i1), yield (%val : i32), ^next

    // Wait for 10 nanoseconds
    sim.suspend delay 10ns, yield (%val : i32), ^next

    // Wait for either value change or timeout
    sim.suspend observed(%data : i8), delay 100ns, yield (%result : i32), ^next

    // Schedule in NBA region
    sim.suspend region "nba", yield (%val : i32), ^next
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$yieldOperands,
    OptionalAttr<I64Attr>:$delayFemtoseconds,
    Variadic<HWValueType>:$observed,
    Variadic<AnyType>:$destOperands,
    OptionalAttr<StrAttr>:$region
  );
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    (`region` $region^)?
    (`observed` `(` $observed^ `:` qualified(type($observed)) `)` `,`)?
    (`delay` $delayFemtoseconds^ `fs` `,`)?
    `yield` `(` $yieldOperands `:` type($yieldOperands) `)` `,`
    $dest (`(` $destOperands^ `:` qualified(type($destOperands)) `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

def SimYieldOp : SimOp<"yield", [
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Yield results from a combinational simulation process";
  let description = [{
    `sim.yield` terminates a `sim.comb_process` and yields the result values.
    This operation is analogous to `llhd.yield` for combinational processes.

    Example:
    ```mlir
    sim.comb_process -> i32 {
      %result = comb.add %a, %b : i32
      sim.yield %result : i32
    }
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{
    ($results^ `:` type($results))?
    attr-dict
  }];
  let hasVerifier = 1;
}

def SimDriveOp : SimOp<"drive", []> {
  let summary = "Drive a value onto a signal with optional delay";
  let description = [{
    `sim.drive` schedules a value to be driven onto a signal. The drive can
    be immediate (combinational) or delayed.

    The `mode` attribute specifies the drive behavior:
    - "blocking": Immediate assignment, blocks until complete (like Verilog `=`)
    - "nonblocking": Scheduled for NBA region (like Verilog `<=`)
    - "continuous": Continuous assignment that persists

    The optional `delay` specifies when the drive takes effect, in femtoseconds.
    For nonblocking assignments, the delay is relative to the NBA region.

    The optional `strength` attribute models Verilog signal strengths.

    Example:
    ```mlir
    // Blocking assignment
    sim.drive %signal, %value : i32

    // Non-blocking assignment
    sim.drive %signal, %value mode "nonblocking" : i32

    // Delayed assignment (10ns delay)
    sim.drive %signal, %value delay 10000000 : i32

    // Continuous assignment with strength
    sim.drive %signal, %value mode "continuous" strength "strong" : i32
    ```
  }];
  let arguments = (ins
    AnyType:$signal,
    AnyType:$value,
    OptionalAttr<I64Attr>:$delayFemtoseconds,
    DefaultValuedStrAttr<StrAttr, "blocking">:$mode,
    OptionalAttr<StrAttr>:$strength
  );
  let assemblyFormat = [{
    $signal `,` $value
    (`delay` $delayFemtoseconds^)?
    (`mode` $mode^)?
    (`strength` $strength^)?
    attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

def SimSampleOp : SimOp<"sample", [Pure]> {
  let summary = "Sample the current value of a signal";
  let description = [{
    `sim.sample` reads the current value of a signal at the current simulation
    time. This represents a synchronous read of signal state.

    The `region` attribute can specify when sampling occurs according to
    IEEE 1800 scheduling regions:
    - "preponed": Sample before any changes in the current time slot
    - "observed": Sample in the observed region (for assertions)
    - "postponed": Sample after all updates (for $strobe-like behavior)

    By default, sampling occurs in the active region.

    Example:
    ```mlir
    %value = sim.sample %signal : i32
    %strobed = sim.sample %signal region "postponed" : i32
    ```
  }];
  let arguments = (ins
    AnyType:$signal,
    OptionalAttr<StrAttr>:$region
  );
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $signal (`region` $region^)? attr-dict `:` type($value)
  }];
}

def SimTimeOp : SimOp<"time", [Pure]> {
  let summary = "Get the current simulation time";
  let description = [{
    `sim.time` returns the current simulation time. The time is returned
    as an integer value in the specified unit.

    Supported units:
    - "fs": femtoseconds (10^-15 s)
    - "ps": picoseconds (10^-12 s)
    - "ns": nanoseconds (10^-9 s)
    - "us": microseconds (10^-6 s)
    - "ms": milliseconds (10^-3 s)
    - "s": seconds

    Example:
    ```mlir
    %time_ns = sim.time unit "ns" : i64
    %time_fs = sim.time unit "fs" : i64
    ```
  }];
  let arguments = (ins
    DefaultValuedStrAttr<StrAttr, "ns">:$unit
  );
  let results = (outs AnyInteger:$time);
  let assemblyFormat = [{
    (`unit` $unit^)? attr-dict `:` type($time)
  }];
}

def SimWaitEventOp : SimOp<"wait_event", []> {
  let summary = "Wait for a specific event condition";
  let description = [{
    `sim.wait_event` suspends execution until the specified event condition
    is met. This is a more specialized form of suspension for common patterns.

    The `edge` attribute specifies edge sensitivity:
    - "posedge": Rising edge (0->1 transition)
    - "negedge": Falling edge (1->0 transition)
    - "anyedge": Any value change

    Example:
    ```mlir
    // Wait for positive clock edge
    sim.wait_event %clk edge "posedge"

    // Wait for any change on data
    sim.wait_event %data edge "anyedge"
    ```
  }];
  let arguments = (ins
    I1:$signal,
    DefaultValuedStrAttr<StrAttr, "posedge">:$edge
  );
  let assemblyFormat = [{
    $signal (`edge` $edge^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Sensitivity List and Edge Detection Operations
//===----------------------------------------------------------------------===//

def SimSensitivityListOp : SimOp<"sensitivity", [Pure]> {
  let summary = "Define a sensitivity list for process triggering";
  let description = [{
    `sim.sensitivity` defines a sensitivity list that specifies when a process
    should be triggered. This corresponds to the @(...) construct in Verilog.

    The sensitivity list can contain multiple signals with different edge
    sensitivities. The process is triggered when ANY of the specified
    conditions is met (OR semantics).

    Edge types:
    - "posedge": Rising edge (0->1)
    - "negedge": Falling edge (1->0)
    - "anyedge": Any value change (default)
    - "level": Level-sensitive (same as anyedge for simulation)

    Example:
    ```mlir
    // Sensitivity to positive clock edge
    %sens = sim.sensitivity posedge %clk : i1

    // Sensitivity to both edges of reset and clock
    %sens = sim.sensitivity posedge %clk, negedge %rst : i1, i1

    // Combinational sensitivity (any change)
    %sens = sim.sensitivity anyedge %a, %b, %c : i8, i8, i8
    ```
  }];
  let arguments = (ins
    Variadic<HWValueType>:$signals,
    ArrayAttr:$edges
  );
  let results = (outs I1:$triggered);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SimEdgeDetectOp : SimOp<"edge_detect", [Pure]> {
  let summary = "Detect edge transitions on a signal";
  let description = [{
    `sim.edge_detect` detects edge transitions on a signal by comparing the
    current value with the previous value. Returns true if the specified
    edge type occurred.

    This operation is typically used in conjunction with signal sampling
    to implement @(posedge clk) style triggers.

    Edge types:
    - "posedge": Returns true on 0->1 transition
    - "negedge": Returns true on 1->0 transition
    - "anyedge": Returns true on any value change

    Example:
    ```mlir
    // Detect positive edge
    %posedge = sim.edge_detect %current, %previous edge "posedge" : i1

    // Detect any change on multi-bit signal
    %changed = sim.edge_detect %data_new, %data_old edge "anyedge" : i8
    ```
  }];
  let arguments = (ins
    HWValueType:$current,
    HWValueType:$previous,
    DefaultValuedStrAttr<StrAttr, "posedge">:$edge
  );
  let results = (outs I1:$detected);
  let assemblyFormat = [{
    $current `,` $previous (`edge` $edge^)? attr-dict `:` type($current)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def SimSignalRegisterOp : SimOp<"signal_register", []> {
  let summary = "Register a signal value for edge detection";
  let description = [{
    `sim.signal_register` captures the current value of a signal and stores
    it for later comparison. This is used to track signal history for
    edge detection.

    The operation maintains state across simulation cycles, providing the
    previous value of a signal when queried.

    Example:
    ```mlir
    // Register clock signal for edge detection
    %prev_clk = sim.signal_register %clk : i1

    // The previous value can be used with edge_detect
    %posedge = sim.edge_detect %clk, %prev_clk edge "posedge" : i1
    ```
  }];
  let arguments = (ins HWValueType:$signal);
  let results = (outs HWValueType:$previous);
  let assemblyFormat = [{
    $signal attr-dict `:` type($signal)
  }];
}

def SimTriggeredProcessOp : SimOp<"triggered_process", [
  NoRegionArguments,
  SimProcessRegion,
  RecursiveMemoryEffects,
]> {
  let summary = "A process triggered by a sensitivity list";
  let description = [{
    `sim.triggered_process` defines a process that executes when its
    sensitivity list is triggered. This is similar to a Verilog
    `always @(sensitivity_list)` block.

    The sensitivity list specifies which signals and edge types trigger
    the process. Multiple triggers are OR'ed together - the process
    executes when any trigger condition is met.

    The body must terminate with either `sim.yield` (for outputs) or
    `sim.suspend` (for explicit control flow).

    Example:
    ```mlir
    // Sequential logic with clock and reset
    sim.triggered_process @(posedge %clk, negedge %rst_n) -> i32 {
      %c1 = hw.constant 1 : i32
      %rst_active = comb.icmp eq %rst_n, %c0_i1 : i1
      %next = scf.if %rst_active -> i32 {
        scf.yield %c0_i32 : i32
      } else {
        %inc = comb.add %state, %c1 : i32
        scf.yield %inc : i32
      }
      sim.yield %next : i32
    }
    ```
  }];
  let arguments = (ins
    Variadic<HWValueType>:$sensitivitySignals,
    ArrayAttr:$sensitivityEdges
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SimPrevValueOp : SimOp<"prev", [Pure]> {
  let summary = "Get the previous value of a signal";
  let description = [{
    `sim.prev` returns the value of a signal from the previous delta cycle
    or time step. This is useful for implementing sequential logic and
    edge detection.

    The optional `delta` attribute specifies how many delta cycles back
    to look. Default is 1 (immediate previous value).

    Example:
    ```mlir
    // Get previous clock value for edge detection
    %prev_clk = sim.prev %clk : i1

    // Get value from 2 deltas ago
    %old_data = sim.prev %data delta 2 : i32
    ```
  }];
  let arguments = (ins
    HWValueType:$signal,
    DefaultValuedAttr<I32Attr, "1">:$delta
  );
  let results = (outs HWValueType:$previous);
  let assemblyFormat = [{
    $signal (`delta` $delta^)? attr-dict `:` type($signal)
  }];
}

def SimDeltaCycleOp : SimOp<"delta_cycle", []> {
  let summary = "Execute a delta cycle boundary";
  let description = [{
    `sim.delta_cycle` marks a delta cycle boundary in simulation. All
    scheduled updates from the current delta cycle are applied, and
    triggered processes are re-evaluated.

    This is primarily used for explicit control over simulation
    scheduling when needed.

    Example:
    ```mlir
    // Drive a value and force delta cycle update
    sim.drive %sig, %val mode "nonblocking" : i32
    sim.delta_cycle
    // Now the driven value is visible
    %updated = sim.sample %sig : i32
    ```
  }];
  let assemblyFormat = "attr-dict";
}

def SimIterationLimitOp : SimOp<"iteration_limit", []> {
  let summary = "Set iteration limit for delta cycle execution";
  let description = [{
    `sim.iteration_limit` sets the maximum number of delta cycles that
    can execute at a single simulation time before the simulator
    declares an infinite loop. This is a safety mechanism to prevent
    runaway combinational loops.

    If the limit is reached, simulation will either error out or
    advance time (depending on implementation).

    Example:
    ```mlir
    // Set limit to 1000 delta cycles
    sim.iteration_limit 1000

    // Set no limit (use with caution)
    sim.iteration_limit 0
    ```
  }];
  let arguments = (ins I32Attr:$limit);
  let assemblyFormat = "$limit attr-dict";
}

#endif // CIRCT_DIALECT_SIM_SIMOPS_TD
