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
include "circt/Dialect/Sim/SimOpInterfaces.td"
include "circt/Dialect/Sim/SimTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

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

def FormatHexOp : SimOp<"fmt.hex",
  [Pure, DeclareOpInterfaceMethods<SimValueFormatter>]
> {
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

def FormatOctOp : SimOp<"fmt.oct",
  [Pure, DeclareOpInterfaceMethods<SimValueFormatter>]
> {
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

def FormatBinOp : SimOp<"fmt.bin",
    [Pure, DeclareOpInterfaceMethods<SimValueFormatter>]
> {
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

def FormatScientificOp : SimOp<"fmt.exp",
  [Pure, DeclareOpInterfaceMethods<SimValueFormatter>]
> {
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

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatFloatOp : SimOp<"fmt.flt",
  [Pure, DeclareOpInterfaceMethods<SimValueFormatter>
]> {
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

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatGeneralOp : SimOp<"fmt.gen",
  [Pure, DeclareOpInterfaceMethods<SimValueFormatter>]
> {
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

  let assemblyFormat = [{
    $value
    (`isLeftAligned` $isLeftAligned^)?
    (`fieldWidth` $fieldWidth^)?
    (`fracDigits` $fracDigits^)?
    attr-dict `:` qualified(type($value))
  }];
}

def FormatDecOp : SimOp<"fmt.dec",
    [Pure, DeclareOpInterfaceMethods<SimValueFormatter>]
> {
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

def FormatCharOp : SimOp<"fmt.char",
  [Pure, DeclareOpInterfaceMethods<SimValueFormatter>]
> {
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

def FormatDynStringOp : SimOp<"fmt.dyn_string", [Pure]> {
  let summary = "Format a dynamic string as a format string";
  let description = [{
    Convert a dynamic string value to a format string for use in formatted
    printing operations. This is used to format dynamic string values such as
    enum names returned by `.name()` method calls.

    The input can be a `sim::DynamicStringType` or an LLVM struct `{ptr, i64}`
    representing a dynamic string (pointer to character data and length).
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs FormatStringType:$result);

  let assemblyFormat = "$value attr-dict `:` type($value)";
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
// Dynamic Strings
//===----------------------------------------------------------------------===//

def StringConstantOp : SimOp<"string.literal", [Pure, ConstantLike]> {
  let summary = "Literal string fragment";
  let description = [{
    Creates a dynamic string from a constant string literal.
  }];

  let arguments = (ins StrAttr:$literal);
  let results = (outs DynamicStringType:$result);

  let hasFolder = true;

  let assemblyFormat = "$literal attr-dict";
}

def StringLengthOp : SimOp<"string.length", [Pure]> {
  let summary = "Length of a string";
  let description = [{
    Returns the length of a dynamic string as an `i64` value.
  }];

  let arguments = (ins DynamicStringType:$input);
  let results = (outs I64:$result);

  let hasFolder = true;

  let assemblyFormat = "$input attr-dict";
}

def StringConcatOp : SimOp<"string.concat", [Pure]> {
  let summary = "Concatenate strings";
  let description = [{
    Concatenates multiple strings into a single, longer string. The first
    operand appears first in the output string. The last character of the first
    operand is immediately followed by the first character of the second
    operand. If the operation has a single operand, its result is equal to that
    operand. If the operation has zero operands, it returns an empty string.
    The length of the result is the sum of each operand's length.

    Example:

    ```mlir
    %0 = sim.string.literal "abc"
    %1 = sim.string.literal "def"
    %2 = sim.string.literal "ghj"
    %3 = sim.string.concat (%0, %1, %2)  // %3 is "abcdefghj"
    %4 = sim.string.concat (%0)          // %4 is "abc"
    %5 = sim.string.concat ()            // %5 is ""
    ```
  }];

  let assemblyFormat = "` ` `(` $inputs `)` attr-dict";

  let arguments = (ins Variadic<DynamicStringType>:$inputs);
  let results = (outs DynamicStringType:$result);

  let hasFolder = true;
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
    attr-dict `:` type($signal) `,` type($value)
  }];
  let hasVerifier = 1;
}

def SimSampleOp : SimOp<"sample", [Pure, SameOperandsAndResultType]> {
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

def SimEdgeDetectOp : SimOp<"edge_detect", [Pure, SameTypeOperands]> {
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

def SimSignalRegisterOp : SimOp<"signal_register", [SameOperandsAndResultType]> {
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

def SimPrevValueOp : SimOp<"prev", [Pure, SameOperandsAndResultType]> {
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

//===----------------------------------------------------------------------===//
// Process Control: Fork/Join Operations
//===----------------------------------------------------------------------===//

def SimForkOp : SimOp<"fork", [
  RecursiveMemoryEffects,
]> {
  let summary = "Spawn concurrent processes (SystemVerilog fork)";
  let description = [{
    `sim.fork` spawns multiple concurrent processes that execute in parallel.
    This corresponds to the SystemVerilog `fork...join`, `fork...join_any`,
    and `fork...join_none` constructs.

    Each region within the fork represents a separate concurrent process.
    The `join_type` attribute specifies the synchronization behavior:
    - "join": Wait for all spawned processes to complete (default)
    - "join_any": Wait for any one process to complete
    - "join_none": Don't wait, continue immediately

    The fork operation returns a handle that can be used with `sim.wait_fork`
    or `sim.disable_fork` for later synchronization or control.

    Example:
    ```mlir
    // Fork with join (wait for all)
    %handle = sim.fork join_type "join" {
      // Process 1
      sim.delay 10 : i64
    }, {
      // Process 2
      sim.delay 20 : i64
    }, {
      // Process 3
      sim.delay 15 : i64
    }

    // Fork with join_any (wait for first)
    %handle2 = sim.fork join_type "join_any" {
      sim.delay 10 : i64
    }, {
      sim.delay 20 : i64
    }

    // Fork with join_none (continue immediately)
    %handle3 = sim.fork join_type "join_none" {
      sim.delay 100 : i64
    }
    ```
  }];
  let arguments = (ins
    DefaultValuedStrAttr<StrAttr, "join">:$joinType,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs I64:$handle);
  let regions = (region VariadicRegion<AnyRegion>:$branches);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SimForkTerminatorOp : SimOp<"fork.terminator", [
  Pure,
  Terminator,
  ParentOneOf<["SimForkOp"]>,
]> {
  let summary = "Terminator for fork branch regions";
  let description = [{
    `sim.fork.terminator` marks the end of a branch within a `sim.fork` operation.
    This is an implicit terminator that is automatically inserted.
  }];
  let assemblyFormat = "attr-dict";
}

def SimJoinOp : SimOp<"join", []> {
  let summary = "Wait for all processes in a fork to complete";
  let description = [{
    `sim.join` waits for all processes spawned by a fork to complete.
    This is used when the fork was created with `join_none` to later
    synchronize with the spawned processes.

    Example:
    ```mlir
    %handle = sim.fork join_type "join_none" {
      sim.delay 100 : i64
    }
    // ... do other work ...
    sim.join %handle  // Now wait for the fork to complete
    ```
  }];
  let arguments = (ins I64:$handle);
  let assemblyFormat = "$handle attr-dict";
}

def SimJoinAnyOp : SimOp<"join_any", []> {
  let summary = "Wait for any process in a fork to complete";
  let description = [{
    `sim.join_any` waits for any one of the processes spawned by a fork
    to complete. The remaining processes continue executing.

    Example:
    ```mlir
    %handle = sim.fork join_type "join_none" {
      sim.delay 100 : i64
    }, {
      sim.delay 50 : i64
    }
    sim.join_any %handle  // Waits until first completes (50 time units)
    ```
  }];
  let arguments = (ins I64:$handle);
  let assemblyFormat = "$handle attr-dict";
}

def SimJoinNoneOp : SimOp<"join_none", [Pure]> {
  let summary = "Continue without waiting (no-op marker)";
  let description = [{
    `sim.join_none` is a no-op that explicitly indicates the intent
    to not wait for forked processes. This is primarily used for
    clarity in code that was converted from SystemVerilog.

    Example:
    ```mlir
    %handle = sim.fork join_type "join_none" {
      sim.delay 100 : i64
    }
    sim.join_none %handle  // Explicit no-wait (no-op)
    ```
  }];
  let arguments = (ins I64:$handle);
  let assemblyFormat = "$handle attr-dict";
}

//===----------------------------------------------------------------------===//
// Process Control: Wait Operations
//===----------------------------------------------------------------------===//

def SimWaitOp : SimOp<"wait", []> {
  let summary = "Wait for a condition to become true";
  let description = [{
    `sim.wait` suspends the current process until the specified condition
    becomes true. The condition is evaluated each time any of the signals
    it depends on changes.

    This corresponds to the SystemVerilog `wait(condition)` statement.

    The optional `timeout` specifies a maximum time to wait in femtoseconds.
    If the timeout expires before the condition becomes true, the process
    resumes and the `timed_out` result is set to true.

    Example:
    ```mlir
    // Wait for condition to be true
    sim.wait %condition

    // Wait with timeout (returns true if timed out)
    %timed_out = sim.wait %condition timeout 1000000000 : i1
    ```
  }];
  let arguments = (ins
    I1:$condition,
    OptionalAttr<I64Attr>:$timeoutFemtoseconds
  );
  let results = (outs Optional<I1>:$timedOut);
  let assemblyFormat = [{
    $condition
    (`timeout` $timeoutFemtoseconds^)?
    attr-dict (`:` type($timedOut)^)?
  }];
  let hasVerifier = 1;
}

def SimWaitForkOp : SimOp<"wait_fork", []> {
  let summary = "Wait for all child processes to complete";
  let description = [{
    `sim.wait_fork` waits for all processes spawned by `fork...join_none`
    in the current scope to complete. This is equivalent to the
    SystemVerilog `wait fork` statement.

    Unlike `sim.join` which operates on a specific fork handle,
    `wait_fork` waits for ALL outstanding child processes.

    Example:
    ```mlir
    %h1 = sim.fork join_type "join_none" { sim.delay 10 : i64 }
    %h2 = sim.fork join_type "join_none" { sim.delay 20 : i64 }
    %h3 = sim.fork join_type "join_none" { sim.delay 15 : i64 }

    // Wait for all three forks to complete
    sim.wait_fork
    ```
  }];
  let assemblyFormat = "attr-dict";
}

def SimDelayOp : SimOp<"delay", []> {
  let summary = "Delay execution for a specified time";
  let description = [{
    `sim.delay` suspends the current process for the specified amount of
    simulation time. This corresponds to the SystemVerilog `#delay` construct.

    The delay is specified in femtoseconds for maximum precision.
    Common time unit helpers:
    - 1 ns = 1,000,000 fs
    - 1 us = 1,000,000,000 fs
    - 1 ms = 1,000,000,000,000 fs

    Example:
    ```mlir
    // Delay for 10 nanoseconds
    sim.delay 10000000 : i64

    // Delay for value read from variable
    sim.delay %time_var : i64
    ```
  }];
  let arguments = (ins AnyInteger:$delay);
  let assemblyFormat = "$delay attr-dict `:` type($delay)";
}

//===----------------------------------------------------------------------===//
// Process Control: Disable Operations
//===----------------------------------------------------------------------===//

def SimDisableOp : SimOp<"disable", []> {
  let summary = "Disable a named block or task";
  let description = [{
    `sim.disable` terminates execution of a named block or task.
    This corresponds to the SystemVerilog `disable` statement.

    When a block/task is disabled:
    - Execution jumps to the statement following the disabled block
    - All nested blocks within the disabled block are also terminated
    - Variables retain their values at the time of disable

    The target must be the name of a `sim.named_block` block.

    Example:
    ```mlir
    sim.named_block "outer_loop" {
      sim.named_block "inner_loop" {
        // ... some condition ...
        sim.disable "outer_loop"  // Exit both loops
      }
    }
    ```
  }];
  let arguments = (ins StrAttr:$target);
  let assemblyFormat = "$target attr-dict";
}

def SimDisableForkOp : SimOp<"disable_fork", []> {
  let summary = "Disable all child processes";
  let description = [{
    `sim.disable_fork` terminates all processes spawned by `fork...join_none`
    in the current scope. This corresponds to the SystemVerilog `disable fork`
    statement.

    All child processes (and their descendants) are immediately terminated.

    Example:
    ```mlir
    %h1 = sim.fork join_type "join_none" { sim.delay 100 : i64 }
    %h2 = sim.fork join_type "join_none" { sim.delay 200 : i64 }

    // Terminate all child processes
    sim.disable_fork
    ```
  }];
  let assemblyFormat = "attr-dict";
}

def SimNamedBlockOp : SimOp<"named_block", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"SimNamedBlockTerminatorOp">,
]> {
  let summary = "A named block that can be disabled";
  let description = [{
    `sim.named_block` defines a named scope that can be targeted by
    `sim.disable`. This corresponds to named begin...end blocks in
    SystemVerilog.

    Named blocks can be nested, and disabling an outer block also
    terminates all nested blocks.

    Example:
    ```mlir
    sim.named_block "search_loop" {
      // ... loop body ...
      scf.if %found {
        sim.disable "search_loop"
      }
    }
    ```
  }];
  let arguments = (ins StrAttr:$blockName);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = "$blockName attr-dict-with-keyword $body";
}

def SimNamedBlockTerminatorOp : SimOp<"named_block.terminator", [
  Pure,
  Terminator,
  ParentOneOf<["SimNamedBlockOp"]>,
]> {
  let summary = "Terminator for named block regions";
  let description = [{
    `sim.named_block.terminator` marks the end of a `sim.named_block`.
    This is an implicit terminator.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Synchronization Primitives: Semaphores
//===----------------------------------------------------------------------===//

def SimSemaphoreCreateOp : SimOp<"semaphore.create", []> {
  let summary = "Create a semaphore with initial count";
  let description = [{
    `sim.semaphore.create` creates a counting semaphore with the specified
    initial key count. Semaphores are used to control access to shared
    resources and for inter-process synchronization.

    This corresponds to SystemVerilog's `semaphore` class constructor:
    `semaphore s = new(keyCount)`

    The returned handle is used with `sim.semaphore.get`, `sim.semaphore.put`,
    and `sim.semaphore.try_get`.

    Example:
    ```mlir
    // Create a binary semaphore (mutex)
    %mutex = sim.semaphore.create 1 : i64

    // Create a counting semaphore with 4 keys
    %sem = sim.semaphore.create 4 : i64
    ```
  }];
  let arguments = (ins AnyInteger:$keyCount);
  let results = (outs I64:$handle);
  let assemblyFormat = "$keyCount attr-dict `:` type($keyCount)";
}

def SimSemaphoreGetOp : SimOp<"semaphore.get", []> {
  let summary = "Get keys from a semaphore (blocking)";
  let description = [{
    `sim.semaphore.get` obtains the specified number of keys from a semaphore.
    If insufficient keys are available, the calling process blocks until
    enough keys become available.

    This corresponds to SystemVerilog's `semaphore.get(keyCount)` method.

    The default key count is 1 if not specified.

    Example:
    ```mlir
    // Get one key (blocking)
    sim.semaphore.get %sem

    // Get 3 keys (blocking)
    sim.semaphore.get %sem, 3 : i32
    ```
  }];
  let arguments = (ins
    I64:$handle,
    Optional<AnyInteger>:$keyCount
  );
  let assemblyFormat = [{
    $handle (`,` $keyCount^ `:` type($keyCount))? attr-dict
  }];
}

def SimSemaphoreTryGetOp : SimOp<"semaphore.try_get", []> {
  let summary = "Try to get keys from a semaphore (non-blocking)";
  let description = [{
    `sim.semaphore.try_get` attempts to obtain the specified number of keys
    from a semaphore without blocking. Returns true if successful, false
    if insufficient keys are available.

    This corresponds to SystemVerilog's `semaphore.try_get(keyCount)` method.

    Example:
    ```mlir
    // Try to get one key
    %success = sim.semaphore.try_get %sem : i1

    // Try to get 3 keys
    %success = sim.semaphore.try_get %sem, 3 : i32 -> i1
    ```
  }];
  let arguments = (ins
    I64:$handle,
    Optional<AnyInteger>:$keyCount
  );
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $handle (`,` $keyCount^ `:` type($keyCount))? attr-dict `->` type($success)
  }];
}

def SimSemaphorePutOp : SimOp<"semaphore.put", []> {
  let summary = "Return keys to a semaphore";
  let description = [{
    `sim.semaphore.put` returns the specified number of keys to a semaphore.
    This may unblock processes waiting on `sim.semaphore.get`.

    This corresponds to SystemVerilog's `semaphore.put(keyCount)` method.

    The default key count is 1 if not specified.

    Example:
    ```mlir
    // Return one key
    sim.semaphore.put %sem

    // Return 3 keys
    sim.semaphore.put %sem, 3 : i32
    ```
  }];
  let arguments = (ins
    I64:$handle,
    Optional<AnyInteger>:$keyCount
  );
  let assemblyFormat = [{
    $handle (`,` $keyCount^ `:` type($keyCount))? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Primitives: Mailboxes
//===----------------------------------------------------------------------===//

def SimMailboxCreateOp : SimOp<"mailbox.create", []> {
  let summary = "Create a mailbox for inter-process communication";
  let description = [{
    `sim.mailbox.create` creates a mailbox that can be used for passing
    messages between processes. This corresponds to SystemVerilog's
    `mailbox` class.

    The optional `bound` parameter limits the mailbox size. A bounded
    mailbox will block on `put` when full. Unbounded mailboxes (bound=0)
    can grow without limit.

    Example:
    ```mlir
    // Create unbounded mailbox
    %mbox = sim.mailbox.create : i64

    // Create bounded mailbox (max 16 entries)
    %bounded_mbox = sim.mailbox.create bound 16 : i64
    ```
  }];
  let arguments = (ins
    OptionalAttr<I32Attr>:$bound
  );
  let results = (outs I64:$handle);
  let assemblyFormat = [{
    (`bound` $bound^)? attr-dict `:` type($handle)
  }];
}

def SimMailboxPutOp : SimOp<"mailbox.put", []> {
  let summary = "Put a message into a mailbox (blocking)";
  let description = [{
    `sim.mailbox.put` places a message into a mailbox. For bounded mailboxes,
    this operation blocks if the mailbox is full.

    This corresponds to SystemVerilog's `mailbox.put(message)` method.

    The message can be any type that the mailbox was parameterized with.

    Example:
    ```mlir
    sim.mailbox.put %mbox, %data : i32
    ```
  }];
  let arguments = (ins
    I64:$handle,
    AnyType:$message
  );
  let assemblyFormat = "$handle `,` $message attr-dict `:` type($message)";
}

def SimMailboxTryPutOp : SimOp<"mailbox.try_put", []> {
  let summary = "Try to put a message into a mailbox (non-blocking)";
  let description = [{
    `sim.mailbox.try_put` attempts to place a message into a mailbox without
    blocking. Returns true if successful, false if the mailbox is full.

    This corresponds to SystemVerilog's `mailbox.try_put(message)` method.

    Example:
    ```mlir
    %success = sim.mailbox.try_put %mbox, %data : i32 -> i1
    ```
  }];
  let arguments = (ins
    I64:$handle,
    AnyType:$message
  );
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $handle `,` $message attr-dict `:` type($message) `->` type($success)
  }];
}

def SimMailboxGetOp : SimOp<"mailbox.get", []> {
  let summary = "Get a message from a mailbox (blocking)";
  let description = [{
    `sim.mailbox.get` retrieves and removes a message from a mailbox.
    If the mailbox is empty, the operation blocks until a message
    is available.

    This corresponds to SystemVerilog's `mailbox.get(message)` method.

    Example:
    ```mlir
    %data = sim.mailbox.get %mbox : i32
    ```
  }];
  let arguments = (ins I64:$handle);
  let results = (outs AnyType:$message);
  let assemblyFormat = "$handle attr-dict `:` type($message)";
}

def SimMailboxTryGetOp : SimOp<"mailbox.try_get", []> {
  let summary = "Try to get a message from a mailbox (non-blocking)";
  let description = [{
    `sim.mailbox.try_get` attempts to retrieve a message from a mailbox
    without blocking. Returns true and the message if available, false
    and undefined data if empty.

    This corresponds to SystemVerilog's `mailbox.try_get(message)` method.

    Example:
    ```mlir
    %success, %data = sim.mailbox.try_get %mbox : i1, i32
    ```
  }];
  let arguments = (ins I64:$handle);
  let results = (outs I1:$success, AnyType:$message);
  let assemblyFormat = "$handle attr-dict `:` type($success) `,` type($message)";
}

def SimMailboxPeekOp : SimOp<"mailbox.peek", []> {
  let summary = "Peek at a message without removing it (blocking)";
  let description = [{
    `sim.mailbox.peek` retrieves a copy of the message at the front of
    the mailbox without removing it. If the mailbox is empty, the
    operation blocks until a message is available.

    This corresponds to SystemVerilog's `mailbox.peek(message)` method.

    Example:
    ```mlir
    %data = sim.mailbox.peek %mbox : i32
    ```
  }];
  let arguments = (ins I64:$handle);
  let results = (outs AnyType:$message);
  let assemblyFormat = "$handle attr-dict `:` type($message)";
}

def SimMailboxTryPeekOp : SimOp<"mailbox.try_peek", []> {
  let summary = "Try to peek at a message without blocking";
  let description = [{
    `sim.mailbox.try_peek` attempts to retrieve a copy of the message
    at the front of the mailbox without removing it or blocking.

    This corresponds to SystemVerilog's `mailbox.try_peek(message)` method.

    Example:
    ```mlir
    %success, %data = sim.mailbox.try_peek %mbox : i1, i32
    ```
  }];
  let arguments = (ins I64:$handle);
  let results = (outs I1:$success, AnyType:$message);
  let assemblyFormat = "$handle attr-dict `:` type($success) `,` type($message)";
}

def SimMailboxNumOp : SimOp<"mailbox.num", [Pure]> {
  let summary = "Get the number of messages in a mailbox";
  let description = [{
    `sim.mailbox.num` returns the current number of messages in the mailbox.

    This corresponds to SystemVerilog's `mailbox.num()` method.

    Example:
    ```mlir
    %count = sim.mailbox.num %mbox : i32
    ```
  }];
  let arguments = (ins I64:$handle);
  let results = (outs AnyInteger:$count);
  let assemblyFormat = "$handle attr-dict `:` type($count)";
}

//===----------------------------------------------------------------------===//
// Synchronization Primitives: Events
//===----------------------------------------------------------------------===//

def SimEventCreateOp : SimOp<"event.create", []> {
  let summary = "Create a named event for synchronization";
  let description = [{
    `sim.event.create` creates a named event that can be used for
    process synchronization. This corresponds to SystemVerilog's
    `event` type.

    Events can be triggered with `sim.event.trigger` and waited
    upon with `sim.event.wait`.

    Example:
    ```mlir
    %evt = sim.event.create : i64
    ```
  }];
  let results = (outs I64:$handle);
  let assemblyFormat = "attr-dict `:` type($handle)";
}

def SimEventTriggerOp : SimOp<"event.trigger", []> {
  let summary = "Trigger an event";
  let description = [{
    `sim.event.trigger` triggers the specified event, unblocking any
    processes waiting on it. This corresponds to SystemVerilog's
    `->event` or `->>(nonblocking)event` trigger.

    The `nonblocking` attribute makes this a non-blocking trigger
    (scheduled for the NBA region).

    Example:
    ```mlir
    // Blocking trigger
    sim.event.trigger %evt

    // Non-blocking trigger
    sim.event.trigger %evt nonblocking
    ```
  }];
  let arguments = (ins
    I64:$handle,
    UnitAttr:$nonblocking
  );
  let assemblyFormat = "$handle (`nonblocking` $nonblocking^)? attr-dict";
}

def SimEventWaitOp : SimOp<"event.wait", []> {
  let summary = "Wait for an event to be triggered";
  let description = [{
    `sim.event.wait` suspends the current process until the specified
    event is triggered. This corresponds to SystemVerilog's `@(event)`
    or `wait(event.triggered)`.

    Example:
    ```mlir
    sim.event.wait %evt
    ```
  }];
  let arguments = (ins I64:$handle);
  let assemblyFormat = "$handle attr-dict";
}

#endif // CIRCT_DIALECT_SIM_SIMOPS_TD
