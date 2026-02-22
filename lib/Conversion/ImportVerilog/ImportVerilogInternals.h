//===- ImportVerilogInternals.h - Internal implementation details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
#define CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "slang/ast/ASTVisitor.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <queue>

#define DEBUG_TYPE "import-verilog"

namespace slang {
namespace ast {
class LocalAssertionVarSymbol;
class AssertionExpr;
class AssertionPortSymbol;
class Expression;
class Pattern;
class InstanceSymbol;
class HierarchicalReference;
class TimingControl;
enum class CaseStatementCondition;
} // namespace ast
} // namespace slang

namespace circt {
namespace ImportVerilog {

using moore::Domain;

/// Port lowering information.
struct PortLowering {
  const slang::ast::Symbol *symbol;
  Location loc;
  BlockArgument arg;
};

/// Module lowering information.
struct ModuleLowering {
  moore::SVModuleOp op;
  SmallVector<PortLowering> ports;
  DenseMap<const slang::syntax::SyntaxNode *, const slang::ast::Symbol *>
      portsBySyntaxNode;
};

/// Function lowering information.
struct FunctionLowering {
  mlir::func::FuncOp op;
  llvm::SmallVector<Value, 4> captures;
  llvm::DenseMap<Value, unsigned> captureIndex;
  bool capturesFinalized = false;
  bool isConverting = false;
};

// Class lowering information.
struct ClassLowering {
  circt::moore::ClassDeclOp op;
  bool bodyConverted = false;
};

// Interface lowering information.
struct InterfaceLowering {
  circt::moore::InterfaceDeclOp op;
  bool bodyConverted = false;
};

struct PendingInterfacePortConnection {
  const slang::ast::InstanceSymbol *instSym;
  Value instRef;
  Location loc;
};

// Covergroup lowering information.
struct CovergroupLowering {
  circt::moore::CovergroupDeclOp op;
};

/// Information about a loops continuation and exit blocks relevant while
/// lowering the loop's body statements.
struct LoopFrame {
  /// The block to jump to from a `continue` statement.
  Block *continueBlock;
  /// The block to jump to from a `break` statement.
  Block *breakBlock;
};

/// Information about a disable target relevant while lowering nested
/// statements.
struct DisableFrame {
  /// The named block symbol that may be targeted by `disable <name>`.
  const slang::ast::Symbol *symbol;
  /// The block to jump to when this target is disabled.
  Block *targetBlock;
};

/// Hierarchical path information.
/// The "hierName" means a different hierarchical name at different module
/// levels.
/// The "idx" means where the current hierarchical name is on the portlists.
/// The "direction" means hierarchical names whether downward(In) or
/// upward(Out).
struct HierPathInfo {
  mlir::StringAttr hierName;
  std::optional<unsigned int> idx;
  slang::ast::ArgumentDirection direction;
  const slang::ast::ValueSymbol *valueSym;
};

/// Hierarchical path information for interface instances.
/// These paths thread a specific interface instance through module ports so
/// hierarchical interface member references can be resolved.
struct HierInterfacePathInfo {
  mlir::StringAttr hierName;
  std::optional<unsigned int> idx;
  slang::ast::ArgumentDirection direction;
  const slang::ast::InstanceSymbol *ifaceInst;
};

/// Information about interface ports needed from bind scopes.
/// When a bound instance references an interface port from its bind scope,
/// that interface needs to be threaded through the target module's ports.
struct BindScopeInterfacePortInfo {
  /// The interface port symbol from the bind scope.
  const slang::ast::InterfacePortSymbol *ifacePort;
  /// The index in the target module's port list (set during header conversion).
  std::optional<unsigned int> idx;
};

struct AssertionLocalVarBinding {
  Value value;
  uint64_t offset = 0;
};

struct AssertionPortBinding {
  enum class Kind { Expr, AssertionExpr, TimingControl };
  Kind kind = Kind::Expr;
  const slang::ast::Expression *expr = nullptr;
  const slang::ast::AssertionExpr *assertionExpr = nullptr;
  const slang::ast::TimingControl *timingControl = nullptr;
};

/// A helper class to facilitate the conversion from a Slang AST to MLIR
/// operations. Keeps track of the destination MLIR module, builders, and
/// various worklists and utilities needed for conversion.
struct Context {
  Context(const ImportVerilogOptions &options,
          slang::ast::Compilation &compilation, mlir::ModuleOp intoModuleOp,
          const slang::SourceManager &sourceManager)
      : options(options), compilation(compilation), intoModuleOp(intoModuleOp),
        sourceManager(sourceManager),
        builder(OpBuilder::atBlockEnd(intoModuleOp.getBody())),
        symbolTable(intoModuleOp) {}
  Context(const Context &) = delete;

  /// Return the MLIR context.
  MLIRContext *getContext() { return intoModuleOp.getContext(); }

  /// Convert a slang `SourceLocation` into an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc);
  /// Convert a slang `SourceRange` into an MLIR `Location`.
  Location convertLocation(slang::SourceRange range);

  /// Convert a slang type into an MLIR type. Returns null on failure. Uses the
  /// provided location for error reporting, or tries to guess one from the
  /// given type. Types tend to have unreliable location information, so it's
  /// generally a good idea to pass in a location.
  Type convertType(const slang::ast::Type &type, LocationAttr loc = {});
  Type convertType(const slang::ast::DeclaredType &type);

  /// Convert hierarchy and structure AST nodes to MLIR ops.
  LogicalResult convertCompilation();
  ModuleLowering *
  convertModuleHeader(const slang::ast::InstanceBodySymbol *module);
  LogicalResult convertModuleBody(const slang::ast::InstanceBodySymbol *module);
  Value getOrCreateHierarchicalPlaceholder(const slang::ast::ValueSymbol *sym,
                                           Location loc);
  LogicalResult convertPackage(const slang::ast::PackageSymbol &package);
  FunctionLowering *
  declareFunction(const slang::ast::SubroutineSymbol &subroutine);
  LogicalResult convertFunction(const slang::ast::SubroutineSymbol &subroutine);
  LogicalResult finalizeFunctionBodyCaptures(FunctionLowering &lowering);
  LogicalResult convertClassDeclaration(const slang::ast::ClassType &classdecl);
  ClassLowering *declareClass(const slang::ast::ClassType &cls);
  LogicalResult convertGlobalVariable(const slang::ast::VariableSymbol &var);
  LogicalResult
  convertStaticClassProperty(const slang::ast::ClassPropertySymbol &prop);
  void captureRef(Value ref);

  /// Convert interface declarations
  InterfaceLowering *
  convertInterfaceHeader(const slang::ast::InstanceBodySymbol *iface);
  LogicalResult convertInterfaceBody(const slang::ast::InstanceBodySymbol *iface);
  Value resolveInterfaceInstance(const slang::ast::InstanceSymbol *instSym,
                                 Location loc);
  Value resolveInterfaceInstance(const slang::ast::HierarchicalReference &ref,
                                 Location loc);

  /// Convert covergroup declarations
  LogicalResult convertCovergroup(const slang::ast::CovergroupType &covergroup);

  /// Checks whether one class (actualTy) is derived from another class
  /// (baseTy). True if it's a subclass, false otherwise.
  bool isClassDerivedFrom(const moore::ClassHandleType &actualTy,
                          const moore::ClassHandleType &baseTy);

  /// Tries to find the closest base class of actualTy that carries a property
  /// with name fieldName. The location is used for error reporting.
  moore::ClassHandleType
  getAncestorClassWithProperty(const moore::ClassHandleType &actualTy,
                               StringRef fieldName, Location loc);

  Value getImplicitThisRef() const {
    return currentThisRef; // block arg added in declareFunction
  }
  void setImplicitThisRef(Value value) { currentThisRef = value; }
  Value getInlineConstraintThisRef() const { return inlineConstraintThisRef; }
  void setInlineConstraintThisRef(Value value) {
    inlineConstraintThisRef = value;
  }
  const slang::ast::Symbol *getInlineConstraintThisSymbol() const {
    return inlineConstraintThisSymbol;
  }
  void setInlineConstraintThisSymbol(const slang::ast::Symbol *symbol) {
    inlineConstraintThisSymbol = symbol;
  }
  // Convert a statement AST node to MLIR ops.
  LogicalResult convertStatement(const slang::ast::Statement &stmt);

  // Convert an expression AST node to MLIR ops.
  Value convertRvalueExpression(const slang::ast::Expression &expr,
                                Type requiredType = {});
  Value convertLvalueExpression(const slang::ast::Expression &expr);

  // Synthesize a struct initial value from field defaults (IEEE 1800-2017
  // §7.2.1). Returns null if the type is not a struct or has no field defaults.
  Value synthesizeStructFieldDefaults(const slang::ast::Type &slangType,
                                      Type mooreType, Location loc);

  // Match a pattern against a value and return the match result (boolean).
  FailureOr<Value> matchPattern(const slang::ast::Pattern &pattern, Value value,
                                const slang::ast::Type &targetType,
                                slang::ast::CaseStatementCondition condKind,
                                Location loc);

  // Convert an assertion expression AST node to MLIR ops.
  Value convertAssertionExpression(const slang::ast::AssertionExpr &expr,
                                   Location loc, bool applyDefaults = true);

  // Convert an assertion expression AST node to MLIR ops.
  Value convertAssertionCallExpression(
      const slang::ast::CallExpression &expr,
      const slang::ast::CallExpression::SystemCallInfo &info, Location loc);

  void pushAssertionLocalVarScope() { assertionLocalVarScopes.emplace_back(); }
  void popAssertionLocalVarScope() { assertionLocalVarScopes.pop_back(); }
  AssertionLocalVarBinding *
  lookupAssertionLocalVarBinding(const slang::ast::LocalAssertionVarSymbol *sym) {
    for (auto it = assertionLocalVarScopes.rbegin();
         it != assertionLocalVarScopes.rend(); ++it) {
      auto entry = it->find(sym);
      if (entry != it->end())
        return &entry->second;
    }
    return nullptr;
  }
  const AssertionLocalVarBinding *lookupAssertionLocalVarBinding(
      const slang::ast::LocalAssertionVarSymbol *sym) const {
    for (auto it = assertionLocalVarScopes.rbegin();
         it != assertionLocalVarScopes.rend(); ++it) {
      auto entry = it->find(sym);
      if (entry != it->end())
        return &entry->second;
    }
    return nullptr;
  }
  void setAssertionLocalVarBinding(
      const slang::ast::LocalAssertionVarSymbol *sym, Value value,
      uint64_t offset) {
    if (assertionLocalVarScopes.empty())
      return;
    assertionLocalVarScopes.back()[sym] = {value, offset};
  }
  Value getPendingAssertionLocalVarLvalue(
      const slang::ast::LocalAssertionVarSymbol *sym) const {
    auto it = pendingAssertionLocalVarLvalues.find(sym);
    if (it == pendingAssertionLocalVarLvalues.end())
      return {};
    return it->second;
  }
  void setPendingAssertionLocalVarLvalue(
      const slang::ast::LocalAssertionVarSymbol *sym, Value ref) {
    pendingAssertionLocalVarLvalues[sym] = ref;
  }
  void clearPendingAssertionLocalVarLvalues() {
    pendingAssertionLocalVarLvalues.clear();
  }
  LogicalResult flushPendingAssertionLocalVarLvalues(Location loc) {
    for (auto &[sym, ref] : pendingAssertionLocalVarLvalues) {
      auto refTy = dyn_cast<moore::RefType>(ref.getType());
      if (!refTy) {
        mlir::emitError(loc, "invalid local assertion variable lvalue type")
            << ref.getType();
        return failure();
      }
      auto value = moore::ReadOp::create(builder, loc, ref).getResult();
      setAssertionLocalVarBinding(sym, value, getAssertionSequenceOffset());
    }
    pendingAssertionLocalVarLvalues.clear();
    return success();
  }
  void pushAssertionPortScope() { assertionPortScopes.emplace_back(); }
  void popAssertionPortScope() { assertionPortScopes.pop_back(); }
  const AssertionPortBinding *lookupAssertionPortBinding(
      const slang::ast::AssertionPortSymbol *sym) const {
    for (auto it = assertionPortScopes.rbegin(); it != assertionPortScopes.rend();
         ++it) {
      auto entry = it->find(sym);
      if (entry != it->end())
        return &entry->second;
    }
    return nullptr;
  }
  void setAssertionPortBinding(const slang::ast::AssertionPortSymbol *sym,
                               const AssertionPortBinding &binding) {
    if (assertionPortScopes.empty())
      return;
    assertionPortScopes.back()[sym] = binding;
  }
  void pushAssertionSequenceOffset(uint64_t offset) {
    assertionSequenceOffsetStack.push_back(offset);
  }
  void popAssertionSequenceOffset() { assertionSequenceOffsetStack.pop_back(); }
  uint64_t getAssertionSequenceOffset() const {
    if (assertionSequenceOffsetStack.empty())
      return 0;
    return assertionSequenceOffsetStack.back();
  }
  void setAssertionSequenceOffset(uint64_t offset) {
    if (assertionSequenceOffsetStack.empty())
      return;
    assertionSequenceOffsetStack.back() = offset;
  }
  void pushAssertionDisableExpr(const slang::ast::Expression *expr) {
    if (expr)
      assertionDisableExprStack.push_back(expr);
  }
  void popAssertionDisableExpr() {
    if (!assertionDisableExprStack.empty())
      assertionDisableExprStack.pop_back();
  }
  std::span<const slang::ast::Expression *const>
  getAssertionDisableExprs() const {
    return assertionDisableExprStack;
  }

  // Traverse the whole AST to collect hierarchical names.
  LogicalResult
  collectHierarchicalValues(const slang::ast::Expression &expr,
                            const slang::ast::Symbol &outermostModule,
                            const slang::ast::Scope *bindScope = nullptr);
  LogicalResult
  collectHierarchicalValuesFromStatement(const slang::ast::Statement &stmt,
                                         const slang::ast::Symbol &outermostModule);
  LogicalResult traverseInstanceBody(const slang::ast::Symbol &symbol);

  // Convert timing controls into a corresponding set of ops that delay
  // execution of the current block. Produces an error if the implicit event
  // control `@*` or `@(*)` is used.
  LogicalResult convertTimingControl(const slang::ast::TimingControl &ctrl);
  // Convert timing controls into a corresponding set of ops that delay
  // execution of the current block. Then converts the given statement, taking
  // note of the rvalues it reads and adding them to a wait op in case an
  // implicit event control `@*` or `@(*)` is used.
  LogicalResult convertTimingControl(const slang::ast::TimingControl &ctrl,
                                     const slang::ast::Statement &stmt);

  /// Helper function to convert a value to a MLIR I1 value.
  Value convertToI1(Value value);

  // Convert a slang timing control for LTL
  Value convertLTLTimingControl(const slang::ast::TimingControl &ctrl,
                                const Value &seqOrPro);

  /// Helper function to convert a value to its "truthy" boolean value.
  Value convertToBool(Value value);

  /// Helper function to convert a value to its "truthy" boolean value and
  /// convert it to the given domain.
  Value convertToBool(Value value, Domain domain);

  /// Helper function to convert a value to its simple bit vector
  /// representation, if it has one. Otherwise returns null. Also returns null
  /// if the given value is null.
  Value convertToSimpleBitVector(Value value);

  /// Helper function to insert the necessary operations to cast a value from
  /// one type to another.
  Value materializeConversion(Type type, Value value, bool isSigned,
                              Location loc);

  /// Helper function to materialize an `SVInt` as an SSA value.
  Value materializeSVInt(const slang::SVInt &svint,
                         const slang::ast::Type &type, Location loc);

  /// Helper function to materialize a real value as an SSA value.
  Value materializeSVReal(const slang::ConstantValue &svreal,
                          const slang::ast::Type &type, Location loc);

  /// Helper function to materialize a string as an SSA value.
  Value materializeString(const slang::ConstantValue &string,
                          const slang::ast::Type &astType, Location loc);

  /// Helper function to materialize an unpacked array of `SVInt`s as an SSA
  /// value.
  Value materializeFixedSizeUnpackedArrayType(
      const slang::ConstantValue &constant,
      const slang::ast::FixedSizeUnpackedArrayType &astType, Location loc);

  /// Helper function to materialize a `ConstantValue` as an SSA value. Returns
  /// null if the constant cannot be materialized.
  Value materializeConstant(const slang::ConstantValue &constant,
                            const slang::ast::Type &type, Location loc);

  /// Convert a list of string literal arguments with formatting specifiers and
  /// arguments to be interpolated into a `!moore.format_string` value. Returns
  /// failure if an error occurs. Returns a null value if the formatted string
  /// is trivially empty. Otherwise returns the formatted string.
  /// The optional scope parameter is used for %m format specifier to determine
  /// the hierarchical path.
  FailureOr<Value> convertFormatString(
      std::span<const slang::ast::Expression *const> arguments, Location loc,
      moore::IntFormat defaultFormat = moore::IntFormat::Decimal,
      bool appendNewline = false,
      const slang::ast::Scope *scope = nullptr);

  /// Convert system function calls only have arity-0.
  FailureOr<Value>
  convertSystemCallArity0(const slang::ast::SystemSubroutine &subroutine,
                          Location loc);

  /// Convert system function calls only have arity-1.
  FailureOr<Value>
  convertSystemCallArity1(const slang::ast::SystemSubroutine &subroutine,
                          Location loc, Value value);

  /// Convert system function calls with arity-2.
  FailureOr<Value>
  convertSystemCallArity2(const slang::ast::SystemSubroutine &subroutine,
                          Location loc, Value value1, Value value2);

  /// Convert system function calls with arity-3.
  FailureOr<Value>
  convertSystemCallArity3(const slang::ast::SystemSubroutine &subroutine,
                          Location loc, Value value1, Value value2,
                          Value value3);

  /// Convert queue method calls with one argument (e.g., push_back, push_front).
  FailureOr<Value>
  convertQueueMethodCall(const slang::ast::SystemSubroutine &subroutine,
                         Location loc, Value queueRef, Value element);

  /// Convert queue method calls with no arguments (e.g., pop_back, pop_front).
  FailureOr<Value>
  convertQueueMethodCallNoArg(const slang::ast::SystemSubroutine &subroutine,
                              Location loc, Value queueRef, Type elementType);

  /// Convert array/queue void method calls (e.g., delete, sort).
  FailureOr<Value>
  convertArrayVoidMethodCall(const slang::ast::SystemSubroutine &subroutine,
                             Location loc, Value arrayRef);

  /// Convert system function calls within properties and assertion with a
  /// single argument.
  FailureOr<Value> convertAssertionSystemCallArity1(
      const slang::ast::SystemSubroutine &subroutine, Location loc,
      Value value);

  /// Evaluate the constant value of an expression.
  slang::ConstantValue evaluateConstant(const slang::ast::Expression &expr);

  /// Convert a slang constraint (from inline or block constraints) to MLIR ops.
  /// This handles all constraint kinds: expression, implication, conditional,
  /// uniqueness, foreach, solve-before, etc.
  LogicalResult convertConstraint(const slang::ast::Constraint &constraint,
                                  Location loc);

  const ImportVerilogOptions &options;
  slang::ast::Compilation &compilation;
  mlir::ModuleOp intoModuleOp;
  const slang::SourceManager &sourceManager;

  /// The builder used to create IR operations.
  OpBuilder builder;
  /// A symbol table of the MLIR module we are emitting into.
  SymbolTable symbolTable;

  /// The top-level operations ordered by their Slang source location. This is
  /// used to produce IR that follows the source file order.
  std::map<slang::SourceLocation, Operation *> orderedRootOps;

  /// How we have lowered modules to MLIR.
  DenseMap<const slang::ast::InstanceBodySymbol *,
           std::unique_ptr<ModuleLowering>>
      modules;
  /// A list of modules for which the header has been created, but the body has
  /// not been converted yet.
  std::queue<const slang::ast::InstanceBodySymbol *> moduleWorklist;

  /// Functions that have already been converted.
  DenseMap<const slang::ast::SubroutineSymbol *,
           std::unique_ptr<FunctionLowering>>
      functions;

  /// Classes that have already been converted.
  DenseMap<const slang::ast::ClassType *, std::unique_ptr<ClassLowering>>
      classes;

  /// Map from specialized class symbol to generic class symbol.
  /// Used to track which class specializations came from the same generic
  /// class template (e.g., uvm_pool_18 -> uvm_pool).
  DenseMap<mlir::StringAttr, mlir::StringAttr> classSpecializationToGeneric;

  /// Interfaces that have already been converted, indexed by instance body.
  /// Multiple instance bodies may map to the same InterfaceLowering if they
  /// share the same DefinitionSymbol.
  DenseMap<const slang::ast::InstanceBodySymbol *,
           std::unique_ptr<InterfaceLowering>>
      interfaces;
  /// Interfaces indexed by their definition symbol. This is used to deduplicate
  /// interface declarations when multiple virtual interface variables reference
  /// the same interface definition.
  DenseMap<const slang::ast::DefinitionSymbol *,
           SmallVector<std::pair<const slang::ast::InstanceBodySymbol *,
                                 InterfaceLowering *>>>
      interfacesByDefinition;
  /// A list of interfaces for which the header has been created, but the body
  /// has not been converted yet.
  std::queue<const slang::ast::InstanceBodySymbol *> interfaceWorklist;

  /// Counter used to generate unique names for synthesized opaque interface
  /// declarations when generic interface ports are allowed at top level.
  uint64_t synthesizedGenericInterfaceCount = 0;

  /// Covergroups that have already been converted.
  DenseMap<const slang::ast::CovergroupType *,
           std::unique_ptr<CovergroupLowering>>
      covergroups;

  /// A table of defined values, such as variables, that may be referred to by
  /// name in expressions. The expressions use this table to lookup the MLIR
  /// value that was created for a given declaration in the Slang AST node.
  using ValueSymbols =
      llvm::ScopedHashTable<const slang::ast::ValueSymbol *, Value>;
  using ValueSymbolScope = ValueSymbols::ScopeTy;
  ValueSymbols valueSymbols;

  /// A table mapping iterator variables to their index values for use with
  /// the `item.index` property in array locator methods.
  /// This is populated when converting array locator predicates.
  using IteratorIndexSymbols =
      llvm::ScopedHashTable<const slang::ast::ValueSymbol *, Value>;
  using IteratorIndexSymbolScope = IteratorIndexSymbols::ScopeTy;
  IteratorIndexSymbols iteratorIndexSymbols;

  /// The current iterator index value for use with item.index in array
  /// locator predicates. This is set when entering an array locator region.
  Value currentIteratorIndex;

  /// A table of defined global variables that may be referred to by name in
  /// expressions.
  DenseMap<const slang::ast::ValueSymbol *, moore::GlobalVariableOp>
      globalVariables;
  /// Global variables indexed by their fully qualified symbol name for
  /// deduplication across parameterized class specializations.
  DenseMap<mlir::StringAttr, moore::GlobalVariableOp> globalVariablesByName;
  /// Synthetic global that tracks runtime enablement of procedural immediate
  /// assertions controlled by $asserton/$assertoff/$assertcontrol.
  moore::GlobalVariableOp proceduralAssertionsEnabledGlobal;
  /// Synthetic global that tracks whether assertion fail messages are displayed.
  /// Controlled by $assertfailoff/$assertfailon.
  moore::GlobalVariableOp assertionFailMessagesEnabledGlobal;
  /// Synthetic global that tracks whether assertion pass messages are
  /// displayed. Controlled by $assertpassoff/$assertpasson.
  moore::GlobalVariableOp assertionPassMessagesEnabledGlobal;
  /// Synthetic global that tracks whether vacuous assertion passes are enabled.
  /// Controlled by $assertvacuousoff/$assertnonvacuouson.
  moore::GlobalVariableOp assertionVacuousPassEnabledGlobal;
  /// A set of static class properties that are currently being converted.
  /// This is used to detect and handle recursive conversions when a property's
  /// type conversion triggers conversion of classes whose methods reference
  /// the property.
  DenseSet<const slang::ast::ValueSymbol *> staticPropertyInProgress;
  SmallVector<DenseMap<const slang::ast::LocalAssertionVarSymbol *,
                       AssertionLocalVarBinding>,
              2>
      assertionLocalVarScopes;
  SmallVector<DenseMap<const slang::ast::AssertionPortSymbol *,
                       AssertionPortBinding>,
              2>
      assertionPortScopes;
  DenseMap<const slang::ast::LocalAssertionVarSymbol *, Value>
      pendingAssertionLocalVarLvalues;
  SmallVector<uint64_t, 4> assertionSequenceOffsetStack;
  SmallVector<const slang::ast::Expression *, 2> assertionDisableExprStack;
  /// A list of global variables that still need their initializers to be
  /// converted.
  SmallVector<const slang::ast::ValueSymbol *> globalVariableWorklist;

  /// Placeholders for hierarchical references that are resolved after instance
  /// creation.
  DenseMap<const slang::ast::ValueSymbol *, Value> hierValuePlaceholders;
  unsigned int nextHierPlaceholderId = 0;

  /// Collect all hierarchical names used for the per module/instance.
  DenseMap<const slang::ast::InstanceBodySymbol *, SmallVector<HierPathInfo>>
      hierPaths;

  /// Collect hierarchical interface instances that need to be threaded through
  /// module ports.
  DenseMap<const slang::ast::InstanceBodySymbol *,
           SmallVector<HierInterfacePathInfo>>
      hierInterfacePaths;

  /// Interface ports from bind scopes that need to be threaded through
  /// target modules. Maps target module body to the interface ports needed.
  DenseMap<const slang::ast::InstanceBodySymbol *,
           SmallVector<BindScopeInterfacePortInfo>>
      bindScopeInterfacePorts;

  /// It's used to collect the repeat hierarchical names on the same path.
  /// Such as `Top.sub.a` and `sub.a`, they are equivalent. The variable "a"
  /// will be added to the port list. But we only record once. If we don't do
  /// that. We will view the strange IR, such as `module @Sub(out y, out y)`;
  DenseSet<StringAttr> sameHierPaths;

  /// A table of interface instances that may be referenced in expressions.
  /// When an interface is assigned to a virtual interface variable, slang
  /// represents the reference as an ArbitrarySymbolExpression. This map
  /// stores the MLIR Value (ref to virtual interface) for each interface
  /// instance symbol.
  DenseMap<const slang::ast::InstanceSymbol *, Value> interfaceInstances;
  /// Interface instances threaded into the current scope via hierarchical
  /// interface ports (e.g. bind port connections across sibling modules).
  DenseSet<const slang::ast::InstanceSymbol *> threadedInterfaceInstances;
  DenseMap<const slang::ast::InterfacePortSymbol *, Value> interfacePortValues;
  SmallVector<PendingInterfacePortConnection, 4>
      pendingInterfacePortConnections;

  /// A stack of assignment left-hand side values. Each assignment will push its
  /// lowered left-hand side onto this stack before lowering its right-hand
  /// side. This allows expressions to resolve the opaque
  /// `LValueReferenceExpression`s in the AST.
  SmallVector<Value> lvalueStack;

  /// A stack of loop continuation and exit blocks. Each loop will push the
  /// relevant info onto this stack, lower its loop body statements, and pop the
  /// info off the stack again. Continue and break statements encountered as
  /// part of the loop body statements will use this information to branch to
  /// the correct block.
  SmallVector<LoopFrame> loopStack;

  /// A stack of randsequence exit blocks. Each randsequence statement pushes
  /// its exit block onto this stack. A 'return' statement within a randsequence
  /// production branches to this block to exit the entire randsequence.
  SmallVector<Block *> randSequenceReturnStack;

  /// A stack of disable targets for named begin/end blocks.
  /// `disable <name>` branches to the innermost matching target block.
  SmallVector<DisableFrame> disableStack;

  /// A stack of break target blocks for randsequence productions. A 'break'
  /// statement within a randsequence production code block branches to this
  /// block to exit the current production. Unlike loop break which continues
  /// after the loop, randsequence break exits the current production but
  /// continues with subsequent productions in the rule.
  SmallVector<Block *> randSequenceBreakStack;

  /// A listener called for every variable or net being read. This can be used
  /// to collect all variables read as part of an expression or statement, for
  /// example to populate the list of observed signals in an implicit event
  /// control `@*`.
  std::function<void(moore::ReadOp)> rvalueReadCallback;
  /// A listener called for every variable or net being assigned. This can be
  /// used to collect all variables assigned in a task scope.
  std::function<void(mlir::Operation *)> variableAssignCallback;

  /// The time scale currently in effect.
  slang::TimeScale timeScale;

  /// The current queue target value for evaluating `$` (UnboundedLiteral).
  /// When indexing into a queue like `q[$]`, this holds the queue value so
  /// that `$` can be evaluated as `queue.size() - 1`.
  Value queueTargetValue = {};

  /// Variable to track the value of the current function's implicit `this`
  /// reference
  Value currentThisRef = {};

  /// Temporary override for method receiver selection. Used when emitting
  /// constructor calls so that argument evaluation continues to use the
  /// caller's `this`, while the callee still receives the correct new object.
  Value methodReceiverOverride = {};

  /// Temporary override for inline randomize constraints so unqualified
  /// properties resolve against the randomized object.
  Value inlineConstraintThisRef = {};
  const slang::ast::Symbol *inlineConstraintThisSymbol = nullptr;

  /// True while converting an assertion expression.
  bool inAssertionExpr = false;

  /// True while converting a constraint expression inside a constraint block.
  /// This affects method call generation to use ConstraintMethodCallOp.
  bool inConstraintExpr = false;

  /// The function currently being converted (if any). Used to propagate
  /// captures from callee functions to the caller when the caller is also
  /// a function that captures variables.
  FunctionLowering *currentFunctionLowering = nullptr;

  /// The current scope being processed. This is used by the %m format
  /// specifier to determine the hierarchical path.
  const slang::ast::Scope *currentScope = nullptr;

  /// Guard condition for procedural concurrent assertions.
  Value currentAssertionGuard = {};

  /// The current clocking event for assertions within a timed statement.
  const slang::ast::SignalEventControl *currentAssertionClock = nullptr;

  /// The current timing control for clocked assertion expressions.
  const slang::ast::TimingControl *currentAssertionTimingControl = nullptr;

  /// The current interface body being processed (if any). This is used when
  /// converting tasks/functions defined inside an interface to determine
  /// when signal accesses should use the implicit interface argument.
  const slang::ast::InstanceBodySymbol *currentInterfaceBody = nullptr;

  /// The implicit interface argument value when inside an interface task/func.
  /// This is the virtual interface reference that signal accesses should use.
  Value currentInterfaceArg = nullptr;

  /// Map from interface signal symbols to their names for signal lookup
  /// when inside interface tasks/functions.
  DenseMap<const slang::ast::Symbol *, StringRef> interfaceSignalNames;

private:
  /// Helper function to extract the commonalities in lowering of functions and
  /// methods
  FunctionLowering *
  declareCallableImpl(const slang::ast::SubroutineSymbol &subroutine,
                      mlir::StringRef qualifiedName,
                      llvm::SmallVectorImpl<Type> &extraParams);
};

/// Construct a fully qualified class name containing the instance hierarchy
/// and the class name formatted as H1::H2::@C
mlir::StringAttr fullyQualifiedClassName(Context &ctx,
                                         const slang::ast::Type &ty);

/// Construct a fully qualified symbol name for generic class definitions
mlir::StringAttr fullyQualifiedSymbolName(Context &ctx,
                                          const slang::ast::Symbol &sym);

} // namespace ImportVerilog
} // namespace circt
#endif // CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
