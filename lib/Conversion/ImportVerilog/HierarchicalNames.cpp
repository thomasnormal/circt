//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTContext.h"
#include "slang/ast/Expression.h"
#include "slang/ast/expressions/AssertionExpr.h"
#include "slang/syntax/AllSyntax.h"

using namespace circt;
using namespace ImportVerilog;

namespace {
struct HierPathValueExprVisitor
    : public slang::ast::ASTVisitor<HierPathValueExprVisitor, false, true,
                                    true> {
  Context &context;
  Location loc;
  OpBuilder &builder;
  LogicalResult result = success();

  // Such as `sub.a`, the `sub` is the outermost module for the hierarchical
  // variable `a`.
  const slang::ast::Symbol &outermostModule;

  HierPathValueExprVisitor(Context &context, Location loc,
                           const slang::ast::Symbol &outermostModule)
      : context(context), loc(loc), builder(context.builder),
        outermostModule(outermostModule) {}

  // Handle hierarchical values
  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    if (failed(result))
      return;
    auto *currentInstBody =
        expr.symbol.getParentScope()->getContainingInstance();
    if (!currentInstBody) {
      for (auto it = expr.ref.path.rbegin(); it != expr.ref.path.rend(); ++it) {
        if (auto *instSym =
                (*it).symbol->as_if<slang::ast::InstanceSymbol>()) {
          currentInstBody = &instSym->body;
          break;
        }
      }
    }
    auto *outermostInstBody =
        outermostModule.as_if<slang::ast::InstanceBodySymbol>();

    // Like module Foo; int a; Foo.a; endmodule.
    // Ignore "Foo.a" invoked by this module itself.
    if (currentInstBody == outermostInstBody)
      return;

    auto getInterfaceDefBody =
        [&](const slang::ast::Symbol &sym)
        -> const slang::ast::InstanceBodySymbol * {
      const slang::ast::Scope *parentScope = nullptr;
      if (auto *var = sym.as_if<slang::ast::VariableSymbol>())
        parentScope = var->getParentScope();
      else if (auto *net = sym.as_if<slang::ast::NetSymbol>())
        parentScope = net->getParentScope();
      else if (auto *modportPort = sym.as_if<slang::ast::ModportPortSymbol>())
        if (auto *internalSym = modportPort->internalSymbol)
          parentScope = internalSym->getParentScope();
      if (!parentScope)
        return nullptr;
      if (auto *instBody =
              parentScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>())
        if (instBody->getDefinition().definitionKind ==
            slang::ast::DefinitionKind::Interface)
          return instBody;
      return nullptr;
    };

    // Handle hierarchical interface member references by threading the
    // interface instance through module ports.
    if (auto *ifaceDefBody = getInterfaceDefBody(expr.symbol)) {
      bool handledInterface = false;
      const slang::ast::InstanceSymbol *ifaceInst = nullptr;
      size_t ifaceInstIndex = 0;
      for (size_t i = 0; i < expr.ref.path.size(); ++i) {
        if (auto *instSym =
                expr.ref.path[i].symbol->as_if<slang::ast::InstanceSymbol>()) {
          if (instSym->getDefinition().definitionKind ==
                  slang::ast::DefinitionKind::Interface &&
              &instSym->getDefinition() == &ifaceDefBody->getDefinition()) {
            ifaceInst = instSym;
            ifaceInstIndex = i;
          }
        } else if (auto *ifacePort =
                       expr.ref.path[i].symbol
                           ->as_if<slang::ast::InterfacePortSymbol>()) {
          auto [ifaceConn, modport] = ifacePort->getConnection();
          (void)modport;
          if (auto *ifaceInstSym =
                  ifaceConn
                      ? ifaceConn->as_if<slang::ast::InstanceSymbol>()
                      : nullptr) {
            if (ifaceInstSym->getDefinition().definitionKind ==
                    slang::ast::DefinitionKind::Interface &&
                &ifaceInstSym->getDefinition() ==
                    &ifaceDefBody->getDefinition()) {
              ifaceInst = ifaceInstSym;
              ifaceInstIndex = i;
            }
          }
        }
      }
      if (ifaceInst && outermostInstBody) {
        auto *ifaceParentBody =
            ifaceInst->getParentScope()->getContainingInstance();
        if (ifaceParentBody && ifaceParentBody != outermostInstBody) {
          auto addHierIfacePath =
              [&](const slang::ast::InstanceBodySymbol *sym,
                  mlir::StringAttr nameAttr) {
                auto &paths = context.hierInterfacePaths[sym];
                bool exists = llvm::any_of(paths, [&](const auto &info) {
                  return info.hierName == nameAttr &&
                         info.ifaceInst == ifaceInst;
                });
                if (!exists)
                  paths.push_back(HierInterfacePathInfo{
                      nameAttr, {}, slang::ast::ArgumentDirection::In,
                      ifaceInst});
              };

          SmallVector<const slang::ast::InstanceBodySymbol *, 8> ifaceChain;
          SmallVector<const slang::ast::InstanceBodySymbol *, 8> outerChain;
          for (auto *body = ifaceParentBody; body;
               body = body->parentInstance
                          ? body->parentInstance->getParentScope()
                                ->getContainingInstance()
                          : nullptr)
            ifaceChain.push_back(body);
          for (auto *body = outermostInstBody; body;
               body = body->parentInstance
                          ? body->parentInstance->getParentScope()
                                ->getContainingInstance()
                          : nullptr)
            outerChain.push_back(body);

          DenseMap<const slang::ast::InstanceBodySymbol *, size_t> ifaceIndex;
          ifaceIndex.reserve(ifaceChain.size());
          for (size_t i = 0; i < ifaceChain.size(); ++i)
            ifaceIndex[ifaceChain[i]] = i;

          const slang::ast::InstanceBodySymbol *lca = nullptr;
          size_t lcaOuterIndex = 0;
          for (size_t i = 0; i < outerChain.size(); ++i) {
            if (ifaceIndex.count(outerChain[i])) {
              lca = outerChain[i];
              lcaOuterIndex = i;
              break;
            }
          }

          if (lca) {
            SmallString<64> ifaceName;
            for (size_t i = 0; i < expr.ref.path.size(); ++i) {
              if (auto *instSym =
                      expr.ref.path[i].symbol->as_if<
                          slang::ast::InstanceSymbol>()) {
                if (!ifaceName.empty())
                  ifaceName += ".";
                ifaceName += instSym->name;
              } else if (auto *ifacePort =
                             expr.ref.path[i].symbol
                                 ->as_if<slang::ast::InterfacePortSymbol>()) {
                if (!ifaceName.empty())
                  ifaceName += ".";
                ifaceName += ifacePort->name;
              }
              if (i == ifaceInstIndex)
                break;
            }
            if (!ifaceName.empty()) {
              auto nameAttr = builder.getStringAttr(ifaceName);
              for (size_t i = lcaOuterIndex; i > 0; --i) {
                auto *body = outerChain[i - 1];
                addHierIfacePath(body, nameAttr);
              }
              handledInterface = true;
            }
          }
        } else if (ifaceParentBody == outermostInstBody) {
          handledInterface = true;
        }
      }
      // If we successfully handled the interface reference, we're done.
      if (handledInterface)
        return;
    }

    // Skip interface port member accesses that are already in scope. When
    // accessing a member of an interface port (e.g., `iface.data` where `iface`
    // is an interface port), slang marks this as a hierarchical reference.
    // These should be handled via VirtualInterfaceSignalRefOp if the interface
    // port is available in the current scope. Otherwise, allow hierarchical
    // handling to thread the reference.
    if (expr.ref.isViaIfacePort()) {
      const auto *ifacePort =
          expr.ref.path.front().symbol->as_if<slang::ast::InterfacePortSymbol>();
      const auto *portScope = ifacePort ? ifacePort->getParentScope() : nullptr;
      const auto *portBody =
          portScope ? portScope->asSymbol().as_if<slang::ast::InstanceBodySymbol>()
                    : nullptr;
      if (portBody && portBody == outermostInstBody)
        return;
    }

    if (!currentInstBody || !outermostInstBody)
      return;

    auto addHierPath = [&](const slang::ast::InstanceBodySymbol *sym,
                           mlir::StringAttr nameAttr,
                           slang::ast::ArgumentDirection dir) {
      auto &paths = context.hierPaths[sym];
      bool exists = llvm::any_of(paths, [&](const auto &info) {
        return info.hierName == nameAttr;
      });
      if (!exists)
        paths.push_back(HierPathInfo{nameAttr, {}, dir, &expr.symbol});
    };


    // Build ancestor chains for the symbol's body and the referencing scope.
    SmallVector<const slang::ast::InstanceBodySymbol *, 8> currentChain;
    SmallVector<const slang::ast::InstanceBodySymbol *, 8> outerChain;
    for (auto *body = currentInstBody; body;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr)
      currentChain.push_back(body);
    for (auto *body = outermostInstBody; body;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr)
      outerChain.push_back(body);

    DenseMap<const slang::ast::InstanceBodySymbol *, size_t> currentIndex;
    currentIndex.reserve(currentChain.size());
    for (size_t i = 0; i < currentChain.size(); ++i)
      currentIndex[currentChain[i]] = i;

    const slang::ast::InstanceBodySymbol *lca = nullptr;
    size_t lcaOuterIndex = 0;
    for (size_t i = 0; i < outerChain.size(); ++i) {
      if (currentIndex.count(outerChain[i])) {
        lca = outerChain[i];
        lcaOuterIndex = i;
        break;
      }
    }
    if (!lca)
      return;

    // Construct the full hierarchical name for the reference.
    SmallString<64> fullName;
    for (const auto &elem : expr.ref.path) {
      if (auto *instSym = elem.symbol->as_if<slang::ast::InstanceSymbol>()) {
        if (!fullName.empty())
          fullName += ".";
        fullName += instSym->name;
      }
    }
    if (!fullName.empty())
      fullName += ".";
    fullName += expr.symbol.name;
    auto fullNameAttr = builder.getStringAttr(fullName);

    // Propagate upward from the symbol's instance body to the LCA.
    SmallString<64> upwardName(expr.symbol.name);
    for (auto *body = currentInstBody; body && body != lca;
         body = body->parentInstance
                    ? body->parentInstance->getParentScope()
                          ->getContainingInstance()
                    : nullptr) {
      addHierPath(body, builder.getStringAttr(upwardName),
                  slang::ast::ArgumentDirection::Out);
      if (body->parentInstance) {
        SmallString<64> nextName;
        nextName += body->parentInstance->name;
        nextName += ".";
        nextName += upwardName;
        upwardName = nextName;
      }
    }

    // Propagate downward from the LCA to the referencing scope.
    for (size_t i = lcaOuterIndex; i > 0; --i) {
      auto *body = outerChain[i - 1];
      addHierPath(body, fullNameAttr, slang::ast::ArgumentDirection::In);
    }
  }

  void handle(const slang::ast::InvalidExpression &expr) {
    if (failed(result))
      return;
    mlir::emitError(loc, "invalid expression");
    result = failure();
  }
};

/// Visitor to traverse statements and collect hierarchical value references.
/// This visitor visits both statements and expressions within them, reusing
/// the HierPathValueExprVisitor logic for expression handling.
struct HierPathValueStmtVisitor
    : public slang::ast::ASTVisitor<HierPathValueStmtVisitor, true, true,
                                    true> {
  Context &context;
  Location loc;
  LogicalResult result = success();

  // The outermost module for determining hierarchical paths.
  const slang::ast::Symbol &outermostModule;

  HierPathValueStmtVisitor(Context &context, Location loc,
                           const slang::ast::Symbol &outermostModule)
      : context(context), loc(loc), outermostModule(outermostModule) {}

  // Handle hierarchical values found in expressions within statements.
  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    if (failed(result))
      return;
    // Delegate to the expression visitor logic via collectHierarchicalValues
    if (failed(context.collectHierarchicalValues(expr, outermostModule)))
      result = failure();
  }

  void handle(const slang::ast::InvalidExpression &expr) {
    if (failed(result))
      return;
    mlir::emitError(loc, "invalid expression");
    result = failure();
  }
};
} // namespace

struct BindAssertionExprVisitor
    : public slang::ast::ASTVisitor<BindAssertionExprVisitor, false, true,
                                    true> {
  HierPathValueExprVisitor exprVisitor;
  LogicalResult result = success();

  BindAssertionExprVisitor(Context &context, Location loc,
                           const slang::ast::Symbol &outermostModule)
      : exprVisitor(context, loc, outermostModule) {}

  void handle(const slang::ast::HierarchicalValueExpression &expr) {
    if (failed(result))
      return;
    exprVisitor.handle(expr);
    if (failed(exprVisitor.result))
      result = failure();
  }

  void handle(const slang::ast::InvalidExpression &expr) {
    if (failed(result))
      return;
    exprVisitor.handle(expr);
    result = failure();
  }

  void handle(const slang::ast::InvalidAssertionExpr &) { result = failure(); }
};

LogicalResult
Context::collectHierarchicalValues(const slang::ast::Expression &expr,
                                   const slang::ast::Symbol &outermostModule) {
  auto loc = convertLocation(expr.sourceRange);
  HierPathValueExprVisitor visitor(*this, loc, outermostModule);
  expr.visit(visitor);
  return visitor.result;
}

LogicalResult Context::collectHierarchicalValuesFromStatement(
    const slang::ast::Statement &stmt,
    const slang::ast::Symbol &outermostModule) {
  auto loc = convertLocation(stmt.sourceRange);
  HierPathValueStmtVisitor visitor(*this, loc, outermostModule);
  stmt.visit(visitor);
  return visitor.result;
}

static LogicalResult collectBindDirectiveHierarchicalValues(
    Context &context, const slang::ast::InstanceBodySymbol &body) {
  auto *node = body.hierarchyOverrideNode;
  if (!node)
    return success();

  for (const auto &entry : node->binds) {
    const auto &bindInfo = entry.first;
    const auto *targetDefSyntax = entry.second;
    if (targetDefSyntax)
      continue;
    auto *bindSyntax = bindInfo.bindSyntax;
    if (!bindSyntax)
      continue;
    const auto *bindScope = context.compilation.getBindDirectiveScope(bindSyntax);
    if (!bindScope)
      continue;

  slang::ast::ASTContext astContext(*bindScope,
                                    slang::ast::LookupLocation::max);

  auto collectFromPropertyExpr =
      [&](const slang::syntax::PropertyExprSyntax &syntax) -> LogicalResult {
    if (syntax.kind == slang::syntax::SyntaxKind::SimplePropertyExpr) {
      auto seqExpr = syntax.as<slang::syntax::SimplePropertyExprSyntax>().expr;
      if (seqExpr->kind == slang::syntax::SyntaxKind::SimpleSequenceExpr) {
        auto &simpleSeq =
            seqExpr->as<slang::syntax::SimpleSequenceExprSyntax>();
        if (!simpleSeq.repetition && simpleSeq.expr) {
          const auto &expr =
              slang::ast::Expression::bind(*simpleSeq.expr, astContext);
          if (expr.bad())
            return failure();
          return context.collectHierarchicalValues(expr, body);
        }
      }
    }

    const auto &expr = slang::ast::AssertionExpr::bind(
        syntax, astContext, /*allowDisable=*/true);
    auto loc = context.convertLocation(syntax.sourceRange());
    BindAssertionExprVisitor visitor(context, loc, body);
    expr.visit(visitor);
    return visitor.result;
  };

  auto collectFromInstances = [&](const auto &instances) -> LogicalResult {
    for (auto *inst : instances) {
      for (auto *conn : inst->connections) {
        if (auto *ordered =
                conn->template as_if<slang::syntax::OrderedPortConnectionSyntax>()) {
          if (!ordered->expr)
            continue;
          if (failed(collectFromPropertyExpr(*ordered->expr)))
            return failure();
          continue;
        }
        if (auto *named =
                conn->template as_if<slang::syntax::NamedPortConnectionSyntax>()) {
          if (!named->expr)
            continue;
          if (failed(collectFromPropertyExpr(*named->expr)))
            return failure();
          continue;
        }
      }
    }
      return success();
    };

    if (auto *inst =
            bindSyntax->instantiation
                ->template as_if<slang::syntax::HierarchyInstantiationSyntax>()) {
      if (failed(collectFromInstances(inst->instances)))
        return failure();
    } else if (auto *checker =
                   bindSyntax->instantiation
                       ->template as_if<slang::syntax::CheckerInstantiationSyntax>()) {
      if (failed(collectFromInstances(checker->instances)))
        return failure();
    }
  }

  return success();
}

/// Traverse the instance body.
namespace {
struct InstBodyVisitor {
  Context &context;
  Location loc;

  InstBodyVisitor(Context &context, Location loc)
      : context(context), loc(loc) {}

  const slang::ast::Symbol &
  getOutermostModule(const slang::ast::Scope &scope) const {
    if (auto *body = scope.getContainingInstance())
      return body->asSymbol();
    return scope.asSymbol();
  }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    auto &outermostModule = getOutermostModule(*instNode.getParentScope());
    for (const auto *con : instNode.getPortConnections()) {
      if (const auto *expr = con->getExpression())
        if (failed(context.collectHierarchicalValues(*expr, outermostModule)))
          return failure();
    }

    // Check if this is a bound instance (has a bind scope)
    if (auto *bindScope = instNode.getBindScope()) {
      // For bound instances, analyze port connections that might reference
      // interface ports from the bind scope
      if (bindScope->getContainingInstance()) {
        for (const auto *con : instNode.getPortConnections()) {
          // Check if the port connection expression references an interface
          // port from the bind scope
          const auto *expr = con->getExpression();
          if (!expr)
            continue;

          // Look for hierarchical value expressions that go through interface
          // ports
          auto checkExpr =
              [&](const slang::ast::HierarchicalValueExpression &hierExpr) {
                if (!hierExpr.ref.isViaIfacePort())
                  return;
                // Find the interface port in the path
                for (const auto &elem : hierExpr.ref.path) {
                  if (auto *ifacePort =
                          elem.symbol
                              ->as_if<slang::ast::InterfacePortSymbol>()) {
                    // Check if this interface port is from the bind scope
                    auto *portParent = ifacePort->getParentScope();
                    if (portParent && portParent == bindScope) {
                      // This interface port needs to be threaded to the target
                      // module
                      auto *targetBody =
                          instNode.getParentScope()->getContainingInstance();
                      if (targetBody) {
                        // Check if we already have this interface port
                        auto &ports =
                            context.bindScopeInterfacePorts[targetBody];
                        bool exists = llvm::any_of(
                            ports, [&](const BindScopeInterfacePortInfo &info) {
                              return info.ifacePort == ifacePort;
                            });
                        if (!exists) {
                          ports.push_back({ifacePort, std::nullopt});
                        }
                      }
                    }
                    break;
                  }
                }
              };

          // Recursively visit the expression to find hierarchical value exprs
          struct HierExprVisitor
              : public slang::ast::ASTVisitor<HierExprVisitor, false, true,
                                              true> {
            std::function<void(const slang::ast::HierarchicalValueExpression &)>
                callback;
            void handle(const slang::ast::HierarchicalValueExpression &expr) {
              callback(expr);
            }
          };
          HierExprVisitor visitor;
          visitor.callback = checkExpr;
          expr->visit(visitor);
        }
      }
    }
    return context.traverseInstanceBody(instNode.body);
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto &outermostModule = getOutermostModule(*varNode.getParentScope());
    if (const auto *init = varNode.getInitializer())
      if (failed(context.collectHierarchicalValues(*init, outermostModule)))
        return failure();
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto &outermostModule = getOutermostModule(*netNode.getParentScope());
    if (const auto *init = netNode.getInitializer())
      if (failed(context.collectHierarchicalValues(*init, outermostModule)))
        return failure();
    return success();
  }

  // Handle continuous assignments.
  LogicalResult visit(const slang::ast::ContinuousAssignSymbol &assignNode) {
    const auto *expr =
        assignNode.getAssignment().as_if<slang::ast::AssignmentExpression>();
    if (!expr) {
      if (context.options.allowNonProceduralDynamic.value_or(false)) {
        mlir::emitWarning(loc)
            << "skipping continuous assignment without an assignment "
               "expression after DynamicNotProcedural downgrade";
        return success();
      }
      mlir::emitError(loc)
          << "expected assignment expression in continuous assignment";
      return failure();
    }

    // Such as `sub.a`, the `sub` is the outermost module for the hierarchical
    // variable `a`.
    auto &outermostModule = getOutermostModule(*assignNode.getParentScope());
    if (failed(context.collectHierarchicalValues(expr->left(), outermostModule)))
      return failure();

    if (failed(
            context.collectHierarchicalValues(expr->right(), outermostModule)))
      return failure();

    return success();
  }

  // Handle procedural blocks (always, initial, final, etc.).
  // Traverse the procedure body to collect hierarchical references
  // from event triggers, wait statements, and other expressions.
  LogicalResult visit(const slang::ast::ProceduralBlockSymbol &procNode) {
    auto &outermostModule = getOutermostModule(*procNode.getParentScope());
    return context.collectHierarchicalValuesFromStatement(procNode.getBody(),
                                                          outermostModule);
  }

  // Traverse generate blocks.
  LogicalResult visit(const slang::ast::GenerateBlockSymbol &genNode) {
    for (auto &member : genNode.members()) {
      auto loc = context.convertLocation(member.location);
      if (failed(member.visit(InstBodyVisitor(context, loc))))
        return failure();
    }
    return success();
  }

  // Traverse generate block arrays.
  LogicalResult visit(const slang::ast::GenerateBlockArraySymbol &genArray) {
    for (auto *entry : genArray.entries) {
      auto loc = context.convertLocation(entry->location);
      if (failed(entry->visit(InstBodyVisitor(context, loc))))
        return failure();
    }
    return success();
  }

  // Traverse instance arrays.
  LogicalResult visit(const slang::ast::InstanceArraySymbol &arrNode) {
    for (auto *element : arrNode.elements) {
      auto loc = context.convertLocation(element->location);
      if (failed(element->visit(InstBodyVisitor(context, loc))))
        return failure();
    }
    return success();
  }

  /// TODO:Skip all others.
  /// But we should output a warning to display which symbol had been skipped.
  /// However, to ensure we can test smoothly, we didn't do that.
  template <typename T>
  LogicalResult visit(T &&node) {
    return success();
  }
};
} // namespace

LogicalResult Context::traverseInstanceBody(const slang::ast::Symbol &symbol) {
  if (auto *instBodySymbol = symbol.as_if<slang::ast::InstanceBodySymbol>()) {
    if (failed(collectBindDirectiveHierarchicalValues(*this,
                                                      *instBodySymbol)))
      return failure();
    for (auto &member : instBodySymbol->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(InstBodyVisitor(*this, loc))))
        return failure();
    }
  }
  return success();
}
