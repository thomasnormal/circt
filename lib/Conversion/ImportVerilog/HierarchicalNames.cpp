//===- Expressions.cpp - Slang expression conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"

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

    // Skip interface port member accesses. When accessing a member of an
    // interface port (e.g., `iface.data` where `iface` is an interface port),
    // slang marks this as a hierarchical reference. However, these should be
    // handled as regular interface port member accesses via
    // VirtualInterfaceSignalRefOp, not as hierarchical ports that need to be
    // threaded through the module hierarchy.
    if (expr.ref.isViaIfacePort())
      return;

    auto hierName = builder.getStringAttr(expr.symbol.name);
    const slang::ast::InstanceBodySymbol *parentInstBody = nullptr;

    // Collect hierarchical names that are added to the port list.
    std::function<void(const slang::ast::InstanceBodySymbol *, bool)>
        collectHierarchicalPaths = [&](auto sym, bool isUpward) {
          // Avoid collecting duplicate hierarchical names for the same module.
          auto &paths = context.hierPaths[sym];
          bool exists = llvm::any_of(paths, [&](const auto &info) {
            return info.hierName == hierName;
          });
          if (!exists)
            paths.push_back(
                HierPathInfo{hierName,
                             {},
                             isUpward ? slang::ast::ArgumentDirection::Out
                                      : slang::ast::ArgumentDirection::In,
                             &expr.symbol});

          // Iterate up from the current instance body symbol until meeting the
          // outermost module.
          parentInstBody =
              sym->parentInstance->getParentScope()->getContainingInstance();
          if (!parentInstBody)
            return;

          if (isUpward) {
            // Avoid collecting hierarchical names into the outermost module.
            if (parentInstBody && parentInstBody != outermostInstBody) {
              hierName =
                  builder.getStringAttr(sym->parentInstance->name +
                                        llvm::Twine(".") + hierName.getValue());
              collectHierarchicalPaths(parentInstBody, isUpward);
            }
          } else {
            if (parentInstBody && parentInstBody != currentInstBody)
              collectHierarchicalPaths(parentInstBody, isUpward);
          }
        };

    // Determine whether hierarchical names are upward or downward.
    auto *tempInstBody = currentInstBody;
    while (tempInstBody) {
      tempInstBody = tempInstBody->parentInstance->getParentScope()
                         ->getContainingInstance();
      if (tempInstBody == outermostInstBody) {
        collectHierarchicalPaths(currentInstBody, true);
        return;
      }
    }

    if (!currentInstBody)
      return;
    auto baseHierName = hierName;
    collectHierarchicalPaths(currentInstBody, true);
    StringRef parentName;
    if (currentInstBody->parentInstance)
      parentName = currentInstBody->parentInstance->name;
    else {
      for (const auto &elem : expr.ref.path) {
        if (auto *instSym =
                elem.symbol->as_if<slang::ast::InstanceSymbol>()) {
          parentName = instSym->name;
          break;
        }
      }
    }
    if (parentName.empty())
      return;
    hierName =
        builder.getStringAttr(parentName + llvm::Twine(".") +
                              baseHierName.getValue());
    collectHierarchicalPaths(outermostInstBody, false);
  }

  void handle(const slang::ast::InvalidExpression &expr) {
    if (failed(result))
      return;
    mlir::emitError(loc, "invalid expression");
    result = failure();
  }
};
} // namespace

LogicalResult
Context::collectHierarchicalValues(const slang::ast::Expression &expr,
                                   const slang::ast::Symbol &outermostModule) {
  auto loc = convertLocation(expr.sourceRange);
  HierPathValueExprVisitor visitor(*this, loc, outermostModule);
  expr.visit(visitor);
  return visitor.result;
}

/// Traverse the instance body.
namespace {
struct InstBodyVisitor {
  Context &context;
  Location loc;

  InstBodyVisitor(Context &context, Location loc)
      : context(context), loc(loc) {}

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    return context.traverseInstanceBody(instNode.body);
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto &outermostModule = varNode.getParentScope()->asSymbol();
    if (const auto *init = varNode.getInitializer())
      if (failed(context.collectHierarchicalValues(*init, outermostModule)))
        return failure();
    return success();
  }

  // Handle nets.
  LogicalResult visit(const slang::ast::NetSymbol &netNode) {
    auto &outermostModule = netNode.getParentScope()->asSymbol();
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
    auto &outermostModule = assignNode.getParentScope()->asSymbol();
    if (failed(context.collectHierarchicalValues(expr->left(), outermostModule)))
      return failure();

    if (failed(
            context.collectHierarchicalValues(expr->right(), outermostModule)))
      return failure();

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
  if (auto *instBodySymbol = symbol.as_if<slang::ast::InstanceBodySymbol>())
    for (auto &member : instBodySymbol->members()) {
      auto loc = convertLocation(member.location);
      if (failed(member.visit(InstBodyVisitor(*this, loc))))
        return failure();
    }
  return success();
}
