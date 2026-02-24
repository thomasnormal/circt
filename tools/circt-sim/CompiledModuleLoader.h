//===- CompiledModuleLoader.h - Load AOT-compiled .so modules ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loads a shared object (.so) produced by circt-sim-compile and provides
// lookup of compiled function pointers. The .so exports a
// CirctSimCompiledModule descriptor via circt_sim_get_compiled_module().
//
// This loader has NO dependency on MLIR ExecutionEngine or LLVM JIT.
// It uses only dlopen/dlsym/dlclose.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SIM_COMPILED_MODULE_LOADER_H
#define CIRCT_SIM_COMPILED_MODULE_LOADER_H

#include "circt/Runtime/CirctSimABI.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace circt {
namespace sim {

/// Loads and manages a compiled simulation .so module.
///
/// Usage:
///   auto loader = CompiledModuleLoader::load("foo_native.so");
///   if (!loader) { /* error already logged */ }
///   void *fn = loader->lookupFunction("some_func");
class CompiledModuleLoader {
public:
  /// Load a compiled .so file. Returns nullptr on failure (logs errors to
  /// llvm::errs()).
  static std::unique_ptr<CompiledModuleLoader> load(llvm::StringRef path);

  ~CompiledModuleLoader();

  // Non-copyable.
  CompiledModuleLoader(const CompiledModuleLoader &) = delete;
  CompiledModuleLoader &operator=(const CompiledModuleLoader &) = delete;

  /// Get the compiled module descriptor.
  const CirctSimCompiledModule *getModule() const { return compiledModule; }

  /// Look up a compiled function by name. Returns nullptr if not found.
  void *lookupFunction(llvm::StringRef name) const;

  /// Get the build ID string from the .so.
  llvm::StringRef getBuildId() const { return buildId; }

  /// Get number of compiled functions.
  uint32_t getNumFunctions() const {
    return compiledModule ? compiledModule->num_funcs : 0;
  }

  /// Get number of compiled processes.
  uint32_t getNumProcesses() const {
    return compiledModule ? compiledModule->num_procs : 0;
  }

  /// Check if ABI version matches the runtime.
  bool isCompatible() const {
    return compiledModule &&
           compiledModule->abi_version == CIRCT_SIM_ABI_VERSION;
  }

  /// Get the .so file path.
  llvm::StringRef getPath() const { return soPath; }

  /// Get number of trampolines (compiled-to-interpreted fallbacks).
  uint32_t getNumTrampolines() const {
    return compiledModule ? compiledModule->num_trampolines : 0;
  }

  /// Get the trampoline function name for a given func_id. Returns empty
  /// string if func_id is out of range.
  llvm::StringRef getTrampolineName(uint32_t funcId) const {
    if (!compiledModule || funcId >= compiledModule->num_trampolines ||
        !compiledModule->trampoline_names)
      return {};
    return compiledModule->trampoline_names[funcId];
  }

  /// Look up a trampoline ID by function name. Returns -1 if not found.
  int32_t lookupTrampolineId(llvm::StringRef name) const {
    auto it = trampolineIdMap.find(name);
    if (it != trampolineIdMap.end())
      return it->second;
    return -1;
  }

  /// Set the __circt_sim_ctx global in the loaded .so. Must be called before
  /// any compiled function that uses trampolines is invoked.
  void setRuntimeContext(void *ctx) {
    if (ctxGlobalPtr)
      *ctxGlobalPtr = ctx;
  }

  /// Get a pointer to the __circt_sim_ctx global slot in the loaded .so.
  /// The returned double-pointer can be captured in callbacks that need to
  /// dereference the context at call time (after setRuntimeContext has run).
  /// Returns nullptr if no context slot was found in the .so.
  void **getRuntimeContextPtr() const { return ctxGlobalPtr; }

private:
  CompiledModuleLoader() = default;

  void *dlHandle = nullptr;
  const CirctSimCompiledModule *compiledModule = nullptr;
  std::string buildId;
  std::string soPath;
  llvm::StringMap<void *> funcMap;          // name → native function pointer
  llvm::StringMap<int32_t> trampolineIdMap; // name → trampoline func_id
  void **ctxGlobalPtr = nullptr;            // pointer to __circt_sim_ctx in .so
};

} // namespace sim
} // namespace circt

#endif // CIRCT_SIM_COMPILED_MODULE_LOADER_H
