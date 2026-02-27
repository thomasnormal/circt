//===- CompiledModuleLoader.h - Load AOT-compiled .so modules ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loads a shared object (.so) produced by circt-compile and provides
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
#include "llvm/ADT/StringSet.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace circt {
namespace sim {

// Forward declaration to avoid circular include with LLHDProcessInterpreter.h.
struct MemoryBlock;

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
    if (!compiledModule)
      return false;
    // Keep runtime policy aligned with load(): accept legacy v4 modules and
    // current ABI modules.
    return compiledModule->abi_version == 4 ||
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

  /// Apply global patches: copy interpreter state into the .so's mutable
  /// globals. Must be called AFTER interpreter initialization
  /// (executeModuleLevelLLVMOps) and BEFORE loadCompiledFunctions().
  void applyGlobalPatches(
      const llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const;

  /// Redirect interpreter globals to use .so storage (single-copy aliasing).
  /// After this call, interpreter reads/writes go directly to the .so's
  /// mutable global storage, eliminating divergence between the two copies.
  void aliasGlobals(llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const;

  /// Pre-alias globals to .so storage BEFORE interpreter initialization.
  /// Creates MemoryBlock entries pointing to .so addresses so that
  /// initializeGlobals() writes directly to .so memory, preventing
  /// dangling inter-global pointers from post-init aliasing.
  void preAliasGlobals(llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const;

  /// Set up arena-backed MemoryBlocks for all arena globals (v5+).
  /// Creates pre-aliased MemoryBlock entries pointing into the runtime-
  /// allocated arena. Call this instead of preAliasGlobals for arena globals.
  void setupArenaGlobals(llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const;

  /// Check if a global name is backed by the arena (and should be skipped
  /// by the legacy patch/alias mechanism).
  bool isArenaGlobal(llvm::StringRef name) const {
    return arenaGlobalNames.count(name);
  }

  /// Get the arena base pointer (nullptr if no arena).
  void *getArenaBase() const { return arenaBase; }

  /// Get the arena allocation size in bytes (0 if no arena).
  uint32_t getArenaSize() const { return arenaAllocSize; }

  /// Get number of arena globals in the .so.
  uint32_t getNumArenaGlobals() const {
    return compiledModule ? compiledModule->num_arena_globals : 0;
  }

  /// Get number of global patches in the .so.
  uint32_t getNumGlobalPatches() const {
    return compiledModule ? compiledModule->num_global_patches : 0;
  }
  /// Get the name of global patch i.
  const char *getGlobalPatchName(uint32_t i) const {
    return compiledModule->global_patch_names[i];
  }
  /// Get the .so address of global patch i.
  void *getGlobalPatchAddr(uint32_t i) const {
    return compiledModule->global_patch_addrs[i];
  }
  /// Get the size of global patch i.
  uint32_t getGlobalPatchSize(uint32_t i) const {
    return compiledModule->global_patch_sizes[i];
  }

  /// Get total byte size of legacy global patches in the .so.
  uint64_t getGlobalPatchBytes() const {
    if (!compiledModule || compiledModule->num_global_patches == 0 ||
        !compiledModule->global_patch_sizes)
      return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < compiledModule->num_global_patches; ++i)
      total += compiledModule->global_patch_sizes[i];
    return total;
  }

  /// Get the total number of functions in the unified entry table.
  uint32_t getNumAllFuncs() const {
    return compiledModule ? compiledModule->num_all_funcs : 0;
  }

  /// Get a function entry pointer by FuncId. Returns nullptr if out of range.
  void *getFuncEntry(uint32_t fid) const {
    if (!compiledModule || fid >= compiledModule->num_all_funcs ||
        !compiledModule->all_func_entries)
      return nullptr;
    return const_cast<void *>(compiledModule->all_func_entries[fid]);
  }

  /// Get the symbol name for a FuncId. Returns empty string if out of range.
  const char *getFuncEntryName(uint32_t fid) const {
    if (!compiledModule || fid >= compiledModule->num_all_funcs ||
        !compiledModule->all_func_entry_names)
      return nullptr;
    return compiledModule->all_func_entry_names[fid];
  }

  /// Get the raw entry table pointer (for bulk assignment to interpreter).
  const void *const *getFuncEntries() const {
    return compiledModule ? compiledModule->all_func_entries : nullptr;
  }

  /// Look up the optional native module-init entrypoint for a specific
  /// hw.module symbol name. Returns nullptr if no native init is available.
  void *lookupModuleInit(llvm::StringRef hwModuleName) const;

private:
  CompiledModuleLoader() = default;

  void *dlHandle = nullptr;
  const CirctSimCompiledModule *compiledModule = nullptr;
  std::string buildId;
  std::string soPath;
  llvm::StringMap<void *> funcMap;          // name → native function pointer
  llvm::StringMap<int32_t> trampolineIdMap; // name → trampoline func_id
  void **ctxGlobalPtr = nullptr;            // pointer to __circt_sim_ctx in .so

  // Arena state (v5+).
  void *arenaBase = nullptr;                // runtime-allocated arena memory
  uint32_t arenaAllocSize = 0;              // size of arenaBase allocation
  llvm::StringSet<> arenaGlobalNames;       // names of globals in the arena
};

} // namespace sim
} // namespace circt

#endif // CIRCT_SIM_COMPILED_MODULE_LOADER_H
