//===- CompiledModuleLoader.cpp - Load AOT-compiled .so modules ---*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompiledModuleLoader.h"
#include "LLHDProcessInterpreter.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstring>
#include <dlfcn.h>

using namespace circt;
using namespace circt::sim;

static std::string encodeModuleInitSymbol(llvm::StringRef moduleName) {
  std::string out = "__circt_sim_module_init__";
  out.reserve(out.size() + moduleName.size() * 3);
  static constexpr char hex[] = "0123456789ABCDEF";
  for (unsigned char c : moduleName.bytes()) {
    if (std::isalnum(c) || c == '_') {
      out.push_back(static_cast<char>(c));
      continue;
    }
    out.push_back('_');
    out.push_back(hex[(c >> 4) & 0xF]);
    out.push_back(hex[c & 0xF]);
  }
  return out;
}

std::unique_ptr<CompiledModuleLoader>
CompiledModuleLoader::load(llvm::StringRef path) {
  auto loader = std::unique_ptr<CompiledModuleLoader>(
      new CompiledModuleLoader());
  loader->soPath = path.str();

  // Open the shared object. RTLD_LAZY defers symbol resolution until first
  // call, allowing unresolved UVM symbols that are handled by the interpreter.
  // RTLD_LOCAL keeps symbols private to avoid collisions.
  loader->dlHandle = dlopen(path.str().c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!loader->dlHandle) {
    llvm::errs() << "[circt-sim] Failed to load compiled module '" << path
                 << "': " << dlerror() << "\n";
    return nullptr;
  }

  // Look up the compiled module descriptor entrypoint.
  using GetModuleFn = const CirctSimCompiledModule *(*)(void);
  auto *getModuleFn = reinterpret_cast<GetModuleFn>(
      dlsym(loader->dlHandle, "circt_sim_get_compiled_module"));
  if (!getModuleFn) {
    llvm::errs() << "[circt-sim] .so missing circt_sim_get_compiled_module: "
                 << dlerror() << "\n";
    dlclose(loader->dlHandle);
    return nullptr;
  }

  loader->compiledModule = getModuleFn();
  if (!loader->compiledModule) {
    llvm::errs() << "[circt-sim] circt_sim_get_compiled_module returned null\n";
    dlclose(loader->dlHandle);
    return nullptr;
  }

  // Validate ABI version. Accept v4 (legacy) and v5+ (arena).
  uint32_t abiVer = loader->compiledModule->abi_version;
  if (abiVer != 4 && abiVer != CIRCT_SIM_ABI_VERSION) {
    llvm::errs() << "[circt-sim] ABI version mismatch: .so has v" << abiVer
                 << ", runtime expects v4 or v" << CIRCT_SIM_ABI_VERSION << "\n";
    dlclose(loader->dlHandle);
    return nullptr;
  }

  // Look up the build ID.
  using GetBuildIdFn = const char *(*)(void);
  auto *getBuildIdFn = reinterpret_cast<GetBuildIdFn>(
      dlsym(loader->dlHandle, "circt_sim_get_build_id"));
  if (getBuildIdFn) {
    const char *id = getBuildIdFn();
    if (id)
      loader->buildId = id;
  }

  // Build the function lookup map from the descriptor's parallel arrays.
  uint32_t numFuncs = loader->compiledModule->num_funcs;
  if (numFuncs > 0 && loader->compiledModule->func_names &&
      loader->compiledModule->func_entry) {
    for (uint32_t i = 0; i < numFuncs; ++i) {
      const char *name = loader->compiledModule->func_names[i];
      const void *entry = loader->compiledModule->func_entry[i];
      if (name && entry)
        loader->funcMap[name] = const_cast<void *>(entry);
    }
  }

  // Build the trampoline ID map from the descriptor.
  uint32_t numTrampolines = loader->compiledModule->num_trampolines;
  if (numTrampolines > 0 && loader->compiledModule->trampoline_names) {
    for (uint32_t i = 0; i < numTrampolines; ++i) {
      const char *name = loader->compiledModule->trampoline_names[i];
      if (name)
        loader->trampolineIdMap[name] = static_cast<int32_t>(i);
    }
  }

  // Set the __circt_sim_ctx global in the .so so trampolines can find
  // the runtime context. The actual value is set later when the interpreter
  // is ready.
  auto *ctxGlobalSym = dlsym(loader->dlHandle, "__circt_sim_ctx");
  if (ctxGlobalSym)
    loader->ctxGlobalPtr = reinterpret_cast<void **>(ctxGlobalSym);

  // v5 arena setup: allocate a single contiguous block for all mutable globals.
  if (abiVer >= 5 && loader->compiledModule->arena_size > 0) {
    uint32_t aSize = loader->compiledModule->arena_size;
    loader->arenaBase = std::aligned_alloc(16, aSize);
    if (!loader->arenaBase) {
      llvm::errs() << "[circt-sim] Failed to allocate arena (" << aSize
                   << " bytes)\n";
      dlclose(loader->dlHandle);
      return nullptr;
    }
    std::memset(loader->arenaBase, 0, aSize);
    loader->arenaAllocSize = aSize;

    // Write the arena base pointer into the .so's __circt_sim_arena_base.
    auto *arenaBaseSym = dlsym(loader->dlHandle, "__circt_sim_arena_base");
    if (arenaBaseSym) {
      *reinterpret_cast<void **>(arenaBaseSym) = loader->arenaBase;
    } else {
      llvm::errs()
          << "[circt-sim] Warning: .so missing __circt_sim_arena_base symbol\n";
    }

    // Build the arena global name set for skip-checking in patch methods.
    uint32_t numArenaGlobals = loader->compiledModule->num_arena_globals;
    if (numArenaGlobals > 0 && loader->compiledModule->arena_global_names) {
      for (uint32_t i = 0; i < numArenaGlobals; ++i) {
        const char *name = loader->compiledModule->arena_global_names[i];
        if (name)
          loader->arenaGlobalNames.insert(name);
      }
    }

    llvm::errs() << "[circt-sim] Allocated arena: " << aSize << " bytes, "
                 << numArenaGlobals << " arena globals\n";
  }

  llvm::errs() << "[circt-sim] Loaded compiled module '" << path << "': "
               << loader->funcMap.size() << " functions";
  if (numTrampolines > 0)
    llvm::errs() << ", " << numTrampolines << " trampolines";
  if (loader->compiledModule->num_procs > 0)
    llvm::errs() << ", " << loader->compiledModule->num_procs << " processes";
  if (loader->compiledModule->num_all_funcs > 0)
    llvm::errs() << ", " << loader->compiledModule->num_all_funcs
                 << " entry table entries";
  if (!loader->buildId.empty())
    llvm::errs() << " (build: " << loader->buildId << ")";
  llvm::errs() << "\n";

  return loader;
}

CompiledModuleLoader::~CompiledModuleLoader() {
  std::free(arenaBase); // no-op if nullptr
  if (dlHandle)
    dlclose(dlHandle);
}

void *CompiledModuleLoader::lookupFunction(llvm::StringRef name) const {
  auto it = funcMap.find(name);
  if (it != funcMap.end())
    return it->second;
  return nullptr;
}

void *CompiledModuleLoader::lookupModuleInit(llvm::StringRef hwModuleName) const {
  if (!dlHandle)
    return nullptr;
  std::string sym = encodeModuleInitSymbol(hwModuleName);
  return dlsym(dlHandle, sym.c_str());
}

void CompiledModuleLoader::applyGlobalPatches(
    const llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const {
  if (!compiledModule || compiledModule->num_global_patches == 0)
    return;

  unsigned patched = 0, missed = 0, skippedArena = 0;
  for (uint32_t i = 0; i < compiledModule->num_global_patches; ++i) {
    const char *name = compiledModule->global_patch_names[i];
    void *soAddr = compiledModule->global_patch_addrs[i];
    uint32_t size = compiledModule->global_patch_sizes[i];

    // Skip globals that live in the arena (already wired up).
    if (arenaGlobalNames.count(name)) {
      ++skippedArena;
      continue;
    }

    auto it = globalMemoryBlocks.find(name);
    if (it == globalMemoryBlocks.end()) {
      ++missed;
      continue;
    }

    const auto &block = it->second;
    uint32_t copySize = std::min(size, static_cast<uint32_t>(block.size));
    std::memcpy(soAddr, block.bytes(), copySize);
    ++patched;
  }

  llvm::errs() << "[circt-sim] Applied " << patched
               << " global patches (" << missed << " not found";
  if (skippedArena)
    llvm::errs() << ", " << skippedArena << " in arena";
  llvm::errs() << ")\n";
}

void CompiledModuleLoader::aliasGlobals(
    llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const {
  if (!compiledModule || compiledModule->num_global_patches == 0)
    return;

  unsigned newlyAliased = 0, alreadyAliased = 0, missed = 0, skippedArena = 0;
  for (uint32_t i = 0; i < compiledModule->num_global_patches; ++i) {
    const char *name = compiledModule->global_patch_names[i];
    void *soAddr = compiledModule->global_patch_addrs[i];
    uint32_t soSize = compiledModule->global_patch_sizes[i];

    // Skip globals that live in the arena.
    if (arenaGlobalNames.count(name)) {
      ++skippedArena;
      continue;
    }

    auto it = globalMemoryBlocks.find(name);
    if (it == globalMemoryBlocks.end()) {
      ++missed;
      continue;
    }

    // Skip globals that were already pre-aliased before initialization.
    if (it->second.aliasedStorage) {
      ++alreadyAliased;
      continue;
    }

    it->second.aliasTo(soAddr, soSize);
    ++newlyAliased;
  }

  llvm::errs() << "[circt-sim] Aliased " << newlyAliased
               << " globals to .so storage";
  if (alreadyAliased)
    llvm::errs() << " (" << alreadyAliased << " already pre-aliased, skipped)";
  if (skippedArena)
    llvm::errs() << " (" << skippedArena << " in arena, skipped)";
  if (missed)
    llvm::errs() << " (" << missed << " not found in interpreter)";
  llvm::errs() << "\n";
}

void CompiledModuleLoader::preAliasGlobals(
    llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const {
  if (!compiledModule || compiledModule->num_global_patches == 0)
    return;

  unsigned preAliased = 0, skippedArena = 0;
  for (uint32_t i = 0; i < compiledModule->num_global_patches; ++i) {
    const char *name = compiledModule->global_patch_names[i];
    void *soAddr = compiledModule->global_patch_addrs[i];
    uint32_t soSize = compiledModule->global_patch_sizes[i];

    // Skip globals that live in the arena.
    if (arenaGlobalNames.count(name)) {
      ++skippedArena;
      continue;
    }

    // Create a pre-aliased MemoryBlock pointing directly to .so storage.
    // initializeGlobals() will detect this and write initializer data
    // directly into .so storage rather than creating a separate copy.
    MemoryBlock block;
    block.preAlias(soAddr, soSize);
    globalMemoryBlocks[name] = std::move(block);
    ++preAliased;
  }

  llvm::errs() << "[circt-sim] Pre-aliased " << preAliased
               << " globals to .so storage";
  if (skippedArena)
    llvm::errs() << " (" << skippedArena << " in arena, skipped)";
  llvm::errs() << "\n";
}

void CompiledModuleLoader::setupArenaGlobals(
    llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const {
  if (!compiledModule || !arenaBase ||
      compiledModule->num_arena_globals == 0)
    return;

  unsigned mapped = 0, preserved = 0;
  for (uint32_t i = 0; i < compiledModule->num_arena_globals; ++i) {
    const char *name = compiledModule->arena_global_names[i];
    uint32_t offset = compiledModule->arena_global_offsets[i];
    uint32_t size = compiledModule->arena_global_sizes[i];
    if (!name)
      continue;

    // Point the MemoryBlock directly into the arena allocation.
    void *addr = static_cast<char *>(arenaBase) + offset;

    // Preserve any already-initialized interpreter state when rebasing an
    // existing global onto arena storage. This keeps pre-init writes coherent.
    auto existingIt = globalMemoryBlocks.find(name);
    if (existingIt != globalMemoryBlocks.end()) {
      const MemoryBlock &existing = existingIt->second;
      if (existing.bytes() != addr) {
        uint32_t copySize =
            std::min(size, static_cast<uint32_t>(existing.size));
        if (copySize > 0)
          std::memcpy(addr, existing.bytes(), copySize);
      }
      if (existing.initialized)
        ++preserved;
    }

    MemoryBlock block;
    block.preAlias(addr, size);
    if (existingIt != globalMemoryBlocks.end())
      block.initialized = existingIt->second.initialized;
    globalMemoryBlocks[name] = std::move(block);
    ++mapped;
  }

  llvm::errs() << "[circt-sim] Mapped " << mapped
               << " arena globals to memory blocks";
  if (preserved)
    llvm::errs() << " (" << preserved << " preserved from preexisting state)";
  llvm::errs() << "\n";
}
