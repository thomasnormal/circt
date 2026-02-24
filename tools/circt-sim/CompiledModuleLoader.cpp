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
#include <cstring>
#include <dlfcn.h>

using namespace circt;
using namespace circt::sim;

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

  // Validate ABI version.
  if (loader->compiledModule->abi_version != CIRCT_SIM_ABI_VERSION) {
    llvm::errs() << "[circt-sim] ABI version mismatch: .so has v"
                 << loader->compiledModule->abi_version << ", runtime expects v"
                 << CIRCT_SIM_ABI_VERSION << "\n";
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

  llvm::errs() << "[circt-sim] Loaded compiled module '" << path << "': "
               << loader->funcMap.size() << " functions";
  if (numTrampolines > 0)
    llvm::errs() << ", " << numTrampolines << " trampolines";
  if (loader->compiledModule->num_procs > 0)
    llvm::errs() << ", " << loader->compiledModule->num_procs << " processes";
  if (!loader->buildId.empty())
    llvm::errs() << " (build: " << loader->buildId << ")";
  llvm::errs() << "\n";

  return loader;
}

CompiledModuleLoader::~CompiledModuleLoader() {
  if (dlHandle)
    dlclose(dlHandle);
}

void *CompiledModuleLoader::lookupFunction(llvm::StringRef name) const {
  auto it = funcMap.find(name);
  if (it != funcMap.end())
    return it->second;
  return nullptr;
}

void CompiledModuleLoader::applyGlobalPatches(
    const llvm::StringMap<MemoryBlock> &globalMemoryBlocks) const {
  if (!compiledModule || compiledModule->num_global_patches == 0)
    return;

  unsigned patched = 0, missed = 0;
  for (uint32_t i = 0; i < compiledModule->num_global_patches; ++i) {
    const char *name = compiledModule->global_patch_names[i];
    void *soAddr = compiledModule->global_patch_addrs[i];
    uint32_t size = compiledModule->global_patch_sizes[i];

    auto it = globalMemoryBlocks.find(name);
    if (it == globalMemoryBlocks.end()) {
      ++missed;
      continue;
    }

    const auto &block = it->second;
    uint32_t copySize = std::min(size, static_cast<uint32_t>(block.size));
    std::memcpy(soAddr, block.data.data(), copySize);
    ++patched;
  }

  llvm::errs() << "[circt-sim] Applied " << patched
               << " global patches (" << missed << " not found)\n";
}
