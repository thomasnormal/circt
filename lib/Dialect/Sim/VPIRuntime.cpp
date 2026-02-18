//===- VPIRuntime.cpp - VPI Runtime Support for Simulation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Verilog Procedural Interface (VPI) runtime for
// circt-sim. Bridges VPI calls to CIRCT's ProcessScheduler.
//
// Based on IEEE 1800-2017 Section 36 (PLI/VPI).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/VPIRuntime.h"
#include "circt/Runtime/VPIDispatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstring>
#include <dlfcn.h>

#define DEBUG_TYPE "vpi-runtime"

using namespace circt::sim;

// gVPIDispatch is defined in MooreRuntime.cpp (so that linking MooreRuntime
// alone doesn't pull in a VPIRuntime dependency).

//===----------------------------------------------------------------------===//
// Handle Arena — allocates handle storage that persists for the runtime's
// lifetime. Each handle is a uint32_t* pointing into the arena.
//===----------------------------------------------------------------------===//

namespace {
/// Simple arena for VPI handle storage. Each handle is a pointer to a
/// uint32_t containing the object ID.
class HandleArena {
public:
  vpiHandle allocate(uint32_t id) {
    auto *ptr = new uint32_t(id);
    handles.push_back(ptr);
    return static_cast<vpiHandle>(ptr);
  }

  ~HandleArena() {
    for (auto *ptr : handles)
      delete ptr;
  }

private:
  std::vector<uint32_t *> handles;
};

static HandleArena &getArena() {
  static HandleArena arena;
  return arena;
}
} // namespace

//===----------------------------------------------------------------------===//
// VPIRuntime Implementation
//===----------------------------------------------------------------------===//

VPIRuntime::VPIRuntime() = default;
VPIRuntime::~VPIRuntime() = default;

VPIRuntime &VPIRuntime::getInstance() {
  static VPIRuntime instance;
  return instance;
}

vpiHandle VPIRuntime::makeHandle(uint32_t id) {
  return getArena().allocate(id);
}

uint32_t VPIRuntime::getHandleId(vpiHandle h) {
  if (!h)
    return 0;
  return *static_cast<uint32_t *>(h);
}

uint32_t VPIRuntime::nextObjectId() { return nextObjId++; }
uint32_t VPIRuntime::nextCallbackId() { return nextCbId++; }

//===----------------------------------------------------------------------===//
// Object Management
//===----------------------------------------------------------------------===//

uint32_t VPIRuntime::registerModule(const std::string &name,
                                     const std::string &fullName,
                                     uint32_t parentId) {
  uint32_t id = nextObjectId();
  auto obj = std::make_unique<VPIObject>();
  obj->id = id;
  obj->type = VPIObjectType::Module;
  obj->name = name;
  obj->fullName = fullName;
  obj->parentId = parentId;
  if (parentId != 0) {
    auto *parent = findById(parentId);
    if (parent)
      parent->children.push_back(id);
  } else {
    rootModules.push_back(id);
  }
  nameToId[fullName] = id;
  objects[id] = std::move(obj);
  stats.objectsCreated++;
  return id;
}

uint32_t VPIRuntime::registerSignal(const std::string &name,
                                     const std::string &fullName,
                                     SignalId signalId, uint32_t width,
                                     VPIObjectType type,
                                     uint32_t parentModuleId) {
  uint32_t id = nextObjectId();
  auto obj = std::make_unique<VPIObject>();
  obj->id = id;
  obj->type = type;
  obj->name = name;
  obj->fullName = fullName;
  obj->signalId = signalId;
  obj->width = width;
  obj->parentId = parentModuleId;
  if (parentModuleId != 0) {
    auto *parent = findById(parentModuleId);
    if (parent)
      parent->children.push_back(id);
  }
  nameToId[fullName] = id;
  signalToObjectIds[signalId].push_back(id);
  objects[id] = std::move(obj);
  stats.objectsCreated++;
  return id;
}

uint32_t VPIRuntime::registerParameter(const std::string &name,
                                       const std::string &fullName,
                                       int64_t value, uint32_t width,
                                       uint32_t parentModuleId) {
  uint32_t id = nextObjectId();
  auto obj = std::make_unique<VPIObject>();
  obj->id = id;
  obj->type = VPIObjectType::Parameter;
  obj->name = name;
  obj->fullName = fullName;
  obj->paramValue = value;
  obj->width = width;
  obj->parentId = parentModuleId;
  if (parentModuleId != 0) {
    auto *parent = findById(parentModuleId);
    if (parent)
      parent->children.push_back(id);
  }
  nameToId[fullName] = id;
  objects[id] = std::move(obj);
  stats.objectsCreated++;
  return id;
}

VPIObject *VPIRuntime::findByName(const std::string &fullName) {
  auto it = nameToId.find(fullName);
  if (it == nameToId.end())
    return nullptr;
  return findById(it->second);
}

VPIObject *VPIRuntime::findById(uint32_t id) {
  auto it = objects.find(id);
  if (it == objects.end())
    return nullptr;
  return it->second.get();
}

//===----------------------------------------------------------------------===//
// Hierarchy Building
//===----------------------------------------------------------------------===//

void VPIRuntime::buildHierarchy() {
  if (!scheduler)
    return;

  LLVM_DEBUG(llvm::dbgs() << "VPIRuntime: Building hierarchy from scheduler\n");

  // Determine the default module name from --top flags.
  std::string defaultModuleName =
      topModuleNames.empty() ? "top" : topModuleNames[0];

  // Create top-level module objects for each registered module.
  // The scheduler provides signal names in hierarchical "inst.signal" format.
  // We build a proper parent-child tree: top-level signals belong to the
  // default module, and dotted prefixes like "i_module_a.sig" create child
  // module scopes under the default module.
  llvm::StringMap<uint32_t> moduleIds;

  // Helper: ensure a module exists for the given hierarchical path, creating
  // intermediate scopes as needed.  Returns the module's VPI object id.
  auto ensureModule = [&](llvm::StringRef path) -> uint32_t {
    auto it = moduleIds.find(path);
    if (it != moduleIds.end())
      return it->second;

    // Walk the path components and create missing intermediates.
    // E.g. for "top.i_module_a.sub", create "top", then "top.i_module_a",
    // then "top.i_module_a.sub", each as a child of its parent.
    llvm::SmallVector<llvm::StringRef, 4> parts;
    path.split(parts, '.');
    std::string accum;
    uint32_t parentId = 0;
    for (auto part : parts) {
      if (accum.empty())
        accum = part.str();
      else
        accum = accum + "." + part.str();
      auto modIt = moduleIds.find(accum);
      if (modIt != moduleIds.end()) {
        parentId = modIt->second;
      } else {
        uint32_t mid = registerModule(part.str(), accum, parentId);
        moduleIds[accum] = mid;
        LLVM_DEBUG(llvm::dbgs()
                   << "  Created module '" << accum << "' id=" << mid
                   << " parent=" << parentId << "\n");
        parentId = mid;
      }
    }
    return parentId;
  };

  for (const auto &entry : scheduler->getSignalNames()) {
    SignalId sigId = entry.first;
    llvm::StringRef fullName = entry.second;
    const SignalValue &sv = scheduler->getSignalValue(sigId);
    uint32_t width = sv.getWidth();
    // Use the pre-computed logical width if available (strips 4-state overhead
    // from both simple and nested struct types).  Fall back to the old halving
    // heuristic for signals that don't have a logical width recorded.
    uint32_t logicalWidth = scheduler->getSignalLogicalWidth(sigId);
    if (logicalWidth > 0) {
      width = logicalWidth;
    } else {
      SignalEncoding enc = scheduler->getSignalEncoding(sigId);
      if (enc == SignalEncoding::FourStateStruct && width >= 2 &&
          (width % 2) == 0)
        width /= 2;
    }

    // Split "inst.sub.signal" into module path and signal name.
    // Signals without dots belong to the default module.
    auto dotPos = fullName.rfind('.');
    std::string modulePath, signalName;
    if (dotPos != llvm::StringRef::npos) {
      std::string prefix = fullName.substr(0, dotPos).str();
      signalName = fullName.substr(dotPos + 1).str();
      // Nest the instance under the default (top) module.
      modulePath = defaultModuleName + "." + prefix;
    } else {
      modulePath = defaultModuleName;
      signalName = fullName.str();
    }

    // Ensure module hierarchy exists.
    uint32_t moduleId = ensureModule(modulePath);

    // Check if this signal is an unpacked array.
    std::string qualifiedName = modulePath + "." + signalName;
    const auto *arrayInfo = scheduler->getSignalArrayInfo(sigId);
    if (arrayInfo && arrayInfo->numElements > 1) {
      // Create an Array parent object.
      uint32_t arrayObjId =
          registerSignal(signalName, qualifiedName, sigId, width,
                         VPIObjectType::Array, moduleId);
      if (nameToId.find(signalName) == nameToId.end())
        nameToId[signalName] = arrayObjId;

      // Create child Reg objects for each array element.
      uint32_t elemPhysW = arrayInfo->elementPhysWidth;
      uint32_t elemLogW = arrayInfo->elementLogicalWidth;
      // Determine the SV index range.  If bounds are available (from
      // vpi.array_bounds), use them; otherwise fall back to 0-based.
      int32_t leftBound = arrayInfo->leftBound;
      int32_t rightBound = arrayInfo->rightBound;
      bool hasBounds = (rightBound != -1);
      // Direction: downto if left > right, ascending if left < right.
      int32_t step = (hasBounds && leftBound > rightBound) ? -1 : 1;
      // Store bounds on the array parent for range queries.
      auto *arrayParent = findById(arrayObjId);
      if (arrayParent) {
        arrayParent->leftBound = hasBounds ? leftBound : 0;
        arrayParent->rightBound =
            hasBounds ? rightBound
                      : static_cast<int32_t>(arrayInfo->numElements - 1);
      }
      for (uint32_t i = 0; i < arrayInfo->numElements; ++i) {
        int32_t svIndex = hasBounds ? (leftBound + step * (int32_t)i)
                                    : (int32_t)i;
        std::string elemName =
            signalName + "[" + std::to_string(svIndex) + "]";
        std::string elemQName =
            qualifiedName + "[" + std::to_string(svIndex) + "]";
        uint32_t elemId = nextObjectId();
        auto elemObj = std::make_unique<VPIObject>();
        elemObj->id = elemId;
        elemObj->type = VPIObjectType::Reg;
        elemObj->name = elemName;
        elemObj->fullName = elemQName;
        elemObj->signalId = sigId;
        elemObj->width = elemLogW;
        // HW convention: element 0 is in the highest bits (field order
        // high-to-low). Bit offset = (N-1-i) * elemPhysWidth.
        elemObj->bitOffset =
            (arrayInfo->numElements - 1 - i) * elemPhysW;
        elemObj->parentId = arrayObjId;
        if (arrayParent)
          arrayParent->children.push_back(elemId);
        nameToId[elemQName] = elemId;
        if (nameToId.find(elemName) == nameToId.end())
          nameToId[elemName] = elemId;
        objects[elemId] = std::move(elemObj);
        stats.objectsCreated++;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  Created array '" << qualifiedName << "' id="
                 << arrayObjId << " sigId=" << sigId << " elements="
                 << arrayInfo->numElements << " elemWidth=" << elemLogW
                 << "\n");
    } else if (const auto *structFields =
                   scheduler->getSignalStructFields(sigId)) {
      // Unpacked struct: create a StructVar parent with field children.
      uint32_t structObjId =
          registerSignal(signalName, qualifiedName, sigId, width,
                         VPIObjectType::StructVar, moduleId);
      if (nameToId.find(signalName) == nameToId.end())
        nameToId[signalName] = structObjId;
      auto *structParent = findById(structObjId);

      // HW convention: first field is in the highest bits.
      // Accumulate physical offset from the top.
      uint32_t totalPhysWidth = 0;
      for (const auto &f : *structFields)
        totalPhysWidth += f.physicalWidth;
      uint32_t bitOff = totalPhysWidth;
      for (const auto &field : *structFields) {
        bitOff -= field.physicalWidth;
        std::string fieldName = field.name;
        std::string fieldQName = qualifiedName + "." + fieldName;
        uint32_t fieldId = nextObjectId();
        auto fieldObj = std::make_unique<VPIObject>();
        fieldObj->id = fieldId;
        fieldObj->type = VPIObjectType::Reg;
        fieldObj->name = fieldName;
        fieldObj->fullName = fieldQName;
        fieldObj->signalId = sigId;
        fieldObj->width = field.logicalWidth;
        fieldObj->bitOffset = bitOff;
        fieldObj->parentId = structObjId;
        if (structParent)
          structParent->children.push_back(fieldId);
        nameToId[fieldQName] = fieldId;
        objects[fieldId] = std::move(fieldObj);
        stats.objectsCreated++;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  Created struct '" << qualifiedName << "' id="
                 << structObjId << " sigId=" << sigId << " fields="
                 << structFields->size() << "\n");
    } else {
      // Register as a regular signal.
      uint32_t sigObjId =
          registerSignal(signalName, qualifiedName, sigId, width,
                         VPIObjectType::Reg, moduleId);
      if (nameToId.find(signalName) == nameToId.end())
        nameToId[signalName] = sigObjId;
      LLVM_DEBUG(llvm::dbgs() << "  Created signal '" << qualifiedName
                              << "' id=" << sigObjId << " sigId=" << sigId
                              << " width=" << width << "\n");
    }
  }

  // Ensure all --top modules exist even if they have no signals.
  for (const auto &modName : topModuleNames) {
    ensureModule(modName);
  }

  // Register child instance scopes that have no signals (so they appear
  // as VPI module children even without any registered signals).
  for (const auto &instPath : scheduler->getInstanceScopes()) {
    std::string fullPath = defaultModuleName + "." + instPath;
    ensureModule(fullPath);
  }

  // Register VPI parameters on child instance modules.
  for (const auto &param : scheduler->getVPIParameters()) {
    std::string instFullPath = defaultModuleName + "." + param.instancePath;
    auto modIt = moduleIds.find(instFullPath);
    if (modIt != moduleIds.end()) {
      uint32_t parentModId = modIt->second;
      std::string paramQName = instFullPath + "." + param.paramName;
      registerParameter(param.paramName, paramQName, param.value,
                        param.width, parentModId);
    }
  }

  // Process signal aliases (e.g., sv.namehint wire names).
  // These create additional nameToId entries so VPI can find signals by
  // their original SystemVerilog wire names (e.g., "counter_plus_two"
  // instead of "i_module_a.data_out").
  for (const auto &alias : scheduler->getSignalAliases()) {
    llvm::StringRef aliasName = alias.first;
    SignalId sigId = alias.second;
    // Find the VPI object for this signal.
    for (auto &objEntry : objects) {
      auto *obj = objEntry.second.get();
      if (obj && obj->signalId == sigId &&
          obj->type != VPIObjectType::Iterator &&
          obj->type != VPIObjectType::Callback &&
          obj->type != VPIObjectType::Module) {
        // Register the alias under the top module's qualified name.
        std::string qualAlias = defaultModuleName + "." + aliasName.str();
        nameToId[qualAlias] = obj->id;
        if (nameToId.find(aliasName) == nameToId.end())
          nameToId[aliasName] = obj->id;
        // Also add as a child of the top module so iterate() finds it.
        auto topIt = moduleIds.find(defaultModuleName);
        if (topIt != moduleIds.end()) {
          auto *topMod = findById(topIt->second);
          if (topMod) {
            // Check it's not already a child of top.
            bool alreadyChild = false;
            for (uint32_t cid : topMod->children) {
              if (cid == obj->id) {
                alreadyChild = true;
                break;
              }
            }
            if (!alreadyChild)
              topMod->children.push_back(obj->id);
          }
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  Alias '" << aliasName << "' -> signal id="
                   << obj->id << " (sigId=" << sigId << ")\n");
        break;
      }
    }
  }

  // Generate scope array support: detect instance names like "arr_1", "arr_2"
  // and create bracket-indexed aliases "arr[1]", "arr[2]" with a parent scope
  // "arr" that contains them. This reconstructs Verilog generate for-loop
  // hierarchy from CIRCT's flattened naming convention.
  {
    // Group module names by parent and detect numeric suffix patterns.
    // For each parent module, collect children and group by prefix.
    llvm::StringMap<llvm::SmallVector<std::pair<std::string, int>, 4>>
        prefixGroups;
    // Track underscore→bracket path mappings for nested fixup.
    llvm::StringMap<std::string> underscoreToBracket;
    for (auto &entry : moduleIds) {
      llvm::StringRef fullPath = entry.first();
      auto lastDot = fullPath.rfind('.');
      if (lastDot == llvm::StringRef::npos)
        continue;
      llvm::StringRef parentPath = fullPath.substr(0, lastDot);
      llvm::StringRef childName = fullPath.substr(lastDot + 1);

      // Check if childName matches "prefix_N" where N is a number.
      auto lastUnderscore = childName.rfind('_');
      if (lastUnderscore == llvm::StringRef::npos || lastUnderscore == 0)
        continue;
      llvm::StringRef prefix = childName.substr(0, lastUnderscore);
      llvm::StringRef suffix = childName.substr(lastUnderscore + 1);
      int index;
      if (suffix.getAsInteger(10, index))
        continue;

      std::string key = (parentPath + "." + prefix).str();
      prefixGroups[key].push_back({fullPath.str(), index});
    }

    // For groups with 2+ entries, create bracket-indexed aliases.
    for (auto &group : prefixGroups) {
      if (group.second.size() < 2)
        continue;

      llvm::StringRef arrayPath = group.first();
      // Ensure the parent array scope exists (GenScopeArray type).
      uint32_t arrayModId = 0;
      if (moduleIds.find(arrayPath) == moduleIds.end()) {
        auto parentDot = arrayPath.rfind('.');
        uint32_t parentId = 0;
        if (parentDot != llvm::StringRef::npos) {
          auto parentIt = moduleIds.find(arrayPath.substr(0, parentDot));
          if (parentIt != moduleIds.end())
            parentId = parentIt->second;
        }
        llvm::StringRef scopeName = arrayPath.substr(
            arrayPath.rfind('.') != llvm::StringRef::npos
                ? arrayPath.rfind('.') + 1
                : 0);
        arrayModId =
            registerModule(scopeName.str(), arrayPath.str(), parentId);
        moduleIds[arrayPath] = arrayModId;
        // Mark as GenScopeArray.
        auto *arrayObj = findById(arrayModId);
        if (arrayObj)
          arrayObj->type = VPIObjectType::GenScopeArray;
      } else {
        arrayModId = moduleIds[arrayPath];
      }

      // Create bracket-indexed aliases for each member.
      for (auto &member : group.second) {
        uint32_t childId = moduleIds[member.first];
        // Create alias: "parent.prefix[N]" → same module ID
        std::string bracketName =
            arrayPath.str() + "[" + std::to_string(member.second) + "]";
        moduleIds[bracketName] = childId;
        nameToId[bracketName] = childId;
        // Record the underscore→bracket mapping for nested path fixup.
        // e.g. "sample_module.outer_scope_1" → "sample_module.outer_scope[1]"
        underscoreToBracket[member.first] = bracketName;
        // Also register the short form for VPI by-name access.
        auto shortDot = bracketName.rfind('.');
        if (shortDot != llvm::StringRef::npos) {
          std::string shortName = bracketName.substr(shortDot + 1);
          if (nameToId.find(shortName) == nameToId.end())
            nameToId[shortName] = childId;
        }
        // Update the object's name to use bracket notation and GenScope type.
        auto *obj = findById(childId);
        if (obj) {
          llvm::StringRef origName = obj->name;
          auto underscore = origName.rfind('_');
          if (underscore != llvm::StringRef::npos) {
            llvm::StringRef origPrefix = origName.substr(0, underscore);
            llvm::StringRef origSuffix = origName.substr(underscore + 1);
            int origIdx;
            if (!origSuffix.getAsInteger(10, origIdx)) {
              obj->name = (origPrefix + "[" + std::to_string(origIdx) + "]").str();
              obj->fullName = bracketName;
            }
          }
          obj->type = VPIObjectType::GenScope;
          // Re-parent under the array scope.
          obj->parentId = arrayModId;
          auto *arrayMod = findById(arrayModId);
          if (arrayMod) {
            bool alreadyChild = false;
            for (uint32_t cid : arrayMod->children)
              if (cid == childId) { alreadyChild = true; break; }
            if (!alreadyChild)
              arrayMod->children.push_back(childId);
          }
          // Remove from old parent's children list.
          auto parentDot = member.first.rfind('.');
          if (parentDot != std::string::npos) {
            std::string oldParentPath = member.first.substr(0, parentDot);
            auto oldParentIt = moduleIds.find(oldParentPath);
            if (oldParentIt != moduleIds.end()) {
              auto *oldParent = findById(oldParentIt->second);
              if (oldParent) {
                oldParent->children.erase(
                    std::remove(oldParent->children.begin(),
                                oldParent->children.end(), childId),
                    oldParent->children.end());
              }
            }
          }
        }
      }

      // Set leftBound/rightBound on the GenScopeArray from children indices.
      auto *gsa = findById(arrayModId);
      if (gsa && !group.second.empty()) {
        int32_t minIdx = group.second[0].second;
        int32_t maxIdx = group.second[0].second;
        for (auto &m : group.second) {
          minIdx = std::min(minIdx, m.second);
          maxIdx = std::max(maxIdx, m.second);
        }
        gsa->leftBound = minIdx;
        gsa->rightBound = maxIdx;
      }
    }

    // Second pass: create bracket-form aliases for nested paths.
    // E.g. if we mapped "sample_module.outer_scope_1" →
    //   "sample_module.outer_scope[1]", then any nameToId/moduleIds entry
    //   containing "outer_scope_1." as a path component should also get an
    //   alias with "outer_scope[1]." substituted.
    if (!underscoreToBracket.empty()) {
      llvm::StringMap<uint32_t> extraNameEntries;
      llvm::StringMap<uint32_t> extraModuleEntries;
      for (auto &mapping : underscoreToBracket) {
        std::string underscorePrefix = mapping.first().str() + ".";
        std::string bracketPrefix = mapping.second + ".";
        // Scan nameToId for entries with the underscore prefix.
        for (auto &ne : nameToId) {
          llvm::StringRef path = ne.first();
          if (path.contains(underscorePrefix)) {
            std::string newPath = path.str();
            size_t pos = newPath.find(underscorePrefix);
            while (pos != std::string::npos) {
              newPath.replace(pos, underscorePrefix.size(),
                              bracketPrefix);
              pos = newPath.find(underscorePrefix,
                                 pos + bracketPrefix.size());
            }
            extraNameEntries[newPath] = ne.second;
          }
        }
        // Scan moduleIds too.
        for (auto &me : moduleIds) {
          llvm::StringRef path = me.first();
          if (path.contains(underscorePrefix)) {
            std::string newPath = path.str();
            size_t pos = newPath.find(underscorePrefix);
            while (pos != std::string::npos) {
              newPath.replace(pos, underscorePrefix.size(),
                              bracketPrefix);
              pos = newPath.find(underscorePrefix,
                                 pos + bracketPrefix.size());
            }
            extraModuleEntries[newPath] = me.second;
          }
        }
      }
      for (auto &e : extraNameEntries)
        nameToId[e.first()] = e.second;
      for (auto &e : extraModuleEntries)
        moduleIds[e.first()] = e.second;
    }
  }

  // Build the nameToSiblingSignals map for port-to-internal propagation.
  // When multiple signals share the same base name (e.g., a hw.module port
  // "mode_in" and an llhd.sig name "mode_in"), VPI writes to one should
  // propagate to all siblings so that the combinational logic sees the update.
  nameToSiblingSignals.clear();
  for (const auto &entry : scheduler->getSignalNames()) {
    llvm::StringRef name = entry.second;
    // Use the base signal name (after the last dot, if any).
    auto dotPos = name.rfind('.');
    std::string baseName =
        dotPos != llvm::StringRef::npos ? name.substr(dotPos + 1).str()
                                        : name.str();
    nameToSiblingSignals[baseName].push_back(entry.first);
  }
  // Remove entries with only one signal (no siblings to propagate to).
  llvm::SmallVector<llvm::StringRef, 16> toRemove;
  for (auto &entry : nameToSiblingSignals) {
    if (entry.second.size() <= 1)
      toRemove.push_back(entry.first());
  }
  for (auto &key : toRemove)
    nameToSiblingSignals.erase(key);

  LLVM_DEBUG({
    for (auto &entry : nameToSiblingSignals) {
      llvm::dbgs() << "VPIRuntime: Sibling signals for '" << entry.first()
                    << "': ";
      for (SignalId s : entry.second)
        llvm::dbgs() << s << " ";
      llvm::dbgs() << "\n";
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "VPIRuntime: Built hierarchy with "
                          << objects.size() << " objects\n");

}

//===----------------------------------------------------------------------===//
// Library Loading
//===----------------------------------------------------------------------===//

bool VPIRuntime::loadVPILibrary(const std::string &path) {
  LLVM_DEBUG(llvm::dbgs() << "VPIRuntime: Loading VPI library '" << path
                          << "'\n");

  // dlopen with RTLD_NOW | RTLD_GLOBAL so that the library can resolve
  // vpi_* symbols from the main binary.
  std::string errMsg;
  auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(),
                                                             &errMsg);
  if (!lib.isValid()) {
    lastErrorMessage = "Failed to load VPI library: " + errMsg;
    lastErrorLevel = vpiError;
    llvm::errs() << "[VPI] " << lastErrorMessage << "\n";
    return false;
  }

  loadedLibraries.push_back(lib);
  active = true;

  // Look for vlog_startup_routines_bootstrap (single entry point).
  using BootstrapFn = void (*)();
  auto *bootstrap = reinterpret_cast<BootstrapFn>(
      lib.getAddressOfSymbol("vlog_startup_routines_bootstrap"));
  if (bootstrap) {
    LLVM_DEBUG(llvm::dbgs()
               << "VPIRuntime: Calling vlog_startup_routines_bootstrap\n");
    bootstrap();
    return true;
  }

  // Fallback: look for vlog_startup_routines array.
  // The symbol is an array of function pointers: void (*routines[])() = {..., NULL};
  using StartupRoutine = void (*)();
  auto *routines = reinterpret_cast<StartupRoutine *>(
      lib.getAddressOfSymbol("vlog_startup_routines"));
  if (routines) {
    LLVM_DEBUG(llvm::dbgs()
               << "VPIRuntime: Calling vlog_startup_routines\n");
    for (size_t i = 0; routines[i]; ++i)
      routines[i]();
    return true;
  }

  lastErrorMessage = "VPI library has no startup routines";
  lastErrorLevel = vpiWarning;
  llvm::errs() << "[VPI] Warning: " << lastErrorMessage << "\n";
  return true; // Library loaded but no startup routines.
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Handle Access
//===----------------------------------------------------------------------===//

uint32_t VPIRuntime::handleByName(const char *name, uint32_t scopeId) {
  if (!name || !*name)
    return 0;

  std::string fullName;
  if (scopeId != 0) {
    auto *scope = findById(scopeId);
    if (scope)
      fullName = scope->fullName + "." + name;
    else
      fullName = name;
  } else {
    fullName = name;
  }

  auto *obj = findByName(fullName);
  if (obj)
    return obj->id;

  // Try without scope prefix.
  obj = findByName(std::string(name));
  if (obj)
    return obj->id;

  return 0;
}

uint32_t VPIRuntime::handleByIndex(uint32_t objectId, int32_t index) {
  auto *obj = findById(objectId);
  if (!obj)
    return 0;

  // For GenScopeArray and Array, the index is the actual SV index (e.g., 7
  // for array_7_downto_4[7]) not a 0-based vector position.  Find the child
  // whose name ends with [index].
  if (obj->type == VPIObjectType::GenScopeArray ||
      obj->type == VPIObjectType::Array) {
    std::string suffix = "[" + std::to_string(index) + "]";
    for (uint32_t childId : obj->children) {
      auto *child = findById(childId);
      if (child && llvm::StringRef(child->name).ends_with(suffix))
        return childId;
    }
    return 0;
  }

  if (index < 0 || static_cast<size_t>(index) >= obj->children.size())
    return 0;
  return obj->children[index];
}

uint32_t VPIRuntime::handle(int32_t type, uint32_t refId) {
  auto *obj = findById(refId);
  if (!obj)
    return 0;

  if (type == vpiScope || type == vpiModule) {
    // Return parent module.
    return obj->parentId;
  }

  // Return range bound handles for arrays and GenScopeArrays. cocotb calls
  // vpi_handle(vpiLeftRange, array) then vpi_get_value() on the result.
  if ((type == vpiLeftRange || type == vpiRightRange) &&
      (obj->type == VPIObjectType::Array ||
       obj->type == VPIObjectType::GenScopeArray)) {
    // Look up or lazily create the range bound parameter object.
    std::string boundName =
        obj->name + (type == vpiLeftRange ? ".__left" : ".__right");
    auto it = nameToId.find(boundName);
    if (it != nameToId.end())
      return it->second;
    int64_t boundValue;
    if (obj->type == VPIObjectType::GenScopeArray) {
      // Extract min/max indices from children names (e.g., "arr[1]" → 1).
      int64_t minIdx = INT64_MAX, maxIdx = INT64_MIN;
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (!child)
          continue;
        auto lbracket = child->name.rfind('[');
        auto rbracket = child->name.rfind(']');
        if (lbracket != std::string::npos && rbracket != std::string::npos) {
          int64_t idx;
          if (!llvm::StringRef(child->name)
                   .substr(lbracket + 1, rbracket - lbracket - 1)
                   .getAsInteger(10, idx)) {
            minIdx = std::min(minIdx, idx);
            maxIdx = std::max(maxIdx, idx);
          }
        }
      }
      boundValue = type == vpiLeftRange ? minIdx : maxIdx;
    } else {
      // Use stored SV bounds if available.
      boundValue = type == vpiLeftRange
                       ? static_cast<int64_t>(obj->leftBound)
                       : static_cast<int64_t>(obj->rightBound);
    }
    return registerParameter(boundName, boundName, boundValue, 32, obj->id);
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Iteration
//===----------------------------------------------------------------------===//

uint32_t VPIRuntime::iterate(int32_t type, uint32_t refId) {
  llvm::SmallVector<uint32_t, 8> elements;

  if (refId == 0) {
    // Iterate top-level modules/instances.
    if (type == vpiModule || type == vpiInstance) {
      elements.append(rootModules.begin(), rootModules.end());
    }
  } else {
    auto *obj = findById(refId);
    if (!obj)
      return 0;

    if (type == vpiModule || type == vpiInternalScope || type == vpiInstance) {
      // Return child modules/instances/generate scopes.
      // For GenScopeArray: don't return the array itself (it would be
      // GPI_ARRAY/ArrayObject with range issues); instead flatten its
      // GenScope children into the parent's results.  cocotb's fallback
      // in get_child_by_name will auto-create the pseudo-region parent.
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (!child)
          continue;
        if (child->type == VPIObjectType::Module ||
            child->type == VPIObjectType::GenScope)
          elements.push_back(childId);
        else if (child->type == VPIObjectType::GenScopeArray) {
          // Flatten: include the GenScope children instead.
          for (uint32_t gcId : child->children) {
            auto *gc = findById(gcId);
            if (gc && gc->type == VPIObjectType::GenScope)
              elements.push_back(gcId);
          }
        }
      }
    } else if (type == vpiGenScopeArray) {
      // Return child generate scope arrays.
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (child && child->type == VPIObjectType::GenScopeArray)
          elements.push_back(childId);
      }
    } else if (type == vpiStructVar || type == vpiStructNet) {
      // Return only struct-typed child signals.
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (child && child->type == VPIObjectType::StructVar)
          elements.push_back(childId);
      }
    } else if (type == vpiNet || type == vpiReg) {
      // Return child signals (including arrays, but NOT structs).
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (child && (child->type == VPIObjectType::Net ||
                      child->type == VPIObjectType::Reg ||
                      child->type == VPIObjectType::Array))
          elements.push_back(childId);
      }
    } else if (type == vpiMember) {
      // Return struct member fields.
      if (obj->type == VPIObjectType::StructVar) {
        for (uint32_t childId : obj->children) {
          auto *child = findById(childId);
          if (child)
            elements.push_back(childId);
        }
      }
    } else if (type == vpiRegArray) {
      // Return child arrays only.
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (child && child->type == VPIObjectType::Array)
          elements.push_back(childId);
      }
    } else if (type == vpiParameter) {
      // Return child parameters.
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (child && child->type == VPIObjectType::Parameter)
          elements.push_back(childId);
      }
    }
  }

  if (elements.empty())
    return 0;

  auto iter = std::make_unique<VPIIterator>();
  uint32_t iterId = nextObjectId();
  iter->id = iterId;
  iter->elements.assign(elements.begin(), elements.end());
  iter->currentIndex = 0;
  iterators[iterId] = std::move(iter);
  return iterId;
}

uint32_t VPIRuntime::scan(uint32_t iteratorId) {
  auto it = iterators.find(iteratorId);
  if (it == iterators.end())
    return 0;

  auto &iter = *it->second;
  if (iter.currentIndex >= iter.elements.size()) {
    iterators.erase(it); // Auto-free exhausted iterator.
    return 0;
  }
  return iter.elements[iter.currentIndex++];
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Properties
//===----------------------------------------------------------------------===//

int32_t VPIRuntime::getProperty(int32_t property, uint32_t objectId) {
  // Handle global properties that don't require an object.
  if (property == vpiTimePrecision)
    return -12; // ps (picoseconds) — matches `timescale 1ns/1ps`.
  if (property == vpiTimeUnit)
    return -9; // ns (nanoseconds) — matches `timescale 1ns/1ps`.

  auto *obj = findById(objectId);
  if (!obj)
    return vpiUndefined;

  switch (property) {
  case vpiType:
    switch (obj->type) {
    case VPIObjectType::Module:
      return vpiModule;
    case VPIObjectType::Net:
      return vpiNet;
    case VPIObjectType::Reg:
      return vpiReg;
    case VPIObjectType::Port:
      return vpiPort;
    case VPIObjectType::Parameter:
      return vpiParameter;
    case VPIObjectType::Array:
      return vpiRegArray;
    case VPIObjectType::GenScope:
      return vpiGenScope;
    case VPIObjectType::GenScopeArray:
      return vpiGenScopeArray;
    case VPIObjectType::StructVar:
      return vpiStructVar;
    default:
      return vpiUndefined;
    }
  case vpiSize:
    if (obj->type == VPIObjectType::Array ||
        obj->type == VPIObjectType::GenScopeArray)
      return static_cast<int32_t>(obj->children.size());
    return static_cast<int32_t>(obj->width);
  case vpiLeftRange:
    if (obj->type == VPIObjectType::GenScopeArray ||
        obj->type == VPIObjectType::Array)
      return obj->leftBound;
    return obj->width > 0 ? static_cast<int32_t>(obj->width - 1) : 0;
  case vpiRightRange:
    if (obj->type == VPIObjectType::GenScopeArray ||
        obj->type == VPIObjectType::Array)
      return obj->rightBound;
    return 0;
  case vpiDirection:
    return obj->direction;
  case vpiVector:
    return obj->width > 1 ? 1 : 0;
  case vpiScalar:
    return obj->width == 1 ? 1 : 0;
  case vpiSigned:
    return 0; // Unsigned by default.
  case vpiPacked:
    return 0; // Unpacked structs only; packed structs stay as plain Reg.
  case vpiTopModule:
    return obj->parentId == 0 ? 1 : 0;
  default:
    return vpiUndefined;
  }
}

const char *VPIRuntime::getStrProperty(int32_t property, uint32_t objectId) {
  auto *obj = findById(objectId);
  if (!obj) {
    strBuffer.clear();
    return strBuffer.c_str();
  }

  switch (property) {
  case vpiName:
    strBuffer = obj->name;
    break;
  case vpiFullName:
    strBuffer = obj->fullName;
    break;
  case vpiDefName:
    strBuffer = obj->name; // Use name as defName for modules.
    break;
  case vpiType:
    switch (obj->type) {
    case VPIObjectType::Module:
      strBuffer = "vpiModule";
      break;
    case VPIObjectType::Net:
      strBuffer = "vpiNet";
      break;
    case VPIObjectType::Reg:
      strBuffer = "vpiReg";
      break;
    case VPIObjectType::Array:
      strBuffer = "vpiRegArray";
      break;
    case VPIObjectType::GenScope:
      strBuffer = "vpiGenScope";
      break;
    case VPIObjectType::GenScopeArray:
      strBuffer = "vpiGenScopeArray";
      break;
    case VPIObjectType::StructVar:
      strBuffer = "vpiStructVar";
      break;
    default:
      strBuffer = "vpiUndefined";
    }
    break;
  default:
    strBuffer.clear();
  }
  return strBuffer.c_str();
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Value Access
//===----------------------------------------------------------------------===//

void VPIRuntime::getValue(uint32_t objectId, struct t_vpi_value *value) {
  stats.valueReads++;
  auto *obj = findById(objectId);
  if (!obj || !value)
    return;

  static bool traceVPI = []() {
    const char *env = std::getenv("CIRCT_VPI_TRACE");
    return env && env[0] != '\0' && env[0] != '0';
  }();

  // Handle parameters: return the elaborated constant value.
  if (obj->type == VPIObjectType::Parameter) {
    switch (value->format) {
    case vpiIntVal:
      value->value.integer = static_cast<PLI_INT32>(obj->paramValue);
      break;
    case vpiDecStrVal: {
      strBuffer = std::to_string(obj->paramValue);
      value->value.str = const_cast<PLI_BYTE8 *>(strBuffer.c_str());
      break;
    }
    case vpiBinStrVal: {
      uint32_t w = obj->width > 0 ? obj->width : 32;
      strBuffer.clear();
      strBuffer.reserve(w);
      for (int i = static_cast<int>(w) - 1; i >= 0; --i)
        strBuffer.push_back((obj->paramValue >> i) & 1 ? '1' : '0');
      value->value.str = const_cast<PLI_BYTE8 *>(strBuffer.c_str());
      break;
    }
    case vpiHexStrVal: {
      llvm::SmallString<64> hexStr;
      llvm::APInt(obj->width > 0 ? obj->width : 32, obj->paramValue)
          .toString(hexStr, 16, /*Signed=*/false);
      strBuffer = std::string(hexStr.begin(), hexStr.end());
      value->value.str = const_cast<PLI_BYTE8 *>(strBuffer.c_str());
      break;
    }
    default:
      value->value.integer = static_cast<PLI_INT32>(obj->paramValue);
      break;
    }
    return;
  }

  if (!obj->signalId || !scheduler)
    return;

  const SignalValue &sv = scheduler->getSignalValue(obj->signalId);
  uint32_t width = obj->width; // Logical width (already halved for 4-state).

  // For array element sub-signals, extract the element's bits from the parent
  // signal and determine 4-state encoding from the element's physical width.
  SignalEncoding enc = scheduler->getSignalEncoding(obj->signalId);
  bool isFourState = (enc == SignalEncoding::FourStateStruct);
  llvm::APInt valueBits(width, 0);
  llvm::APInt unknownBits(width, 0);
  bool hasUnknown = false;

  // Check if this is an array element (not the array parent itself).
  const auto *arrayInfo = scheduler->getSignalArrayInfo(obj->signalId);
  if (arrayInfo && obj->type != VPIObjectType::Array) {
    // Array element: extract element bits from the parent signal.
    {
      uint32_t elemPhysW = arrayInfo->elementPhysWidth;
      const llvm::APInt &raw = sv.getAPInt();
      if (obj->bitOffset + elemPhysW <= raw.getBitWidth()) {
        llvm::APInt elemBits = raw.extractBits(elemPhysW, obj->bitOffset);
        // Check if element is 4-state (elemPhysW = 2 * elemLogicalW).
        if (elemPhysW == width * 2) {
          valueBits = elemBits.extractBits(width, width);
          unknownBits = elemBits.extractBits(width, 0);
          hasUnknown = !unknownBits.isZero();
        } else {
          valueBits = elemBits.zextOrTrunc(width);
        }
      }
    }
  } else if (isFourState) {
    const llvm::APInt &raw = sv.getAPInt();
    uint32_t physWidth = raw.getBitWidth();
    if (physWidth >= width * 2) {
      // Upper N bits = value, lower N bits = unknown flags.
      valueBits = raw.extractBits(width, width);
      unknownBits = raw.extractBits(width, 0);
      hasUnknown = !unknownBits.isZero();
    } else {
      valueBits = raw.zextOrTrunc(width);
    }
  } else {
    valueBits = sv.getAPInt().zextOrTrunc(width);
    hasUnknown = sv.isUnknown();
  }


  switch (value->format) {
  case vpiBinStrVal: {
    strBuffer.clear();
    strBuffer.reserve(width);
    for (int i = static_cast<int>(width) - 1; i >= 0; --i) {
      if (isFourState && unknownBits[i])
        strBuffer.push_back('x');
      else if (!isFourState && hasUnknown)
        strBuffer.push_back('x');
      else
        strBuffer.push_back(valueBits[i] ? '1' : '0');
    }
    value->value.str = const_cast<PLI_BYTE8 *>(strBuffer.c_str());
    break;
  }
  case vpiHexStrVal: {
    strBuffer.clear();
    if (hasUnknown) {
      strBuffer.assign((width + 3) / 4, 'x');
    } else {
      llvm::SmallString<64> hexStr;
      valueBits.toString(hexStr, 16, /*Signed=*/false);
      strBuffer = std::string(hexStr.begin(), hexStr.end());
    }
    value->value.str = const_cast<PLI_BYTE8 *>(strBuffer.c_str());
    break;
  }
  case vpiDecStrVal: {
    strBuffer.clear();
    if (hasUnknown) {
      strBuffer = "x";
    } else {
      llvm::SmallString<64> decStr;
      valueBits.toString(decStr, 10, /*Signed=*/false);
      strBuffer = std::string(decStr.begin(), decStr.end());
    }
    value->value.str = const_cast<PLI_BYTE8 *>(strBuffer.c_str());
    break;
  }
  case vpiIntVal:
    if (hasUnknown)
      value->value.integer = 0;
    else
      value->value.integer =
          static_cast<PLI_INT32>(valueBits.getZExtValue());
    break;
  case vpiScalarVal:
    if (hasUnknown)
      value->value.scalar = vpiX;
    else
      value->value.scalar = valueBits.getBoolValue() ? vpi1 : vpi0;
    break;
  case vpiVectorVal: {
    if (!value->value.vector)
      break;
    // For four-state, pack aval/bval from value and unknown bits.
    uint32_t numWords = (width + 31) / 32;
    for (uint32_t i = 0; i < numWords; ++i) {
      uint32_t bitsThisWord = std::min(32u, width - i * 32);
      value->value.vector[i].aval = static_cast<PLI_UINT32>(
          valueBits.extractBitsAsZExtValue(bitsThisWord, i * 32));
      if (isFourState)
        value->value.vector[i].bval = static_cast<PLI_UINT32>(
            unknownBits.extractBitsAsZExtValue(bitsThisWord, i * 32));
      else
        value->value.vector[i].bval = hasUnknown ? 0xFFFFFFFF : 0;
    }
    break;
  }
  default:
    break;
  }

  if (traceVPI && obj->signalId) {
    const SignalValue &svDbg = scheduler->getSignalValue(obj->signalId);
    llvm::SmallString<64> rawHex, valHex;
    svDbg.getAPInt().toString(rawHex, 16, false);
    valueBits.toString(valHex, 16, false);
    llvm::errs() << "[VPI-GET] obj=" << objectId << " name=" << obj->name
                 << " sig=" << obj->signalId << " raw=0x" << rawHex
                 << " val=0x" << valHex;
    if (hasUnknown) llvm::errs() << " UNK";
    llvm::errs() << " t=" << scheduler->getCurrentTime().realTime << "\n";
  }
}

uint32_t VPIRuntime::putValue(uint32_t objectId, struct t_vpi_value *value,
                               struct t_vpi_time *time, int32_t flags) {
  stats.valueWrites++;

  auto *obj = findById(objectId);
  if (!obj || !obj->signalId || !scheduler || !value)
    return 0;

  static bool traceVPI = []() {
    const char *env = std::getenv("CIRCT_VPI_TRACE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (traceVPI) {
    llvm::errs() << "[VPI-PUT] obj=" << objectId << " name=" << obj->name
                 << " sig=" << obj->signalId << " flags=" << flags
                 << " t=" << scheduler->getCurrentTime().realTime << "\n";
  }

  uint32_t logicalWidth = obj->width;
  SignalEncoding enc = scheduler->getSignalEncoding(obj->signalId);
  bool isFourState = (enc == SignalEncoding::FourStateStruct);
  uint32_t physWidth = isFourState ? logicalWidth * 2 : logicalWidth;

  llvm::APInt valueBits(logicalWidth, 0);

  switch (value->format) {
  case vpiBinStrVal: {
    const char *str = value->value.str;
    if (!str)
      return 0;
    size_t len = strlen(str);
    for (size_t i = 0; i < len && i < logicalWidth; ++i) {
      if (str[len - 1 - i] == '1')
        valueBits.setBit(i);
    }
    break;
  }
  case vpiIntVal: {
    uint64_t masked = static_cast<uint64_t>(value->value.integer) &
                      llvm::maskTrailingOnes<uint64_t>(logicalWidth);
    valueBits = llvm::APInt(logicalWidth, masked);
    break;
  }
  case vpiScalarVal:
    valueBits = llvm::APInt(logicalWidth, value->value.scalar == vpi1);
    break;
  case vpiVectorVal: {
    if (!value->value.vector)
      return 0;
    uint32_t numWords = (logicalWidth + 31) / 32;
    for (uint32_t i = 0; i < numWords; ++i) {
      uint32_t bitsThisWord = std::min(32u, logicalWidth - i * 32);
      uint64_t aval = static_cast<uint64_t>(value->value.vector[i].aval) &
                      llvm::maskTrailingOnes<uint64_t>(logicalWidth);
      llvm::APInt word(logicalWidth, aval);
      word <<= (i * 32);
      valueBits |= word;
      (void)bitsThisWord;
    }
    break;
  }
  default:
    return 0;
  }

  // Construct the physical signal value.
  // For array element sub-signals, do a read-modify-write on the parent signal.
  SignalValue writtenValue(llvm::APInt(physWidth, 0));
  const auto *arrayInfoPut = scheduler->getSignalArrayInfo(obj->signalId);
  if (arrayInfoPut && obj->type != VPIObjectType::Array) {
    // Array element write: read parent, modify element slice, write back.
    const SignalValue &parentSv = scheduler->getSignalValue(obj->signalId);
    llvm::APInt parentBits = parentSv.getAPInt();
    uint32_t elemPhysW = arrayInfoPut->elementPhysWidth;

    // Build the element's physical bits.
    llvm::APInt elemPhysBits(elemPhysW, 0);
    if (elemPhysW == logicalWidth * 2) {
      // 4-state element: [value | 0_unknown]
      elemPhysBits |= valueBits.zext(elemPhysW) << logicalWidth;
    } else {
      elemPhysBits = valueBits.zextOrTrunc(elemPhysW);
    }

    // Clear the old element bits and insert the new ones.
    llvm::APInt mask =
        llvm::APInt::getBitsSet(parentBits.getBitWidth(), obj->bitOffset,
                                obj->bitOffset + elemPhysW);
    parentBits &= ~mask;
    parentBits |= elemPhysBits.zext(parentBits.getBitWidth()) << obj->bitOffset;
    writtenValue = SignalValue(parentBits);
    scheduler->updateSignal(obj->signalId, writtenValue);
  } else {
    SignalValue newVal(llvm::APInt(physWidth, 0));
    if (isFourState) {
      // Pack: [value_bits | zero_unknown_bits]
      llvm::APInt physBits(physWidth, 0);
      physBits |= valueBits.zext(physWidth) << logicalWidth;
      // Unknown bits = 0 (all known).
      newVal = SignalValue(physBits);
    } else {
      newVal = SignalValue(valueBits);
    }

    // Apply the value immediately (vpiNoDelay).
    writtenValue = newVal;
    scheduler->updateSignal(obj->signalId, writtenValue);
  }

  // Propagate to sibling signals with the same name. This handles the case
  // where a hw.module port signal and an internal llhd.sig have the same name
  // (e.g., port "mode_in" and llhd.sig name "mode_in"). Without this,
  // VPI writes to the port signal don't reach the internal signal that
  // combinational logic reads from.
  auto nameIt = scheduler->getSignalNames().find(obj->signalId);
  if (nameIt != scheduler->getSignalNames().end()) {
    llvm::StringRef sigName = nameIt->second;
    auto dotPos = sigName.rfind('.');
    std::string baseName =
        dotPos != llvm::StringRef::npos ? sigName.substr(dotPos + 1).str()
                                        : sigName.str();
    auto sibIt = nameToSiblingSignals.find(baseName);
    if (sibIt != nameToSiblingSignals.end()) {
      for (SignalId sibId : sibIt->second) {
        if (sibId == obj->signalId)
          continue;
        const SignalValue &sibVal = scheduler->getSignalValue(sibId);
        if (sibVal.getWidth() == writtenValue.getWidth())
          scheduler->updateSignal(sibId, writtenValue);
      }
    }
  }

  // Flush combinational logic: execute delta cycles so that downstream
  // combinational processes see the updated value. This matches Verilog
  // semantics where vpi_put_value(vpiNoDelay) causes immediate propagation
  // through combinational logic before returning.
  //
  // Track the VPI-written signal (and its siblings) so we can restore their
  // values if executeCurrentTime() fires stale scheduled drives that
  // overwrite them.
  vpiDrivenSignals[obj->signalId] = writtenValue;
  auto nameIt2 = scheduler->getSignalNames().find(obj->signalId);
  if (nameIt2 != scheduler->getSignalNames().end()) {
    llvm::StringRef sigName2 = nameIt2->second;
    auto dotPos2 = sigName2.rfind('.');
    std::string baseName2 =
        dotPos2 != llvm::StringRef::npos ? sigName2.substr(dotPos2 + 1).str()
                                         : sigName2.str();
    auto sibIt2 = nameToSiblingSignals.find(baseName2);
    if (sibIt2 != nameToSiblingSignals.end()) {
      for (SignalId sibId : sibIt2->second) {
        if (sibId != obj->signalId) {
          const SignalValue &sibVal = scheduler->getSignalValue(sibId);
          if (sibVal.getWidth() == writtenValue.getWidth())
            vpiDrivenSignals[sibId] = writtenValue;
        }
      }
    }
  }

  scheduler->executeCurrentTime();

  // Re-assert VPI-driven values that were overwritten by stale scheduled
  // drives during executeCurrentTime(). This ensures VPI writes have
  // highest priority — testbench-driven values always win.
  for (auto &[sigId, expectedVal] : vpiDrivenSignals) {
    const SignalValue &currentVal = scheduler->getSignalValue(sigId);
    if (!(currentVal == expectedVal)) {
      scheduler->updateSignal(sigId, expectedVal);
    }
  }

  return objectId;
}

// signalValueToVecval and vecvalToSignalValue are now inlined in
// getValue/putValue with four-state encoding support.

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Time
//===----------------------------------------------------------------------===//

void VPIRuntime::getTime(uint32_t /*objectId*/, struct t_vpi_time *time) {
  if (!time || !scheduler)
    return;
  SimTime now = scheduler->getCurrentTime();
  if (time->type == vpiSimTime) {
    // Return time in VPI time precision units. We report timePrecision=-12
    // (picoseconds). Internal time is in femtoseconds. 1 ps = 1000 fs.
    uint64_t ps = now.realTime / 1000;
    time->high = static_cast<PLI_UINT32>(ps >> 32);
    time->low = static_cast<PLI_UINT32>(ps & 0xFFFFFFFF);
  } else if (time->type == vpiScaledRealTime) {
    time->real = static_cast<double>(now.realTime) * 1e-15; // fs to seconds.
  }
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Callbacks
//===----------------------------------------------------------------------===//

uint32_t VPIRuntime::registerCb(struct t_cb_data *cbData) {
  if (!cbData || !cbData->cb_rtn)
    return 0;

  stats.callbacksRegistered++;
  auto cb = std::make_unique<VPICallback>();
  uint32_t cbId = nextCallbackId();
  cb->id = cbId;
  cb->reason = cbData->reason;
  cb->cbFunc = cbData->cb_rtn;
  cb->userData = cbData->user_data;
  cb->active = true;
  cb->oneShot = false;

  if (cbData->reason == cbValueChange && cbData->obj) {
    cb->objectId = getHandleId(static_cast<vpiHandle>(cbData->obj));
    objectToCallbackIds[cb->objectId].push_back(cbId);
  }

  if (cbData->reason == cbAfterDelay && cbData->time && scheduler) {
    cb->oneShot = true;
    uint64_t delayFs = 0;
    if (cbData->time->type == vpiSimTime) {
      // VPI time is in time precision units (ps for timePrecision=-12).
      // Convert to internal femtoseconds: 1 ps = 1000 fs.
      uint64_t delayPs = (static_cast<uint64_t>(cbData->time->high) << 32) |
                          cbData->time->low;
      delayFs = delayPs * 1000;
    } else if (cbData->time->type == vpiScaledRealTime) {
      delayFs = static_cast<uint64_t>(cbData->time->real * 1e15);
    }
    SimTime targetTime =
        scheduler->getCurrentTime().advanceTime(delayFs);
    // Schedule the callback at the target time.
    uint32_t capturedId = cbId;
    scheduler->getEventScheduler().schedule(
        targetTime, SchedulingRegion::Active,
        Event([this, capturedId]() {
          auto it = callbacks.find(capturedId);
          if (it == callbacks.end() || !it->second->active)
            return;
          // Capture fields before calling cbFunc, which may modify callbacks
          // DenseMap (via vpi_register_cb/vpi_remove_cb) and invalidate `it`.
          auto cbFunc = it->second->cbFunc;
          void *userData = it->second->userData;
          int32_t reason = it->second->reason;
          bool isOneShot = it->second->oneShot;
          t_cb_data data = {};
          data.reason = reason;
          data.user_data = static_cast<PLI_BYTE8 *>(userData);
          cbFunc(&data);
          stats.callbacksFired++;
          // Fire ReadWriteSynch callbacks after the delay callback completes.
          // cocotb 2.0.1 defers DEPOSIT writes — they are only applied when
          // a ReadWrite trigger fires (_apply_scheduled_writes). Each round
          // of ReadWriteSynch may cause signal changes (via the applied
          // writes) that wake coroutines which defer MORE writes, registering
          // new cbReadWriteSynch callbacks. We must loop until all deferred
          // writes have been flushed before firing ReadOnlySynch (where
          // cocotb reads signal values for assertions).
          // Note: putValue() calls executeCurrentTime() internally, so
          // combinational propagation happens within each write.
          for (int rwIter = 0; rwIter < 100; ++rwIter) {
            fireCallbacks(cbReadWriteSynch);
            if (!hasActiveCallbacks(cbReadWriteSynch))
              break;
          }
          fireCallbacks(cbReadOnlySynch);
          if (isOneShot) {
            // Re-find after potential DenseMap rehash.
            auto eraseIt = callbacks.find(capturedId);
            if (eraseIt != callbacks.end())
              callbacks.erase(eraseIt);
          }
        }));
  }

  if (cbData->reason == cbReadWriteSynch || cbData->reason == cbReadOnlySynch) {
    cb->oneShot = true;
  }

  callbacksByReason[cbData->reason].push_back(cbId);
  callbacks[cbId] = std::move(cb);

  LLVM_DEBUG(llvm::dbgs() << "VPIRuntime: Registered callback id=" << cbId
                          << " reason=" << cbData->reason << "\n");
  return cbId;
}

int32_t VPIRuntime::removeCb(uint32_t cbId) {
  auto it = callbacks.find(cbId);
  if (it == callbacks.end())
    return 0;
  it->second->active = false;
  callbacks.erase(it);
  return 1;
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Error Checking
//===----------------------------------------------------------------------===//

int32_t VPIRuntime::chkError(struct t_vpi_error_info *errorInfo) {
  if (!errorInfo)
    return 0;
  if (lastErrorMessage.empty())
    return 0;
  errorInfo->state = vpiRun;
  errorInfo->level = lastErrorLevel;
  errorInfo->message =
      const_cast<PLI_BYTE8 *>(lastErrorMessage.c_str());
  errorInfo->product = const_cast<PLI_BYTE8 *>("circt-sim");
  errorInfo->code = const_cast<PLI_BYTE8 *>("");
  errorInfo->file = const_cast<PLI_BYTE8 *>("");
  errorInfo->line = 0;
  lastErrorMessage.clear();
  return lastErrorLevel;
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Misc
//===----------------------------------------------------------------------===//

int32_t VPIRuntime::getVlogInfo(struct t_vpi_vlog_info *info) {
  if (!info)
    return 0;
  static const char *product = "circt-sim";
  static const char *version = "1.0";
  static char *argv0 = const_cast<char *>("circt-sim");
  info->argc = 1;
  info->argv = &argv0;
  info->product = const_cast<PLI_BYTE8 *>(product);
  info->version = const_cast<PLI_BYTE8 *>(version);
  return 1;
}

int32_t VPIRuntime::freeObject(uint32_t objectId) {
  // Free iterators when their handle is freed.
  auto iterIt = iterators.find(objectId);
  if (iterIt != iterators.end()) {
    iterators.erase(iterIt);
    return 1;
  }
  // Persistent objects (modules, signals) are not freed.
  return 1;
}

int32_t VPIRuntime::control(int32_t operation) {
  if (operation == vpiStop || operation == vpiFinish) {
    LLVM_DEBUG(llvm::dbgs() << "VPIRuntime: vpi_control("
                            << (operation == vpiStop ? "vpiStop" : "vpiFinish")
                            << ")\n");
    // Note: VPIRuntime doesn't own SimulationControl. The main loop's
    // shouldContinue() will eventually detect simulation completion.
    llvm::errs() << "[VPI] vpi_control: simulation stop requested\n";
    return 1;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Callback Dispatch
//===----------------------------------------------------------------------===//

void VPIRuntime::fireCallbacks(int32_t reason) {
  auto it = callbacksByReason.find(reason);
  if (it == callbacksByReason.end())
    return;

  // Copy the callback IDs to avoid iterator invalidation.
  llvm::SmallVector<uint32_t, 8> ids(it->second.begin(), it->second.end());

  for (uint32_t cbId : ids) {
    auto cbIt = callbacks.find(cbId);
    if (cbIt == callbacks.end() || !cbIt->second->active)
      continue;

    // Capture fields before calling the callback, which may modify the
    // callbacks DenseMap (via vpi_register_cb/vpi_remove_cb), invalidating
    // iterators.
    bool isOneShot = cbIt->second->oneShot;
    auto cbFunc = cbIt->second->cbFunc;
    void *userData = cbIt->second->userData;

    t_cb_data data = {};
    data.reason = reason;
    data.user_data = static_cast<PLI_BYTE8 *>(userData);
    cbFunc(&data);
    stats.callbacksFired++;

    if (isOneShot) {
      cbIt = callbacks.find(cbId);
      if (cbIt != callbacks.end())
        callbacks.erase(cbIt);
    }
  }

  // Clean up removed callbacks from the reason list.
  auto &list = callbacksByReason[reason];
  list.erase(
      std::remove_if(list.begin(), list.end(),
                     [this](uint32_t id) {
                       return callbacks.find(id) == callbacks.end();
                     }),
      list.end());
}

bool VPIRuntime::hasActiveCallbacks(int32_t reason) const {
  auto it = callbacksByReason.find(reason);
  if (it == callbacksByReason.end())
    return false;
  for (uint32_t id : it->second) {
    if (callbacks.find(id) != callbacks.end())
      return true;
  }
  return false;
}

void VPIRuntime::fireValueChangeCallbacks(SignalId signalId) {
  auto sigIt = signalToObjectIds.find(signalId);
  if (sigIt == signalToObjectIds.end())
    return;

  // Copy object IDs to avoid iterator invalidation — callbacks may call
  // vpi_register_cb/vpi_remove_cb which modify objectToCallbackIds.
  llvm::SmallVector<uint32_t, 4> objIds(sigIt->second.begin(),
                                        sigIt->second.end());

  for (uint32_t objId : objIds) {
    auto cbIt = objectToCallbackIds.find(objId);
    if (cbIt == objectToCallbackIds.end())
      continue;

    // Copy callback IDs for same reason.
    llvm::SmallVector<uint32_t, 4> cbIds(cbIt->second.begin(),
                                         cbIt->second.end());

    for (uint32_t cbId : cbIds) {
      auto it = callbacks.find(cbId);
      if (it == callbacks.end() || !it->second->active)
        continue;
      if (it->second->reason != cbValueChange)
        continue;

      // Capture fields before calling cbFunc (may invalidate iterators).
      auto cbFunc = it->second->cbFunc;
      void *userData = it->second->userData;

      // Provide current signal value in the callback data. cocotb's GPI
      // layer reads this to determine edge type (rising/falling/change).
      auto *obj = findById(objId);
      s_vpi_value cbValue = {};
      cbValue.format = vpiIntVal;
      if (obj && obj->signalId && scheduler) {
        const SignalValue &sv = scheduler->getSignalValue(obj->signalId);
        SignalEncoding enc = scheduler->getSignalEncoding(obj->signalId);
        bool isFourState = (enc == SignalEncoding::FourStateStruct);
        uint32_t width = obj->width;
        if (isFourState) {
          const llvm::APInt &raw = sv.getAPInt();
          uint32_t physWidth = raw.getBitWidth();
          if (physWidth >= width * 2) {
            llvm::APInt valueBits = raw.extractBits(width, width);
            cbValue.value.integer =
                static_cast<PLI_INT32>(valueBits.getZExtValue());
          }
        } else {
          cbValue.value.integer =
              static_cast<PLI_INT32>(sv.getAPInt().zextOrTrunc(width)
                                         .getZExtValue());
        }
      }

      t_cb_data data = {};
      data.reason = cbValueChange;
      data.obj = makeHandle(objId);
      data.user_data = static_cast<PLI_BYTE8 *>(userData);
      data.value = &cbValue;
      cbFunc(&data);
      stats.callbacksFired++;
    }
  }
}

void VPIRuntime::fireStartOfSimulation() {
  fireCallbacks(cbStartOfSimulation);
}

void VPIRuntime::fireEndOfSimulation() {
  fireCallbacks(cbEndOfSimulation);
}

void VPIRuntime::processReadWriteSynchCallbacks() {
  fireCallbacks(cbReadWriteSynch);
}

void VPIRuntime::processReadOnlySynchCallbacks() {
  fireCallbacks(cbReadOnlySynch);
}

void VPIRuntime::processAfterDelayCallbacks() {
  fireCallbacks(cbAfterDelay);
}

//===----------------------------------------------------------------------===//
// Dispatch Table Installation
//===----------------------------------------------------------------------===//

void VPIRuntime::installDispatchTable() {
  gVPIDispatch.isActive = true;
  gVPIDispatch.handleByName = [](const char *name, void *scope) -> void * {
    auto &vpi = VPIRuntime::getInstance();
    uint32_t scopeId = scope ? getHandleId(static_cast<vpiHandle>(scope)) : 0;
    uint32_t id = vpi.handleByName(name, scopeId);
    return id ? makeHandle(id) : nullptr;
  };
  gVPIDispatch.getProperty = [](int32_t prop, void *obj) -> int32_t {
    return VPIRuntime::getInstance().getProperty(
        prop, getHandleId(static_cast<vpiHandle>(obj)));
  };
  gVPIDispatch.getStrProperty = [](int32_t prop, void *obj) -> const char * {
    return VPIRuntime::getInstance().getStrProperty(
        prop, getHandleId(static_cast<vpiHandle>(obj)));
  };
  gVPIDispatch.getValue = [](void *obj, void *value_p) {
    VPIRuntime::getInstance().getValue(
        getHandleId(static_cast<vpiHandle>(obj)),
        static_cast<struct t_vpi_value *>(value_p));
  };
  gVPIDispatch.putValue = [](void *obj, void *value_p, void *time_p,
                              int32_t flags) -> int32_t {
    return VPIRuntime::getInstance().putValue(
        getHandleId(static_cast<vpiHandle>(obj)),
        static_cast<struct t_vpi_value *>(value_p),
        static_cast<struct t_vpi_time *>(time_p), flags);
  };
  gVPIDispatch.freeObject = [](void *obj) -> int32_t {
    return VPIRuntime::getInstance().freeObject(
        getHandleId(static_cast<vpiHandle>(obj)));
  };
  gVPIDispatch.getTime = [](void *obj, void *time_p) {
    VPIRuntime::getInstance().getTime(
        getHandleId(static_cast<vpiHandle>(obj)),
        static_cast<struct t_vpi_time *>(time_p));
  };
  gVPIDispatch.getVlogInfo = [](void *vlog_info_p) -> int32_t {
    return VPIRuntime::getInstance().getVlogInfo(
        static_cast<struct t_vpi_vlog_info *>(vlog_info_p));
  };
  gVPIDispatch.handle = [](int32_t type, void *ref) -> void * {
    auto &vpi = VPIRuntime::getInstance();
    uint32_t id =
        vpi.handle(type, getHandleId(static_cast<vpiHandle>(ref)));
    return id ? makeHandle(id) : nullptr;
  };
  gVPIDispatch.iterate = [](int32_t type, void *ref) -> void * {
    auto &vpi = VPIRuntime::getInstance();
    uint32_t refId = ref ? getHandleId(static_cast<vpiHandle>(ref)) : 0;
    uint32_t id = vpi.iterate(type, refId);
    return id ? makeHandle(id) : nullptr;
  };
  gVPIDispatch.scan = [](void *iterator) -> void * {
    auto &vpi = VPIRuntime::getInstance();
    uint32_t id =
        vpi.scan(getHandleId(static_cast<vpiHandle>(iterator)));
    return id ? makeHandle(id) : nullptr;
  };
  gVPIDispatch.registerCb = [](void *cb_data_p) -> void * {
    auto &vpi = VPIRuntime::getInstance();
    uint32_t id =
        vpi.registerCb(static_cast<struct t_cb_data *>(cb_data_p));
    return id ? makeHandle(id) : nullptr;
  };
  gVPIDispatch.removeCb = [](void *cb_obj) -> int32_t {
    return VPIRuntime::getInstance().removeCb(
        getHandleId(static_cast<vpiHandle>(cb_obj)));
  };
  gVPIDispatch.chkError = [](void *error_info_p) -> int32_t {
    return VPIRuntime::getInstance().chkError(
        static_cast<struct t_vpi_error_info *>(error_info_p));
  };
  gVPIDispatch.control = [](int32_t operation) -> int32_t {
    return VPIRuntime::getInstance().control(operation);
  };
  gVPIDispatch.releaseHandle = [](void *obj) {
    VPIRuntime::getInstance().freeObject(
        getHandleId(static_cast<vpiHandle>(obj)));
  };
}

//===----------------------------------------------------------------------===//
// C API VPI Functions — These are the symbols cocotb's .so resolves against.
// Only define functions NOT already provided by MooreRuntime.cpp.
//===----------------------------------------------------------------------===//

extern "C" {

vpiHandle vpi_register_cb(p_cb_data cb_data_p) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return nullptr;
  uint32_t id = vpi.registerCb(cb_data_p);
  return id ? VPIRuntime::makeHandle(id) : nullptr;
}

PLI_INT32 vpi_remove_cb(vpiHandle cb_obj) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return 0;
  return vpi.removeCb(VPIRuntime::getHandleId(cb_obj));
}

vpiHandle vpi_handle(PLI_INT32 type, vpiHandle refHandle) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return nullptr;
  uint32_t refId = refHandle ? VPIRuntime::getHandleId(refHandle) : 0;
  uint32_t id = vpi.handle(type, refId);
  return id ? VPIRuntime::makeHandle(id) : nullptr;
}

vpiHandle vpi_handle_by_index(vpiHandle object, PLI_INT32 indx) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return nullptr;
  uint32_t id =
      vpi.handleByIndex(VPIRuntime::getHandleId(object), indx);
  return id ? VPIRuntime::makeHandle(id) : nullptr;
}

vpiHandle vpi_iterate(PLI_INT32 type, vpiHandle refHandle) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return nullptr;
  uint32_t refId = refHandle ? VPIRuntime::getHandleId(refHandle) : 0;
  uint32_t id = vpi.iterate(type, refId);
  return id ? VPIRuntime::makeHandle(id) : nullptr;
}

vpiHandle vpi_scan(vpiHandle iterator) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return nullptr;
  uint32_t id = vpi.scan(VPIRuntime::getHandleId(iterator));
  return id ? VPIRuntime::makeHandle(id) : nullptr;
}

PLI_INT32 vpi_free_object(vpiHandle object) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return 0;
  return vpi.freeObject(VPIRuntime::getHandleId(object));
}

void vpi_get_time(vpiHandle object, p_vpi_time time_p) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive() || !time_p)
    return;
  vpi.getTime(object ? VPIRuntime::getHandleId(object) : 0, time_p);
}

PLI_INT32 vpi_chk_error(p_vpi_error_info error_info_p) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return 0;
  return vpi.chkError(error_info_p);
}

PLI_INT32 vpi_get_vlog_info(p_vpi_vlog_info vlog_info_p) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return 0;
  return vpi.getVlogInfo(vlog_info_p);
}

PLI_INT32 vpi_control(PLI_INT32 operation, ...) {
  auto &vpi = VPIRuntime::getInstance();
  if (!vpi.isActive())
    return 0;
  return vpi.control(operation);
}

} // extern "C"
