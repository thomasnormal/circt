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
  // The scheduler provides signal names in "module.signal" format.
  llvm::StringMap<uint32_t> moduleIds;

  for (const auto &entry : scheduler->getSignalNames()) {
    SignalId sigId = entry.first;
    llvm::StringRef fullName = entry.second;
    const SignalValue &sv = scheduler->getSignalValue(sigId);
    uint32_t width = sv.getWidth();
    // For four-state signals, the physical width is 2x the logical width
    // (value bits + unknown bits). Report the logical width to VPI.
    SignalEncoding enc = scheduler->getSignalEncoding(sigId);
    if (enc == SignalEncoding::FourStateStruct && width >= 2 && (width % 2) == 0)
      width /= 2;

    // Split "module.signal" into module and signal parts.
    auto dotPos = fullName.rfind('.');
    std::string moduleName, signalName;
    if (dotPos != llvm::StringRef::npos) {
      moduleName = fullName.substr(0, dotPos).str();
      signalName = fullName.substr(dotPos + 1).str();
    } else {
      moduleName = defaultModuleName;
      signalName = fullName.str();
    }

    // Ensure module exists.
    uint32_t moduleId;
    auto modIt = moduleIds.find(moduleName);
    if (modIt == moduleIds.end()) {
      moduleId = registerModule(moduleName, moduleName);
      moduleIds[moduleName] = moduleId;
      LLVM_DEBUG(llvm::dbgs()
                 << "  Created module '" << moduleName << "' id=" << moduleId
                 << "\n");
    } else {
      moduleId = modIt->second;
    }

    // Register signal under both short name and module-qualified name.
    std::string qualifiedName = moduleName + "." + signalName;
    uint32_t sigObjId =
        registerSignal(signalName, qualifiedName, sigId, width,
                       VPIObjectType::Reg, moduleId);
    // Also register under the unqualified name for vpi_handle_by_name.
    if (nameToId.find(signalName) == nameToId.end())
      nameToId[signalName] = sigObjId;
    LLVM_DEBUG(llvm::dbgs() << "  Created signal '" << qualifiedName
                            << "' id=" << sigObjId << " sigId=" << sigId
                            << " width=" << width << "\n");
  }

  // Ensure all --top modules exist even if they have no signals.
  for (const auto &modName : topModuleNames) {
    if (moduleIds.find(modName) == moduleIds.end()) {
      uint32_t moduleId = registerModule(modName, modName);
      moduleIds[modName] = moduleId;
      LLVM_DEBUG(llvm::dbgs()
                 << "  Created empty module '" << modName
                 << "' id=" << moduleId << "\n");
    }
  }

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
  return 0;
}

//===----------------------------------------------------------------------===//
// VPI C API Implementation — Iteration
//===----------------------------------------------------------------------===//

uint32_t VPIRuntime::iterate(int32_t type, uint32_t refId) {
  llvm::SmallVector<uint32_t, 8> elements;

  if (refId == 0) {
    // Iterate top-level modules.
    if (type == vpiModule) {
      elements.append(rootModules.begin(), rootModules.end());
    }
  } else {
    auto *obj = findById(refId);
    if (!obj)
      return 0;

    if (type == vpiModule || type == vpiInternalScope) {
      // Return child modules.
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (child && child->type == VPIObjectType::Module)
          elements.push_back(childId);
      }
    } else if (type == vpiNet || type == vpiReg) {
      // Return child signals.
      for (uint32_t childId : obj->children) {
        auto *child = findById(childId);
        if (child && (child->type == VPIObjectType::Net ||
                      child->type == VPIObjectType::Reg))
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
    default:
      return vpiUndefined;
    }
  case vpiSize:
    return static_cast<int32_t>(obj->width);
  case vpiDirection:
    return obj->direction;
  case vpiVector:
    return obj->width > 1 ? 1 : 0;
  case vpiScalar:
    return obj->width == 1 ? 1 : 0;
  case vpiSigned:
    return 0; // Unsigned by default.
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
  if (!obj || !obj->signalId || !scheduler || !value)
    return;

  const SignalValue &sv = scheduler->getSignalValue(obj->signalId);
  uint32_t width = obj->width; // Logical width (already halved for 4-state).

  // Extract logical value and unknown mask for four-state signals.
  SignalEncoding enc = scheduler->getSignalEncoding(obj->signalId);
  bool isFourState = (enc == SignalEncoding::FourStateStruct);
  llvm::APInt valueBits(width, 0);
  llvm::APInt unknownBits(width, 0);
  bool hasUnknown = false;

  if (isFourState) {
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
}

uint32_t VPIRuntime::putValue(uint32_t objectId, struct t_vpi_value *value,
                               struct t_vpi_time *time, int32_t flags) {
  stats.valueWrites++;
  auto *obj = findById(objectId);
  if (!obj || !obj->signalId || !scheduler || !value)
    return 0;

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
  case vpiIntVal:
    valueBits = llvm::APInt(logicalWidth, value->value.integer);
    break;
  case vpiScalarVal:
    valueBits = llvm::APInt(logicalWidth, value->value.scalar == vpi1);
    break;
  case vpiVectorVal: {
    if (!value->value.vector)
      return 0;
    uint32_t numWords = (logicalWidth + 31) / 32;
    for (uint32_t i = 0; i < numWords; ++i) {
      uint32_t bitsThisWord = std::min(32u, logicalWidth - i * 32);
      llvm::APInt word(logicalWidth, value->value.vector[i].aval);
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
  scheduler->updateSignal(obj->signalId, newVal);
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
    // Return time in simulation units (femtoseconds).
    uint64_t fs = now.realTime;
    time->high = static_cast<PLI_UINT32>(fs >> 32);
    time->low = static_cast<PLI_UINT32>(fs & 0xFFFFFFFF);
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
    uint64_t delay = 0;
    if (cbData->time->type == vpiSimTime) {
      delay = (static_cast<uint64_t>(cbData->time->high) << 32) |
              cbData->time->low;
    } else if (cbData->time->type == vpiScaledRealTime) {
      delay = static_cast<uint64_t>(cbData->time->real * 1e15);
    }
    SimTime targetTime =
        scheduler->getCurrentTime().advanceTime(delay);
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
          // cocotb defers signal writes (vpi_put_value) to the ReadWriteSynch
          // region. Without this, writes made during cbAfterDelay callbacks
          // would never be flushed when advanceTime() processes multiple
          // time steps internally without returning to the main loop.
          fireCallbacks(cbReadWriteSynch);
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

void VPIRuntime::fireValueChangeCallbacks(SignalId signalId) {
  auto sigIt = signalToObjectIds.find(signalId);
  if (sigIt == signalToObjectIds.end())
    return;

  for (uint32_t objId : sigIt->second) {
    auto cbIt = objectToCallbackIds.find(objId);
    if (cbIt == objectToCallbackIds.end())
      continue;

    for (uint32_t cbId : cbIt->second) {
      auto it = callbacks.find(cbId);
      if (it == callbacks.end() || !it->second->active)
        continue;
      if (it->second->reason != cbValueChange)
        continue;

      t_cb_data data = {};
      data.reason = cbValueChange;
      data.obj = makeHandle(objId);
      data.user_data = static_cast<PLI_BYTE8 *>(it->second->userData);
      it->second->cbFunc(&data);
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
