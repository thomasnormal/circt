//===- VPIRuntime.h - VPI Runtime Support for Simulation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Verilog Procedural Interface (VPI) runtime for
// circt-sim. It implements the IEEE 1364/1800 VPI C API, enabling external
// verification frameworks (e.g., cocotb) to interact with the simulator.
//
// The implementation bridges VPI calls to CIRCT's ProcessScheduler signal
// management infrastructure.
//
// Based on IEEE 1800-2017 Section 36 (PLI/VPI).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_VPIRUNTIME_H
#define CIRCT_DIALECT_SIM_VPIRUNTIME_H

#include "circt/Dialect/Sim/EventQueue.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// VPI Types and Constants (IEEE 1800-2017 Section 36)
//
// If a system vpi_user.h (e.g., from slang) is already included, we use its
// definitions. Otherwise we provide our own minimal subset for cocotb.
//===----------------------------------------------------------------------===//

// VPI type definitions. We provide our own types that are compatible with
// both IEEE 1800 vpi_user.h and MooreRuntime.h's VPI stubs. We avoid
// redefining vpiHandle since MooreRuntime.h may have already defined it.
#ifndef CIRCT_VPI_TYPES_DEFINED
#define CIRCT_VPI_TYPES_DEFINED

// PLI base types — only define if no system vpi_user.h.
#ifndef VPI_USER_H
typedef int PLI_INT32;
typedef unsigned int PLI_UINT32;
typedef short PLI_INT16;
typedef unsigned short PLI_UINT16;
typedef char PLI_BYTE8;
typedef unsigned char PLI_UBYTE8;
typedef int64_t PLI_INT64;
typedef uint64_t PLI_UINT64;
typedef void *vpiHandle;
#endif // VPI_USER_H

// VPI structures — only define the full IEEE structs if not already provided.
#ifndef VPI_USER_H

struct t_vpi_time {
  PLI_INT32 type;
  PLI_UINT32 high, low;
  double real;
};
typedef struct t_vpi_time s_vpi_time;
typedef struct t_vpi_time *p_vpi_time;

struct t_vpi_vecval {
  PLI_UINT32 aval, bval;
};
typedef struct t_vpi_vecval s_vpi_vecval;
typedef struct t_vpi_vecval *p_vpi_vecval;

struct t_vpi_value {
  PLI_INT32 format;
  union {
    PLI_BYTE8 *str;
    PLI_INT32 scalar;
    PLI_INT32 integer;
    double real;
    struct t_vpi_time *time;
    struct t_vpi_vecval *vector;
    PLI_BYTE8 *misc;
  } value;
};
typedef struct t_vpi_value s_vpi_value;
typedef struct t_vpi_value *p_vpi_value;

struct t_vpi_vlog_info {
  PLI_INT32 argc;
  PLI_BYTE8 **argv;
  PLI_BYTE8 *product;
  PLI_BYTE8 *version;
};
typedef struct t_vpi_vlog_info s_vpi_vlog_info;
typedef struct t_vpi_vlog_info *p_vpi_vlog_info;

struct t_vpi_error_info {
  PLI_INT32 state;
  PLI_INT32 level;
  PLI_BYTE8 *message;
  PLI_BYTE8 *product;
  PLI_BYTE8 *code;
  PLI_BYTE8 *file;
  PLI_INT32 line;
};
typedef struct t_vpi_error_info s_vpi_error_info;
typedef struct t_vpi_error_info *p_vpi_error_info;

struct t_cb_data {
  PLI_INT32 reason;
  PLI_INT32 (*cb_rtn)(struct t_cb_data *);
  void *obj; // vpiHandle — use void* for compatibility
  p_vpi_time time;
  p_vpi_value value;
  PLI_INT32 index;
  PLI_BYTE8 *user_data;
};
typedef struct t_cb_data s_cb_data;
typedef struct t_cb_data *p_cb_data;

#endif // VPI_USER_H
#endif // CIRCT_VPI_TYPES_DEFINED

// VPI constants — define only if not already defined (e.g., by vpi_user.h).
// We use #ifndef to be compatible with both standalone and slang-included builds.

#ifndef vpiModule
#define vpiModule 32
#endif
#ifndef vpiNet
#define vpiNet 36
#endif
#ifndef vpiReg
#define vpiReg 48
#endif
#ifndef vpiPort
#define vpiPort 44
#endif
#ifndef vpiParameter
#define vpiParameter 41
#endif
#ifndef vpiIterator
#define vpiIterator 27
#endif
#ifndef vpiCallback
#define vpiCallback 107
#endif
#ifndef vpiScope
#define vpiScope 84
#endif
#ifndef vpiInternalScope
#define vpiInternalScope 92
#endif
#ifndef vpiUndefined
#define vpiUndefined (-1)
#endif
#ifndef vpiType
#define vpiType 1
#endif
#ifndef vpiName
#define vpiName 2
#endif
#ifndef vpiFullName
#define vpiFullName 3
#endif
#ifndef vpiSize
#define vpiSize 4
#endif
#ifndef vpiDirection
#define vpiDirection 20
#endif
#ifndef vpiInput
#define vpiInput 1
#endif
#ifndef vpiOutput
#define vpiOutput 2
#endif
#ifndef vpiInout
#define vpiInout 3
#endif
#ifndef vpiNoDirection
#define vpiNoDirection 5
#endif
#ifndef vpiVector
#define vpiVector 18
#endif
#ifndef vpiScalar
#define vpiScalar 17
#endif
#ifndef vpiTopModule
#define vpiTopModule 7
#endif
#ifndef vpiSigned
#define vpiSigned 65
#endif
#ifndef vpiLeftRange
#define vpiLeftRange 79
#endif
#ifndef vpiRightRange
#define vpiRightRange 83
#endif
#ifndef vpiArray
#define vpiArray 28
#endif
#ifndef vpiRegArray
#define vpiRegArray 116
#endif
#ifndef vpiStructVar
#define vpiStructVar 618
#endif
#ifndef vpiStructNet
#define vpiStructNet 683
#endif
#ifndef vpiMember
#define vpiMember 742
#endif
#ifndef vpiPacked
#define vpiPacked 630
#endif
#ifndef vpiDefName
#define vpiDefName 9
#endif
#ifndef vpiScaledRealTime
#define vpiScaledRealTime 1
#endif
#ifndef vpiSimTime
#define vpiSimTime 2
#endif
#ifndef vpiBinStrVal
#define vpiBinStrVal 1
#endif
#ifndef vpiOctStrVal
#define vpiOctStrVal 2
#endif
#ifndef vpiDecStrVal
#define vpiDecStrVal 3
#endif
#ifndef vpiHexStrVal
#define vpiHexStrVal 4
#endif
#ifndef vpiScalarVal
#define vpiScalarVal 5
#endif
#ifndef vpiIntVal
#define vpiIntVal 6
#endif
#ifndef vpiRealVal
#define vpiRealVal 7
#endif
#ifndef vpiVectorVal
#define vpiVectorVal 9
#endif
#ifndef vpi0
#define vpi0 0
#endif
#ifndef vpi1
#define vpi1 1
#endif
#ifndef vpiZ
#define vpiZ 2
#endif
#ifndef vpiX
#define vpiX 3
#endif
#ifndef vpiStop
#define vpiStop 66
#endif
#ifndef vpiFinish
#define vpiFinish 67
#endif
#ifndef vpiNotice
#define vpiNotice 1
#endif
#ifndef vpiWarning
#define vpiWarning 2
#endif
#ifndef vpiError
#define vpiError 3
#endif
#ifndef vpiRun
#define vpiRun 3
#endif
#ifndef cbValueChange
#define cbValueChange 1
#endif
#ifndef cbReadWriteSynch
#define cbReadWriteSynch 6
#endif
#ifndef cbReadOnlySynch
#define cbReadOnlySynch 7
#endif
#ifndef cbAfterDelay
#define cbAfterDelay 9
#endif
#ifndef cbStartOfSimulation
#define cbStartOfSimulation 11
#endif
#ifndef cbEndOfSimulation
#define cbEndOfSimulation 12
#endif
#ifndef vpiTimePrecision
#define vpiTimePrecision 12
#endif
#ifndef vpiTimeUnit
#define vpiTimeUnit 11
#endif
// SystemVerilog extensions (sv_vpi_user.h)
#ifndef vpiInstance
#define vpiInstance 745
#endif

#ifndef vpiGenScope
#define vpiGenScope 134
#endif

#ifndef vpiGenScopeArray
#define vpiGenScopeArray 133
#endif

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// VPIObjectType - Types of VPI objects we track
//===----------------------------------------------------------------------===//

enum class VPIObjectType : uint8_t {
  Module = 0,
  Net = 1,
  Reg = 2,
  Port = 3,
  Parameter = 4,
  Iterator = 5,
  Callback = 6,
  Array = 7,
  GenScope = 8,
  GenScopeArray = 9,
  StructVar = 10,
};

//===----------------------------------------------------------------------===//
// VPIObject - Internal representation of a VPI handle target
//===----------------------------------------------------------------------===//

/// Represents an object accessible through VPI handles.
/// Each object maps to either a module instance or a signal in the
/// ProcessScheduler.
struct VPIObject {
  /// Unique ID for this object (used as the handle value).
  uint32_t id;

  /// Object type.
  VPIObjectType type;

  /// Name of the object (short name, e.g., "clk").
  std::string name;

  /// Full hierarchical name (e.g., "top.dut.clk").
  std::string fullName;

  /// For signals: the ProcessScheduler signal ID.
  SignalId signalId = 0;

  /// For signals: bit width.
  uint32_t width = 0;

  /// For ports: direction (1=input, 2=output, 3=inout).
  int32_t direction = 0;

  /// For parameters: the elaborated constant value.
  int64_t paramValue = 0;

  /// For array element sub-signals: bit offset within the parent signal.
  /// Used to read/write the correct slice of the parent signal's bits.
  uint32_t bitOffset = 0;

  /// For Array objects: SV-declared left and right bounds.
  int32_t leftBound = 0;
  int32_t rightBound = 0;

  /// Parent object ID (0 = no parent / root).
  uint32_t parentId = 0;

  /// Child object IDs (for modules: ports/nets; for iterators: elements).
  llvm::SmallVector<uint32_t, 8> children;
};

//===----------------------------------------------------------------------===//
// VPIIterator - State for vpi_iterate/vpi_scan traversal
//===----------------------------------------------------------------------===//

struct VPIIterator {
  uint32_t id;
  std::vector<uint32_t> elements;
  size_t currentIndex = 0;
};

//===----------------------------------------------------------------------===//
// VPICallback - Registered VPI callback
//===----------------------------------------------------------------------===//

struct VPICallback {
  uint32_t id;
  int32_t reason; // cbValueChange, cbReadWriteSynch, etc.
  uint32_t objectId; // Object being monitored (for cbValueChange).
  int32_t (*cbFunc)(struct t_cb_data *);
  void *userData;
  bool oneShot; // Auto-remove after firing.
  bool active;
};

//===----------------------------------------------------------------------===//
// VPIRuntime - Main VPI runtime that bridges to ProcessScheduler
//===----------------------------------------------------------------------===//

/// The VPI runtime manages VPI objects, callbacks, and provides the
/// implementation for all VPI C API functions. It bridges between the
/// IEEE VPI interface and CIRCT's ProcessScheduler.
class VPIRuntime {
public:
  VPIRuntime();
  ~VPIRuntime();

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  /// Set the ProcessScheduler that this VPI runtime bridges to.
  void setScheduler(ProcessScheduler *scheduler) { this->scheduler = scheduler; }

  /// Get the ProcessScheduler.
  ProcessScheduler *getScheduler() const { return scheduler; }

  /// Set the top module names (from --top CLI flags).
  void setTopModuleNames(const llvm::SmallVector<std::string, 4> &names) {
    topModuleNames = names;
  }

  /// Build the VPI object hierarchy from the ProcessScheduler's signals.
  /// Call this after all signals are registered in the scheduler.
  void buildHierarchy();

  /// Load a cocotb/VPI shared library and call its startup routines.
  bool loadVPILibrary(const std::string &path);

  //===--------------------------------------------------------------------===//
  // Object Management
  //===--------------------------------------------------------------------===//

  /// Register a module object. Returns its handle ID.
  uint32_t registerModule(const std::string &name,
                          const std::string &fullName,
                          uint32_t parentId = 0);

  /// Register a signal object (net/reg). Returns its handle ID.
  uint32_t registerSignal(const std::string &name,
                          const std::string &fullName,
                          SignalId signalId, uint32_t width,
                          VPIObjectType type = VPIObjectType::Net,
                          uint32_t parentModuleId = 0);

  /// Register a parameter object. Returns its handle ID.
  uint32_t registerParameter(const std::string &name,
                             const std::string &fullName,
                             int64_t value, uint32_t width,
                             uint32_t parentModuleId = 0);

  /// Look up an object by full hierarchical name.
  VPIObject *findByName(const std::string &fullName);

  /// Look up an object by handle ID.
  VPIObject *findById(uint32_t id);

  /// Add an alias name mapping (e.g., unqualified name → object ID).
  void addNameMapping(const std::string &name, uint32_t objectId) {
    nameToId[name] = objectId;
  }

  //===--------------------------------------------------------------------===//
  // VPI C API Implementation
  //===--------------------------------------------------------------------===//

  // Handle access
  uint32_t handleByName(const char *name, uint32_t scopeId);
  uint32_t handleByIndex(uint32_t objectId, int32_t index);
  uint32_t handle(int32_t type, uint32_t refId);

  // Iteration
  uint32_t iterate(int32_t type, uint32_t refId);
  uint32_t scan(uint32_t iteratorId);

  // Properties
  int32_t getProperty(int32_t property, uint32_t objectId);
  const char *getStrProperty(int32_t property, uint32_t objectId);

  // Value access
  void getValue(uint32_t objectId, struct t_vpi_value *value);
  uint32_t putValue(uint32_t objectId, struct t_vpi_value *value,
                    struct t_vpi_time *time, int32_t flags);

  // Time
  void getTime(uint32_t objectId, struct t_vpi_time *time);

  // Callbacks
  uint32_t registerCb(struct t_cb_data *cbData);
  int32_t removeCb(uint32_t cbId);

  // Error checking
  int32_t chkError(struct t_vpi_error_info *errorInfo);

  // Simulator info
  int32_t getVlogInfo(struct t_vpi_vlog_info *info);

  // Object lifecycle
  int32_t freeObject(uint32_t objectId);

  // Simulator control
  int32_t control(int32_t operation);

  //===--------------------------------------------------------------------===//
  // Callback Dispatch
  //===--------------------------------------------------------------------===//

  /// Fire all callbacks of the given reason.
  void fireCallbacks(int32_t reason);

  /// Check if there are any active callbacks registered for the given reason.
  bool hasActiveCallbacks(int32_t reason) const;

  /// Fire value-change callbacks for a specific signal.
  void fireValueChangeCallbacks(SignalId signalId);

  /// Fire all start-of-simulation callbacks.
  void fireStartOfSimulation();

  /// Fire all end-of-simulation callbacks.
  void fireEndOfSimulation();

  /// Process any pending ReadWriteSynch callbacks (Active/NBA region).
  void processReadWriteSynchCallbacks();

  /// Process any pending ReadOnlySynch callbacks (Postponed region).
  void processReadOnlySynchCallbacks();

  /// Process any pending AfterDelay callbacks.
  void processAfterDelayCallbacks();

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  struct Statistics {
    size_t objectsCreated = 0;
    size_t callbacksRegistered = 0;
    size_t callbacksFired = 0;
    size_t valueReads = 0;
    size_t valueWrites = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Whether VPI is active (a library was loaded).
  bool isActive() const { return active; }

  /// Install VPIRuntime dispatch functions into the global VPI dispatch table
  /// (gVPIDispatch). This allows MooreRuntime's VPI stubs to delegate to
  /// VPIRuntime when active.
  void installDispatchTable();

  /// Get the singleton instance.
  static VPIRuntime &getInstance();

  /// Convert an object ID to a VPI handle (void*).
  /// Allocates storage in an internal arena.
  static vpiHandle makeHandle(uint32_t id);

  /// Extract an object ID from a VPI handle.
  static uint32_t getHandleId(vpiHandle h);

private:
  /// Generate the next object ID.
  uint32_t nextObjectId();

  /// Generate the next callback ID.
  uint32_t nextCallbackId();

  /// Convert ProcessScheduler SignalValue to VPI vecval format.
  void signalValueToVecval(const SignalValue &sv, struct t_vpi_vecval *vecval,
                           uint32_t width);

  /// Convert VPI vecval format to ProcessScheduler SignalValue.
  SignalValue vecvalToSignalValue(const struct t_vpi_vecval *vecval,
                                  uint32_t width);

  /// The ProcessScheduler we bridge to.
  ProcessScheduler *scheduler = nullptr;

  /// All VPI objects indexed by ID.
  llvm::DenseMap<uint32_t, std::unique_ptr<VPIObject>> objects;

  /// Name-to-ID lookup.
  llvm::StringMap<uint32_t> nameToId;

  /// Active iterators.
  llvm::DenseMap<uint32_t, std::unique_ptr<VPIIterator>> iterators;

  /// Registered callbacks.
  llvm::DenseMap<uint32_t, std::unique_ptr<VPICallback>> callbacks;

  /// Signal ID to object ID mapping (for value-change callback dispatch).
  llvm::DenseMap<SignalId, llvm::SmallVector<uint32_t, 4>> signalToObjectIds;

  /// Map from signal name to all SignalIds with that name.
  /// Used to propagate VPI writes from port signals to internal copies
  /// (e.g., hw.module port "mode_in" → llhd.sig name "mode_in").
  llvm::StringMap<llvm::SmallVector<SignalId, 2>> nameToSiblingSignals;

  /// Signals actively driven by VPI putValue and their expected values.
  /// Used to re-assert VPI-written values after executeCurrentTime() fires
  /// stale scheduled drives. Cleared at the start of each time step.
  llvm::DenseMap<SignalId, SignalValue> vpiDrivenSignals;

  /// Object ID to callback IDs mapping (for value-change callbacks).
  llvm::DenseMap<uint32_t, llvm::SmallVector<uint32_t, 4>> objectToCallbackIds;

  /// Callbacks by reason (for efficient dispatch).
  llvm::DenseMap<int32_t, llvm::SmallVector<uint32_t, 4>> callbacksByReason;

  /// Root module IDs.
  llvm::SmallVector<uint32_t, 4> rootModules;

  /// Top module names from --top CLI flags.
  llvm::SmallVector<std::string, 4> topModuleNames;

  /// ID counters.
  uint32_t nextObjId = 1;
  uint32_t nextCbId = 1;

  /// Last error info.
  std::string lastErrorMessage;
  int32_t lastErrorLevel = 0;

  /// Static string buffer for vpi_get_str return values.
  std::string strBuffer;

  /// Whether VPI is active.
  bool active = false;

  /// Statistics.
  Statistics stats;

  /// Loaded VPI libraries.
  std::vector<llvm::sys::DynamicLibrary> loadedLibraries;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_VPIRUNTIME_H
