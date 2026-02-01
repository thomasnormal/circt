//===- HWEliminateInOutPorts.cpp - Generator Callout Pass
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/FourStateUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include <optional>

namespace circt {
namespace sv {
#define GEN_PASS_DEF_HWELIMINATEINOUTPORTS
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace sv
} // namespace circt

using namespace circt;
using namespace sv;
using namespace hw;
using namespace igraph;

namespace {

struct HWEliminateInOutPortsPass
    : public circt::sv::impl::HWEliminateInOutPortsBase<
          HWEliminateInOutPortsPass> {
  using HWEliminateInOutPortsBase<
      HWEliminateInOutPortsPass>::HWEliminateInOutPortsBase;
  void runOnOperation() override;
};

struct AccessGroup {
  enum class StepKind { Field, ArrayIndex };
  struct Step {
    StepKind kind;
    StringAttr field;
    APInt index;
    bool dynamic = false;
    unsigned dynamicId = 0;
    Value dynamicIndexValue;
  };

  SmallVector<Step, 2> path;
  SmallVector<sv::ReadInOutOp, 4> readers;
  SmallVector<sv::AssignOp, 4> writers;
  hw::PortInfo readPort;
  hw::PortInfo writePort;
  Value internalDrive;
  Value readValue;
  std::optional<Location> driveLoc;
  Type portType;
  Type elementType;
  bool dynamicIndex = false;
  unsigned dynamicId = 0;
  int dynamicPathIndex = -1;
};

class HWInOutPortConversion : public PortConversion {
public:
  HWInOutPortConversion(PortConverterImpl &converter, hw::PortInfo port,
                        llvm::StringRef readSuffix,
                        llvm::StringRef writeSuffix,
                        bool allowMultipleWritersSameValue,
                        bool resolveReadWrite);

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

  LogicalResult init() override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  Value buildDerivedInOut(OpBuilder &b, Location loc, Value base,
                          ArrayRef<AccessGroup::Step> path);
  AccessGroup &getAccessGroup(StringRef pathKey,
                              ArrayRef<AccessGroup::Step> path, Type elementType);

  // Suffix to be used when creating read ports.
  llvm::StringRef readSuffix;
  // Suffix to be used when creating write ports.
  llvm::StringRef writeSuffix;

  bool allowMultipleWritersSameValue;
  bool resolveReadWrite;
  llvm::StringMap<unsigned> accessGroupIndex;
  SmallVector<AccessGroup, 4> accessGroups;
  SmallVector<Operation *, 4> derivedInouts;
};

HWInOutPortConversion::HWInOutPortConversion(
    PortConverterImpl &converter, hw::PortInfo port, llvm::StringRef readSuffix,
    llvm::StringRef writeSuffix, bool allowMultipleWritersSameValue,
    bool resolveReadWrite)
    : PortConversion(converter, port), readSuffix(readSuffix),
      writeSuffix(writeSuffix),
      allowMultipleWritersSameValue(allowMultipleWritersSameValue),
      resolveReadWrite(resolveReadWrite) {}

static std::string buildPathKey(ArrayRef<AccessGroup::Step> path) {
  std::string key;
  for (const auto &step : path) {
    if (!key.empty())
      key += '.';
    if (step.kind == AccessGroup::StepKind::Field) {
      key += step.field.getValue().str();
    } else {
      if (step.dynamic) {
        key += "[dyn";
        key += std::to_string(step.dynamicId);
        key += "]";
      } else {
        llvm::SmallString<16> buffer;
        step.index.toString(buffer, 10, false);
        key.push_back('[');
        key += buffer.str().str();
        key.push_back(']');
      }
    }
  }
  return key;
}

AccessGroup &HWInOutPortConversion::getAccessGroup(
    StringRef pathKey, ArrayRef<AccessGroup::Step> path, Type elementType) {
  auto it = accessGroupIndex.find(pathKey);
  if (it == accessGroupIndex.end()) {
    AccessGroup group;
    group.path.assign(path.begin(), path.end());
    group.elementType = elementType;
    group.portType = elementType;
    accessGroups.push_back(std::move(group));
    unsigned index = accessGroups.size() - 1;
    accessGroupIndex.try_emplace(pathKey, index);
    return accessGroups[index];
  }
  AccessGroup &group = accessGroups[it->second];
  assert((!group.elementType || group.elementType == elementType) &&
         "inout access group element type mismatch");
  return group;
}

Value HWInOutPortConversion::buildDerivedInOut(OpBuilder &b, Location loc,
                                               Value base,
                                               ArrayRef<AccessGroup::Step> path) {
  Value current = base;
  for (const auto &step : path) {
    if (step.kind == AccessGroup::StepKind::Field) {
      current = sv::StructFieldInOutOp::create(b, loc, current, step.field);
      continue;
    }
    Value idx = hw::ConstantOp::create(b, loc, step.index).getResult();
    current = sv::ArrayIndexInOutOp::create(b, loc, current, idx);
  }
  return current;
}

static bool allWritersSameValue(ArrayRef<sv::AssignOp> writers) {
  if (writers.empty())
    return true;
  sv::AssignOp first = writers.front();
  Value src = first.getSrc();
  auto attrs = first->getAttrDictionary();
  auto areEquivalentValues = [](Value lhs, Value rhs) -> bool {
    if (lhs == rhs)
      return true;
    Operation *lhsOp = lhs.getDefiningOp();
    Operation *rhsOp = rhs.getDefiningOp();
    if (!lhsOp || !rhsOp)
      return false;
    if (lhsOp->getNumOperands() != 0 || rhsOp->getNumOperands() != 0)
      return false;
    return mlir::OperationEquivalence::isEquivalentTo(
        lhsOp, rhsOp, mlir::OperationEquivalence::IgnoreLocations);
  };
  for (auto writer : writers.drop_front()) {
    if (!areEquivalentValues(writer.getSrc(), src))
      return false;
    if (writer->getAttrDictionary() != attrs)
      return false;
  }
  return true;
}

static bool isFourStateType(Type type) {
  return circt::isFourStateStructType(type);
}

static bool dependsOnAny(Value value,
                         const llvm::SmallPtrSetImpl<Value> &targets) {
  llvm::SmallVector<Value, 8> worklist;
  llvm::SmallPtrSet<Value, 16> visited;
  worklist.push_back(value);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (targets.contains(current))
      return true;
    Operation *def = current.getDefiningOp();
    if (!def)
      continue;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

LogicalResult HWInOutPortConversion::init() {
  auto accessLabel = [&](const AccessGroup &group) -> std::string {
    std::string label = origPort.name.getValue().str();
    if (group.path.empty())
      return group.dynamicIndex
                 ? label + "[dyn" + std::to_string(group.dynamicId) + "]"
                 : label;
    for (const auto &step : group.path) {
      if (step.kind == AccessGroup::StepKind::Field) {
        label.push_back('.');
        label += step.field.getValue().str();
      } else {
        if (step.dynamic) {
          label += "[dyn";
          label += std::to_string(step.dynamicId);
          label += "]";
        } else {
          llvm::SmallString<16> buffer;
          step.index.toString(buffer, 10, false);
          label.push_back('[');
          label += buffer.str().str();
          label.push_back(']');
        }
      }
    }
    if (group.dynamicIndex)
      label += "[dyn" + std::to_string(group.dynamicId) + "]";
    return label;
  };

  struct DynamicIndexInfo {
    unsigned id = 0;
    Type portType;
    unsigned pathIndex = 0;
  };

  unsigned nextDynamicId = 0;
  auto collectAccesses =
      [&](auto &&self, Value inoutValue, ArrayRef<AccessGroup::Step> path,
          Type elementType, std::optional<DynamicIndexInfo> dynamicInfo)
      -> LogicalResult {
    for (Operation *user : inoutValue.getUsers()) {
      if (auto read = dyn_cast<sv::ReadInOutOp>(user)) {
        std::string key = buildPathKey(path);
        AccessGroup &group = getAccessGroup(key, path, elementType);
        if (dynamicInfo) {
          group.dynamicIndex = true;
          group.dynamicId = dynamicInfo->id;
          group.dynamicPathIndex = dynamicInfo->pathIndex;
          group.portType = dynamicInfo->portType;
        }
        group.readers.push_back(read);
        continue;
      }
      if (auto write = dyn_cast<sv::AssignOp>(user)) {
        std::string key = buildPathKey(path);
        AccessGroup &group = getAccessGroup(key, path, elementType);
        if (dynamicInfo) {
          group.dynamicIndex = true;
          group.dynamicId = dynamicInfo->id;
          group.dynamicPathIndex = dynamicInfo->pathIndex;
          group.portType = dynamicInfo->portType;
        }
        group.writers.push_back(write);
        continue;
      }
      if (auto fieldOp = dyn_cast<sv::StructFieldInOutOp>(user)) {
        SmallVector<AccessGroup::Step, 2> nextPath(path.begin(), path.end());
        nextPath.push_back(
            {AccessGroup::StepKind::Field, fieldOp.getFieldAttr(), APInt(),
             false, 0, Value()});
        auto inoutTy = dyn_cast<hw::InOutType>(fieldOp.getType());
        if (!inoutTy)
          return fieldOp.emitOpError("expected hw.inout type");
        derivedInouts.push_back(fieldOp.getOperation());
        if (failed(self(self, fieldOp.getResult(), nextPath,
                        inoutTy.getElementType(), dynamicInfo)))
          return failure();
        continue;
      }
      if (auto arrayOp = dyn_cast<sv::ArrayIndexInOutOp>(user)) {
        auto inoutTy = dyn_cast<hw::InOutType>(arrayOp.getType());
        if (!inoutTy)
          return arrayOp.emitOpError("expected hw.inout type");
        auto indexConst = arrayOp.getIndex().getDefiningOp<hw::ConstantOp>();
        if (!indexConst) {
          auto baseInOutTy =
              dyn_cast<hw::InOutType>(arrayOp.getInput().getType());
          if (!baseInOutTy)
            return arrayOp.emitOpError("expected hw.inout type");
          DynamicIndexInfo dynInfo;
          if (dynamicInfo) {
            dynInfo = *dynamicInfo;
          } else {
            dynInfo.id = nextDynamicId++;
            dynInfo.portType = baseInOutTy.getElementType();
            dynInfo.pathIndex = path.size();
          }
          unsigned stepId = dynamicInfo ? nextDynamicId++ : dynInfo.id;
          derivedInouts.push_back(arrayOp.getOperation());
          SmallVector<AccessGroup::Step, 2> nextPath(path.begin(), path.end());
          nextPath.push_back({AccessGroup::StepKind::ArrayIndex, StringAttr(),
                              APInt(), true, stepId, arrayOp.getIndex()});
          if (failed(self(self, arrayOp.getResult(), nextPath,
                          inoutTy.getElementType(), dynInfo)))
            return failure();
          continue;
        }
        SmallVector<AccessGroup::Step, 2> nextPath(path.begin(), path.end());
        nextPath.push_back({AccessGroup::StepKind::ArrayIndex, StringAttr(),
                            indexConst.getValue(), false, 0, Value()});
        derivedInouts.push_back(arrayOp.getOperation());
        if (failed(self(self, arrayOp.getResult(), nextPath,
                        inoutTy.getElementType(), dynamicInfo)))
          return failure();
        continue;
      }
      return user->emitOpError()
             << "uses hw.inout port " << origPort.name
             << " but the operation itself is unsupported.";
    }
    return success();
  };

  if (failed(collectAccesses(collectAccesses,
                             body->getArgument(origPort.argNum), {},
                             origPort.type, std::nullopt)))
    return failure();

  auto dynamicBaseKey = [&](const AccessGroup &group) -> std::string {
    assert(group.dynamicPathIndex >= 0 && "dynamic path index not set");
    return buildPathKey(ArrayRef<AccessGroup::Step>(group.path)
                            .take_front(group.dynamicPathIndex));
  };
  llvm::StringMap<unsigned> dynamicWriterCounts;
  llvm::StringMap<bool> dynamicWriterNeedsResolve;
  auto collectArrayBaseKeys =
      [&](const AccessGroup &group,
          SmallVectorImpl<std::string> &keys) -> void {
    if (isa<hw::ArrayType>(group.elementType))
      keys.push_back(buildPathKey(group.path));
    SmallVector<AccessGroup::Step, 4> prefix;
    prefix.reserve(group.path.size());
    for (const auto &step : group.path) {
      if (step.kind == AccessGroup::StepKind::ArrayIndex)
        keys.push_back(buildPathKey(prefix));
      prefix.push_back(step);
    }
  };

  for (auto &group : accessGroups) {
    bool hasReaders = !group.readers.empty();
    bool hasWriters = !group.writers.empty();
    bool canResolveMultipleWriters =
        resolveReadWrite && isFourStateType(group.elementType);

    if (resolveReadWrite && hasReaders && hasWriters &&
        !isFourStateType(group.elementType))
      return converter.getModule()->emitOpError()
             << "inout port " << accessLabel(group)
             << " requires 4-state type for read/write resolution.";

    if (resolveReadWrite && hasReaders && hasWriters) {
      llvm::SmallPtrSet<Value, 8> readValues;
      for (auto read : group.readers)
        readValues.insert(read.getResult());
      for (auto writer : group.writers) {
        if (dependsOnAny(writer.getSrc(), readValues))
          return converter.getModule()->emitOpError()
                 << "inout port " << accessLabel(group)
                 << " has write depending on read; rerun without "
                    "--resolve-read-write or refactor.";
      }
    }

    if (group.writers.size() > 1 && !canResolveMultipleWriters &&
        (!allowMultipleWritersSameValue ||
         !allWritersSameValue(group.writers)))
      return converter.getModule()->emitOpError()
             << "multiple writers of inout port " << accessLabel(group)
             << " is unsupported.";

    if (group.dynamicIndex && hasWriters) {
      std::string baseKey = dynamicBaseKey(group);
      dynamicWriterCounts[baseKey]++;
      if (!isFourStateType(group.elementType))
        dynamicWriterNeedsResolve[baseKey] = true;
    }
  }

  if (!dynamicWriterCounts.empty()) {
    for (auto &[baseKey, count] : dynamicWriterCounts) {
      if (count > 1 &&
          (!resolveReadWrite || dynamicWriterNeedsResolve.lookup(baseKey)))
        return converter.getModule()->emitOpError()
               << "multiple dynamic writers of inout port "
               << (baseKey.empty() ? origPort.name.getValue() : baseKey)
               << " require 4-state resolution; rerun with "
                  "--resolve-read-write";
    }
    for (auto &group : accessGroups) {
      if (group.writers.empty())
        continue;
      if (group.dynamicIndex)
        continue;
      SmallVector<std::string, 4> baseKeys;
      collectArrayBaseKeys(group, baseKeys);
      for (const auto &baseKey : baseKeys) {
        if (!dynamicWriterCounts.contains(baseKey))
          continue;
        return converter.getModule()->emitOpError()
               << "dynamic inout writer conflicts with other writer for path "
               << (baseKey.empty() ? origPort.name.getValue() : baseKey);
      }
    }
  }

  return success();
}

void HWInOutPortConversion::buildInputSignals() {
  OpBuilder builder(body, body->getTerminator()->getIterator());

  auto buildSuffix = [&](StringRef baseSuffix,
                         const AccessGroup &group) -> std::string {
    std::string suffix = baseSuffix.str();
    if (!group.path.empty()) {
      for (const auto &step : group.path) {
        std::string token;
        if (step.kind == AccessGroup::StepKind::Field) {
          token = step.field.getValue().str();
        } else if (!step.dynamic) {
          llvm::SmallString<16> buffer;
          step.index.toString(buffer, 10, false);
          token = "idx";
          token += buffer.str().str();
        }
        if (token.empty())
          continue;
        suffix.push_back('_');
        suffix += token;
      }
    }
    if (group.dynamicIndex) {
      suffix += "_dyn";
      suffix += std::to_string(group.dynamicId);
    }
    return suffix;
  };

  auto buildDynamicBaseSuffix = [&](StringRef baseSuffix,
                                    const AccessGroup &group) -> std::string {
    assert(group.dynamicIndex && "expected dynamic index group");
    std::string suffix = baseSuffix.str();
    auto prefix =
        ArrayRef<AccessGroup::Step>(group.path).take_front(group.dynamicPathIndex);
    if (!prefix.empty()) {
      for (const auto &step : prefix) {
        std::string token;
        if (step.kind == AccessGroup::StepKind::Field) {
          token = step.field.getValue().str();
        } else if (!step.dynamic) {
          llvm::SmallString<16> buffer;
          step.index.toString(buffer, 10, false);
          token = "idx";
          token += buffer.str().str();
        }
        if (token.empty())
          continue;
        suffix.push_back('_');
        suffix += token;
      }
    }
    suffix += "_dyn";
    suffix += std::to_string(group.dynamicId);
    return suffix;
  };

  auto dynamicBaseKey = [&](const AccessGroup &group) -> std::string {
    assert(group.dynamicPathIndex >= 0 && "dynamic path index not set");
    return buildPathKey(ArrayRef<AccessGroup::Step>(group.path)
                            .take_front(group.dynamicPathIndex));
  };

  llvm::StringMap<unsigned> dynamicBaseReadCounts;
  for (auto &group : accessGroups) {
    bool needsReadPort =
        !group.readers.empty() || (group.dynamicIndex && !group.writers.empty());
    if (group.dynamicIndex && needsReadPort)
      dynamicBaseReadCounts[dynamicBaseKey(group)]++;
  }

  struct DynamicBaseReadPort {
    Value value;
    hw::PortInfo port;
    unsigned dynamicId = 0;
    Type portType;
  };
  llvm::StringMap<DynamicBaseReadPort> dynamicBaseReadPorts;
  auto getDynamicBaseRead = [&](AccessGroup &group) -> DynamicBaseReadPort & {
    assert(group.dynamicIndex && "expected dynamic index group");
    std::string baseKey = dynamicBaseKey(group);
    auto it = dynamicBaseReadPorts.find(baseKey);
    if (it != dynamicBaseReadPorts.end()) {
      assert(it->second.portType == group.portType &&
             "mismatched dynamic base port type");
      return it->second;
    }
    DynamicBaseReadPort entry;
    entry.dynamicId = group.dynamicId;
    entry.portType = group.portType;
    entry.value = converter.createNewInput(
        origPort, buildDynamicBaseSuffix(readSuffix, group), group.portType,
        entry.port);
    auto [inserted, didInsert] =
        dynamicBaseReadPorts.try_emplace(baseKey, std::move(entry));
    (void)didInsert;
    return inserted->second;
  };

  auto applySuffixExtract =
      [&](Value base, ArrayRef<AccessGroup::Step> suffix,
          Location loc) -> Value {
    Value current = base;
    for (const auto &step : suffix) {
      if (step.kind == AccessGroup::StepKind::Field) {
        current =
            hw::StructExtractOp::create(builder, loc, current, step.field);
        continue;
      }
      Value idx;
      if (step.dynamic) {
        assert(step.dynamicIndexValue && "dynamic index value not set");
        idx = step.dynamicIndexValue;
      } else {
        idx = hw::ConstantOp::create(builder, loc, step.index).getResult();
      }
      current = hw::ArrayGetOp::create(builder, loc, current, idx);
    }
    return current;
  };

  auto applySuffixInject = [&](auto &&self, Value base,
                               ArrayRef<AccessGroup::Step> suffix,
                               Value newValue, Location loc) -> Value {
    if (suffix.empty())
      return newValue;
    const auto &step = suffix.front();
    if (step.kind == AccessGroup::StepKind::Field) {
      Value fieldValue =
          hw::StructExtractOp::create(builder, loc, base, step.field);
      Value updatedField =
          self(self, fieldValue, suffix.drop_front(), newValue, loc);
      return hw::StructInjectOp::create(builder, loc, base, step.field,
                                        updatedField);
    }
    Value idx;
    if (step.dynamic) {
      assert(step.dynamicIndexValue && "dynamic index value not set");
      idx = step.dynamicIndexValue;
    } else {
      idx = hw::ConstantOp::create(builder, loc, step.index).getResult();
    }
    Value element = hw::ArrayGetOp::create(builder, loc, base, idx);
    Value updatedElement =
        self(self, element, suffix.drop_front(), newValue, loc);
    return hw::ArrayInjectOp::create(builder, loc, base, idx, updatedElement);
  };

  llvm::StringMap<SmallVector<AccessGroup *, 4>> dynamicWriterGroups;

  for (auto &group : accessGroups) {
    bool hasReaders = !group.readers.empty();
    bool hasWriters = !group.writers.empty();
    bool needsReadPort = hasReaders || (group.dynamicIndex && hasWriters);
    bool canResolveMultipleWriters =
        resolveReadWrite && isFourStateType(group.elementType);
    group.internalDrive = Value();
    group.readValue = Value();
    group.driveLoc.reset();

    auto computeInternalDrive = [&]() {
      if (!hasWriters)
        return;
      if (group.writers.size() > 1 && canResolveMultipleWriters) {
        SmallVector<Value, 4> driveValues;
        driveValues.reserve(group.writers.size());
        for (auto writer : group.writers)
          driveValues.push_back(writer.getSrc());
        group.internalDrive = resolveFourStateValues(
            builder, group.writers.front().getLoc(), driveValues);
        assert(group.internalDrive &&
               "expected resolved 4-state inout drive value");
      } else {
        group.internalDrive = group.writers.front().getSrc();
      }
    };

    Value readValue;
    if (needsReadPort) {
      if (group.dynamicIndex) {
        std::string baseKey = dynamicBaseKey(group);
        if (dynamicBaseReadCounts.lookup(baseKey) > 1) {
          auto &shared = getDynamicBaseRead(group);
          readValue = shared.value;
          group.readPort = shared.port;
        } else {
          readValue = converter.createNewInput(
              origPort, buildSuffix(readSuffix, group), group.portType,
              group.readPort);
        }
      } else {
        readValue = converter.createNewInput(origPort, buildSuffix(readSuffix, group),
                                             group.elementType, group.readPort);
      }
      group.readValue = readValue;
    }

    bool resolveReaders = resolveReadWrite && hasReaders && hasWriters;
    if (resolveReaders)
      computeInternalDrive();

    if (group.dynamicIndex) {
      assert(group.dynamicPathIndex >= 0 && "dynamic path index not set");
      const auto &dynamicStep = group.path[group.dynamicPathIndex];
      Value dynamicIndexValue = dynamicStep.dynamicIndexValue;
      assert(dynamicIndexValue && "dynamic index value not set");
      auto suffix = ArrayRef<AccessGroup::Step>(group.path)
                        .drop_front(group.dynamicPathIndex + 1);
      if (hasReaders) {
        for (auto reader : group.readers) {
          Value element = hw::ArrayGetOp::create(
              builder, reader.getLoc(), readValue, dynamicIndexValue);
          element = applySuffixExtract(element, suffix, reader.getLoc());
          if (resolveReaders) {
            SmallVector<Value, 2> resolveValues;
            resolveValues.push_back(element);
            resolveValues.push_back(group.internalDrive);
            Value resolved = resolveFourStateValues(builder, reader.getLoc(),
                                                    resolveValues);
            assert(resolved && "expected resolved 4-state inout read value");
            element = resolved;
          }
          reader.replaceAllUsesWith(element);
          reader.erase();
        }
      }
      if (!resolveReaders)
        computeInternalDrive();
      if (hasWriters) {
        group.driveLoc = group.writers.front().getLoc();
        dynamicWriterGroups[dynamicBaseKey(group)].push_back(&group);
        for (auto writer : group.writers)
          writer.erase();
      }
      continue;
    }

    if (!resolveReaders)
      computeInternalDrive();

    if (hasReaders) {
      Value element = readValue;
      if (resolveReaders) {
        SmallVector<Value, 2> resolveValues;
        resolveValues.push_back(readValue);
        resolveValues.push_back(group.internalDrive);
        Value resolved =
            resolveFourStateValues(builder, readValue.getLoc(), resolveValues);
        assert(resolved && "expected resolved 4-state inout read value");
        element = resolved;
      }
      for (auto reader : group.readers) {
        reader.replaceAllUsesWith(element);
        reader.erase();
      }
    }

    if (hasWriters) {
      converter.createNewOutput(origPort, buildSuffix(writeSuffix, group),
                                group.elementType, group.internalDrive,
                                group.writePort);
      group.driveLoc = group.writers.front().getLoc();
      for (auto writer : group.writers)
        writer.erase();
    }
  }

  for (auto &entry : dynamicWriterGroups) {
    auto &groups = entry.second;
    if (groups.empty())
      continue;
    AccessGroup *baseGroup = groups.front();
    Value currentArray = baseGroup->readValue;
    bool resolveDynamicWrites =
        resolveReadWrite && isFourStateType(baseGroup->elementType) &&
        groups.size() > 1;
    for (AccessGroup *group : groups) {
      assert(group->dynamicPathIndex >= 0 && "dynamic path index not set");
      const auto &dynamicStep = group->path[group->dynamicPathIndex];
      Value dynamicIndexValue = dynamicStep.dynamicIndexValue;
      assert(dynamicIndexValue && "dynamic index value not set");
      auto suffix = ArrayRef<AccessGroup::Step>(group->path)
                        .drop_front(group->dynamicPathIndex + 1);
      Location loc =
          group->driveLoc.value_or(builder.getUnknownLoc());
      Value element =
          hw::ArrayGetOp::create(builder, loc, currentArray, dynamicIndexValue);
      Value updatedElement;
      if (resolveDynamicWrites) {
        Value existingLeaf = applySuffixExtract(element, suffix, loc);
        Value resolvedLeaf = resolveFourStateValues(
            builder, loc, {existingLeaf, group->internalDrive});
        assert(resolvedLeaf && "expected resolved dynamic inout drive value");
        updatedElement =
            applySuffixInject(applySuffixInject, element, suffix, resolvedLeaf,
                              loc);
      } else {
        updatedElement =
            applySuffixInject(applySuffixInject, element, suffix,
                              group->internalDrive, loc);
      }
      currentArray = hw::ArrayInjectOp::create(builder, loc, currentArray,
                                               dynamicIndexValue, updatedElement);
    }
    converter.createNewOutput(origPort,
                              buildDynamicBaseSuffix(writeSuffix, *baseGroup),
                              baseGroup->portType, currentArray,
                              baseGroup->writePort);
    for (AccessGroup *group : groups)
      group->writePort = baseGroup->writePort;
  }

  for (auto fieldOp : llvm::reverse(derivedInouts))
    if (fieldOp->use_empty())
      fieldOp->erase();
}

void HWInOutPortConversion::buildOutputSignals() {
  assert(false &&
         "`hw.inout` outputs not yet supported. Currently, `hw.inout` "
         "outputs are handled by UntouchedPortConversion, given that "
         "output `hw.inout` ports have a `ModulePort::Direction::Output` "
         "direction instead of `ModulePort::Direction::InOut`. If this for "
         "some reason changes, then this assert will fire.");
}

void HWInOutPortConversion::mapInputSignals(OpBuilder &b, Operation *inst,
                                            Value instValue,
                                            SmallVectorImpl<Value> &newOperands,
                                            ArrayRef<Backedge> newResults) {
  llvm::DenseSet<unsigned> assignedWritePorts;
  llvm::DenseSet<unsigned> assignedReadPorts;
  for (auto &group : accessGroups) {
    Value targetInOut = instValue;
    ArrayRef<AccessGroup::Step> prefixPath = group.path;
    if (group.dynamicIndex) {
      assert(group.dynamicPathIndex >= 0 && "dynamic path index not set");
      prefixPath = ArrayRef<AccessGroup::Step>(group.path)
                       .take_front(group.dynamicPathIndex);
    }
    if (!prefixPath.empty())
      targetInOut =
          buildDerivedInOut(b, inst->getLoc(), instValue, prefixPath);

    bool needsReadPort =
        !group.readers.empty() || (group.dynamicIndex && !group.writers.empty());
    if (needsReadPort) {
      // Create a read_inout op at the instantiation point. This effectively
      // pushes the read_inout op from the module to the instantiation site.
      unsigned argNum = group.readPort.argNum;
      if (assignedReadPorts.insert(argNum).second)
        newOperands[argNum] =
            ReadInOutOp::create(b, inst->getLoc(), targetInOut).getResult();
    }

    if (!group.writers.empty()) {
      // Create a sv::AssignOp at the instantiation point. This effectively
      // pushes the write op from the module to the instantiation site.
      unsigned argNum = group.writePort.argNum;
      if (!assignedWritePorts.insert(argNum).second)
        continue;
      Value writeFromInsideMod = newResults[argNum];
      sv::AssignOp::create(b, inst->getLoc(), targetInOut, writeFromInsideMod);
    }
  }
}

void HWInOutPortConversion::mapOutputSignals(
    OpBuilder &b, Operation *inst, Value instValue,
    SmallVectorImpl<Value> &newOperands, ArrayRef<Backedge> newResults) {
  // FIXME: hw.inout cannot be used in outputs.
  assert(false &&
         "`hw.inout` outputs not yet supported. Currently, `hw.inout` "
         "outputs are handled by UntouchedPortConversion, given that "
         "output `hw.inout` ports have a `ModulePort::Direction::Output` "
         "direction instead of `ModulePort::Direction::InOut`. If this for "
         "some reason changes, then this assert will fire.");
}

class HWInoutPortConversionBuilder : public PortConversionBuilder {
public:
  HWInoutPortConversionBuilder(PortConverterImpl &converter,
                               llvm::StringRef readSuffix,
                               llvm::StringRef writeSuffix,
                               bool allowMultipleWritersSameValue,
                               bool resolveReadWrite)
      : PortConversionBuilder(converter), readSuffix(readSuffix),
        writeSuffix(writeSuffix),
        allowMultipleWritersSameValue(allowMultipleWritersSameValue),
        resolveReadWrite(resolveReadWrite) {}

  FailureOr<std::unique_ptr<PortConversion>> build(hw::PortInfo port) override {
    if (port.dir == hw::ModulePort::Direction::InOut)
      return {std::make_unique<HWInOutPortConversion>(converter, port,
                                                      readSuffix, writeSuffix,
                                                      allowMultipleWritersSameValue,
                                                      resolveReadWrite)};
    return PortConversionBuilder::build(port);
  }

private:
  llvm::StringRef readSuffix;
  llvm::StringRef writeSuffix;
  bool allowMultipleWritersSameValue;
  bool resolveReadWrite;
};

} // namespace

void HWEliminateInOutPortsPass::runOnOperation() {
  // Find all modules and run port conversion on them.
  circt::hw::InstanceGraph &instanceGraph =
      getAnalysis<circt::hw::InstanceGraph>();
  llvm::DenseSet<InstanceGraphNode *> visited;
  FailureOr<llvm::ArrayRef<InstanceGraphNode *>> res =
      instanceGraph.getInferredTopLevelNodes();

  if (failed(res)) {
    signalPassFailure();
    return;
  }

  // Visit the instance hierarchy in a depth-first manner, modifying child
  // modules and their ports before their parents.

  // Doing this DFS ensures that all module instance uses of an inout value has
  // been converted before the current instance use. E.g. say you have m1 -> m2
  // -> m3 where both m3 and m2 reads an inout value defined in m1. If we don't
  // do DFS, and we just randomly pick a module, we have to e.g. select m2, see
  // that it also passes that inout value to other module instances, processes
  // those first (which may bubble up read/writes to that hw.inout op), and then
  // process m2... which in essence is a DFS traversal. So we just go ahead and
  // do the DFS to begin with, ensuring the invariant that all module instance
  // uses of an inout value have been converted before converting any given
  // module.

  for (InstanceGraphNode *topModule : res.value()) {
    for (InstanceGraphNode *node : llvm::post_order(topModule)) {
      if (visited.count(node))
        continue;
      auto mutableModule =
          dyn_cast_or_null<hw::HWMutableModuleLike>(*node->getModule());
      if (!mutableModule)
        continue;
      if (failed(PortConverter<HWInoutPortConversionBuilder>(
                     instanceGraph, mutableModule, readSuffix.getValue(),
                     writeSuffix.getValue(), allowMultipleWritersSameValue,
                     resolveReadWrite)
                     .run()))
        return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> circt::sv::createHWEliminateInOutPortsPass(
    const HWEliminateInOutPortsOptions &options) {
  return std::make_unique<HWEliminateInOutPortsPass>(options);
}
