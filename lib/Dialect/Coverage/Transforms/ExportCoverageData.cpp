//===- ExportCoverageData.cpp - Export coverage metadata ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass exports coverage instrumentation metadata for runtime use.
// It produces a CoverageDatabase that can be used with the circt-cov tool
// for merging, reporting, and trend tracking.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Coverage/CoverageOps.h"
#include "circt/Dialect/Coverage/CoveragePasses.h"
#include "circt/Support/CoverageDatabase.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

namespace circt {
namespace coverage {
#define GEN_PASS_DEF_EXPORTCOVERAGEDATA
#include "circt/Dialect/Coverage/CoveragePasses.h.inc"
} // namespace coverage
} // namespace circt

using namespace circt;
using namespace coverage;

namespace {
struct ExportCoverageDataPass
    : public coverage::impl::ExportCoverageDataBase<ExportCoverageDataPass> {
  using ExportCoverageDataBase::ExportCoverageDataBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Build a CoverageDatabase from the coverage operations
    CoverageDatabase db;

    // Set metadata
    db.setMetadata("source", "circt-coverage-export");
    if (auto moduleName = moduleOp.getName())
      db.setMetadata("module", moduleName->str());

    // Collect all coverage points
    moduleOp.walk([&](mlir::Operation *op) {
      if (auto lineOp = llvm::dyn_cast<LineCoverageOp>(op)) {
        processLineCoverage(lineOp, db);
      } else if (auto toggleOp = llvm::dyn_cast<ToggleCoverageOp>(op)) {
        processToggleCoverage(toggleOp, db);
      } else if (auto branchOp = llvm::dyn_cast<BranchCoverageOp>(op)) {
        processBranchCoverage(branchOp, db);
      } else if (auto fsmStateOp = llvm::dyn_cast<FSMStateCoverageOp>(op)) {
        processFSMStateCoverage(fsmStateOp, db);
      } else if (auto fsmTransOp = llvm::dyn_cast<FSMTransitionCoverageOp>(op)) {
        processFSMTransitionCoverage(fsmTransOp, db);
      } else if (auto exprOp = llvm::dyn_cast<ExpressionCoverageOp>(op)) {
        processExpressionCoverage(exprOp, db);
      } else if (auto assertOp = llvm::dyn_cast<AssertionCoverageOp>(op)) {
        processAssertionCoverage(assertOp, db);
      } else if (auto groupOp = llvm::dyn_cast<CoverageGroupOp>(op)) {
        processCoverageGroup(groupOp, db);
      }
    });

    // Export based on format
    llvm::Error err = llvm::Error::success();
    if (format == "json") {
      err = db.writeToJSON(outputFile);
    } else if (format == "binary" || format == "cov") {
      err = db.writeToFile(outputFile);
    } else if (format == "legacy-json") {
      // Legacy JSON format for backward compatibility
      exportLegacyJSON(db);
      return;
    } else if (format == "legacy-binary") {
      // Legacy binary format for backward compatibility
      exportLegacyBinary(db);
      return;
    } else {
      // Default to new binary format
      err = db.writeToFile(outputFile);
    }

    if (err) {
      emitError(getOperation().getLoc())
          << "failed to write coverage database: "
          << llvm::toString(std::move(err));
      signalPassFailure();
    }
  }

private:
  void processLineCoverage(LineCoverageOp op, CoverageDatabase &db) {
    CoveragePoint point;
    point.name = op.getCoveragePointId();
    point.type = CoverageType::Line;
    point.hits = 0; // Will be filled during simulation
    point.goal = 1;

    point.location.filename = op.getFilename().str();
    point.location.line = op.getLine();

    if (op.getTag())
      point.metadata["tag"] = op.getTag()->str();

    // Try to extract hierarchy from parent operations
    if (auto *parent = op->getParentOp()) {
      if (auto groupOp = llvm::dyn_cast<CoverageGroupOp>(parent)) {
        point.hierarchy = groupOp.getName().str();
      }
    }

    db.addCoveragePoint(point);
  }

  void processToggleCoverage(ToggleCoverageOp op, CoverageDatabase &db) {
    CoveragePoint point;
    point.name = op.getName().str();
    point.type = CoverageType::Toggle;
    point.hits = 0;
    point.goal = 1;
    point.toggle01 = false;
    point.toggle10 = false;

    if (op.getHierarchy())
      point.hierarchy = op.getHierarchy()->str();

    // Store signal width in metadata
    unsigned width = op.getSignalWidth();
    point.metadata["width"] = std::to_string(width);

    // For multi-bit signals, we might want to create separate points per bit
    // For now, treat as a single toggle point

    db.addCoveragePoint(point);
  }

  void processBranchCoverage(BranchCoverageOp op, CoverageDatabase &db) {
    CoveragePoint point;
    point.name = op.getName().str();
    point.type = CoverageType::Branch;
    point.hits = 0;
    point.goal = 1;
    point.branchTrue = false;
    point.branchFalse = false;

    if (op.getFilename())
      point.location.filename = op.getFilename()->str();
    if (op.getLine())
      point.location.line = *op.getLine();

    point.metadata["true_id"] = std::to_string(op.getTrueId());
    point.metadata["false_id"] = std::to_string(op.getFalseId());

    db.addCoveragePoint(point);
  }

  void processFSMStateCoverage(FSMStateCoverageOp op, CoverageDatabase &db) {
    CoveragePoint point;
    point.name = op.getCoveragePointId();
    point.type = CoverageType::FSMState;
    point.hits = 0;
    point.goal = op.getNumStates();

    if (op.getHierarchy())
      point.hierarchy = op.getHierarchy()->str();

    point.metadata["num_states"] = std::to_string(op.getNumStates());
    point.metadata["width"] = std::to_string(op.getStateWidth());

    // Store state names if provided
    if (auto stateNames = op.getStateNames()) {
      std::string names;
      for (auto attr : *stateNames) {
        if (!names.empty())
          names += ",";
        names += llvm::cast<StringAttr>(attr).getValue().str();
      }
      if (!names.empty())
        point.metadata["state_names"] = names;
    }

    db.addCoveragePoint(point);
  }

  void processFSMTransitionCoverage(FSMTransitionCoverageOp op,
                                    CoverageDatabase &db) {
    CoveragePoint point;
    point.name = op.getCoveragePointId();
    point.type = CoverageType::FSMTransition;
    point.hits = 0;
    // Goal is N^2 possible transitions for N states
    int32_t numStates = op.getNumStates();
    point.goal = numStates * numStates;

    if (op.getHierarchy())
      point.hierarchy = op.getHierarchy()->str();

    point.metadata["num_states"] = std::to_string(numStates);
    point.metadata["width"] = std::to_string(op.getStateWidth());

    db.addCoveragePoint(point);
  }

  void processExpressionCoverage(ExpressionCoverageOp op,
                                 CoverageDatabase &db) {
    CoveragePoint point;
    point.name = op.getCoveragePointId();
    point.type = CoverageType::Expression;
    point.hits = 0;
    // Goal for MC/DC is 2*N unique independence pairs (where N is # conditions)
    unsigned numConds = op.getNumConditions();
    point.goal = 2 * numConds;

    if (op.getHierarchy())
      point.hierarchy = op.getHierarchy()->str();

    point.metadata["num_conditions"] = std::to_string(numConds);

    // Store condition names if provided
    if (auto condNames = op.getConditionNames()) {
      std::string names;
      for (auto attr : *condNames) {
        if (!names.empty())
          names += ",";
        names += llvm::cast<StringAttr>(attr).getValue().str();
      }
      if (!names.empty())
        point.metadata["condition_names"] = names;
    }

    db.addCoveragePoint(point);
  }

  void processAssertionCoverage(AssertionCoverageOp op, CoverageDatabase &db) {
    CoveragePoint point;
    point.name = op.getCoveragePointId();
    point.type = CoverageType::Assertion;
    point.hits = 0;
    point.goal = 1;

    if (op.getHierarchy())
      point.hierarchy = op.getHierarchy()->str();

    if (op.getFilename())
      point.location.filename = op.getFilename()->str();
    if (op.getLine())
      point.location.line = *op.getLine();

    db.addCoveragePoint(point);
  }

  void processCoverageGroup(CoverageGroupOp op, CoverageDatabase &db) {
    CoverageGroup group;
    group.name = op.getName().str();
    if (op.getDescription())
      group.description = op.getDescription()->str();

    // Collect all coverage points within this group
    op.walk([&](mlir::Operation *childOp) {
      if (auto lineOp = llvm::dyn_cast<LineCoverageOp>(childOp)) {
        group.pointNames.push_back(lineOp.getCoveragePointId());
      } else if (auto toggleOp = llvm::dyn_cast<ToggleCoverageOp>(childOp)) {
        group.pointNames.push_back(toggleOp.getName().str());
      } else if (auto branchOp = llvm::dyn_cast<BranchCoverageOp>(childOp)) {
        group.pointNames.push_back(branchOp.getName().str());
      }
    });

    db.addCoverageGroup(group);
  }

  // Legacy export methods for backward compatibility
  void exportLegacyJSON(const CoverageDatabase &db) {
    llvm::json::Object root;

    // Export line coverage points
    llvm::json::Array lineArray;
    size_t lineId = 0;
    for (const auto &kv : db.getCoveragePoints()) {
      const auto &point = kv.second;
      if (point.type != CoverageType::Line)
        continue;

      llvm::json::Object obj;
      obj["id"] = static_cast<int64_t>(lineId++);
      obj["filename"] = point.location.filename;
      obj["line"] = static_cast<int64_t>(point.location.line);
      auto tagIt = point.metadata.find("tag");
      if (tagIt != point.metadata.end())
        obj["tag"] = tagIt->second;
      lineArray.push_back(std::move(obj));
    }
    root["line_coverage"] = std::move(lineArray);

    // Export toggle coverage points
    llvm::json::Array toggleArray;
    size_t toggleId = 0;
    for (const auto &kv : db.getCoveragePoints()) {
      const auto &point = kv.second;
      if (point.type != CoverageType::Toggle)
        continue;

      llvm::json::Object obj;
      obj["id"] = static_cast<int64_t>(toggleId++);
      obj["name"] = point.name;
      auto widthIt = point.metadata.find("width");
      if (widthIt != point.metadata.end())
        obj["width"] = std::stoi(widthIt->second);
      if (!point.hierarchy.empty())
        obj["hierarchy"] = point.hierarchy;
      toggleArray.push_back(std::move(obj));
    }
    root["toggle_coverage"] = std::move(toggleArray);

    // Export branch coverage points
    llvm::json::Array branchArray;
    size_t branchId = 0;
    for (const auto &kv : db.getCoveragePoints()) {
      const auto &point = kv.second;
      if (point.type != CoverageType::Branch)
        continue;

      llvm::json::Object obj;
      obj["id"] = static_cast<int64_t>(branchId++);
      obj["name"] = point.name;
      auto trueIdIt = point.metadata.find("true_id");
      if (trueIdIt != point.metadata.end())
        obj["true_id"] = std::stoi(trueIdIt->second);
      auto falseIdIt = point.metadata.find("false_id");
      if (falseIdIt != point.metadata.end())
        obj["false_id"] = std::stoi(falseIdIt->second);
      if (!point.location.filename.empty())
        obj["filename"] = point.location.filename;
      if (point.location.line > 0)
        obj["line"] = static_cast<int64_t>(point.location.line);
      branchArray.push_back(std::move(obj));
    }
    root["branch_coverage"] = std::move(branchArray);

    // Summary
    llvm::json::Object summary;
    summary["total_line_points"] = static_cast<int64_t>(lineId);
    summary["total_toggle_points"] = static_cast<int64_t>(toggleId);
    summary["total_branch_points"] = static_cast<int64_t>(branchId);
    root["summary"] = std::move(summary);

    // Write to file
    std::error_code ec;
    llvm::raw_fd_ostream os(outputFile, ec);
    if (ec) {
      emitError(getOperation().getLoc())
          << "failed to open output file: " << ec.message();
      signalPassFailure();
      return;
    }

    os << llvm::json::Value(std::move(root));
  }

  void exportLegacyBinary(const CoverageDatabase &db) {
    // Binary format:
    // Header:
    //   4 bytes: magic "CCOV"
    //   4 bytes: version (1)
    //   4 bytes: number of line coverage points
    //   4 bytes: number of toggle coverage points
    //   4 bytes: number of branch coverage points
    // Then data for each type follows

    std::error_code ec;
    llvm::raw_fd_ostream os(outputFile, ec);
    if (ec) {
      emitError(getOperation().getLoc())
          << "failed to open output file: " << ec.message();
      signalPassFailure();
      return;
    }

    // Count points by type
    uint32_t numLine = 0, numToggle = 0, numBranch = 0;
    for (const auto &kv : db.getCoveragePoints()) {
      switch (kv.second.type) {
      case CoverageType::Line:
        ++numLine;
        break;
      case CoverageType::Toggle:
        ++numToggle;
        break;
      case CoverageType::Branch:
        ++numBranch;
        break;
      default:
        break;
      }
    }

    // Write header
    os << "CCOV";
    uint32_t version = 1;
    os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    os.write(reinterpret_cast<const char *>(&numLine), sizeof(numLine));
    os.write(reinterpret_cast<const char *>(&numToggle), sizeof(numToggle));
    os.write(reinterpret_cast<const char *>(&numBranch), sizeof(numBranch));

    // Write line coverage data
    for (const auto &kv : db.getCoveragePoints()) {
      const auto &point = kv.second;
      if (point.type != CoverageType::Line)
        continue;

      uint32_t line = point.location.line;
      os.write(reinterpret_cast<const char *>(&line), sizeof(line));
      uint32_t filenameLen = point.location.filename.size();
      os.write(reinterpret_cast<const char *>(&filenameLen),
               sizeof(filenameLen));
      os << point.location.filename;
    }

    // Write toggle coverage data
    for (const auto &kv : db.getCoveragePoints()) {
      const auto &point = kv.second;
      if (point.type != CoverageType::Toggle)
        continue;

      uint32_t width = 1;
      auto widthIt = point.metadata.find("width");
      if (widthIt != point.metadata.end())
        width = std::stoi(widthIt->second);
      os.write(reinterpret_cast<const char *>(&width), sizeof(width));
      uint32_t nameLen = point.name.size();
      os.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
      os << point.name;
    }

    // Write branch coverage data
    for (const auto &kv : db.getCoveragePoints()) {
      const auto &point = kv.second;
      if (point.type != CoverageType::Branch)
        continue;

      uint32_t trueId = 0, falseId = 0;
      auto trueIdIt = point.metadata.find("true_id");
      if (trueIdIt != point.metadata.end())
        trueId = std::stoi(trueIdIt->second);
      auto falseIdIt = point.metadata.find("false_id");
      if (falseIdIt != point.metadata.end())
        falseId = std::stoi(falseIdIt->second);

      os.write(reinterpret_cast<const char *>(&trueId), sizeof(trueId));
      os.write(reinterpret_cast<const char *>(&falseId), sizeof(falseId));
      uint32_t nameLen = point.name.size();
      os.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
      os << point.name;
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
coverage::createExportCoverageDataPass(const ExportCoverageDataOptions &options) {
  return std::make_unique<ExportCoverageDataPass>(options);
}
