//===- CoverageDatabase.cpp - Coverage data storage -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CoverageDatabase class for storing and manipulating
// coverage data.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CoverageDatabase.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace circt;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

StringRef circt::getCoverageTypeName(CoverageType type) {
  switch (type) {
  case CoverageType::Line:
    return "line";
  case CoverageType::Toggle:
    return "toggle";
  case CoverageType::Branch:
    return "branch";
  case CoverageType::Condition:
    return "condition";
  case CoverageType::FSM:
    return "fsm";
  case CoverageType::Assertion:
    return "assertion";
  case CoverageType::Coverpoint:
    return "coverpoint";
  }
  return "unknown";
}

std::optional<CoverageType> circt::parseCoverageType(StringRef name) {
  if (name == "line")
    return CoverageType::Line;
  if (name == "toggle")
    return CoverageType::Toggle;
  if (name == "branch")
    return CoverageType::Branch;
  if (name == "condition")
    return CoverageType::Condition;
  if (name == "fsm")
    return CoverageType::FSM;
  if (name == "assertion")
    return CoverageType::Assertion;
  if (name == "coverpoint")
    return CoverageType::Coverpoint;
  return std::nullopt;
}

static std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

//===----------------------------------------------------------------------===//
// CoveragePoint Implementation
//===----------------------------------------------------------------------===//

bool CoveragePoint::isCovered() const {
  switch (type) {
  case CoverageType::Line:
  case CoverageType::Assertion:
  case CoverageType::Coverpoint:
    return hits >= goal;

  case CoverageType::Toggle:
    return toggle01 && toggle10;

  case CoverageType::Branch:
    return branchTrue && branchFalse;

  case CoverageType::Condition:
  case CoverageType::FSM:
    return hits >= goal;
  }
  return false;
}

double CoveragePoint::getCoveragePercent() const {
  switch (type) {
  case CoverageType::Line:
  case CoverageType::Assertion:
  case CoverageType::Coverpoint:
  case CoverageType::Condition:
  case CoverageType::FSM:
    if (goal == 0)
      return 100.0;
    return std::min(100.0, (static_cast<double>(hits) / goal) * 100.0);

  case CoverageType::Toggle: {
    int covered = (toggle01 ? 1 : 0) + (toggle10 ? 1 : 0);
    return covered * 50.0;
  }

  case CoverageType::Branch: {
    int covered = (branchTrue ? 1 : 0) + (branchFalse ? 1 : 0);
    return covered * 50.0;
  }
  }
  return 0.0;
}

void CoveragePoint::merge(const CoveragePoint &other) {
  // Accumulate hits
  hits += other.hits;

  // Merge toggle coverage
  toggle01 = toggle01 || other.toggle01;
  toggle10 = toggle10 || other.toggle10;

  // Merge branch coverage
  branchTrue = branchTrue || other.branchTrue;
  branchFalse = branchFalse || other.branchFalse;

  // Merge metadata
  for (const auto &kv : other.metadata) {
    if (!metadata.count(kv.first()))
      metadata[kv.first()] = kv.second;
  }
}

//===----------------------------------------------------------------------===//
// CoverageGroup Implementation
//===----------------------------------------------------------------------===//

double CoverageGroup::getCoveragePercent(
    const llvm::StringMap<CoveragePoint> &allPoints) const {
  if (pointNames.empty())
    return 100.0;

  double totalPercent = 0.0;
  size_t validPoints = 0;

  for (const auto &name : pointNames) {
    auto it = allPoints.find(name);
    if (it != allPoints.end()) {
      totalPercent += it->second.getCoveragePercent();
      ++validPoints;
    }
  }

  if (validPoints == 0)
    return 100.0;

  return totalPercent / validPoints;
}

//===----------------------------------------------------------------------===//
// CoverageDatabase Implementation
//===----------------------------------------------------------------------===//

void CoverageDatabase::addCoveragePoint(const CoveragePoint &point) {
  coveragePoints[point.name] = point;
}

void CoverageDatabase::recordHit(StringRef name, uint64_t count) {
  auto it = coveragePoints.find(name);
  if (it != coveragePoints.end()) {
    it->second.hits += count;
  }
}

const CoveragePoint *CoverageDatabase::getCoveragePoint(StringRef name) const {
  auto it = coveragePoints.find(name);
  if (it != coveragePoints.end())
    return &it->second;
  return nullptr;
}

bool CoverageDatabase::hasCoveragePoint(StringRef name) const {
  return coveragePoints.count(name) > 0;
}

std::vector<const CoveragePoint *>
CoverageDatabase::getCoveragePointsByType(CoverageType type) const {
  std::vector<const CoveragePoint *> result;
  for (const auto &kv : coveragePoints) {
    if (kv.second.type == type)
      result.push_back(&kv.second);
  }
  return result;
}

std::vector<const CoveragePoint *>
CoverageDatabase::getCoveragePointsByHierarchy(StringRef hierarchyPrefix) const {
  std::vector<const CoveragePoint *> result;
  for (const auto &kv : coveragePoints) {
    if (StringRef(kv.second.hierarchy).starts_with(hierarchyPrefix))
      result.push_back(&kv.second);
  }
  return result;
}

void CoverageDatabase::addCoverageGroup(const CoverageGroup &group) {
  coverageGroups[group.name] = group;
}

const CoverageGroup *CoverageDatabase::getCoverageGroup(StringRef name) const {
  auto it = coverageGroups.find(name);
  if (it != coverageGroups.end())
    return &it->second;
  return nullptr;
}

void CoverageDatabase::addExclusion(const CoverageExclusion &exclusion) {
  // Check if already excluded
  if (exclusionIndex.count(exclusion.pointName))
    return;

  exclusionIndex[exclusion.pointName] = exclusions.size();
  exclusions.push_back(exclusion);
}

bool CoverageDatabase::isExcluded(StringRef pointName) const {
  return exclusionIndex.count(pointName) > 0;
}

const CoverageExclusion *
CoverageDatabase::getExclusion(StringRef pointName) const {
  auto it = exclusionIndex.find(pointName);
  if (it != exclusionIndex.end())
    return &exclusions[it->second];
  return nullptr;
}

llvm::Error CoverageDatabase::loadExclusions(StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return llvm::createStringError(bufferOrErr.getError(),
                                   "Failed to open exclusions file");

  auto jsonOrErr = llvm::json::parse(bufferOrErr.get()->getBuffer());
  if (!jsonOrErr)
    return jsonOrErr.takeError();

  auto *array = jsonOrErr->getAsArray();
  if (!array)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Exclusions file must contain a JSON array");

  for (const auto &item : *array) {
    auto *obj = item.getAsObject();
    if (!obj)
      continue;

    CoverageExclusion exclusion;
    if (auto name = obj->getString("name"))
      exclusion.pointName = name->str();
    if (auto reason = obj->getString("reason"))
      exclusion.reason = reason->str();
    if (auto author = obj->getString("author"))
      exclusion.author = author->str();
    if (auto date = obj->getString("date"))
      exclusion.date = date->str();
    if (auto ticket = obj->getString("ticket"))
      exclusion.ticketId = ticket->str();

    if (!exclusion.pointName.empty())
      addExclusion(exclusion);
  }

  return llvm::Error::success();
}

llvm::Error CoverageDatabase::saveExclusions(StringRef path) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    return llvm::createStringError(ec, "Failed to open file for writing");

  llvm::json::Array array;
  for (const auto &exclusion : exclusions) {
    llvm::json::Object obj;
    obj["name"] = exclusion.pointName;
    obj["reason"] = exclusion.reason;
    obj["author"] = exclusion.author;
    obj["date"] = exclusion.date;
    if (!exclusion.ticketId.empty())
      obj["ticket"] = exclusion.ticketId;
    array.push_back(std::move(obj));
  }

  os << llvm::json::Value(std::move(array));
  return llvm::Error::success();
}

void CoverageDatabase::addTrendPoint(const CoverageTrendPoint &trend) {
  trends.push_back(trend);
}

CoverageTrendPoint
CoverageDatabase::createCurrentTrendPoint(StringRef runId,
                                          StringRef commitHash) const {
  CoverageTrendPoint trend;
  trend.timestamp = getCurrentTimestamp();
  trend.runId = runId.str();
  trend.commitHash = commitHash.str();
  trend.lineCoverage = getCoverageByType(CoverageType::Line);
  trend.toggleCoverage = getCoverageByType(CoverageType::Toggle);
  trend.branchCoverage = getCoverageByType(CoverageType::Branch);
  trend.overallCoverage = getOverallCoverage();
  trend.totalPoints = getTotalPointCount();
  trend.coveredPoints = getCoveredPointCount();
  return trend;
}

double CoverageDatabase::getOverallCoverage() const {
  if (coveragePoints.empty())
    return 100.0;

  double totalPercent = 0.0;
  size_t count = 0;

  for (const auto &kv : coveragePoints) {
    if (!isExcluded(kv.first())) {
      totalPercent += kv.second.getCoveragePercent();
      ++count;
    }
  }

  if (count == 0)
    return 100.0;

  return totalPercent / count;
}

double CoverageDatabase::getCoverageByType(CoverageType type) const {
  double totalPercent = 0.0;
  size_t count = 0;

  for (const auto &kv : coveragePoints) {
    if (kv.second.type == type && !isExcluded(kv.first())) {
      totalPercent += kv.second.getCoveragePercent();
      ++count;
    }
  }

  if (count == 0)
    return 100.0;

  return totalPercent / count;
}

size_t CoverageDatabase::getCoveredPointCount() const {
  size_t count = 0;
  for (const auto &kv : coveragePoints) {
    if (kv.second.isCovered() || isExcluded(kv.first()))
      ++count;
  }
  return count;
}

size_t CoverageDatabase::getCoveredPointCountByType(CoverageType type) const {
  size_t count = 0;
  for (const auto &kv : coveragePoints) {
    if (kv.second.type == type &&
        (kv.second.isCovered() || isExcluded(kv.first())))
      ++count;
  }
  return count;
}

size_t CoverageDatabase::getTotalPointCountByType(CoverageType type) const {
  size_t count = 0;
  for (const auto &kv : coveragePoints) {
    if (kv.second.type == type)
      ++count;
  }
  return count;
}

void CoverageDatabase::merge(const CoverageDatabase &other) {
  // Merge coverage points
  for (const auto &kv : other.coveragePoints) {
    auto it = coveragePoints.find(kv.first());
    if (it != coveragePoints.end()) {
      it->second.merge(kv.second);
    } else {
      coveragePoints[kv.first()] = kv.second;
    }
  }

  // Merge groups (take union)
  for (const auto &kv : other.coverageGroups) {
    if (!coverageGroups.count(kv.first()))
      coverageGroups[kv.first()] = kv.second;
  }

  // Merge exclusions
  for (const auto &exclusion : other.exclusions) {
    addExclusion(exclusion);
  }

  // Append trends
  for (const auto &trend : other.trends) {
    trends.push_back(trend);
  }

  // Merge metadata
  for (const auto &kv : other.metadata) {
    if (!metadata.count(kv.first()))
      metadata[kv.first()] = kv.second;
  }
}

CoverageDatabase::DiffResult
CoverageDatabase::diff(const CoverageDatabase &other) const {
  DiffResult result;

  // Find points in this database
  for (const auto &kv : coveragePoints) {
    auto otherIt = other.coveragePoints.find(kv.first());
    if (otherIt == other.coveragePoints.end()) {
      result.onlyInThis.push_back(kv.first().str());
    } else {
      bool thisCovered = kv.second.isCovered();
      bool otherCovered = otherIt->second.isCovered();

      if (thisCovered && !otherCovered)
        result.newlyCovered.push_back(kv.first().str());
      else if (!thisCovered && otherCovered)
        result.newlyUncovered.push_back(kv.first().str());
    }
  }

  // Find points only in other database
  for (const auto &kv : other.coveragePoints) {
    if (!coveragePoints.count(kv.first()))
      result.onlyInOther.push_back(kv.first().str());
  }

  result.coverageDelta = getOverallCoverage() - other.getOverallCoverage();

  return result;
}

llvm::Error CoverageDatabase::writeToFile(StringRef path) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return llvm::createStringError(ec, "Failed to open file for writing");

  // Write header
  uint32_t magic = MAGIC;
  uint32_t version = VERSION;
  os.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  os.write(reinterpret_cast<const char *>(&version), sizeof(version));

  // Write counts
  uint32_t pointCount = coveragePoints.size();
  uint32_t groupCount = coverageGroups.size();
  uint32_t exclusionCount = exclusions.size();
  uint32_t trendCount = trends.size();

  os.write(reinterpret_cast<const char *>(&pointCount), sizeof(pointCount));
  os.write(reinterpret_cast<const char *>(&groupCount), sizeof(groupCount));
  os.write(reinterpret_cast<const char *>(&exclusionCount),
           sizeof(exclusionCount));
  os.write(reinterpret_cast<const char *>(&trendCount), sizeof(trendCount));

  // Helper to write a string
  auto writeString = [&os](const std::string &str) {
    uint32_t len = str.size();
    os.write(reinterpret_cast<const char *>(&len), sizeof(len));
    os.write(str.data(), len);
  };

  // Write coverage points
  for (const auto &kv : coveragePoints) {
    const auto &point = kv.second;
    writeString(point.name);

    uint8_t typeVal = static_cast<uint8_t>(point.type);
    os.write(reinterpret_cast<const char *>(&typeVal), sizeof(typeVal));

    os.write(reinterpret_cast<const char *>(&point.hits), sizeof(point.hits));
    os.write(reinterpret_cast<const char *>(&point.goal), sizeof(point.goal));

    writeString(point.location.filename);
    os.write(reinterpret_cast<const char *>(&point.location.line),
             sizeof(point.location.line));
    os.write(reinterpret_cast<const char *>(&point.location.column),
             sizeof(point.location.column));

    writeString(point.hierarchy);
    writeString(point.description);

    uint8_t flags = (point.toggle01 ? 1 : 0) | (point.toggle10 ? 2 : 0) |
                    (point.branchTrue ? 4 : 0) | (point.branchFalse ? 8 : 0);
    os.write(reinterpret_cast<const char *>(&flags), sizeof(flags));
  }

  // Write groups
  for (const auto &kv : coverageGroups) {
    const auto &group = kv.second;
    writeString(group.name);
    writeString(group.description);

    uint32_t numPoints = group.pointNames.size();
    os.write(reinterpret_cast<const char *>(&numPoints), sizeof(numPoints));
    for (const auto &pointName : group.pointNames) {
      writeString(pointName);
    }
  }

  // Write exclusions
  for (const auto &exclusion : exclusions) {
    writeString(exclusion.pointName);
    writeString(exclusion.reason);
    writeString(exclusion.author);
    writeString(exclusion.date);
    writeString(exclusion.ticketId);
  }

  // Write trends
  for (const auto &trend : trends) {
    writeString(trend.timestamp);
    writeString(trend.runId);
    writeString(trend.commitHash);
    os.write(reinterpret_cast<const char *>(&trend.lineCoverage),
             sizeof(trend.lineCoverage));
    os.write(reinterpret_cast<const char *>(&trend.toggleCoverage),
             sizeof(trend.toggleCoverage));
    os.write(reinterpret_cast<const char *>(&trend.branchCoverage),
             sizeof(trend.branchCoverage));
    os.write(reinterpret_cast<const char *>(&trend.overallCoverage),
             sizeof(trend.overallCoverage));
    os.write(reinterpret_cast<const char *>(&trend.totalPoints),
             sizeof(trend.totalPoints));
    os.write(reinterpret_cast<const char *>(&trend.coveredPoints),
             sizeof(trend.coveredPoints));
  }

  return llvm::Error::success();
}

llvm::Expected<CoverageDatabase> CoverageDatabase::readFromFile(StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return llvm::createStringError(bufferOrErr.getError(),
                                   "Failed to open coverage database file");

  const char *data = bufferOrErr.get()->getBufferStart();
  const char *end = bufferOrErr.get()->getBufferEnd();
  const char *ptr = data;

  // Helper to read data
  auto readBytes = [&ptr, end](void *dest, size_t size) -> bool {
    if (ptr + size > end)
      return false;
    memcpy(dest, ptr, size);
    ptr += size;
    return true;
  };

  auto readString = [&ptr, end]() -> std::optional<std::string> {
    if (ptr + sizeof(uint32_t) > end)
      return std::nullopt;
    uint32_t len;
    memcpy(&len, ptr, sizeof(len));
    ptr += sizeof(len);
    if (ptr + len > end)
      return std::nullopt;
    std::string result(ptr, len);
    ptr += len;
    return result;
  };

  // Read header
  uint32_t magic, version;
  if (!readBytes(&magic, sizeof(magic)) || !readBytes(&version, sizeof(version)))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Truncated file header");

  if (magic != MAGIC)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Invalid magic number");

  if (version != VERSION)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Unsupported database version");

  // Read counts
  uint32_t pointCount, groupCount, exclusionCount, trendCount;
  if (!readBytes(&pointCount, sizeof(pointCount)) ||
      !readBytes(&groupCount, sizeof(groupCount)) ||
      !readBytes(&exclusionCount, sizeof(exclusionCount)) ||
      !readBytes(&trendCount, sizeof(trendCount)))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Truncated counts section");

  CoverageDatabase db;

  // Read coverage points
  for (uint32_t i = 0; i < pointCount; ++i) {
    CoveragePoint point;

    auto name = readString();
    if (!name)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read point name");
    point.name = *name;

    uint8_t typeVal;
    if (!readBytes(&typeVal, sizeof(typeVal)))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read point type");
    point.type = static_cast<CoverageType>(typeVal);

    if (!readBytes(&point.hits, sizeof(point.hits)) ||
        !readBytes(&point.goal, sizeof(point.goal)))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read point hits/goal");

    auto filename = readString();
    if (!filename)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read location filename");
    point.location.filename = *filename;

    if (!readBytes(&point.location.line, sizeof(point.location.line)) ||
        !readBytes(&point.location.column, sizeof(point.location.column)))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read location");

    auto hierarchy = readString();
    if (!hierarchy)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read hierarchy");
    point.hierarchy = *hierarchy;

    auto description = readString();
    if (!description)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read description");
    point.description = *description;

    uint8_t flags;
    if (!readBytes(&flags, sizeof(flags)))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read flags");
    point.toggle01 = (flags & 1) != 0;
    point.toggle10 = (flags & 2) != 0;
    point.branchTrue = (flags & 4) != 0;
    point.branchFalse = (flags & 8) != 0;

    db.addCoveragePoint(point);
  }

  // Read groups
  for (uint32_t i = 0; i < groupCount; ++i) {
    CoverageGroup group;

    auto name = readString();
    if (!name)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read group name");
    group.name = *name;

    auto description = readString();
    if (!description)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read group description");
    group.description = *description;

    uint32_t numPoints;
    if (!readBytes(&numPoints, sizeof(numPoints)))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read group point count");

    for (uint32_t j = 0; j < numPoints; ++j) {
      auto pointName = readString();
      if (!pointName)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Failed to read group point name");
      group.pointNames.push_back(*pointName);
    }

    db.addCoverageGroup(group);
  }

  // Read exclusions
  for (uint32_t i = 0; i < exclusionCount; ++i) {
    CoverageExclusion exclusion;

    auto pointName = readString();
    if (!pointName)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read exclusion point name");
    exclusion.pointName = *pointName;

    auto reason = readString();
    if (!reason)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read exclusion reason");
    exclusion.reason = *reason;

    auto author = readString();
    if (!author)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read exclusion author");
    exclusion.author = *author;

    auto date = readString();
    if (!date)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read exclusion date");
    exclusion.date = *date;

    auto ticketId = readString();
    if (!ticketId)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read exclusion ticket");
    exclusion.ticketId = *ticketId;

    db.addExclusion(exclusion);
  }

  // Read trends
  for (uint32_t i = 0; i < trendCount; ++i) {
    CoverageTrendPoint trend;

    auto timestamp = readString();
    if (!timestamp)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read trend timestamp");
    trend.timestamp = *timestamp;

    auto runId = readString();
    if (!runId)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read trend runId");
    trend.runId = *runId;

    auto commitHash = readString();
    if (!commitHash)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read trend commitHash");
    trend.commitHash = *commitHash;

    if (!readBytes(&trend.lineCoverage, sizeof(trend.lineCoverage)) ||
        !readBytes(&trend.toggleCoverage, sizeof(trend.toggleCoverage)) ||
        !readBytes(&trend.branchCoverage, sizeof(trend.branchCoverage)) ||
        !readBytes(&trend.overallCoverage, sizeof(trend.overallCoverage)) ||
        !readBytes(&trend.totalPoints, sizeof(trend.totalPoints)) ||
        !readBytes(&trend.coveredPoints, sizeof(trend.coveredPoints)))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Failed to read trend data");

    db.addTrendPoint(trend);
  }

  return db;
}

llvm::Error CoverageDatabase::writeToJSON(StringRef path) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    return llvm::createStringError(ec, "Failed to open file for writing");

  os << toJSON();
  return llvm::Error::success();
}

llvm::Expected<CoverageDatabase> CoverageDatabase::readFromJSON(StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return llvm::createStringError(bufferOrErr.getError(),
                                   "Failed to open JSON file");

  auto jsonOrErr = llvm::json::parse(bufferOrErr.get()->getBuffer());
  if (!jsonOrErr)
    return jsonOrErr.takeError();

  return fromJSON(*jsonOrErr);
}

llvm::json::Value CoverageDatabase::toJSON() const {
  llvm::json::Object root;

  // Version info
  root["version"] = static_cast<int64_t>(VERSION);

  // Coverage points
  llvm::json::Array pointsArray;
  for (const auto &kv : coveragePoints) {
    const auto &point = kv.second;
    llvm::json::Object obj;
    obj["name"] = point.name;
    obj["type"] = getCoverageTypeName(point.type).str();
    obj["hits"] = static_cast<int64_t>(point.hits);
    obj["goal"] = static_cast<int64_t>(point.goal);
    obj["covered"] = point.isCovered();
    obj["coverage_percent"] = point.getCoveragePercent();

    llvm::json::Object locObj;
    locObj["filename"] = point.location.filename;
    locObj["line"] = static_cast<int64_t>(point.location.line);
    locObj["column"] = static_cast<int64_t>(point.location.column);
    obj["location"] = std::move(locObj);

    obj["hierarchy"] = point.hierarchy;
    obj["description"] = point.description;

    if (point.type == CoverageType::Toggle) {
      obj["toggle_01"] = point.toggle01;
      obj["toggle_10"] = point.toggle10;
    }
    if (point.type == CoverageType::Branch) {
      obj["branch_true"] = point.branchTrue;
      obj["branch_false"] = point.branchFalse;
    }

    if (!point.metadata.empty()) {
      llvm::json::Object metaObj;
      for (const auto &meta : point.metadata) {
        metaObj[meta.first()] = meta.second;
      }
      obj["metadata"] = std::move(metaObj);
    }

    pointsArray.push_back(std::move(obj));
  }
  root["coverage_points"] = std::move(pointsArray);

  // Groups
  llvm::json::Array groupsArray;
  for (const auto &kv : coverageGroups) {
    const auto &group = kv.second;
    llvm::json::Object obj;
    obj["name"] = group.name;
    obj["description"] = group.description;
    obj["coverage_percent"] = group.getCoveragePercent(coveragePoints);

    llvm::json::Array namesArray;
    for (const auto &name : group.pointNames) {
      namesArray.push_back(name);
    }
    obj["points"] = std::move(namesArray);

    groupsArray.push_back(std::move(obj));
  }
  root["groups"] = std::move(groupsArray);

  // Exclusions
  llvm::json::Array exclusionsArray;
  for (const auto &exclusion : exclusions) {
    llvm::json::Object obj;
    obj["name"] = exclusion.pointName;
    obj["reason"] = exclusion.reason;
    obj["author"] = exclusion.author;
    obj["date"] = exclusion.date;
    if (!exclusion.ticketId.empty())
      obj["ticket"] = exclusion.ticketId;
    exclusionsArray.push_back(std::move(obj));
  }
  root["exclusions"] = std::move(exclusionsArray);

  // Trends
  llvm::json::Array trendsArray;
  for (const auto &trend : trends) {
    llvm::json::Object obj;
    obj["timestamp"] = trend.timestamp;
    obj["run_id"] = trend.runId;
    obj["commit_hash"] = trend.commitHash;
    obj["line_coverage"] = trend.lineCoverage;
    obj["toggle_coverage"] = trend.toggleCoverage;
    obj["branch_coverage"] = trend.branchCoverage;
    obj["overall_coverage"] = trend.overallCoverage;
    obj["total_points"] = static_cast<int64_t>(trend.totalPoints);
    obj["covered_points"] = static_cast<int64_t>(trend.coveredPoints);
    trendsArray.push_back(std::move(obj));
  }
  root["trends"] = std::move(trendsArray);

  // Summary
  llvm::json::Object summary;
  summary["total_points"] = static_cast<int64_t>(getTotalPointCount());
  summary["covered_points"] = static_cast<int64_t>(getCoveredPointCount());
  summary["overall_coverage"] = getOverallCoverage();
  summary["line_coverage"] = getCoverageByType(CoverageType::Line);
  summary["toggle_coverage"] = getCoverageByType(CoverageType::Toggle);
  summary["branch_coverage"] = getCoverageByType(CoverageType::Branch);
  root["summary"] = std::move(summary);

  // Metadata
  if (!metadata.empty()) {
    llvm::json::Object metaObj;
    for (const auto &kv : metadata) {
      metaObj[kv.first()] = kv.second;
    }
    root["metadata"] = std::move(metaObj);
  }

  return llvm::json::Value(std::move(root));
}

llvm::Expected<CoverageDatabase>
CoverageDatabase::fromJSON(const llvm::json::Value &json) {
  auto *root = json.getAsObject();
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "JSON root must be an object");

  CoverageDatabase db;

  // Parse coverage points
  if (auto *pointsArray = root->getArray("coverage_points")) {
    for (const auto &item : *pointsArray) {
      auto *obj = item.getAsObject();
      if (!obj)
        continue;

      CoveragePoint point;

      if (auto name = obj->getString("name"))
        point.name = name->str();

      if (auto typeStr = obj->getString("type")) {
        if (auto type = parseCoverageType(*typeStr))
          point.type = *type;
      }

      if (auto hits = obj->getInteger("hits"))
        point.hits = static_cast<uint64_t>(*hits);

      if (auto goal = obj->getInteger("goal"))
        point.goal = static_cast<uint64_t>(*goal);

      if (auto *locObj = obj->getObject("location")) {
        if (auto filename = locObj->getString("filename"))
          point.location.filename = filename->str();
        if (auto line = locObj->getInteger("line"))
          point.location.line = static_cast<uint32_t>(*line);
        if (auto column = locObj->getInteger("column"))
          point.location.column = static_cast<uint32_t>(*column);
      }

      if (auto hierarchy = obj->getString("hierarchy"))
        point.hierarchy = hierarchy->str();

      if (auto description = obj->getString("description"))
        point.description = description->str();

      if (auto toggle01 = obj->getBoolean("toggle_01"))
        point.toggle01 = *toggle01;
      if (auto toggle10 = obj->getBoolean("toggle_10"))
        point.toggle10 = *toggle10;
      if (auto branchTrue = obj->getBoolean("branch_true"))
        point.branchTrue = *branchTrue;
      if (auto branchFalse = obj->getBoolean("branch_false"))
        point.branchFalse = *branchFalse;

      if (auto *metaObj = obj->getObject("metadata")) {
        for (const auto &kv : *metaObj) {
          if (auto val = kv.second.getAsString())
            point.metadata[kv.first] = val->str();
        }
      }

      if (!point.name.empty())
        db.addCoveragePoint(point);
    }
  }

  // Parse groups
  if (auto *groupsArray = root->getArray("groups")) {
    for (const auto &item : *groupsArray) {
      auto *obj = item.getAsObject();
      if (!obj)
        continue;

      CoverageGroup group;

      if (auto name = obj->getString("name"))
        group.name = name->str();
      if (auto description = obj->getString("description"))
        group.description = description->str();

      if (auto *pointsArray = obj->getArray("points")) {
        for (const auto &point : *pointsArray) {
          if (auto name = point.getAsString())
            group.pointNames.push_back(name->str());
        }
      }

      if (!group.name.empty())
        db.addCoverageGroup(group);
    }
  }

  // Parse exclusions
  if (auto *exclusionsArray = root->getArray("exclusions")) {
    for (const auto &item : *exclusionsArray) {
      auto *obj = item.getAsObject();
      if (!obj)
        continue;

      CoverageExclusion exclusion;

      if (auto name = obj->getString("name"))
        exclusion.pointName = name->str();
      if (auto reason = obj->getString("reason"))
        exclusion.reason = reason->str();
      if (auto author = obj->getString("author"))
        exclusion.author = author->str();
      if (auto date = obj->getString("date"))
        exclusion.date = date->str();
      if (auto ticket = obj->getString("ticket"))
        exclusion.ticketId = ticket->str();

      if (!exclusion.pointName.empty())
        db.addExclusion(exclusion);
    }
  }

  // Parse trends
  if (auto *trendsArray = root->getArray("trends")) {
    for (const auto &item : *trendsArray) {
      auto *obj = item.getAsObject();
      if (!obj)
        continue;

      CoverageTrendPoint trend;

      if (auto timestamp = obj->getString("timestamp"))
        trend.timestamp = timestamp->str();
      if (auto runId = obj->getString("run_id"))
        trend.runId = runId->str();
      if (auto commitHash = obj->getString("commit_hash"))
        trend.commitHash = commitHash->str();
      if (auto lineCoverage = obj->getNumber("line_coverage"))
        trend.lineCoverage = *lineCoverage;
      if (auto toggleCoverage = obj->getNumber("toggle_coverage"))
        trend.toggleCoverage = *toggleCoverage;
      if (auto branchCoverage = obj->getNumber("branch_coverage"))
        trend.branchCoverage = *branchCoverage;
      if (auto overallCoverage = obj->getNumber("overall_coverage"))
        trend.overallCoverage = *overallCoverage;
      if (auto totalPoints = obj->getInteger("total_points"))
        trend.totalPoints = static_cast<uint64_t>(*totalPoints);
      if (auto coveredPoints = obj->getInteger("covered_points"))
        trend.coveredPoints = static_cast<uint64_t>(*coveredPoints);

      db.addTrendPoint(trend);
    }
  }

  // Parse metadata
  if (auto *metaObj = root->getObject("metadata")) {
    for (const auto &kv : *metaObj) {
      if (auto val = kv.second.getAsString())
        db.setMetadata(kv.first, *val);
    }
  }

  return db;
}

void CoverageDatabase::clear() {
  coveragePoints.clear();
  coverageGroups.clear();
  exclusions.clear();
  exclusionIndex.clear();
  trends.clear();
  metadata.clear();
}
