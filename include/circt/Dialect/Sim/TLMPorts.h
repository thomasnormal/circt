//===- TLMPorts.h - TLM Port infrastructure for UVM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TLM (Transaction Level Modeling) port infrastructure
// for UVM communication. This includes TLM FIFOs, analysis ports, and the
// blocking/non-blocking interfaces used for inter-component communication.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_TLMPORTS_H
#define CIRCT_DIALECT_SIM_TLMPORTS_H

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// TLMTransaction - Base class for TLM transactions
//===----------------------------------------------------------------------===//

/// Unique identifier for a transaction.
using TransactionId = uint64_t;

/// Invalid transaction ID constant.
constexpr TransactionId InvalidTransactionId = 0;

/// Base class for TLM transactions.
/// Transactions are the data objects passed through TLM interfaces.
class TLMTransaction {
public:
  TLMTransaction() : id(0), timestamp(0) {}
  explicit TLMTransaction(TransactionId id) : id(id), timestamp(0) {}
  virtual ~TLMTransaction() = default;

  /// Get the transaction ID.
  TransactionId getId() const { return id; }

  /// Set the transaction ID.
  void setId(TransactionId newId) { id = newId; }

  /// Get the timestamp when this transaction was created.
  uint64_t getTimestamp() const { return timestamp; }

  /// Set the timestamp.
  void setTimestamp(uint64_t ts) { timestamp = ts; }

  /// Clone the transaction (for analysis ports).
  virtual std::unique_ptr<TLMTransaction> clone() const {
    auto copy = std::make_unique<TLMTransaction>(id);
    copy->timestamp = timestamp;
    return copy;
  }

protected:
  TransactionId id;
  uint64_t timestamp;
};

//===----------------------------------------------------------------------===//
// TLMBlockingPutIf - Blocking put interface
//===----------------------------------------------------------------------===//

/// Interface for blocking put operations.
template <typename T>
class TLMBlockingPutIf {
public:
  virtual ~TLMBlockingPutIf() = default;

  /// Blocking put - waits until the transaction can be accepted.
  virtual void put(const T &t) = 0;
};

//===----------------------------------------------------------------------===//
// TLMNonBlockingPutIf - Non-blocking put interface
//===----------------------------------------------------------------------===//

/// Interface for non-blocking put operations.
template <typename T>
class TLMNonBlockingPutIf {
public:
  virtual ~TLMNonBlockingPutIf() = default;

  /// Try to put without blocking. Returns true if successful.
  virtual bool tryPut(const T &t) = 0;

  /// Check if put can proceed without blocking.
  virtual bool canPut() const = 0;
};

//===----------------------------------------------------------------------===//
// TLMPutIf - Combined put interface
//===----------------------------------------------------------------------===//

/// Combined blocking and non-blocking put interface.
template <typename T>
class TLMPutIf : public TLMBlockingPutIf<T>, public TLMNonBlockingPutIf<T> {};

//===----------------------------------------------------------------------===//
// TLMBlockingGetIf - Blocking get interface
//===----------------------------------------------------------------------===//

/// Interface for blocking get operations.
template <typename T>
class TLMBlockingGetIf {
public:
  virtual ~TLMBlockingGetIf() = default;

  /// Blocking get - waits until a transaction is available.
  virtual T get() = 0;

  /// Blocking peek - returns a copy without removing.
  virtual T peek() const = 0;
};

//===----------------------------------------------------------------------===//
// TLMNonBlockingGetIf - Non-blocking get interface
//===----------------------------------------------------------------------===//

/// Interface for non-blocking get operations.
template <typename T>
class TLMNonBlockingGetIf {
public:
  virtual ~TLMNonBlockingGetIf() = default;

  /// Try to get without blocking. Returns true if successful.
  virtual bool tryGet(T &t) = 0;

  /// Try to peek without blocking. Returns true if successful.
  virtual bool tryPeek(T &t) const = 0;

  /// Check if get can proceed without blocking.
  virtual bool canGet() const = 0;
};

//===----------------------------------------------------------------------===//
// TLMGetIf - Combined get interface
//===----------------------------------------------------------------------===//

/// Combined blocking and non-blocking get interface.
template <typename T>
class TLMGetIf : public TLMBlockingGetIf<T>, public TLMNonBlockingGetIf<T> {};

//===----------------------------------------------------------------------===//
// TLMAnalysisIf - Analysis interface for one-to-many broadcast
//===----------------------------------------------------------------------===//

/// Interface for analysis (write) operations.
/// Analysis ports broadcast to multiple subscribers.
template <typename T>
class TLMAnalysisIf {
public:
  virtual ~TLMAnalysisIf() = default;

  /// Write (broadcast) a transaction to all connected subscribers.
  virtual void write(const T &t) = 0;
};

//===----------------------------------------------------------------------===//
// TLMFifo - TLM FIFO implementation
//===----------------------------------------------------------------------===//

/// TLM FIFO for transaction-level communication between components.
/// Implements both put and get interfaces with configurable depth.
template <typename T>
class TLMFifo : public TLMPutIf<T>, public TLMGetIf<T> {
public:
  /// Create a FIFO with unlimited depth.
  TLMFifo() : TLMFifo(0) {}

  /// Create a FIFO with specified depth (0 = unlimited).
  explicit TLMFifo(size_t depth)
      : maxDepth(depth), scheduler(nullptr), waitingPutter(InvalidProcessId),
        waitingGetter(InvalidProcessId) {}

  /// Create a FIFO with scheduler for blocking operations.
  TLMFifo(size_t depth, ProcessScheduler *sched)
      : maxDepth(depth), scheduler(sched), waitingPutter(InvalidProcessId),
        waitingGetter(InvalidProcessId) {}

  ~TLMFifo() override = default;

  //===------------------------------------------------------------------===//
  // Blocking Put Interface
  //===------------------------------------------------------------------===//

  void put(const T &t) override {
    if (!canPut() && scheduler) {
      // Block until space is available
      // In a real implementation, this would suspend the calling process
      // For now, we just fail
      return;
    }
    doPut(t);
  }

  //===------------------------------------------------------------------===//
  // Non-Blocking Put Interface
  //===------------------------------------------------------------------===//

  bool tryPut(const T &t) override {
    if (!canPut())
      return false;
    doPut(t);
    return true;
  }

  bool canPut() const override {
    if (maxDepth == 0)
      return true; // Unlimited
    return fifoData.size() < maxDepth;
  }

  //===------------------------------------------------------------------===//
  // Blocking Get Interface
  //===------------------------------------------------------------------===//

  T get() override {
    if (fifoData.empty() && scheduler) {
      // Block until data is available
      // In a real implementation, this would suspend the calling process
      return T();
    }
    return doGet();
  }

  T peek() const override {
    if (fifoData.empty())
      return T();
    return fifoData.front();
  }

  //===------------------------------------------------------------------===//
  // Non-Blocking Get Interface
  //===------------------------------------------------------------------===//

  bool tryGet(T &t) override {
    if (fifoData.empty())
      return false;
    t = doGet();
    return true;
  }

  bool tryPeek(T &t) const override {
    if (fifoData.empty())
      return false;
    t = fifoData.front();
    return true;
  }

  bool canGet() const override { return !fifoData.empty(); }

  //===------------------------------------------------------------------===//
  // FIFO Status
  //===------------------------------------------------------------------===//

  /// Get the number of items in the FIFO.
  size_t size() const { return fifoData.size(); }

  /// Check if the FIFO is empty.
  bool isEmpty() const { return fifoData.empty(); }

  /// Check if the FIFO is full.
  bool isFull() const {
    if (maxDepth == 0)
      return false;
    return fifoData.size() >= maxDepth;
  }

  /// Get the maximum depth (0 = unlimited).
  size_t getMaxDepth() const { return maxDepth; }

  /// Get the number of free slots (SIZE_MAX for unlimited).
  size_t freeSlots() const {
    if (maxDepth == 0)
      return SIZE_MAX;
    return maxDepth - fifoData.size();
  }

  /// Flush all data from the FIFO.
  void flush() { fifoData.clear(); }

  //===------------------------------------------------------------------===//
  // Scheduler Integration
  //===------------------------------------------------------------------===//

  /// Set the scheduler for blocking operations.
  void setScheduler(ProcessScheduler *sched) { scheduler = sched; }

  /// Set the callback for put events (data available).
  void setDataWrittenCallback(std::function<void()> callback) {
    dataWrittenCallback = std::move(callback);
  }

  /// Set the callback for get events (space available).
  void setDataReadCallback(std::function<void()> callback) {
    dataReadCallback = std::move(callback);
  }

private:
  void doPut(const T &t) {
    fifoData.push_back(t);

    // Notify waiting getter
    if (dataWrittenCallback)
      dataWrittenCallback();
  }

  T doGet() {
    T result = fifoData.front();
    fifoData.pop_front();

    // Notify waiting putter
    if (dataReadCallback)
      dataReadCallback();

    return result;
  }

  std::deque<T> fifoData;
  size_t maxDepth;
  ProcessScheduler *scheduler;

  // For blocking operations
  ProcessId waitingPutter;
  ProcessId waitingGetter;

  // Callbacks
  std::function<void()> dataWrittenCallback;
  std::function<void()> dataReadCallback;
};

//===----------------------------------------------------------------------===//
// TLMAnalysisPort - One-to-many broadcast port
//===----------------------------------------------------------------------===//

/// Analysis port for broadcasting transactions to multiple subscribers.
/// This is used for passive monitoring and coverage collection.
template <typename T>
class TLMAnalysisPort : public TLMAnalysisIf<T> {
public:
  TLMAnalysisPort() = default;
  explicit TLMAnalysisPort(llvm::StringRef name) : portName(name.str()) {}
  ~TLMAnalysisPort() override = default;

  /// Connect a subscriber to this port.
  void connect(TLMAnalysisIf<T> *subscriber) {
    if (subscriber)
      subscribers.push_back(subscriber);
  }

  /// Disconnect a subscriber from this port.
  void disconnect(TLMAnalysisIf<T> *subscriber) {
    subscribers.erase(
        std::remove(subscribers.begin(), subscribers.end(), subscriber),
        subscribers.end());
  }

  /// Write (broadcast) to all subscribers.
  void write(const T &t) override {
    for (auto *sub : subscribers) {
      sub->write(t);
    }
    writeCount++;
  }

  /// Get the port name.
  const std::string &getName() const { return portName; }

  /// Get the number of subscribers.
  size_t getSubscriberCount() const { return subscribers.size(); }

  /// Get the number of writes performed.
  size_t getWriteCount() const { return writeCount; }

private:
  std::string portName;
  llvm::SmallVector<TLMAnalysisIf<T> *, 4> subscribers;
  size_t writeCount = 0;
};

//===----------------------------------------------------------------------===//
// TLMAnalysisExport - Analysis export for receiving broadcasts
//===----------------------------------------------------------------------===//

/// Analysis export that receives broadcasts from analysis ports.
template <typename T>
class TLMAnalysisExport : public TLMAnalysisIf<T> {
public:
  using WriteCallback = std::function<void(const T &)>;

  TLMAnalysisExport() = default;
  explicit TLMAnalysisExport(WriteCallback callback)
      : writeCallback(std::move(callback)) {}
  ~TLMAnalysisExport() override = default;

  /// Set the callback for received writes.
  void setWriteCallback(WriteCallback callback) {
    writeCallback = std::move(callback);
  }

  /// Receive a write from an analysis port.
  void write(const T &t) override {
    if (writeCallback)
      writeCallback(t);
    receiveCount++;
  }

  /// Get the number of writes received.
  size_t getReceiveCount() const { return receiveCount; }

private:
  WriteCallback writeCallback;
  size_t receiveCount = 0;
};

//===----------------------------------------------------------------------===//
// TLMBlockingTransportIf - Blocking transport interface
//===----------------------------------------------------------------------===//

/// Interface for blocking transport calls (TLM-2.0 style).
template <typename REQ, typename RSP = REQ>
class TLMBlockingTransportIf {
public:
  virtual ~TLMBlockingTransportIf() = default;

  /// Blocking transport - send request and wait for response.
  virtual void transport(REQ &req, RSP &rsp) = 0;
};

//===----------------------------------------------------------------------===//
// TLMNonBlockingTransportIf - Non-blocking transport interface
//===----------------------------------------------------------------------===//

/// Interface for non-blocking transport calls.
template <typename REQ, typename RSP = REQ>
class TLMNonBlockingTransportIf {
public:
  virtual ~TLMNonBlockingTransportIf() = default;

  /// Forward path of non-blocking transport.
  virtual bool nbTransportFw(REQ &req) = 0;

  /// Backward path of non-blocking transport.
  virtual bool nbTransportBw(RSP &rsp) = 0;
};

//===----------------------------------------------------------------------===//
// TLMPort - Generic TLM port base class
//===----------------------------------------------------------------------===//

/// Base class for TLM ports with connection management.
template <typename IF>
class TLMPort {
public:
  TLMPort() : connectedIf(nullptr) {}
  explicit TLMPort(llvm::StringRef name) : portName(name.str()), connectedIf(nullptr) {}
  virtual ~TLMPort() = default;

  /// Connect to an interface.
  void connect(IF *iface) { connectedIf = iface; }

  /// Check if connected.
  bool isConnected() const { return connectedIf != nullptr; }

  /// Get the connected interface.
  IF *getInterface() const { return connectedIf; }

  /// Get the port name.
  const std::string &getName() const { return portName; }

  /// Arrow operator for interface access.
  IF *operator->() const { return connectedIf; }

protected:
  std::string portName;
  IF *connectedIf;
};

//===----------------------------------------------------------------------===//
// TLMSeqItemPullPort - Sequence item pull port for UVM sequences
//===----------------------------------------------------------------------===//

/// Port for pulling sequence items from a sequencer.
/// This implements the UVM driver-sequencer communication pattern.
template <typename T>
class TLMSeqItemPullPort {
public:
  using ItemCallback = std::function<bool(T &)>;
  using ItemDoneCallback = std::function<void(const T &)>;

  TLMSeqItemPullPort() = default;
  explicit TLMSeqItemPullPort(llvm::StringRef name) : portName(name.str()) {}

  /// Set the callback for getting next item.
  void setGetNextItemCallback(ItemCallback callback) {
    getNextItemCallback = std::move(callback);
  }

  /// Set the callback for item done notification.
  void setItemDoneCallback(ItemDoneCallback callback) {
    itemDoneCallback = std::move(callback);
  }

  /// Get the next sequence item. Returns false if no item available.
  bool getNextItem(T &item) {
    if (getNextItemCallback)
      return getNextItemCallback(item);
    return false;
  }

  /// Signal that the item has been processed.
  void itemDone(const T &item) {
    if (itemDoneCallback)
      itemDoneCallback(item);
  }

  /// Get the port name.
  const std::string &getName() const { return portName; }

private:
  std::string portName;
  ItemCallback getNextItemCallback;
  ItemDoneCallback itemDoneCallback;
};

//===----------------------------------------------------------------------===//
// TLMManager - Manages TLM ports and connections
//===----------------------------------------------------------------------===//

/// Manages TLM port creation and connection tracking.
class TLMManager {
public:
  TLMManager() = default;
  ~TLMManager() = default;

  /// Create a new FIFO with the given name and depth.
  template <typename T>
  TLMFifo<T> *createFifo(llvm::StringRef name, size_t depth = 0) {
    auto fifo = std::make_unique<TLMFifo<T>>(depth);
    auto *ptr = fifo.get();
    // Store type-erased pointer - in practice, you'd use a polymorphic base
    return ptr;
  }

  /// Get statistics.
  struct Statistics {
    size_t portsCreated = 0;
    size_t connectionsEstablished = 0;
    size_t transactionsProcessed = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Reset the manager.
  void reset() { stats = Statistics(); }

private:
  Statistics stats;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_TLMPORTS_H
