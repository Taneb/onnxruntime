// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorCostModel.h"

namespace onnxruntime {
class Barrier {
 public:
  Barrier(unsigned int count) : state_(count << 1), notified_(false) {
    eigen_plain_assert(((count << 1) >> 1) == count);
  }
#ifdef NDEBUG
  ~Barrier() = default;
#else
  ~Barrier() { eigen_plain_assert((state_ >> 1) == 0); }
#endif

  void Notify() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
      // Clear the lowest bit (waiter flag) and check that the original state
      // value was not zero. If it was zero, it means that notify was called
      // more times than the original count.
      eigen_plain_assert(((v + 2) & ~1) != 0);
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::unique_lock<OrtMutex> l(mu_);
    eigen_plain_assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0)
      return;
    std::unique_lock<OrtMutex> l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  OrtMutex mu_;
  OrtCondVar cv_;
  std::atomic<unsigned int> state_;  // low bit is waiter flag
  bool notified_;
};

// Notification is an object that allows a user to to wait for another
// thread to signal a notification that an event has occurred.
//
// Multiple threads can wait on the same Notification object,
// but only one caller must call Notify() on the object.
struct Notification : Barrier {
  Notification() : Barrier(1){};
};

class EventCount {
 public:
  class Waiter;

  EventCount(Eigen::MaxSizeVector<Waiter>& waiters) : state_(kStackMask), waiters_(waiters) {
    eigen_plain_assert(waiters.size() < (1 << kWaiterBits) - 1);
  }

#ifdef NDEBUG
  ~EventCount() = default;
#else
  ~EventCount() {
    // Ensure there are no waiters.
    eigen_plain_assert(state_.load() == kStackMask);
  }
#endif
  // Prewait prepares for waiting.
  // After calling Prewait, the thread must re-check the wait predicate
  // and then call either CancelWait or CommitWait.
  void Prewait() {
    uint64_t state = state_.load(std::memory_order_relaxed);
    for (;;) {
      CheckState(state);
      uint64_t newstate = state + kWaiterInc;
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_seq_cst))
        return;
    }
  }

  // CommitWait commits waiting after Prewait.
  void CommitWait(Waiter* w) {
    eigen_plain_assert((w->epoch & ~kEpochMask) == 0);
    w->state = Waiter::kNotSignaled;
    const uint64_t me = (w - &waiters_[0]) | w->epoch;
    uint64_t state = state_.load(std::memory_order_seq_cst);
    for (;;) {
      CheckState(state, true);
      uint64_t newstate;
      if ((state & kSignalMask) != 0) {
        // Consume the signal and return immidiately.
        newstate = state - kWaiterInc - kSignalInc;
      } else {
        // Remove this thread from pre-wait counter and add to the waiter stack.
        newstate = ((state & kWaiterMask) - kWaiterInc) | me;
        w->next.store(state & (kStackMask | kEpochMask), std::memory_order_relaxed);
      }
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_acq_rel)) {
        if ((state & kSignalMask) == 0) {
          w->epoch += kEpochInc;
          Park(w);
        }
        return;
      }
    }
  }

  // CancelWait cancels effects of the previous Prewait call.
  void CancelWait() {
    uint64_t state = state_.load(std::memory_order_relaxed);
    for (;;) {
      CheckState(state, true);
      uint64_t newstate = state - kWaiterInc;
      // We don't know if the thread was also notified or not,
      // so we should not consume a signal unconditionaly.
      // Only if number of waiters is equal to number of signals,
      // we know that the thread was notified and we must take away the signal.
      if (((state & kWaiterMask) >> kWaiterShift) == ((state & kSignalMask) >> kSignalShift))
        newstate -= kSignalInc;
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_acq_rel))
        return;
    }
  }

  // Notify wakes one or all waiting threads.
  // Must be called after changing the associated wait predicate.
  void Notify(bool notifyAll) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t state = state_.load(std::memory_order_acquire);
    for (;;) {
      CheckState(state);
      const uint64_t waiters = (state & kWaiterMask) >> kWaiterShift;
      const uint64_t signals = (state & kSignalMask) >> kSignalShift;
      // Easy case: no waiters.
      if ((state & kStackMask) == kStackMask && waiters == signals)
        return;
      uint64_t newstate;
      if (notifyAll) {
        // Empty wait stack and set signal to number of pre-wait threads.
        newstate = (state & kWaiterMask) | (waiters << kSignalShift) | kStackMask;
      } else if (signals < waiters) {
        // There is a thread in pre-wait state, unblock it.
        newstate = state + kSignalInc;
      } else {
        // Pop a waiter from list and unpark it.
        Waiter* w = &waiters_[state & kStackMask];
        uint64_t next = w->next.load(std::memory_order_relaxed);
        newstate = (state & (kWaiterMask | kSignalMask)) | next;
      }
      CheckState(newstate);
      if (state_.compare_exchange_weak(state, newstate, std::memory_order_acq_rel)) {
        if (!notifyAll && (signals < waiters))
          return;  // unblocked pre-wait thread
        if ((state & kStackMask) == kStackMask)
          return;
        Waiter* w = &waiters_[state & kStackMask];
        if (!notifyAll)
          w->next.store(kStackMask, std::memory_order_relaxed);
        Unpark(w);
        return;
      }
    }
  }

  class Waiter {
    friend class EventCount;
    // Align to 128 byte boundary to prevent false sharing with other Waiter
    // objects in the same vector.
    EIGEN_ALIGN_TO_BOUNDARY(128) std::atomic<uint64_t> next;
    OrtMutex mu;
    OrtCondVar cv;
    uint64_t epoch = 0;
    unsigned state = kNotSignaled;
    enum {
      kNotSignaled,
      kWaiting,
      kSignaled,
    };
  };

 private:
  // State_ layout:
  // - low kWaiterBits is a stack of waiters committed wait
  //   (indexes in waiters_ array are used as stack elements,
  //   kStackMask means empty stack).
  // - next kWaiterBits is count of waiters in prewait state.
  // - next kWaiterBits is count of pending signals.
  // - remaining bits are ABA counter for the stack.
  //   (stored in Waiter node and incremented on push).
  static constexpr uint64_t kWaiterBits = 14;
  static constexpr uint64_t kStackMask = (1ull << kWaiterBits) - 1;
  static constexpr uint64_t kWaiterShift = kWaiterBits;
  static constexpr uint64_t kWaiterMask = ((1ull << kWaiterBits) - 1) << kWaiterShift;
  static constexpr uint64_t kWaiterInc = 1ull << kWaiterShift;
  static constexpr uint64_t kSignalShift = 2 * kWaiterBits;
  static constexpr uint64_t kSignalMask = ((1ull << kWaiterBits) - 1) << kSignalShift;
  static constexpr uint64_t kSignalInc = 1ull << kSignalShift;
  static constexpr uint64_t kEpochShift = 3 * kWaiterBits;
  static constexpr uint64_t kEpochBits = 64 - kEpochShift;
  static constexpr uint64_t kEpochMask = ((1ull << kEpochBits) - 1) << kEpochShift;
  static constexpr uint64_t kEpochInc = 1ull << kEpochShift;
  std::atomic<uint64_t> state_;
  Eigen::MaxSizeVector<Waiter>& waiters_;

#ifdef NDEBUG
  static void CheckState(uint64_t , bool) {}
  static void CheckState(uint64_t) {}
#else
  static void CheckState(uint64_t state, bool waiter = false) {
    static_assert(kEpochBits >= 20, "not enough bits to prevent ABA problem");
    const uint64_t waiters = (state & kWaiterMask) >> kWaiterShift;
    const uint64_t signals = (state & kSignalMask) >> kSignalShift;
    eigen_plain_assert(waiters >= signals);
    eigen_plain_assert(waiters < (1 << kWaiterBits) - 1);
    eigen_plain_assert(!waiter || waiters > 0);
    (void)waiters;
    (void)signals;
  }
#endif
  void Park(Waiter* w) {
    std::unique_lock<OrtMutex> lock(w->mu);
    while (w->state != Waiter::kSignaled) {
      w->state = Waiter::kWaiting;
      w->cv.wait(lock);
    }
  }

  void Unpark(Waiter* w) {
    for (Waiter* next; w; w = next) {
      uint64_t wnext = w->next.load(std::memory_order_relaxed) & kStackMask;
      next = wnext == kStackMask ? nullptr : &waiters_[wnext];
      unsigned state;
      {
        std::unique_lock<OrtMutex> lock(w->mu);
        state = w->state;
        w->state = Waiter::kSignaled;
      }
      // Avoid notifying if it wasn't waiting.
      if (state == Waiter::kWaiting)
        w->cv.notify_one();
    }
  }

  EventCount(const EventCount&) = delete;
  void operator=(const EventCount&) = delete;
};
template <typename Work, unsigned kSize>
class RunQueue {
 public:
  RunQueue() : front_(0), back_(0) {
    // require power-of-two for fast masking
    eigen_plain_assert((kSize & (kSize - 1)) == 0);
    eigen_plain_assert(kSize > 2);            // why would you do this?
    eigen_plain_assert(kSize <= (64 << 10));  // leave enough space for counter
    for (unsigned i = 0; i < kSize; i++) array_[i].state.store(kEmpty, std::memory_order_relaxed);
  }

  ~RunQueue() { eigen_plain_assert(Size() == 0); }

  // PushFront inserts w at the beginning of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushFront(Work w) {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[front & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return w;
    front_.store(front + 1 + (kSize << 1), std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);
    return Work();
  }

  // PopFront removes and returns the first element in the queue.
  // If the queue was empty returns default-constructed Work.
  Work PopFront() {
    unsigned front = front_.load(std::memory_order_relaxed);
    Elem* e = &array_[(front - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return Work();
    Work w = std::move(e->w);
    e->state.store(kEmpty, std::memory_order_release);
    front = ((front - 1) & kMask2) | (front & ~kMask2);
    front_.store(front, std::memory_order_relaxed);
    return w;
  }

  // PushBack adds w at the end of the queue.
  // If queue is full returns w, otherwise returns default-constructed Work.
  Work PushBack(Work w) {
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e = &array_[(back - 1) & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kEmpty || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return w;
    back = ((back - 1) & kMask2) | (back & ~kMask2);
    back_.store(back, std::memory_order_relaxed);
    e->w = std::move(w);
    e->state.store(kReady, std::memory_order_release);
    return Work();
  }

  // PopBack removes and returns the last elements in the queue.
  Work PopBack() {
    if (Empty())
      return Work();
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    Elem* e = &array_[back & kMask];
    uint8_t s = e->state.load(std::memory_order_relaxed);
    if (s != kReady || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
      return Work();
    Work w = std::move(e->w);
    e->state.store(kEmpty, std::memory_order_release);
    back_.store(back + 1 + (kSize << 1), std::memory_order_relaxed);
    return w;
  }

  // PopBackHalf removes and returns half last elements in the queue.
  // Returns number of elements removed.
  unsigned PopBackHalf(std::vector<Work>* result) {
    if (Empty())
      return 0;
    std::unique_lock<OrtMutex> lock(mutex_);
    unsigned back = back_.load(std::memory_order_relaxed);
    unsigned size = Size();
    unsigned mid = back;
    if (size > 1)
      mid = back + (size - 1) / 2;
    unsigned n = 0;
    unsigned start = 0;
    for (; static_cast<int>(mid - back) >= 0; mid--) {
      Elem* e = &array_[mid & kMask];
      uint8_t s = e->state.load(std::memory_order_relaxed);
      if (n == 0) {
        if (s != kReady || !e->state.compare_exchange_strong(s, kBusy, std::memory_order_acquire))
          continue;
        start = mid;
      } else {
        // Note: no need to store temporal kBusy, we exclusively own these
        // elements.
        eigen_plain_assert(s == kReady);
      }
      result->push_back(std::move(e->w));
      e->state.store(kEmpty, std::memory_order_release);
      n++;
    }
    if (n != 0)
      back_.store(start + 1 + (kSize << 1), std::memory_order_relaxed);
    return n;
  }

  // Size returns current queue size.
  // Can be called by any thread at any time.
  unsigned Size() const { return SizeOrNotEmpty<true>(); }

  // Empty tests whether container is empty.
  // Can be called by any thread at any time.
  bool Empty() const { return SizeOrNotEmpty<false>() == 0; }

  // Delete all the elements from the queue.
  void Flush() {
    while (!Empty()) {
      PopFront();
    }
  }

 private:
  static const unsigned kMask = kSize - 1;
  static const unsigned kMask2 = (kSize << 1) - 1;
  struct Elem {
    std::atomic<uint8_t> state;
    Work w;
  };
  enum {
    kEmpty,
    kBusy,
    kReady,
  };
  OrtMutex mutex_;
  // Low log(kSize) + 1 bits in front_ and back_ contain rolling index of
  // front/back, respectively. The remaining bits contain modification counters
  // that are incremented on Push operations. This allows us to (1) distinguish
  // between empty and full conditions (if we would use log(kSize) bits for
  // position, these conditions would be indistinguishable); (2) obtain
  // consistent snapshot of front_/back_ for Size operation using the
  // modification counters.
  std::atomic<unsigned> front_;
  std::atomic<unsigned> back_;
  Elem array_[kSize];

  // SizeOrNotEmpty returns current queue size; if NeedSizeEstimate is false,
  // only whether the size is 0 is guaranteed to be correct.
  // Can be called by any thread at any time.
  template <bool NeedSizeEstimate>
  unsigned SizeOrNotEmpty() const {
    // Emptiness plays critical role in thread pool blocking. So we go to great
    // effort to not produce false positives (claim non-empty queue as empty).
    unsigned front = front_.load(std::memory_order_acquire);
    for (;;) {
      // Capture a consistent snapshot of front/tail.
      unsigned back = back_.load(std::memory_order_acquire);
      unsigned front1 = front_.load(std::memory_order_relaxed);
      if (front != front1) {
        front = front1;
        std::atomic_thread_fence(std::memory_order_acquire);
        continue;
      }
      if (NeedSizeEstimate) {
        return CalculateSize(front, back);
      } else {
        // This value will be 0 if the queue is empty, and undefined otherwise.
        unsigned maybe_zero = ((front ^ back) & kMask2);
        // Queue size estimate must agree with maybe zero check on the queue
        // empty/non-empty state.
        eigen_assert((CalculateSize(front, back) == 0) == (maybe_zero == 0));
        return maybe_zero;
      }
    }
  }

  EIGEN_ALWAYS_INLINE
  unsigned CalculateSize(unsigned front, unsigned back) const {
    int size = (front & kMask2) - (back & kMask2);
    // Fix overflow.
    if (size < 0)
      size += 2 * kSize;
    // Order of modification in push/pop is crafted to make the queue look
    // larger than it is during concurrent modifications. E.g. push can
    // increment size before the corresponding pop has decremented it.
    // So the computed size can be up to kSize + 1, fix it.
    if (size > static_cast<int>(kSize))
      size = kSize;
    return static_cast<unsigned>(size);
  }

  RunQueue(const RunQueue&) = delete;
  void operator=(const RunQueue&) = delete;
};

template <typename Environment>
class ThreadPoolTempl : public Eigen::ThreadPoolInterface {
 public:
  typedef typename Environment::Task Task;
  typedef RunQueue<Task, 1024> Queue;

  ThreadPoolTempl(int num_threads, Environment env = Environment())
      : ThreadPoolTempl(num_threads, true, env) {}

  ThreadPoolTempl(int num_threads, bool allow_spinning,
                  Environment env = Environment())
      : env_(env),
        num_threads_(num_threads),
        allow_spinning_(allow_spinning),
        thread_data_(num_threads),
        all_coprimes_(num_threads),
        waiters_(num_threads),
        global_steal_partition_(EncodePartition(0, num_threads_)),
        blocked_(0),
        spinning_(0),
        done_(false),
        cancelled_(false),
        ec_(waiters_) {
    waiters_.resize(num_threads_);
    // Calculate coprimes of all numbers [1, num_threads].
    // Coprimes are used for random walks over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a random starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    eigen_plain_assert(num_threads_ < kMaxThreads);
    for (int i = 1; i <= num_threads_; ++i) {
      all_coprimes_.emplace_back(i);
      ComputeCoprimes(i, &all_coprimes_.back());
    }

    thread_data_.resize(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
      SetStealPartition(i, EncodePartition(0, num_threads_));
      thread_data_[i].thread.reset(
          env_.CreateThread([this, i]() { WorkerLoop(i); }));
    }
  }

  ~ThreadPoolTempl() {
    done_ = true;

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    if (!cancelled_) {
      ec_.Notify(true);
    } else {
      // Since we were cancelled, there might be entries in the queues.
      // Empty them to prevent their destructor from asserting.
      for (size_t i = 0; i < thread_data_.size(); i++) {
        thread_data_[i].queue.Flush();
      }
    }
    // Join threads explicitly (by destroying) to avoid destruction order within
    // this class.
    for (size_t i = 0; i < thread_data_.size(); ++i)
      thread_data_[i].thread.reset();
  }

  void SetStealPartitions(const std::vector<std::pair<unsigned, unsigned>>& partitions) {
    eigen_plain_assert(partitions.size() == static_cast<std::size_t>(num_threads_));

    // Pass this information to each thread queue.
    for (int i = 0; i < num_threads_; i++) {
      const auto& pair = partitions[i];
      unsigned start = pair.first, end = pair.second;
      AssertBounds(start, end);
      unsigned val = EncodePartition(start, end);
      SetStealPartition(i, val);
    }
  }

  void Schedule(std::function<void()> fn) override {
    ScheduleWithHint(std::move(fn), 0, num_threads_);
  }

  void ScheduleWithHint(std::function<void()> fn, int start,
                        int limit) override {
    Task t = env_.CreateTask(std::move(fn));
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue& q = thread_data_[pt->thread_id].queue;
      t = q.PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      eigen_plain_assert(start < limit);
      eigen_plain_assert(limit <= num_threads_);
      int num_queues = limit - start;
      int rnd = Rand(&pt->rand) % num_queues;
      eigen_plain_assert(start + rnd < limit);
      Queue& q = thread_data_[start + rnd].queue;
      t = q.PushBack(std::move(t));
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t.f) {
      ec_.Notify(false);
    } else {
      env_.ExecuteTask(t);  // Push failed, execute directly.
    }
  }

  void Cancel() override {
    cancelled_ = true;
    done_ = true;

    // Let each thread know it's been cancelled.
#ifdef EIGEN_THREAD_ENV_SUPPORTS_CANCELLATION
    for (size_t i = 0; i < thread_data_.size(); i++) {
      thread_data_[i].thread->OnCancel();
    }
#endif

    // Wake up the threads without work to let them exit on their own.
    ec_.Notify(true);
  }

  int NumThreads() const EIGEN_FINAL { return num_threads_; }

  int CurrentThreadId() const EIGEN_FINAL {
    const PerThread* pt = const_cast<ThreadPoolTempl*>(this)->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

 private:
  // Create a single atomic<int> that encodes start and limit information for
  // each thread.
  // We expect num_threads_ < 65536, so we can store them in a single
  // std::atomic<unsigned>.
  // Exposed publicly as static functions so that external callers can reuse
  // this encode/decode logic for maintaining their own thread-safe copies of
  // scheduling and steal domain(s).
  static const int kMaxPartitionBits = 16;
  static const int kMaxThreads = 1 << kMaxPartitionBits;

  inline unsigned EncodePartition(unsigned start, unsigned limit) {
    return (start << kMaxPartitionBits) | limit;
  }

  inline void DecodePartition(unsigned val, unsigned* start, unsigned* limit) {
    *limit = val & (kMaxThreads - 1);
    val >>= kMaxPartitionBits;
    *start = val;
  }
#ifdef NDEBUG
  void AssertBounds(int , int ) {}
#else
  void AssertBounds(int start, int end) {
    eigen_plain_assert(start >= 0);
    eigen_plain_assert(start < end);  // non-zero sized partition
    eigen_plain_assert(end <= num_threads_);
  }
#endif
  inline void SetStealPartition(size_t i, unsigned val) {
    thread_data_[i].steal_partition.store(val, std::memory_order_relaxed);
  }

  inline unsigned GetStealPartition(int i) {
    return thread_data_[i].steal_partition.load(std::memory_order_relaxed);
  }

  void ComputeCoprimes(int N, Eigen::MaxSizeVector<unsigned>* coprimes) {
    for (int i = 1; i <= N; i++) {
      unsigned a = i;
      unsigned b = N;
      // If GCD(a, b) == 1, then a and b are coprimes.
      while (b != 0) {
        unsigned tmp = a;
        a = b;
        b = tmp % b;
      }
      if (a == 1) {
        coprimes->push_back(i);
      }
    }
  }

  typedef typename Environment::EnvThread Thread;

  struct PerThread {
    constexpr PerThread() : pool(NULL), rand(0), thread_id(-1) {}
    ThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    uint64_t rand;          // Random generator state.
    int thread_id;          // Worker thread index in pool.
  };

  struct ThreadData {
    constexpr ThreadData() : thread(), steal_partition(0), queue() {}
    std::unique_ptr<Thread> thread;
    std::atomic<unsigned> steal_partition;
    Queue queue;
  };

  Environment env_;
  const int num_threads_;
  const bool allow_spinning_;
  Eigen::MaxSizeVector<ThreadData> thread_data_;
  Eigen::MaxSizeVector<Eigen::MaxSizeVector<unsigned>> all_coprimes_;
  Eigen::MaxSizeVector<EventCount::Waiter> waiters_;
  unsigned global_steal_partition_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> spinning_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;
  EventCount ec_;


  // Main worker thread loop.
  void WorkerLoop(int thread_id) {
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->rand = GlobalThreadIdHash();
    pt->thread_id = thread_id;
    Queue& q = thread_data_[thread_id].queue;
    EventCount::Waiter* waiter = &waiters_[thread_id];
    // TODO(dvyukov,rmlarsen): The time spent in NonEmptyQueueIndex() is
    // proportional to num_threads_ and we assume that new work is scheduled at
    // a constant rate, so we set spin_count to 5000 / num_threads_. The
    // constant was picked based on a fair dice roll, tune it.
    const int spin_count =
        allow_spinning_ && num_threads_ > 0 ? 5000 / num_threads_ : 0;
    if (num_threads_ == 1) {
      // For num_threads_ == 1 there is no point in going through the expensive
      // steal loop. Moreover, since NonEmptyQueueIndex() calls PopBack() on the
      // victim queues it might reverse the order in which ops are executed
      // compared to the order in which they are scheduled, which tends to be
      // counter-productive for the types of I/O workloads the single thread
      // pools tend to be used for.
      while (!cancelled_) {
        Task t = q.PopFront();
        for (int i = 0; i < spin_count && !t.f; i++) {
          if (!cancelled_.load(std::memory_order_relaxed)) {
            t = q.PopFront();
          }
        }
        if (!t.f) {
          if (!WaitForWork(waiter, &t)) {
            return;
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    } else {
      while (!cancelled_) {
        Task t = q.PopFront();
        if (!t.f) {
          t = LocalSteal();
          if (!t.f) {
            t = GlobalSteal();
            if (!t.f) {
              // Leave one thread spinning. This reduces latency.
              if (allow_spinning_ && !spinning_ && !spinning_.exchange(true)) {
                for (int i = 0; i < spin_count && !t.f; i++) {
                  if (!cancelled_.load(std::memory_order_relaxed)) {
                    t = GlobalSteal();
                  } else {
                    return;
                  }
                }
                spinning_ = false;
              }
              if (!t.f) {
                if (!WaitForWork(waiter, &t)) {
                  return;
                }
              }
            }
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    }
  }

  // Steal tries to steal work from other worker threads in the range [start,
  // limit) in best-effort manner.
  Task Steal(unsigned start, unsigned limit) {
    PerThread* pt = GetPerThread();
    const unsigned size = static_cast<unsigned>(limit - start);
    unsigned r = Rand(&pt->rand);
    unsigned victim = r % size;
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];

    for (unsigned i = 0; i < size; i++) {
      eigen_plain_assert(start + victim < limit);
      Task t = thread_data_[start + victim].queue.PopBack();
      if (t.f) {
        return t;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return Task();
  }

  // Steals work within threads belonging to the partition.
  Task LocalSteal() {
    PerThread* pt = GetPerThread();
    unsigned partition = GetStealPartition(pt->thread_id);
    // If thread steal partition is the same as global partition, there is no
    // need to go through the steal loop twice.
    if (global_steal_partition_ == partition) return Task();
    unsigned start, limit;
    DecodePartition(partition, &start, &limit);
    AssertBounds(start, limit);

    return Steal(start, limit);
  }

  // Steals work from any other thread in the pool.
  Task GlobalSteal() {
    return Steal(0, num_threads_);
  }


  // WaitForWork blocks until new work is available (returns true), or if it is
  // time to exit (returns false). Can optionally return a task to execute in t
  // (in such case t.f != nullptr on return).
  bool WaitForWork(EventCount::Waiter* waiter, Task* t) {
    eigen_plain_assert(!t->f);
    // We already did best-effort emptiness check in Steal, so prepare for
    // blocking.
    ec_.Prewait();
    // Now do a reliable emptiness check.
    int victim = NonEmptyQueueIndex();
    if (victim != -1) {
      ec_.CancelWait();
      if (cancelled_) {
        return false;
      } else {
        *t = thread_data_[victim].queue.PopBack();
        return true;
      }
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    blocked_++;
    // TODO is blocked_ required to be unsigned?
    if (done_ && blocked_ == static_cast<unsigned>(num_threads_)) {
      ec_.CancelWait();
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (NonEmptyQueueIndex() != -1) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        blocked_--;
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(true);
      return false;
    }
    ec_.CommitWait(waiter);
    blocked_--;
    return true;
  }

  int NonEmptyQueueIndex() {
    PerThread* pt = GetPerThread();
    // We intentionally design NonEmptyQueueIndex to steal work from
    // anywhere in the queue so threads don't block in WaitForWork() forever
    // when all threads in their partition go to sleep. Steal is still local.
    const unsigned size = static_cast<unsigned>(thread_data_.size());
    unsigned r = Rand(&pt->rand);
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      if (!thread_data_[victim].queue.Empty()) {
        return victim;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return -1;
  }

  static EIGEN_STRONG_INLINE uint64_t GlobalThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }

  EIGEN_STRONG_INLINE PerThread* GetPerThread() {
    static thread_local PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  static EIGEN_STRONG_INLINE unsigned Rand(uint64_t* state) {
    uint64_t current = *state;
    // Update the internal state
    *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22)) >>
                                 (22 + (current >> 61)));
  }
};

using Eigen::ThreadPoolInterface;

template <typename Function, typename... Args>
struct FunctionWrapperWithBarrier {
  static void run(Barrier* b, Function f, Args... args) {
    f(args...);
    if (b) {
      b->Notify();
    }
  }
};

// An abstract interface to a device specific memory allocator.
class EigenAllocator {
 public:
  virtual ~EigenAllocator() {}
  virtual void* allocate(size_t num_bytes) const = 0;
  virtual void deallocate(void* buffer) const = 0;
};


// Build a thread pool device on top the an existing pool of threads.
struct ThreadPoolDevice {
  // The ownership of the thread pool remains with the caller.
  ThreadPoolDevice(ThreadPoolInterface* pool, int num_cores, EigenAllocator* allocator = nullptr)
      : pool_(pool), num_threads_(num_cores), allocator_(allocator) {}

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return allocator_ ? allocator_->allocate(num_bytes) : Eigen::internal::aligned_malloc(num_bytes);
  }

  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    if (allocator_) {
      allocator_->deallocate(buffer);
    } else {
      Eigen::internal::aligned_free(buffer);
    }
  }

  EIGEN_STRONG_INLINE void* allocate_temp(size_t num_bytes) const { return allocate(num_bytes); }

  EIGEN_STRONG_INLINE void deallocate_temp(void* buffer) const { deallocate(buffer); }

  template <typename Type>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Type get(Type data) const {
    return data;
  }

  EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
#ifdef __ANDROID__
    ::memcpy(dst, src, n);
#else
    // TODO(rmlarsen): Align blocks on cache lines.
    // We have observed that going beyond 4 threads usually just wastes
    // CPU cycles due to the threads competing for memory bandwidth, so we
    // statically schedule at most 4 block copies here.
    const size_t kMinBlockSize = 32768;
    const size_t num_threads = CostModel::numThreads(n, Eigen::TensorOpCost(1.0, 1.0, 0), 4);
    if (n <= kMinBlockSize || num_threads < 2) {
      ::memcpy(dst, src, n);
    } else {
      const char* src_ptr = static_cast<const char*>(src);
      char* dst_ptr = static_cast<char*>(dst);
      const size_t blocksize = (n + (num_threads - 1)) / num_threads;
      Barrier barrier(static_cast<int>(num_threads - 1));
      // Launch the last 3 blocks on worker threads.
      for (size_t i = 1; i < num_threads; ++i) {
        enqueue_with_barrier(&barrier, [n, i, src_ptr, dst_ptr, blocksize] {
          ::memcpy(dst_ptr + i * blocksize, src_ptr + i * blocksize,
                   Eigen::numext::mini(blocksize, n - (i * blocksize)));
        });
      }
      // Launch the first block on the main thread.
      ::memcpy(dst_ptr, src_ptr, blocksize);
      barrier.Wait();
    }
#endif
  }
  EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const { memcpy(dst, src, n); }
  EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const { memcpy(dst, src, n); }

  EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const { ::memset(buffer, c, n); }

  EIGEN_STRONG_INLINE int numThreads() const { return num_threads_; }

  // Number of theads available in the underlying thread pool. This number can
  // be different from the value returned by numThreads().
  EIGEN_STRONG_INLINE int numThreadsInPool() const { return pool_->NumThreads(); }

  EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const { return Eigen::l1CacheSize(); }

  EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // The l3 cache size is shared between all the cores.
    return Eigen::l3CacheSize() / num_threads_;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
    // Should return an enum that encodes the ISA supported by the CPU
    return 1;
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE Notification* enqueue(Function&& f, Args&&... args) const {
    Notification* n = new Notification();
    pool_->Schedule(std::bind(&FunctionWrapperWithNotification<Function, Args...>::run, n, std::move(f), args...));
    return n;
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE void enqueue_with_barrier(Barrier* b, Function&& f, Args&&... args) const {
    pool_->Schedule(std::bind(&FunctionWrapperWithBarrier<Function, Args...>::run, b, std::move(f), args...));
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE void enqueueNoNotification(Function&& f, Args&&... args) const {
    if (sizeof...(args) > 0) {
      pool_->Schedule(std::bind(std::move(f), args...));
    } else {
      pool_->Schedule(std::move(f));
    }
  }

  // Returns a logical thread index between 0 and pool_->NumThreads() - 1 if
  // called from one of the threads in pool_. Returns -1 otherwise.
  EIGEN_STRONG_INLINE int currentThreadId() const { return pool_->CurrentThreadId(); }

  // WARNING: This function is synchronous and will block the calling thread.
  //
  // Synchronous parallelFor executes f with [0, n) arguments in parallel and
  // waits for completion. F accepts a half-open interval [first, last). Block
  // size is chosen based on the iteration cost and resulting parallel
  // efficiency. If block_align is not nullptr, it is called to round up the
  // block size.
  void parallelFor(Eigen::Index n, const Eigen::TensorOpCost& cost,
                   std::function<Eigen::Index(Eigen::Index)> block_align,
                   std::function<void(Eigen::Index, Eigen::Index)> f) const {
    // Compute small problems directly in the caller thread.
    if (n <= 1 || numThreads() == 1 || CostModel::numThreads(n, cost, static_cast<int>(numThreads())) == 1) {
      f(0, n);
      return;
    }

    // Compute block size and total count of blocks.
    ParallelForBlock block = CalculateParallelForBlock(n, cost, block_align);

    // Recursively divide size into halves until we reach block_size.
    // Division code rounds mid to block_size, so we are guaranteed to get
    // block_count leaves that do actual computations.
    Barrier barrier(static_cast<unsigned int>(block.count));
    std::function<void(Eigen::Index, Eigen::Index)> handleRange;
    handleRange = [=, &handleRange, &barrier, &f](Eigen::Index firstIdx, Eigen::Index lastIdx) {
      while (lastIdx - firstIdx > block.size) {
        // Split into halves and schedule the second half on a different thread.
        const Eigen::Index midIdx = firstIdx + Eigen::divup((lastIdx - firstIdx) / 2, block.size) * block.size;
        pool_->Schedule([=, &handleRange]() { handleRange(midIdx, lastIdx); });
        lastIdx = midIdx;
      }
      // Single block or less, execute directly.
      f(firstIdx, lastIdx);
      barrier.Notify();
    };

    if (block.count <= numThreads()) {
      // Avoid a thread hop by running the root of the tree and one block on the
      // main thread.
      handleRange(0, n);
    } else {
      // Execute the root in the thread pool to avoid running work on more than
      // numThreads() threads.
      pool_->Schedule([=, &handleRange]() { handleRange(0, n); });
    }

    barrier.Wait();
  }

  // Convenience wrapper for parallelFor that does not align blocks.
  void parallelFor(Eigen::Index n, const Eigen::TensorOpCost& cost,
                   std::function<void(Eigen::Index, Eigen::Index)> f) const {
    parallelFor(n, cost, nullptr, std::move(f));
  }

  // WARNING: This function is asynchronous and will not block the calling thread.
  //
  // Asynchronous parallelFor executes f with [0, n) arguments in parallel
  // without waiting for completion. When the last block finished, it will call
  // 'done' callback. F accepts a half-open interval [first, last). Block size
  // is chosen based on the iteration cost and resulting parallel efficiency. If
  // block_align is not nullptr, it is called to round up the block size.
  void parallelForAsync(Eigen::Index n, const Eigen::TensorOpCost& cost,
                        std::function<Eigen::Index(Eigen::Index)> block_align,
                        std::function<void(Eigen::Index, Eigen::Index)> f, std::function<void()> done) const {
    // Compute small problems directly in the caller thread.
    if (n <= 1 || numThreads() == 1 || CostModel::numThreads(n, cost, static_cast<int>(numThreads())) == 1) {
      f(0, n);
      done();
      return;
    }

    // Compute block size and total count of blocks.
    ParallelForBlock block = CalculateParallelForBlock(n, cost, block_align);

    ParallelForAsyncContext* const ctx = new ParallelForAsyncContext(block.count, std::move(f), std::move(done));

    // Recursively divide size into halves until we reach block_size.
    // Division code rounds mid to block_size, so we are guaranteed to get
    // block_count leaves that do actual computations.
    ctx->handle_range = [this, ctx, block](Eigen::Index firstIdx, Eigen::Index lastIdx) {
      while (lastIdx - firstIdx > block.size) {
        // Split into halves and schedule the second half on a different thread.
        const Eigen::Index midIdx = firstIdx + Eigen::divup((lastIdx - firstIdx) / 2, block.size) * block.size;
        pool_->Schedule([ctx, midIdx, lastIdx]() { ctx->handle_range(midIdx, lastIdx); });
        lastIdx = midIdx;
      }

      // Single block or less, execute directly.
      ctx->f(firstIdx, lastIdx);

      // Delete async context if it was the last block.
      if (ctx->count.fetch_sub(1) == 1)
        delete ctx;
    };

    if (block.count <= numThreads()) {
      // Avoid a thread hop by running the root of the tree and one block on the
      // main thread.
      ctx->handle_range(0, n);
    } else {
      // Execute the root in the thread pool to avoid running work on more than
      // numThreads() threads.
      pool_->Schedule([ctx, n]() { ctx->handle_range(0, n); });
    }
  }

  // Convenience wrapper for parallelForAsync that does not align blocks.
  void parallelForAsync(Eigen::Index n, const Eigen::TensorOpCost& cost,
                        std::function<void(Eigen::Index, Eigen::Index)> f,
                        std::function<void()> done) const {
    parallelForAsync(n, cost, nullptr, std::move(f), std::move(done));
  }

  // Thread pool accessor.
  ThreadPoolInterface* getPool() const { return pool_; }

  // Allocator accessor.
  EigenAllocator* allocator() const { return allocator_; }

 private:
  typedef Eigen::TensorCostModel<ThreadPoolDevice> CostModel;

  // For parallelForAsync we must keep passed in closures on the heap, and
  // delete them only after `done` callback finished.
  struct ParallelForAsyncContext {
    ParallelForAsyncContext(Eigen::Index block_count, std::function<void(Eigen::Index, Eigen::Index)> block_f,
                            std::function<void()> done_callback)
        : count(block_count), f(std::move(block_f)), done(std::move(done_callback)) {}
    ~ParallelForAsyncContext() { done(); }

    std::atomic<Eigen::Index> count;
    std::function<void(Eigen::Index, Eigen::Index)> f;
    std::function<void()> done;

    std::function<void(Eigen::Index, Eigen::Index)> handle_range;
  };

  struct ParallelForBlock {
    Eigen::Index size;  // block size
    Eigen::Index count;  // number of blocks
  };

  // Calculates block size based on (1) the iteration cost and (2) parallel
  // efficiency. We want blocks to be not too small to mitigate parallelization
  // overheads; not too large to mitigate tail effect and potential load
  // imbalance and we also want number of blocks to be evenly dividable across
  // threads.
  ParallelForBlock CalculateParallelForBlock(const Eigen::Index n, const Eigen::TensorOpCost& cost,
                                             std::function<Eigen::Index(Eigen::Index)> block_align) const {
    const double block_size_f = 1.0 / CostModel::taskSize(1, cost);
    const Eigen::Index max_oversharding_factor = 4;
    Eigen::Index block_size = Eigen::numext::mini(n, Eigen::numext::maxi<Eigen::Index>(Eigen::divup<Eigen::Index>(n, max_oversharding_factor * numThreads()),
                                             block_size_f));
    const Eigen::Index max_block_size = Eigen::numext::mini(n, 2 * block_size);

    if (block_align) {
      Eigen::Index new_block_size = block_align(block_size);
      eigen_assert(new_block_size >= block_size);
      block_size = Eigen::numext::mini(n, new_block_size);
    }

    Eigen::Index block_count = Eigen::divup(n, block_size);

    // Calculate parallel efficiency as fraction of total CPU time used for
    // computations:
    double max_efficiency =
        static_cast<double>(block_count) / (Eigen::divup<int>(block_count, numThreads()) * numThreads());

    // Now try to increase block size up to max_block_size as long as it
    // doesn't decrease parallel efficiency.
    for (Eigen::Index prev_block_count = block_count; max_efficiency < 1.0 && prev_block_count > 1;) {
      // This is the next block size that divides size into a smaller number
      // of blocks than the current block_size.
      Eigen::Index coarser_block_size = Eigen::divup(n, prev_block_count - 1);
      if (block_align) {
        Eigen::Index new_block_size = block_align(coarser_block_size);
        eigen_assert(new_block_size >= coarser_block_size);
        coarser_block_size = Eigen::numext::mini(n, new_block_size);
      }
      if (coarser_block_size > max_block_size) {
        break;  // Reached max block size. Stop.
      }
      // Recalculate parallel efficiency.
      const Eigen::Index coarser_block_count = Eigen::divup(n, coarser_block_size);
      eigen_assert(coarser_block_count < prev_block_count);
      prev_block_count = coarser_block_count;
      const double coarser_efficiency = static_cast<double>(coarser_block_count) /
                                        (Eigen::divup<int>(coarser_block_count, numThreads()) * numThreads());
      if (coarser_efficiency + 0.01 >= max_efficiency) {
        // Taking it.
        block_size = coarser_block_size;
        block_count = coarser_block_count;
        if (max_efficiency < coarser_efficiency) {
          max_efficiency = coarser_efficiency;
        }
      }
    }

    return {block_size, block_count};
  }

  Eigen::ThreadPoolInterface* pool_;
  int num_threads_;
  EigenAllocator* allocator_;
};


}  // namespace Eigen


