/**
 * @file progress.hpp
 * @brief Lock-free progress monitor for Monte Carlo simulations.
 *
 * `ProgressMonitor` tracks how many photons have been completed and fires
 * a user-supplied callback at configurable percentage intervals. All methods
 * are thread-safe; the implementation uses `std::atomic` operations so worker
 * threads never block each other.
 *
 * Typical usage from the Python side:
 * @code
 *   monitor = luminis.ProgressMonitor()
 *   monitor.setup(total=1_000_000, callback=lambda d,t: print(f"{100*d//t}%"), interval_pct=5)
 *   config.progress = monitor
 *   luminis.run_simulation_parallel(config)
 * @endcode
 *
 * @see simulation.hpp  SimConfig::progress
 */
#pragma once
#include <atomic>
#include <cstddef>
#include <functional>

namespace luminis::log
{

  /// Callback signature: (photons_completed, total_photons)
  using ProgressCallback = std::function<void(std::size_t, std::size_t)>;

  /**
   * @brief Lock-free progress monitor for photon transport simulations.
   *
   * Call `setup()` before the simulation starts, then have each worker thread
   * call `tick()` after every photon. The monitor fires the callback only when
   * the completion count crosses the next reporting threshold, so callback
   * overhead is negligible (typically ~20 invocations for the default 5 %
   * interval on a million-photon run).
   */
  class ProgressMonitor
  {
  public:
    ProgressMonitor() = default;

    /**
     * @brief Configure the monitor before a simulation run.
     *
     * @param total         Total number of photons in the simulation.
     * @param cb            Callback invoked at each reporting threshold
     *                      (may be nullptr for silent monitoring).
     * @param interval_pct  Reporting interval as a percentage (1–100).
     *                      Default is 5 %, yielding 20 callbacks per run.
     */
    void setup(std::size_t total, ProgressCallback cb = nullptr, std::size_t interval_pct = 5)
    {
      total_ = total;
      completed_.store(0, std::memory_order_relaxed);
      callback_ = std::move(cb);
      // Compute how many photons correspond to one reporting interval
      interval_ = (total * interval_pct) / 100;
      if (interval_ == 0)
        interval_ = 1;
      next_report_.store(interval_, std::memory_order_relaxed);
      enabled_ = true;
    }

    /**
     * @brief Record one (or more) completed photons. Thread-safe.
     *
     * Uses `fetch_add` (no lock) and only fires the callback when the
     * cumulative count crosses the next reporting threshold. The
     * `compare_exchange_weak` loop ensures exactly one thread fires each
     * threshold, so the callback is never invoked concurrently.
     */
    void tick(std::size_t count = 1)
    {
      if (!enabled_)
        return;
      std::size_t prev = completed_.fetch_add(count, std::memory_order_relaxed);
      std::size_t now = prev + count;

      // Check if we crossed the next reporting threshold
      std::size_t expected = next_report_.load(std::memory_order_relaxed);
      while (now >= expected)
      {
        if (next_report_.compare_exchange_weak(expected, expected + interval_,
                                               std::memory_order_relaxed))
        {
          if (callback_)
          {
            callback_(now, total_);
          }
          break;
        }
        // expected is reloaded by compare_exchange_weak on failure
      }
    }

    /// Number of photons completed so far.
    std::size_t completed() const { return completed_.load(std::memory_order_relaxed); }

    /// Total photon budget.
    std::size_t total() const { return total_; }

    /// Whether the monitor is active.
    bool is_enabled() const { return enabled_; }

    /// Disable the monitor (no further callbacks).
    void disable() { enabled_ = false; }

  private:
    std::size_t total_ = 0;
    std::size_t interval_ = 1;
    std::atomic<std::size_t> completed_{0};
    std::atomic<std::size_t> next_report_{0};
    ProgressCallback callback_ = nullptr;
    bool enabled_ = false;
  };

} // namespace luminis::log
