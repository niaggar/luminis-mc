#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>

namespace luminis::log
{

  /**
   * Lock-free terminal progress reporter for photon simulations.
   *
   * Threads call tick() after each photon. The reporter redraws the
   * progress line on stdout using \r — no newline spam, no mutex.
   *
   * Example output:
   *   Photons  [████████████░░░░░░░░]  600 000 / 1 000 000  60.0%  |  2.1M ph/s  |  ETA 0:09
   */
  class ProgressReporter
  {
  public:
    ProgressReporter() = default;

    void setup(std::size_t total, std::size_t interval_pct = 5)
    {
      total_ = total;
      interval_ = std::max(std::size_t(1), (total * interval_pct) / 100);
      completed_.store(0, std::memory_order_relaxed);
      next_report_.store(interval_, std::memory_order_relaxed);
      start_time_ = std::chrono::steady_clock::now();
      enabled_ = true;
    }

    void tick(std::size_t count = 1)
    {
      if (!enabled_)
        return;

      std::size_t prev = completed_.fetch_add(count, std::memory_order_relaxed);
      std::size_t now = prev + count;

      std::size_t expected = next_report_.load(std::memory_order_relaxed);
      while (now >= expected)
      {
        if (next_report_.compare_exchange_weak(expected, expected + interval_,
                                               std::memory_order_relaxed))
        {
          print_progress(now);
          break;
        }
      }
    }

    // Imprime la línea final con newline al terminar
    void finish()
    {
      if (!enabled_)
        return;
      print_progress(total_);
      std::printf("\n");
      enabled_ = false;
    }

    std::size_t completed() const { return completed_.load(std::memory_order_relaxed); }

  private:
    void print_progress(std::size_t done) const
    {
      double fraction = (total_ > 0) ? double(done) / double(total_) : 1.0;

      // Barra de 20 caracteres
      constexpr int BAR_WIDTH = 20;
      int filled = int(fraction * BAR_WIDTH);
      std::string bar(filled, '#');
      bar += std::string(BAR_WIDTH - filled, '-');

      // Fotones/segundo
      auto now = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(now - start_time_).count();
      double rate = (secs > 0) ? double(done) / secs : 0.0;

      // ETA en segundos
      double remaining = (rate > 0 && done < total_)
                             ? double(total_ - done) / rate
                             : 0.0;
      int eta_m = int(remaining) / 60;
      int eta_s = int(remaining) % 60;

      // Formatear con separadores de miles usando un helper simple
      std::printf("\r  Photons  [%s]  %s / %s  %5.1f%%  |  %.2g ph/s  |  ETA %d:%02d   ",
                  bar.c_str(),
                  format_int(done).c_str(),
                  format_int(total_).c_str(),
                  fraction * 100.0,
                  rate,
                  eta_m, eta_s);
      std::fflush(stdout);
    }

    static std::string format_int(std::size_t n)
    {
      // Formatea con espacios: 1 000 000
      std::string s = std::to_string(n);
      int i = int(s.size()) - 3;
      while (i > 0)
      {
        s.insert(i, " ");
        i -= 3;
      }
      return s;
    }

    std::size_t total_ = 0;
    std::size_t interval_ = 1;
    std::atomic<std::size_t> completed_{0};
    std::atomic<std::size_t> next_report_{0};
    std::chrono::steady_clock::time_point start_time_;
    bool enabled_ = false;
  };

} // namespace luminis