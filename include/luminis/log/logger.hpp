#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <format>
#include <mutex>
#include <string>
#include <string_view>

namespace luminis::log {

enum class Level : int { debug = 1, info = 2, warn = 3, error = 4, off = 6 };

inline std::string_view to_string(Level lv) {
  switch (lv) {
  case Level::debug:
    return "D";
  case Level::info:
    return "I";
  case Level::warn:
    return "W";
  case Level::error:
    return "E";
  default:
    return "O";
  }
}

class Logger {
public:
  static Logger &instance() {
    static Logger L;
    return L;
  }

  void set_level(Level lv) {
    level_.store(lv, std::memory_order_relaxed);
  }

  Level level() const {
    return level_.load(std::memory_order_relaxed);
  }

  template <class... Args>
  void log(Level lv, std::string_view fmt, Args &&...args) {
    log_impl(lv, fmt, std::forward<Args>(args)...);
  }

private:
  std::atomic<Level> level_{Level::info}; // default INFO
  std::mutex mu_;

  template <class... Args>
  void log_impl(Level lv, std::string_view fmt, Args &&...args) {
    if (lv < level())
      return;

    // timestamp (YYYY-MM-DD HH:MM:SS)
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char tbuf[9];
    std::strftime(tbuf, sizeof(tbuf), "%H:%M:%S", &tm);

    std::string body = std::vformat(fmt, std::make_format_args(args...));
    std::string line = std::format("[{} {}] {}\n", tbuf, to_string(lv), body);

    std::scoped_lock lk(mu_);
    std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
  }
};

// convenience macros
#define LLOG_DEBUG(...)                                                        \
  ::luminis::log::Logger::instance().log(::luminis::log::Level::debug,         \
                                         __VA_ARGS__)
#define LLOG_INFO(...)                                                         \
  ::luminis::log::Logger::instance().log(::luminis::log::Level::info,          \
                                         __VA_ARGS__)
#define LLOG_WARN(...)                                                         \
  ::luminis::log::Logger::instance().log(::luminis::log::Level::warn,          \
                                         __VA_ARGS__)
#define LLOG_ERROR(...)                                                        \
  ::luminis::log::Logger::instance().log(::luminis::log::Level::error,         \
                                         __VA_ARGS__)

} // namespace luminis::log
