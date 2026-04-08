/**
 * cpu_stress.cpp
 * 多线程 CPU 压力测试工具
 * 用法:
 *   ./cpu_stress              # 占满所有核心
 *   ./cpu_stress 4            # 指定 4 个线程
 *   ./cpu_stress 4 60         # 4 线程跑 60 秒后自动退出
 *   ./cpu_stress 4 60 75      # 4 线程、目标占用率 75%（duty cycle 模式）
 *
 * Ctrl+C 随时停止。
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <signal.h>

static std::atomic<bool> g_running{true};

void sig_handler(int) { g_running = false; }

// ── 全力模式：纯整数 + 浮点混合运算，防编译器优化 ──────────────────────────
static void burn_full(std::atomic<uint64_t> &counter)
{
    volatile double x = 1.23456789;
    volatile uint64_t n = 0;
    while (g_running)
    {
        // 一组混合运算，难以被优化消除
        x = std::sqrt(x + 1.0) * 0.9999999 + 0.0000001;
        x = std::sin(x) * std::cos(x) + std::tan(x * 0.001);
        n += static_cast<uint64_t>(x * 1e6) ^ 0xDEADBEEFULL;
        ++counter;
    }
    (void)n;
}

// ── Duty-cycle 模式：工作 work_us 微秒，睡眠 sleep_us 微秒，循环 ──────────
static void burn_duty(std::atomic<uint64_t> &counter,
                      uint64_t work_us, uint64_t sleep_us)
{
    volatile double x = 1.23456789;
    volatile uint64_t n = 0;
    while (g_running)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        while (g_running)
        {
            x = std::sqrt(x + 1.0) * 0.9999999 + 0.0000001;
            x = std::sin(x) * std::cos(x) + std::tan(x * 0.001);
            n += static_cast<uint64_t>(x * 1e6) ^ 0xDEADBEEFULL;
            ++counter;
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t_start).count();
            if (static_cast<uint64_t>(elapsed) >= work_us) break;
        }
        if (sleep_us > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }
    (void)n;
}

int main(int argc, char *argv[])
{
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    // ── 参数解析 ────────────────────────────────────────────────────────────
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    int duration_s  = 0;   // 0 = 无限
    int duty_pct    = 100; // 占用百分比

    if (argc >= 2) num_threads = std::atoi(argv[1]);
    if (argc >= 3) duration_s  = std::atoi(argv[2]);
    if (argc >= 4) duty_pct    = std::atoi(argv[3]);

    if (num_threads <= 0) num_threads = 1;
    if (duty_pct < 1)   duty_pct = 1;
    if (duty_pct > 100) duty_pct = 100;

    // duty cycle：以 10ms 为一个周期
    const uint64_t cycle_us = 10000ULL;
    const uint64_t work_us  = cycle_us * static_cast<uint64_t>(duty_pct) / 100;
    const uint64_t sleep_us = cycle_us - work_us;

    // ── 打印信息 ─────────────────────────────────────────────────────────────
    std::cout << "┌─────────────────────────────────────────┐\n";
    std::cout << "│          CPU Stress Tool                │\n";
    std::cout << "├─────────────────────────────────────────┤\n";
    std::cout << "│  threads  : " << std::setw(4) << num_threads
              << " / " << std::thread::hardware_concurrency() << " logical cores    │\n";
    std::cout << "│  duration : ";
    if (duration_s > 0)
        std::cout << std::setw(4) << duration_s << " s                      │\n";
    else
        std::cout << "   ∞  (Ctrl+C to stop)        │\n";
    std::cout << "│  CPU load : " << std::setw(3) << duty_pct << "%                        │\n";
    std::cout << "└─────────────────────────────────────────┘\n";
    std::cout << std::flush;

    // ── 启动工作线程 ──────────────────────────────────────────────────────────
    std::vector<std::atomic<uint64_t>> counters(num_threads);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i)
    {
        if (duty_pct == 100)
            workers.emplace_back(burn_full, std::ref(counters[i]));
        else
            workers.emplace_back(burn_duty, std::ref(counters[i]), work_us, sleep_us);
    }

    // ── 监控线程：每秒打印吞吐量 ──────────────────────────────────────────────
    auto t_start = std::chrono::steady_clock::now();
    int elapsed = 0;
    while (g_running)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ++elapsed;

        uint64_t total = 0;
        for (auto &c : counters) total += c.load(std::memory_order_relaxed);

        auto now = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(now - t_start).count();

        std::cout << "\r[" << std::setw(4) << elapsed << "s]  "
                  << std::setw(12) << total / static_cast<uint64_t>(secs)
                  << " iters/s   total=" << total
                  << "          " << std::flush;

        if (duration_s > 0 && elapsed >= duration_s)
        {
            g_running = false;
            break;
        }
    }

    std::cout << "\nStopping..." << std::flush;
    for (auto &w : workers) w.join();
    std::cout << " done.\n";
    return 0;
}
