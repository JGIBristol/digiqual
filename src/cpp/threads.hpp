#ifndef DIGIQUAL_THREADS_HPP
#define DIGIQUAL_THREADS_HPP

#include <atomic>
#include <thread>

// Upper limit on the threads each kernel call starts (0 = no limit). Set to 1 in
// bootstrap worker processes, so N workers don't each start one thread per core.
inline std::atomic<unsigned int>& digiqual_thread_limit() {
    static std::atomic<unsigned int> limit{0};
    return limit;
}

inline unsigned int digiqual_max_threads() {
    unsigned int n = std::thread::hardware_concurrency();
    if (n == 0) n = 4;
    unsigned int limit = digiqual_thread_limit().load();
    return (limit > 0 && limit < n) ? limit : n;
}

#endif // DIGIQUAL_THREADS_HPP
