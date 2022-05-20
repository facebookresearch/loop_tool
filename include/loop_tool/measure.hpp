// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace dabun
{

namespace detail
{

template <class Fn>
inline double measure_single_run(Fn&& fn)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    fn();
    auto end = high_resolution_clock::now();

    auto nsecs = duration_cast<nanoseconds>(end - start).count();

    return static_cast<double>(nsecs) / 1e9;
}

template <class Fn>
__attribute__((always_inline)) inline void warmup_run(Fn&& fn,
                                                      unsigned iterations = 0)
{
    for (unsigned i = 0; i < iterations; ++i)
    {
        fn();
    }
}

template <class Fn>
__attribute__((always_inline)) inline void
warmup_run_time_limited(Fn&& fn, unsigned iterations = 0, double seconds = 1.0)
{
    double total_time = 0.0;

    for (unsigned i = 0; i < iterations && total_time <= seconds; ++i)
    {
        total_time += measure_single_run(fn);
    }
}

} // namespace detail

struct time_duraton_measurement
{
    double shortest = std::numeric_limits<double>::max();
    double mean     = std::numeric_limits<double>::max();
    double median   = std::numeric_limits<double>::max();
};

struct flops_measurement
{
    double shortest = std::numeric_limits<double>::max();
    double mean     = std::numeric_limits<double>::max();
    double median   = std::numeric_limits<double>::max();
};

template <class Fn>
double measure_fastest(Fn&& fn, unsigned iterations = 1)
{
    double ret = std::numeric_limits<double>::max();

    for (unsigned i = 0; i < iterations; ++i)
    {
        ret = std::min(ret, detail::measure_single_run(fn));
    }

    return ret;
}

template <class Fn>
double measure_fastest_time_limited(Fn&& fn, unsigned iterations = 1,
                                    double seconds = 1.0)
{
    double ret        = std::numeric_limits<double>::max();
    double total_time = 0.0;

    for (unsigned i = 0; i < iterations && total_time <= seconds; ++i)
    {
        auto t = detail::measure_single_run(fn);
        ret    = std::min(ret, t);
        total_time += t;
    }

    return ret;
}

template <class Fn>
double measure_mean(Fn&& fn, unsigned iterations = 1,
                    unsigned warmup_iterations = 1)
{
    if (iterations == 0)
    {
        return std::numeric_limits<double>::max();
    }

    detail::warmup_run(fn, warmup_iterations);

    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        fn();
    }

    auto end   = high_resolution_clock::now();
    auto nsecs = duration_cast<nanoseconds>(end - start).count();

    return static_cast<double>(nsecs) / 1e9 / iterations;
}

template <class Fn>
double measure_mean_time_limited(Fn&& fn, unsigned iterations = 1,
                                 unsigned warmup_iterations = 1,
                                 double   seconds           = 1.0)
{
    double ret = std::numeric_limits<double>::max();

    if (iterations == 0)
    {
        return ret;
    }

    detail::warmup_run_time_limited(fn, warmup_iterations, seconds);

    double total_time = 0.0;

    for (unsigned i = 0; i < iterations && total_time <= seconds; ++i)
    {
        total_time += detail::measure_single_run(fn);
    }

    return total_time / iterations;
}

template <class Fn>
double measure_mean_timed(Fn&& fn, double seconds = 1.0)
{
    using namespace std::chrono;

    std::size_t n_iter = 1;
    auto        start  = high_resolution_clock::now();

    while (1)
    {
        for (std::size_t i = 0; i < n_iter; ++i)
        {
            fn();
        }

        auto end   = high_resolution_clock::now();
        auto nsecs = duration_cast<nanoseconds>(end - start).count();
        if (static_cast<double>(nsecs) / 1e9 > seconds / 2)
        {
            break;
        }

        n_iter *= 2;
    }

    return measure_mean(fn, n_iter * 2);
}

template <class Fn>
double measure_mean_timed_and_bounded(Fn&& fn, double seconds = 1.0,
                                      std::size_t min_iterations = 1,
                                      std::size_t max_iterations =
                                      std::numeric_limits<std::size_t>::max())
{
    using namespace std::chrono;

    std::size_t n_iter = 1;
    auto        start  = high_resolution_clock::now();

    while (1)
    {
        for (std::size_t i = 0; i < n_iter; ++i)
        {
            fn();
        }

        auto end   = high_resolution_clock::now();
        auto nsecs = duration_cast<nanoseconds>(end - start).count();
        if (static_cast<double>(nsecs) / 1e9 > seconds / 2)
        {
            break;
        }

        n_iter *= 2;
    }

    n_iter = std::max(min_iterations, n_iter);
    n_iter = std::min(max_iterations, n_iter);

    return measure_mean(fn, n_iter * 2);
}


template <class Fn>
double measure_median(Fn&& fn, unsigned iterations = 1,
                      unsigned warmup_iterations = 1)
{
    using namespace std::chrono;

    if (iterations <= 0)
    {
        return std::numeric_limits<double>::max();
    }

    std::vector<double> measurements(iterations);

    detail::warmup_run(fn, warmup_iterations);

    for (int i = 0; i < iterations; ++i)
    {
        measurements[i] = detail::measure_single_run(fn);
    }

    std::sort(std::begin(measurements), std::end(measurements));

    return measurements[iterations / 2];
}

template <class Fn>
double measure_median_time_limited(Fn&& fn, unsigned iterations = 1,
                                   unsigned warmup_iterations = 1,
                                   double   seconds           = 1.0)
{
    if (iterations <= 0)
    {
        return std::numeric_limits<double>::max();
    }

    detail::warmup_run_time_limited(fn, warmup_iterations, seconds);

    std::vector<double> measurements(iterations);

    double total_time = 0.0;

    unsigned ran = 0;

    for (; ran < iterations && total_time <= seconds; ++ran)
    {
        measurements[ran] = detail::measure_single_run(fn);
    }

    std::sort(std::begin(measurements), std::end(measurements));

    return measurements[ran / 2];
}

template <class Fn>
std::tuple<double, double, double> measure_all(Fn&& fn, int iterations = 1,
                                               int warmup_iterations = 1)
{
    using namespace std::chrono;

    auto fastest = nanoseconds::max().count();

    std::vector<double> measurements(iterations);

    if (iterations <= 0)
    {
        return {-1., -1., -1.};
    }

    for (int i = 0; i < warmup_iterations; ++i)
    {
        fn();
    }

    for (int i = 0; i < iterations; ++i)
    {
        auto start = high_resolution_clock::now();
        fn();
        auto end        = high_resolution_clock::now();
        auto nsecs      = duration_cast<nanoseconds>(end - start).count();
        measurements[i] = static_cast<double>(nsecs) / 1e9;

        fastest = std::min(fastest, nsecs);
    }

    std::sort(std::begin(measurements), std::end(measurements));

    double the_median = measurements[iterations / 2];
    double the_mean =
        std::accumulate(std::begin(measurements), std::end(measurements), 0.0) /
        measurements.size();
    double the_fastest = static_cast<double>(fastest) / 1e9;

    return {the_fastest, the_mean, the_median};
}

} // namespace dabun
