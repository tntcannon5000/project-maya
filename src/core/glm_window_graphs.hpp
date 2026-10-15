// Bounded, opt-in full verification MoE graphs. This cache owns no input storage.
#pragma once

#include "strata/core/graph.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace strata::core::glmfast {

class WindowMoeGraphs {
public:
    // Fixed-address identities and effective route configuration, not live table contents.
    using Key = std::array<uint64_t, 64>;
    // expected, component layers, component fallbacks, prefix launches, down launches.
    using Accounts = std::array<uint64_t, 5>;
    enum class Result { Enqueued, Fallback, Failed };
    struct Stats {
        uint64_t eligible = 0, captures = 0, replays = 0, requests = 0;
        uint64_t capture_failures = 0, disabled_fallbacks = 0, invalidations = 0, launch_failures = 0;
        uint64_t bypass_scope = 0, bypass_diag = 0, bypass_format = 0, bypass_storage = 0, bypass_async = 0;
        uint64_t credit_waits = 0, credit_timeouts = 0, nodes = 0;
        double build_ms = 0, first_submit_ms = 0, replay_submit_ms = 0, credit_ms = 0;
    } stats;

    // body only records launches and normal host counters. It copies its final
    // counters into accounts even on failure. Capture never executes GPU work;
    // a failed build restores accounts and permanently bypasses this key.
    // A launch attempt is irrevocable: retain its credits and propagate failure,
    // since a later event-record failure may follow an already accepted launch.
    Result enqueue(int layer, int T, const Key& key, cudaStream_t stream, cudaStream_t side,
                   Accounts& accounts, const std::function<bool()>& body, std::string& err) {
        ++stats.eligible;
        const auto id = std::make_pair(layer, T);
        auto it = entries_.find(id);
        if (it != entries_.end() && it->second.key != key) {
            // A pointer/configuration change may follow queued launches. Drain
            // both streams before destroying any executable or reusable event.
            if (!drain(stream, side, err)) return Result::Failed;
            entries_.erase(it);
            ++stats.invalidations;
            it = entries_.end();
        }
        if (it != entries_.end()) {
            if (it->second.disabled) {
                ++stats.disabled_fallbacks;
                return Result::Fallback;
            }
            for (size_t i = 0; i < accounts.size(); ++i) accounts[i] += it->second.delta[i];
            return launch(it->second, stream, T, false, err);
        }

        const Accounts before = accounts;
        Entry entry;
        entry.key = key;
        const auto build_start = Clock::now();
        const bool began = entry.graph.begin(stream, err);
        bool ok = false;
        if (began) {
            ok = body();
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                err = std::string("window MoE capture body: ") + cudaGetErrorString(e);
                ok = false;
            }
            if (accounts[0] != before[0] + (uint64_t) T) {
                err = "window MoE capture request accounting mismatch";
                ok = false;
            }
            if (ok) {
                ok = entry.graph.end(stream, err);
            } else {
                cudaGraph_t abandoned = nullptr;
                const cudaError_t ended = cudaStreamEndCapture(stream, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                if (ended != cudaSuccess && err.empty())
                    err = std::string("window MoE abandon capture: ") + cudaGetErrorString(ended);
            }
        }
        stats.build_ms += ms(build_start);
        if (!ok) {
            accounts = before;
            ++stats.capture_failures;
            // Never synchronize a participating stream while capture is active.
            // Confirm EndCapture cleaned up, including invalidated/unjoined capture.
            cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
            const cudaError_t q = cudaStreamIsCapturing(stream, &status);
            if (q != cudaSuccess || status != cudaStreamCaptureStatusNone) {
                err += "; capture cleanup did not leave the origin stream usable";
                return Result::Failed;
            }
            (void) cudaGetLastError();
            // The body's error guard deliberately skipped this drain during
            // capture. Only now may the side stream be synchronized.
            if (!drain(stream, side, err)) return Result::Failed;
            last_capture_error = err;
            entry.disabled = true;
            entries_.emplace(id, std::move(entry));
            err.clear();
            return Result::Fallback;
        }
        for (size_t i = 0; i < accounts.size(); ++i) entry.delta[i] = accounts[i] - before[i];
        stats.nodes += entry.graph.nodes();
        ++stats.captures;
        auto inserted = entries_.emplace(id, std::move(entry));
        return launch(inserted.first->second, stream, T, true, err);
    }

    // Caller must drain the compute and component streams BEFORE clear/free.
    void clear() { stats.invalidations += entries_.size(); entries_.clear(); }
    size_t size() const { return entries_.size(); }
    std::string last_capture_error;

private:
    using Clock = std::chrono::steady_clock;
    static double ms(Clock::time_point begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }
    static bool drain(cudaStream_t stream, cudaStream_t side, std::string& err) {
        const cudaError_t main_error = cudaStreamSynchronize(stream);
        const cudaError_t side_error = side ? cudaStreamSynchronize(side) : cudaSuccess;
        if (main_error == cudaSuccess && side_error == cudaSuccess) return true;
        err = std::string("window MoE graph drain: ") +
              cudaGetErrorString(main_error != cudaSuccess ? main_error : side_error);
        return false;
    }
    struct Entry {
        Key key{};
        Accounts delta{};
        CapturedGraph graph;
        bool disabled = false;
    };
    Result launch(Entry& entry, cudaStream_t stream, int T, bool first, std::string& err) {
        const auto start = Clock::now();
        const bool ok = entry.graph.launch(stream, err);
        (first ? stats.first_submit_ms : stats.replay_submit_ms) += ms(start);
        stats.requests += (uint64_t) T;
        if (!first) ++stats.replays;
        if (ok) return Result::Enqueued;
        ++stats.launch_failures;
        // Do not enqueue fallback after an attempted live launch, even when the
        // error reports that launch itself failed. The caller aborts the window.
        return Result::Failed;
    }
    std::map<std::pair<int, int>, Entry> entries_;
};

}  // namespace strata::core::glmfast
