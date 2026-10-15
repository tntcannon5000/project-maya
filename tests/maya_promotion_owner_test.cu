// Exercise the exact production metadata transition; no model or inference.
// Event readiness, drained consumers and the metadata lock are caller contracts.
#include "../src/core/glm_fast_state.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <tuple>

using State = strata::core::Glm5Model::FastState;
using Result = State::PromotionRetirement;
static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
struct Fixture {
    State state;
    std::array<uint8_t, 192> ram{};
    std::array<unsigned long long, 8> tables{};
    State::BgMove move{0, 0, 1, 0, 0, true, nullptr, true};
    std::string err;
    int writes = 0;
    Fixture() {
        state.n_keys = 4;
        state.lp.resize(1); state.rc.resize(1);
        auto& p = state.lp[0]; p.n = 3; p.st = {State::kLanding, State::kResident, State::kFree};
        p.key = {1, 1, -1}; p.tick.resize(3);
        auto& r = state.rc[0]; r.n = 3; r.stride = 64; r.base = ram.data();
        r.st = {State::kRPin, State::kRFree, State::kRFree}; r.key = {1, -1, -1}; r.tick.resize(3);
        state.slot_of = {-1, 1, -1, -1}; state.ram_of = {-1, 0, -1, -1};
        tables[1] = 0x1234; tables[5] = (unsigned long long)ram.data();
    }
    Result run() {
        return state.retire_promotion_owner(move, [&](size_t key, unsigned long long value) {
            require(key == 5, "superseded retirement must not publish GPU TAB");
            if (state.ram_of[1] != 0)
                require(state.rc[0].st[0] == State::kRPin, "source released before replacement publication");
            tables[key] = value; ++writes;
        }, err);
    }
    void evict_winner() {
        state.slot_of[1] = -1; tables[1] = 0;
        // Its former GPU slot has already been reused by a different expert.
        state.lp[0].key[1] = 2; state.slot_of[2] = 1; tables[2] = 0x5678;
    }
    auto ownership() const {
        return std::make_tuple(state.lp[0].st, state.lp[0].key, state.lp[0].tick,
            state.rc[0].st, state.rc[0].key, state.rc[0].tick, state.slot_of, state.ram_of,
            tables, writes, state.bg_superseded);
    }
};

int main() {
    int cases = 0;
    for (int winner : {1, -1}) {
        Fixture f; if (winner == -1) f.evict_winner();
        const auto gpu_table = f.tables[1]; const auto winner_key = f.state.lp[0].key[1];
        require(f.run() == Result::Superseded, "same RAM owner retirement");
        require(f.state.slot_of[1] == winner && f.tables[1] == gpu_table &&
            f.state.lp[0].key[1] == winner_key && f.state.lp[0].st[1] == State::kResident,
            "newer GPU mapping or reused slot changed");
        if (winner == -1) require(f.state.slot_of[2] == 1 && f.tables[2] == 0x5678, "replacement expert changed");
        require(f.state.lp[0].st[0] == State::kFree && f.state.lp[0].key[0] == -1, "losing destination not released");
        require(f.state.rc[0].st[0] == State::kRHold && f.state.ram_of[1] == 0, "live RAM backing not retained");
        require(f.writes == 1 && f.state.bg_superseded == 1, "retirement accounting");
        ++cases;
    }
    for (int winner : {1, -1}) for (char replacement : {char(State::kRHold), char(State::kRPin),
             char(State::kRDemote), char(State::kRLoad), char(State::kRNew), char(State::kRFree)}) {
        Fixture f; if (winner == -1) f.evict_winner(); f.state.ram_of[1] = 1;
        const auto gpu_table = f.tables[1]; const auto winner_key = f.state.lp[0].key[1];
        auto& r = f.state.rc[0]; r.st[1] = replacement; r.key[1] = 1;
        require(f.run() == Result::Superseded, "replacement owner retirement");
        const bool ready = replacement == State::kRHold || replacement == State::kRPin;
        require(f.tables[5] == (ready ? (unsigned long long)(f.ram.data()+64) : 0ull), "stale or pending RAM pointer published");
        require(r.st[0] == State::kRFree && r.key[0] == -1, "obsolete source not released");
        require(r.st[1] == replacement && r.key[1] == 1 && f.state.ram_of[1] == 1, "new RAM owner overwritten");
        require(f.state.slot_of[1] == winner && f.tables[1] == gpu_table &&
            f.state.lp[0].key[1] == winner_key && f.state.lp[0].st[1] == State::kResident,
            "winner eviction or reused slot not preserved");
        if (winner == -1) require(f.state.slot_of[2] == 1 && f.tables[2] == 0x5678, "replacement expert changed");
        ++cases;
    }
    for (int replacement : {-1, 1, 99}) {
        Fixture f; f.state.ram_of[1] = replacement;
        // Slot1 is ready but belongs to another key: it must not be published.
        f.state.rc[0].st[1] = State::kRHold; f.state.rc[0].key[1] = 2;
        require(f.run() == Result::Superseded && f.tables[5] == 0, "missing/foreign/out-of-range backing published");
        require(f.state.ram_of[1] == replacement, "replacement metadata unexpectedly changed");
        ++cases;
    }
    for (int fault = 0; fault < 5; ++fault) {
        Fixture f;
        if (fault == 0) f.move.superseded = false; // unmarked competing GPU owner
        if (fault == 1) f.state.lp[0].st[0] = State::kResident;
        if (fault == 2) f.state.rc[0].st[0] = State::kRHold;
        if (fault == 3) f.state.rc[0].key[0] = 2;
        if (fault == 4) f.move.rslot = 99;
        const auto before = f.ownership();
        require(f.run() == Result::Failed && !f.state.bg_copy_error.empty(), "invalid reservation not fatal");
        require(f.ownership() == before, "failed transition mutated ownership");
        f.move.superseded = true; f.move.rslot = 0; f.state.lp[0].st[0] = State::kLanding;
        f.state.rc[0].st[0] = State::kRPin; f.state.rc[0].key[0] = 1;
        const auto corrected = f.ownership();
        require(f.run() == Result::Failed && f.ownership() == corrected, "sticky error allowed retry");
        ++cases;
    }
    {
        Fixture f; f.move.superseded = false; f.state.slot_of[1] = -1;
        const auto before = f.ownership();
        require(f.run() == Result::Publish && f.ownership() == before,
                "ordinary upload should continue without superseded edits");
        ++cases;
    }
    {
        Fixture f; require(f.run() == Result::Superseded, "first retirement");
        const auto before = f.ownership();
        require(f.run() == Result::Failed && f.ownership() == before, "double retirement changed ownership");
        ++cases;
    }
    std::printf("PASS %d production promotion metadata cases; DMA ordering is not covered by this unit test\n", cases);
}
