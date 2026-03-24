/*
 * engine_og.h -- Pure C++ minimax engine (hash-map variant, namespace og).
 *
 * No pybind11, no Emscripten -- just the search engine.
 * Uses ankerl::unordered_dense flat hash maps for all board data.
 */
#pragma once

#include "types.h"

// Include the ankerl stl prerequisites directly to avoid "stl.h" path
// collision with pybind11's stl.h on the include path.
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#define ANKERL_UNORDERED_DENSE_STD_MODULE 1
#include "ankerl_unordered_dense.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

// ── Alias flat hash containers ──
template <typename K, typename V, typename H = ankerl::unordered_dense::hash<K>>
using og_flat_map = ankerl::unordered_dense::map<K, V, H>;

template <typename K, typename H = ankerl::unordered_dense::hash<K>>
using og_flat_set = ankerl::unordered_dense::set<K, H>;

// ═══════════════════════════════════════════════════════════════════════
//  Constants  (mirror ai.py hyperparameters exactly)
// ═══════════════════════════════════════════════════════════════════════
static constexpr int    OG_CANDIDATE_CAP      = 11;
static constexpr int    OG_ROOT_CANDIDATE_CAP = 13;
static constexpr int    OG_NEIGHBOR_DIST      = 2;
static constexpr double OG_DELTA_WEIGHT       = 1.5;
static constexpr int    OG_MAX_QDEPTH         = 16;
static constexpr int    OG_WIN_LENGTH         = 6;
static constexpr double OG_WIN_SCORE          = 100000000.0;
static constexpr double OG_INF_SCORE          = std::numeric_limits<double>::infinity();

// TT flags
static constexpr int8_t OG_TT_EXACT = 0;
static constexpr int8_t OG_TT_LOWER = 1;
static constexpr int8_t OG_TT_UPPER = 2;

// ═══════════════════════════════════════════════════════════════════════
//  Coordinate packing
// ═══════════════════════════════════════════════════════════════════════
using OgCoord = int64_t;

static inline OgCoord og_pack(int q, int r) {
    return (static_cast<int64_t>(static_cast<uint32_t>(q)) << 32) |
            static_cast<uint32_t>(r);
}
static inline int og_pack_q(OgCoord c) { return static_cast<int32_t>(static_cast<uint32_t>(c >> 32)); }
static inline int og_pack_r(OgCoord c) { return static_cast<int32_t>(static_cast<uint32_t>(c)); }

static inline bool og_coord_lt(OgCoord a, OgCoord b) {
    int aq = og_pack_q(a), ar = og_pack_r(a), bq = og_pack_q(b), br = og_pack_r(b);
    return (aq < bq) || (aq == bq && ar < br);
}
static inline OgCoord og_coord_min(OgCoord a, OgCoord b) { return og_coord_lt(a, b) ? a : b; }
static inline OgCoord og_coord_max(OgCoord a, OgCoord b) { return og_coord_lt(a, b) ? b : a; }

// Window key: packs (d_idx, q, r) into int64_t.
static constexpr int OG_WKEY_BIAS = 0x8000000; // 2^27
static inline int64_t og_pack_wkey(int d_idx, int q, int r) {
    return (static_cast<int64_t>(static_cast<uint8_t>(d_idx)) << 56) |
           (static_cast<int64_t>(static_cast<uint32_t>(q + OG_WKEY_BIAS) & 0x0FFFFFFFu) << 28) |
            static_cast<int64_t>(static_cast<uint32_t>(r + OG_WKEY_BIAS) & 0x0FFFFFFFu);
}

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════
using OgTurn = std::pair<OgCoord, OgCoord>;

struct OgTurnHash {
    size_t operator()(const OgTurn& t) const {
        auto h = std::hash<int64_t>{};
        return h(t.first) ^ (h(t.second) * 0x9e3779b97f4a7c15ULL);
    }
};

struct OgWinOff  { int d_idx, oq, or_; };
struct OgEvalOff { int d_idx, k, oq, or_; };
struct OgNbOff   { int dq, dr; };

struct OgSavedState {
    int8_t cur_player;
    int8_t moves_left;
    int8_t winner;
    bool   game_over;
};

struct OgUndoStep {
    OgCoord      cell;
    OgSavedState state;
    int8_t       player;
};

struct OgTTEntry {
    int    depth;
    double score;
    int8_t flag;
    OgTurn move;
    bool   has_move;
};

struct OgTimeUp {};

// ═══════════════════════════════════════════════════════════════════════
//  Direction arrays
// ═══════════════════════════════════════════════════════════════════════
static constexpr int OG_DIR_Q[3] = {1, 0, 1};
static constexpr int OG_DIR_R[3] = {0, 1, -1};
static constexpr int OG_COLONY_DQ[6] = { 1, -1,  0,  0,  1, -1};
static constexpr int OG_COLONY_DR[6] = { 0,  0,  1, -1, -1,  1};

// ═══════════════════════════════════════════════════════════════════════
//  Precomputed offset tables (initialised once)
// ═══════════════════════════════════════════════════════════════════════
static std::vector<OgWinOff> og_g_win_offsets;
static std::vector<OgNbOff>  og_g_nb_offsets;

static inline int og_hex_distance(int dq, int dr) {
    return std::max({std::abs(dq), std::abs(dr), std::abs(dq + dr)});
}

// ═══════════════════════════════════════════════════════════════════════
//  Zobrist tables (lazy, global, never cleared)
// ═══════════════════════════════════════════════════════════════════════
static og_flat_map<OgCoord, uint64_t> og_g_zobrist_a, og_g_zobrist_b;
static std::mt19937_64 og_g_zobrist_rng(12345);

static inline uint64_t og_get_zobrist(OgCoord c, int8_t player) {
    auto& tbl = (player == P_A) ? og_g_zobrist_a : og_g_zobrist_b;
    auto it = tbl.find(c);
    if (it != tbl.end()) return it->second;
    uint64_t v = og_g_zobrist_rng();
    tbl[c] = v;
    return v;
}

static bool og_g_tables_ready = false;
static void og_ensure_tables() {
    if (og_g_tables_ready) return;
    for (int d = 0; d < 3; d++)
        for (int k = 0; k < OG_WIN_LENGTH; k++)
            og_g_win_offsets.push_back({d, k * OG_DIR_Q[d], k * OG_DIR_R[d]});
    for (int dq = -OG_NEIGHBOR_DIST; dq <= OG_NEIGHBOR_DIST; dq++)
        for (int dr = -OG_NEIGHBOR_DIST; dr <= OG_NEIGHBOR_DIST; dr++)
            if ((dq || dr) && og_hex_distance(dq, dr) <= OG_NEIGHBOR_DIST)
                og_g_nb_offsets.push_back({dq, dr});
    og_g_tables_ready = true;
}

// ═══════════════════════════════════════════════════════════════════════
//  MinimaxBot  (namespace og -- hash-map variant)
// ═══════════════════════════════════════════════════════════════════════
namespace og {

class MinimaxBot {
public:
    // ── Public attributes ──
    bool   pair_moves = true;
    double time_limit;
    int    last_depth  = 0;
    int    _nodes      = 0;
    double last_score  = 0;
    double last_ebf    = 0;
    int    max_depth   = 200;

    // ── Constructors ──
    MinimaxBot() : time_limit(0.05), _rng(std::random_device{}()) { og_ensure_tables(); }

    explicit MinimaxBot(double tl)
        : time_limit(tl), _rng(std::random_device{}())
    {
        og_ensure_tables();
    }

    // ── Pattern loading ──
    void load_patterns(const double* values, int count, int eval_length,
                       const std::string& path = "") {
        _pv.assign(values, values + count);
        _eval_length = eval_length;
        _pattern_path_str = path;
        _build_eval_tables();
    }

    void load_patterns(const std::vector<double>& values, int eval_length,
                       const std::string& path = "") {
        load_patterns(values.data(), static_cast<int>(values.size()),
                      eval_length, path);
    }

    // ── Serialisation helpers ──
    EngineState get_state() const {
        return {time_limit, _pv, _eval_length, _pattern_path_str};
    }

    void set_state(const EngineState& es) {
        og_ensure_tables();
        time_limit = es.time_limit;
        _pv = es.pv;
        _eval_length = es.eval_length;
        _pattern_path_str = es.pattern_path_str;
        _rng = std::mt19937(std::random_device{}());
        _build_eval_tables();
    }

    // ── Main entry point ──
    MoveResult get_move(const GameState& gs) {
        if (gs.cells.empty())
            return {0, 0, 0, 0, 1};

        // ── Populate board from GameState ──
        _board.clear();
        _board.reserve(gs.cells.size() + 64);
        for (const auto& cell : gs.cells)
            _board[og_pack(cell.q, cell.r)] = cell.player;

        _cur_player = gs.cur_player;
        _moves_left = gs.moves_left;
        _move_count = gs.move_count;
        _winner     = P_NONE;
        _game_over  = false;

        // ── Deadline ──
        _deadline = Clock::now() + std::chrono::microseconds(
                        static_cast<int64_t>(time_limit * 2000000.0));

        // ── Player tracking / TT management ──
        if (_cur_player != _player) {
            _tt.clear();
            _history.clear();
        }
        _player    = _cur_player;
        _nodes     = 0;
        last_depth = 0;
        last_score = 0;
        last_ebf   = 0;
        if (_tt.size() > 1000000) _tt.clear();

        // ── Zobrist ──
        _hash = 0;
        for (const auto& kv : _board)
            _hash ^= og_get_zobrist(kv.first, kv.second);

        // ── Cell value mapping ──
        if (_player == P_A) { _cell_a = 1; _cell_b = 2; }
        else                { _cell_a = 2; _cell_b = 1; }

        // ── Init 6-cell windows ──
        _wc.clear();
        _hot_a.clear();
        _hot_b.clear();
        {
            og_flat_set<int64_t> seen;
            for (const auto& kv : _board) {
                int bq = og_pack_q(kv.first), br = og_pack_r(kv.first);
                for (const auto& wo : og_g_win_offsets) {
                    int64_t wkey = og_pack_wkey(wo.d_idx, bq - wo.oq, br - wo.or_);
                    if (!seen.insert(wkey).second) continue;
                    int d = wo.d_idx;
                    int sq = bq - wo.oq, sr = br - wo.or_;
                    int ac = 0, bc = 0;
                    for (int j = 0; j < OG_WIN_LENGTH; j++) {
                        auto it = _board.find(og_pack(sq + j * OG_DIR_Q[d], sr + j * OG_DIR_R[d]));
                        if (it != _board.end()) { if (it->second == P_A) ac++; else bc++; }
                    }
                    if (ac || bc) _wc[wkey] = {static_cast<int8_t>(ac), static_cast<int8_t>(bc)};
                }
            }
            for (const auto& kv : _wc) {
                if (kv.second.first  >= 4) _hot_a.insert(kv.first);
                if (kv.second.second >= 4) _hot_b.insert(kv.first);
            }
        }

        // ── Init N-cell eval windows ──
        _wp.clear();
        _eval_score = 0.0;
        {
            const double* pv = _pv.data();
            og_flat_set<int64_t> seen;
            for (const auto& kv : _board) {
                int bq = og_pack_q(kv.first), br = og_pack_r(kv.first);
                for (const auto& eo : _eval_offsets) {
                    int64_t wkey8 = og_pack_wkey(3 + eo.d_idx, bq - eo.oq, br - eo.or_);
                    if (!seen.insert(wkey8).second) continue;
                    int sq = bq - eo.oq, sr = br - eo.or_;
                    int d = eo.d_idx;
                    int pi = 0;
                    bool has = false;
                    for (int j = 0; j < _eval_length; j++) {
                        auto it = _board.find(og_pack(sq + j * OG_DIR_Q[d], sr + j * OG_DIR_R[d]));
                        if (it != _board.end()) {
                            pi += ((it->second == P_A) ? _cell_a : _cell_b) * _pow3[j];
                            has = true;
                        }
                    }
                    if (has) { _wp[wkey8] = pi; _eval_score += pv[pi]; }
                }
            }
        }

        // ── Init candidates ──
        _cand_rc.clear();
        _cand_set.clear();
        _rc_stack.clear();
        for (const auto& kv : _board) {
            int bq = og_pack_q(kv.first), br = og_pack_r(kv.first);
            for (const auto& nb : og_g_nb_offsets) {
                OgCoord nc = og_pack(bq + nb.dq, br + nb.dr);
                if (!_board.count(nc)) {
                    _cand_rc[nc]++;
                    _cand_set.insert(nc);
                }
            }
        }

        if (_cand_set.empty())
            return {0, 0, 0, 0, 1};

        bool maximizing = (_cur_player == _player);
        auto turns = _generate_turns();
        if (turns.empty())
            return {0, 0, 0, 0, 1};

        OgTurn best_move = turns[0];

        // ── Save state for OgTimeUp rollback ──
        auto saved_board    = _board;
        auto saved_st       = OgSavedState{_cur_player, _moves_left, _winner, _game_over};
        int  saved_mc       = _move_count;
        uint64_t saved_hash = _hash;
        double   saved_eval = _eval_score;
        auto saved_wc       = _wc;
        auto saved_wp       = _wp;
        auto saved_cs       = _cand_set;
        auto saved_cr       = _cand_rc;
        auto saved_ha       = _hot_a;
        auto saved_hb       = _hot_b;

        for (int depth = 1; depth <= max_depth; depth++) {
            try {
                int nb4 = _nodes;
                auto root_result = _search_root(turns, depth);
                OgTurn result = root_result.first;
                auto& scores = root_result.second;
                best_move  = result;
                last_depth = depth;
                auto si = scores.find(result);
                last_score = (si != scores.end()) ? si->second : 0.0;
                int nthis = _nodes - nb4;
                if (nthis > 1)
                    last_ebf = std::round(std::pow(static_cast<double>(nthis),
                                                   1.0 / depth) * 10.0) / 10.0;
                std::sort(turns.begin(), turns.end(),
                    [&scores, maximizing](const OgTurn& a, const OgTurn& b) {
                        double sa = 0, sb = 0;
                        auto ia = scores.find(a); if (ia != scores.end()) sa = ia->second;
                        auto ib = scores.find(b); if (ib != scores.end()) sb = ib->second;
                        return maximizing ? (sa > sb) : (sa < sb);
                    });
                if (std::abs(last_score) >= OG_WIN_SCORE) break;
            } catch (const OgTimeUp&) {
                _board      = std::move(saved_board);
                _move_count = saved_mc;
                _cur_player = saved_st.cur_player;
                _moves_left = saved_st.moves_left;
                _winner     = saved_st.winner;
                _game_over  = saved_st.game_over;
                _hash       = saved_hash;
                _eval_score = saved_eval;
                _wc         = std::move(saved_wc);
                _wp         = std::move(saved_wp);
                _cand_set   = std::move(saved_cs);
                _cand_rc    = std::move(saved_cr);
                _hot_a      = std::move(saved_ha);
                _hot_b      = std::move(saved_hb);
                break;
            }
        }

        return {og_pack_q(best_move.first),  og_pack_r(best_move.first),
                og_pack_q(best_move.second), og_pack_r(best_move.second), 2};
    }

private:
    // ── Pattern data ──
    std::vector<double>    _pv;
    int                    _eval_length = 6;
    std::vector<OgEvalOff> _eval_offsets;
    std::vector<int>       _pow3;
    std::string            _pattern_path_str;

    // ── Internal game state ──
    og_flat_map<OgCoord, int8_t> _board;
    int8_t _cur_player  = P_A;
    int8_t _moves_left  = 1;
    int8_t _winner      = P_NONE;
    bool   _game_over   = false;
    int    _move_count  = 0;

    // ── Search state ──
    using Clock = std::chrono::steady_clock;
    Clock::time_point _deadline;
    uint64_t _hash      = 0;
    int8_t   _player    = P_A;
    int8_t   _cell_a    = 1;
    int8_t   _cell_b    = 2;
    double   _eval_score = 0;

    // ── 6-cell window counts ──
    og_flat_map<int64_t, std::pair<int8_t,int8_t>> _wc;
    og_flat_set<int64_t> _hot_a, _hot_b;

    // ── N-cell eval window patterns ──
    og_flat_map<int64_t, int> _wp;

    // ── Candidates ──
    og_flat_map<OgCoord, int> _cand_rc;
    og_flat_set<OgCoord>      _cand_set;
    std::vector<int>          _rc_stack;

    // ── Transposition table & history ──
    og_flat_map<uint64_t, OgTTEntry> _tt;
    og_flat_map<OgCoord, int>        _history;

    // ── RNG ──
    std::mt19937 _rng;

    // ────────────────────────────────────────────────────────────────
    void _build_eval_tables() {
        _eval_offsets.clear();
        for (int d = 0; d < 3; d++)
            for (int k = 0; k < _eval_length; k++)
                _eval_offsets.push_back({d, k, k * OG_DIR_Q[d], k * OG_DIR_R[d]});
        _pow3.resize(_eval_length);
        _pow3[0] = 1;
        for (int i = 1; i < _eval_length; i++)
            _pow3[i] = _pow3[i - 1] * 3;
    }

    inline void _check_time() {
        _nodes++;
        if ((_nodes & 1023) == 0 && Clock::now() >= _deadline)
            throw OgTimeUp{};
    }

    inline uint64_t _tt_key() const {
        return _hash ^ (static_cast<uint64_t>(_cur_player) * 0x9e3779b97f4a7c15ULL)
                      ^ (static_cast<uint64_t>(_moves_left) * 0x517cc1b727220a95ULL);
    }

    // ────────────────────────────────────────────────────────────────
    //  Incremental make / undo
    // ────────────────────────────────────────────────────────────────
    void _make(int q, int r) {
        int8_t player = _cur_player;
        OgCoord cell = og_pack(q, r);

        _hash ^= og_get_zobrist(cell, player);

        int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;

        // ── 6-cell windows ──
        bool won = false;
        if (player == P_A) {
            for (const auto& wo : og_g_win_offsets) {
                int64_t wkey = og_pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];
                counts.first++;
                if (counts.first >= 4) _hot_a.insert(wkey);
                if (counts.first == OG_WIN_LENGTH && counts.second == 0) won = true;
            }
        } else {
            for (const auto& wo : og_g_win_offsets) {
                int64_t wkey = og_pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];
                counts.second++;
                if (counts.second >= 4) _hot_b.insert(wkey);
                if (counts.second == OG_WIN_LENGTH && counts.first == 0) won = true;
            }
        }

        // ── N-cell eval windows ──
        const double* pv = _pv.data();
        for (const auto& eo : _eval_offsets) {
            int64_t wkey8 = og_pack_wkey(3 + eo.d_idx, q - eo.oq, r - eo.or_);
            auto it = _wp.find(wkey8);
            int old_pi = (it != _wp.end()) ? it->second : 0;
            int new_pi = old_pi + cell_val * _pow3[eo.k];
            _eval_score += pv[new_pi] - pv[old_pi];
            if (it != _wp.end())
                it->second = new_pi;
            else
                _wp[wkey8] = new_pi;
        }

        // ── Candidates ──
        _cand_set.erase(cell);
        auto rc_it = _cand_rc.find(cell);
        _rc_stack.push_back((rc_it != _cand_rc.end()) ? rc_it->second : 0);
        if (rc_it != _cand_rc.end()) _cand_rc.erase(rc_it);

        for (const auto& nb : og_g_nb_offsets) {
            OgCoord nc = og_pack(q + nb.dq, r + nb.dr);
            _cand_rc[nc]++;
            if (!_board.count(nc))
                _cand_set.insert(nc);
        }

        // Place stone
        _board[cell] = player;
        _move_count++;

        if (won) {
            _winner    = player;
            _game_over = true;
        } else {
            _moves_left--;
            if (_moves_left <= 0) {
                _cur_player = (player == P_A) ? P_B : P_A;
                _moves_left = 2;
            }
        }
    }

    void _undo(int q, int r, const OgSavedState& st, int8_t player) {
        OgCoord cell = og_pack(q, r);

        _board.erase(cell);
        _move_count--;
        _cur_player = st.cur_player;
        _moves_left = st.moves_left;
        _winner     = st.winner;
        _game_over  = st.game_over;

        _hash ^= og_get_zobrist(cell, player);

        int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;

        // ── 6-cell windows ──
        if (player == P_A) {
            for (const auto& wo : og_g_win_offsets) {
                int64_t wkey = og_pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];
                counts.first--;
                if (counts.first < 4) _hot_a.erase(wkey);
            }
        } else {
            for (const auto& wo : og_g_win_offsets) {
                int64_t wkey = og_pack_wkey(wo.d_idx, q - wo.oq, r - wo.or_);
                auto& counts = _wc[wkey];
                counts.second--;
                if (counts.second < 4) _hot_b.erase(wkey);
            }
        }

        // ── N-cell eval windows ──
        const double* pv = _pv.data();
        for (const auto& eo : _eval_offsets) {
            int64_t wkey8 = og_pack_wkey(3 + eo.d_idx, q - eo.oq, r - eo.or_);
            int old_pi = _wp[wkey8];
            int new_pi = old_pi - cell_val * _pow3[eo.k];
            _eval_score += pv[new_pi] - pv[old_pi];
            if (new_pi == 0)
                _wp.erase(wkey8);
            else
                _wp[wkey8] = new_pi;
        }

        // ── Candidates ──
        for (const auto& nb : og_g_nb_offsets) {
            OgCoord nc = og_pack(q + nb.dq, r + nb.dr);
            auto it = _cand_rc.find(nc);
            int c = it->second - 1;
            if (c == 0) {
                _cand_rc.erase(it);
                _cand_set.erase(nc);
            } else {
                it->second = c;
            }
        }
        int saved_rc = _rc_stack.back();
        _rc_stack.pop_back();
        if (saved_rc > 0) {
            _cand_rc[cell] = saved_rc;
            _cand_set.insert(cell);
        }
    }

    // ────────────────────────────────────────────────────────────────
    int _make_turn(const OgTurn& turn, OgUndoStep steps[2]) {
        int q1 = og_pack_q(turn.first),  r1 = og_pack_r(turn.first);
        int q2 = og_pack_q(turn.second), r2 = og_pack_r(turn.second);

        steps[0] = {turn.first, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
        _make(q1, r1);
        if (_game_over) return 1;

        steps[1] = {turn.second, {_cur_player, _moves_left, _winner, _game_over}, _cur_player};
        _make(q2, r2);
        return 2;
    }

    void _undo_turn(const OgUndoStep steps[], int n) {
        for (int i = n - 1; i >= 0; i--)
            _undo(og_pack_q(steps[i].cell), og_pack_r(steps[i].cell),
                  steps[i].state, steps[i].player);
    }

    // ────────────────────────────────────────────────────────────────
    double _move_delta(int q, int r, bool is_a) const {
        int8_t cell_val = is_a ? _cell_a : _cell_b;
        const double* pv = _pv.data();
        double delta = 0.0;
        for (const auto& eo : _eval_offsets) {
            int64_t wkey8 = og_pack_wkey(3 + eo.d_idx, q - eo.oq, r - eo.or_);
            int old_pi = 0;
            auto it = _wp.find(wkey8);
            if (it != _wp.end()) old_pi = it->second;
            int new_pi = old_pi + cell_val * _pow3[eo.k];
            delta += pv[new_pi] - pv[old_pi];
        }
        return delta;
    }

    // ────────────────────────────────────────────────────────────────
    std::pair<bool, OgTurn> _find_instant_win(int8_t player) const {
        int p_idx = (player == P_A) ? 0 : 1;
        const auto& hot = (player == P_A) ? _hot_a : _hot_b;

        for (int64_t wkey : hot) {
            auto wit = _wc.find(wkey);
            if (wit == _wc.end()) continue;
            int my_count  = (p_idx == 0) ? wit->second.first : wit->second.second;
            int opp_count = (p_idx == 0) ? wit->second.second : wit->second.first;

            if (my_count >= OG_WIN_LENGTH - 2 && opp_count == 0) {
                int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
                int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - OG_WKEY_BIAS;
                int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - OG_WKEY_BIAS;
                int dq = OG_DIR_Q[d_idx], dr = OG_DIR_R[d_idx];

                OgCoord cells[OG_WIN_LENGTH];
                int n = 0;
                for (int j = 0; j < OG_WIN_LENGTH; j++) {
                    OgCoord c = og_pack(sq + j * dq, sr + j * dr);
                    if (!_board.count(c))
                        cells[n++] = c;
                }
                if (n == 1) {
                    OgCoord other = cells[0];
                    for (OgCoord c : _cand_set)
                        if (c != cells[0]) { other = c; break; }
                    return {true, {og_coord_min(cells[0], other),
                                   og_coord_max(cells[0], other)}};
                }
                if (n == 2) {
                    return {true, {og_coord_min(cells[0], cells[1]),
                                   og_coord_max(cells[0], cells[1])}};
                }
            }
        }
        return {false, {}};
    }

    og_flat_set<OgCoord> _find_threat_cells(int8_t player) const {
        og_flat_set<OgCoord> threats;
        int p_idx = (player == P_A) ? 0 : 1;
        const auto& hot = (player == P_A) ? _hot_a : _hot_b;

        for (int64_t wkey : hot) {
            auto wit = _wc.find(wkey);
            if (wit == _wc.end()) continue;
            int opp_count = (p_idx == 0) ? wit->second.second : wit->second.first;
            if (opp_count != 0) continue;

            int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
            int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - OG_WKEY_BIAS;
            int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - OG_WKEY_BIAS;
            int dq = OG_DIR_Q[d_idx], dr = OG_DIR_R[d_idx];

            for (int j = 0; j < OG_WIN_LENGTH; j++) {
                OgCoord c = og_pack(sq + j * dq, sr + j * dr);
                if (!_board.count(c))
                    threats.insert(c);
            }
        }
        return threats;
    }

    std::vector<OgTurn> _filter_turns_by_threats(const std::vector<OgTurn>& turns) const {
        int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
        int p_idx = (opponent == P_A) ? 0 : 1;
        const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;

        std::vector<og_flat_set<OgCoord>> must_hit;
        for (int64_t wkey : hot) {
            auto wit = _wc.find(wkey);
            if (wit == _wc.end()) continue;
            int my_count  = (p_idx == 0) ? wit->second.first  : wit->second.second;
            int opp_count = (p_idx == 0) ? wit->second.second : wit->second.first;
            if (my_count < OG_WIN_LENGTH - 2 || opp_count != 0) continue;

            int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
            int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - OG_WKEY_BIAS;
            int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - OG_WKEY_BIAS;
            int dq = OG_DIR_Q[d_idx], dr = OG_DIR_R[d_idx];

            og_flat_set<OgCoord> empties;
            for (int j = 0; j < OG_WIN_LENGTH; j++) {
                OgCoord c = og_pack(sq + j * dq, sr + j * dr);
                if (!_board.count(c))
                    empties.insert(c);
            }
            must_hit.push_back(std::move(empties));
        }
        if (must_hit.empty()) return turns;

        std::vector<OgTurn> out;
        out.reserve(turns.size());
        for (const auto& t : turns) {
            bool ok = true;
            for (const auto& w : must_hit) {
                if (!w.count(t.first) && !w.count(t.second)) {
                    ok = false; break;
                }
            }
            if (ok) out.push_back(t);
        }
        return out;
    }

    // ────────────────────────────────────────────────────────────────
    std::vector<OgTurn> _generate_turns() {
        auto [found, wt] = _find_instant_win(_cur_player);
        if (found) return {wt};

        std::vector<OgCoord> cands(_cand_set.begin(), _cand_set.end());
        if (cands.size() < 2) {
            if (!cands.empty()) return {{cands[0], cands[0]}};
            return {};
        }

        bool is_a = (_cur_player == P_A);
        bool maximizing = (_cur_player == _player);

        std::vector<std::pair<double, OgCoord>> scored;
        scored.reserve(cands.size());
        for (OgCoord c : cands)
            scored.push_back({_move_delta(og_pack_q(c), og_pack_r(c), is_a), c});
        std::sort(scored.begin(), scored.end(), [maximizing](const auto& a, const auto& b) {
            if (a.first != b.first)
                return maximizing ? (a.first > b.first) : (a.first < b.first);
            return a.second < b.second;
        });

        cands.clear();
        int cap = std::min(static_cast<int>(scored.size()), OG_ROOT_CANDIDATE_CAP);
        for (int i = 0; i < cap; i++)
            cands.push_back(scored[i].second);

        // Colony candidate
        if (!_board.empty()) {
            int64_t sq = 0, sr = 0;
            for (const auto& kv : _board) { sq += og_pack_q(kv.first); sr += og_pack_r(kv.first); }
            int cq = static_cast<int>(sq / static_cast<int64_t>(_board.size()));
            int cr = static_cast<int>(sr / static_cast<int64_t>(_board.size()));
            int max_r = 0;
            for (const auto& kv : _board) {
                int d = og_hex_distance(og_pack_q(kv.first) - cq, og_pack_r(kv.first) - cr);
                if (d > max_r) max_r = d;
            }
            int cd = max_r + 3;
            std::uniform_int_distribution<int> dist(0, 5);
            int di = dist(_rng);
            OgCoord colony = og_pack(cq + OG_COLONY_DQ[di] * cd, cr + OG_COLONY_DR[di] * cd);
            if (!_board.count(colony))
                cands.push_back(colony);
        }

        int n = static_cast<int>(cands.size());
        std::vector<OgTurn> turns;
        turns.reserve(n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                turns.push_back({cands[i], cands[j]});

        return _filter_turns_by_threats(turns);
    }

    std::vector<OgTurn> _generate_threat_turns(
            const og_flat_set<OgCoord>& my_threats,
            const og_flat_set<OgCoord>& opp_threats) {
        auto [found, wt] = _find_instant_win(_cur_player);
        if (found) return {wt};

        bool is_a = (_cur_player == P_A);
        bool maximizing = (_cur_player == _player);
        double sign = maximizing ? 1.0 : -1.0;

        std::vector<OgCoord> opp_cells, my_cells;
        for (OgCoord c : opp_threats) if (_cand_set.count(c)) opp_cells.push_back(c);
        for (OgCoord c : my_threats)  if (_cand_set.count(c)) my_cells.push_back(c);

        std::vector<OgCoord>* primary = nullptr;
        if (!opp_cells.empty())     primary = &opp_cells;
        else if (!my_cells.empty()) primary = &my_cells;
        else return {};

        if (primary->size() >= 2) {
            int n = static_cast<int>(primary->size());
            std::vector<OgTurn> pairs;
            pairs.reserve(n * (n - 1) / 2);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    pairs.push_back({(*primary)[i], (*primary)[j]});
            std::sort(pairs.begin(), pairs.end(),
                [&](const OgTurn& a, const OgTurn& b) {
                    double da = _move_delta(og_pack_q(a.first), og_pack_r(a.first), is_a)
                              + _move_delta(og_pack_q(a.second), og_pack_r(a.second), is_a);
                    double db = _move_delta(og_pack_q(b.first), og_pack_r(b.first), is_a)
                              + _move_delta(og_pack_q(b.second), og_pack_r(b.second), is_a);
                    return maximizing ? (da > db) : (da < db);
                });
            return pairs;
        }

        OgCoord tc = (*primary)[0];
        OgCoord best_comp = tc;
        double best_d = -OG_INF_SCORE;
        for (OgCoord c : _cand_set) {
            if (c != tc) {
                double d = _move_delta(og_pack_q(c), og_pack_r(c), is_a) * sign;
                if (d > best_d) { best_d = d; best_comp = c; }
            }
        }
        if (best_comp == tc) return {};
        return {{og_coord_min(tc, best_comp), og_coord_max(tc, best_comp)}};
    }

    // ────────────────────────────────────────────────────────────────
    double _quiescence(double alpha, double beta, int qdepth) {
        _check_time();

        if (_game_over) {
            if (_winner == _player)    return  OG_WIN_SCORE;
            if (_winner != P_NONE)     return -OG_WIN_SCORE;
            return 0.0;
        }

        auto [found, wt] = _find_instant_win(_cur_player);
        if (found) {
            OgUndoStep steps[2];
            int n = _make_turn(wt, steps);
            double sc = (_winner == _player) ? OG_WIN_SCORE : -OG_WIN_SCORE;
            _undo_turn(steps, n);
            return sc;
        }

        double stand_pat = _eval_score;
        int8_t current  = _cur_player;
        int8_t opponent = (current == P_A) ? P_B : P_A;
        auto my_threats  = _find_threat_cells(current);
        auto opp_threats = _find_threat_cells(opponent);

        if ((my_threats.empty() && opp_threats.empty()) || qdepth <= 0)
            return stand_pat;

        bool maximizing = (current == _player);
        if (maximizing) {
            if (stand_pat >= beta) return stand_pat;
            alpha = std::max(alpha, stand_pat);
        } else {
            if (stand_pat <= alpha) return stand_pat;
            beta = std::min(beta, stand_pat);
        }

        auto threat_turns = _generate_threat_turns(my_threats, opp_threats);
        if (threat_turns.empty()) return stand_pat;

        double value = stand_pat;
        if (maximizing) {
            for (const auto& turn : threat_turns) {
                OgUndoStep steps[2];
                int nm = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? OG_WIN_SCORE : -OG_WIN_SCORE)
                    : _quiescence(alpha, beta, qdepth - 1);
                _undo_turn(steps, nm);
                if (cv > value) value = cv;
                alpha = std::max(alpha, value);
                if (alpha >= beta) break;
            }
        } else {
            for (const auto& turn : threat_turns) {
                OgUndoStep steps[2];
                int nm = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? OG_WIN_SCORE : -OG_WIN_SCORE)
                    : _quiescence(alpha, beta, qdepth - 1);
                _undo_turn(steps, nm);
                if (cv < value) value = cv;
                beta = std::min(beta, value);
                if (alpha >= beta) break;
            }
        }
        return value;
    }

    // ────────────────────────────────────────────────────────────────
    std::pair<OgTurn, og_flat_map<OgTurn, double, OgTurnHash>>
    _search_root(std::vector<OgTurn>& turns, int depth) {
        bool maximizing = (_cur_player == _player);
        OgTurn best = turns[0];
        double alpha = -OG_INF_SCORE, beta = OG_INF_SCORE;

        og_flat_map<OgTurn, double, OgTurnHash> scores;
        scores.reserve(turns.size());

        for (const auto& turn : turns) {
            _check_time();
            OgUndoStep steps[2];
            int n = _make_turn(turn, steps);
            double sc;
            if (_game_over)
                sc = (_winner == _player) ? OG_WIN_SCORE : -OG_WIN_SCORE;
            else
                sc = _minimax(depth - 1, alpha, beta);
            _undo_turn(steps, n);
            scores[turn] = sc;

            if (maximizing && sc > alpha)  { alpha = sc; best = turn; }
            if (!maximizing && sc < beta)  { beta  = sc; best = turn; }
        }

        double best_sc = maximizing ? alpha : beta;
        _tt[_tt_key()] = {depth, best_sc, OG_TT_EXACT, best, true};
        return {best, std::move(scores)};
    }

    // ────────────────────────────────────────────────────────────────
    double _minimax(int depth, double alpha, double beta) {
        _check_time();

        if (_game_over) {
            if (_winner == _player)    return  OG_WIN_SCORE;
            if (_winner != P_NONE)     return -OG_WIN_SCORE;
            return 0.0;
        }

        uint64_t ttk = _tt_key();
        OgTurn tt_move{};
        bool has_tt_move = false;

        auto tt_it = _tt.find(ttk);
        if (tt_it != _tt.end()) {
            const auto& e = tt_it->second;
            has_tt_move = e.has_move;
            tt_move     = e.move;
            if (e.depth >= depth) {
                if (e.flag == OG_TT_EXACT) return e.score;
                if (e.flag == OG_TT_LOWER) alpha = std::max(alpha, e.score);
                if (e.flag == OG_TT_UPPER) beta  = std::min(beta,  e.score);
                if (alpha >= beta) return e.score;
            }
        }

        if (depth == 0) {
            double sc = _quiescence(alpha, beta, OG_MAX_QDEPTH);
            _tt[ttk] = {0, sc, OG_TT_EXACT, {}, false};
            return sc;
        }

        {
            auto [found, wt] = _find_instant_win(_cur_player);
            if (found) {
                OgUndoStep steps[2];
                int n = _make_turn(wt, steps);
                double sc = (_winner == _player) ? OG_WIN_SCORE : -OG_WIN_SCORE;
                _undo_turn(steps, n);
                _tt[ttk] = {depth, sc, OG_TT_EXACT, wt, true};
                return sc;
            }
        }

        int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
        {
            auto [opp_found, opp_wt] = _find_instant_win(opponent);
            if (opp_found) {
                int p_idx = (opponent == P_A) ? 0 : 1;
                const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;
                std::vector<og_flat_set<OgCoord>> must_hit;
                for (int64_t wkey : hot) {
                    auto wit = _wc.find(wkey);
                    if (wit == _wc.end()) continue;
                    int mc = (p_idx == 0) ? wit->second.first  : wit->second.second;
                    int oc = (p_idx == 0) ? wit->second.second : wit->second.first;
                    if (mc < OG_WIN_LENGTH - 2 || oc != 0) continue;

                    int d_idx = static_cast<int>(static_cast<uint8_t>(wkey >> 56));
                    int sq = static_cast<int>((static_cast<uint64_t>(wkey) >> 28) & 0x0FFFFFFFu) - OG_WKEY_BIAS;
                    int sr = static_cast<int>( static_cast<uint64_t>(wkey)        & 0x0FFFFFFFu) - OG_WKEY_BIAS;
                    int dq = OG_DIR_Q[d_idx], dr = OG_DIR_R[d_idx];
                    og_flat_set<OgCoord> empties;
                    for (int j = 0; j < OG_WIN_LENGTH; j++) {
                        OgCoord c = og_pack(sq + j * dq, sr + j * dr);
                        if (!_board.count(c)) empties.insert(c);
                    }
                    must_hit.push_back(std::move(empties));
                }
                if (must_hit.size() > 1) {
                    og_flat_set<OgCoord> all_cells;
                    for (const auto& s : must_hit) all_cells.insert(s.begin(), s.end());
                    bool can_block = false;
                    for (OgCoord c1 : all_cells) {
                        for (OgCoord c2 : all_cells) {
                            bool ok = true;
                            for (const auto& w : must_hit)
                                if (!w.count(c1) && !w.count(c2)) { ok = false; break; }
                            if (ok) { can_block = true; break; }
                        }
                        if (can_block) break;
                    }
                    if (!can_block) {
                        double sc = (opponent != _player) ? -OG_WIN_SCORE : OG_WIN_SCORE;
                        _tt[ttk] = {depth, sc, OG_TT_EXACT, {}, false};
                        return sc;
                    }
                }
            }
        }

        double orig_alpha = alpha, orig_beta = beta;
        bool maximizing = (_cur_player == _player);

        std::vector<OgTurn> turns;
        {
            std::vector<OgCoord> cands(_cand_set.begin(), _cand_set.end());
            if (cands.size() < 2) {
                if (cands.empty()) {
                    double sc = _eval_score;
                    _tt[ttk] = {depth, sc, OG_TT_EXACT, {}, false};
                    return sc;
                }
                turns = {{cands[0], cands[0]}};
            } else {
                bool is_a = (_cur_player == P_A);
                double dsign = maximizing ? OG_DELTA_WEIGHT : -OG_DELTA_WEIGHT;

                std::vector<std::pair<double, OgCoord>> scored;
                scored.reserve(cands.size());
                for (OgCoord c : cands) {
                    double h = 0;
                    auto hi = _history.find(c);
                    if (hi != _history.end()) h = static_cast<double>(hi->second);
                    scored.push_back({h + _move_delta(og_pack_q(c), og_pack_r(c), is_a) * dsign, c});
                }
                std::sort(scored.begin(), scored.end(),
                    [](const auto& a, const auto& b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                    });

                cands.clear();
                int cap = std::min(static_cast<int>(scored.size()), OG_CANDIDATE_CAP);
                for (int i = 0; i < cap; i++) cands.push_back(scored[i].second);

                int n = static_cast<int>(cands.size());
                turns.reserve(n * (n - 1) / 2);
                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++)
                        turns.push_back({cands[i], cands[j]});
                turns = _filter_turns_by_threats(turns);
            }
        }

        if (turns.empty()) {
            double sc = _eval_score;
            _tt[ttk] = {depth, sc, OG_TT_EXACT, {}, false};
            return sc;
        }

        if (has_tt_move) {
            for (size_t i = 0; i < turns.size(); i++)
                if (turns[i] == tt_move) { std::swap(turns[0], turns[i]); break; }
        }

        OgTurn best_move{};
        double value;

        if (maximizing) {
            value = -OG_INF_SCORE;
            for (const auto& turn : turns) {
                OgUndoStep steps[2];
                int n = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? OG_WIN_SCORE : -OG_WIN_SCORE)
                    : _minimax(depth - 1, alpha, beta);
                _undo_turn(steps, n);
                if (cv > value) { value = cv; best_move = turn; }
                alpha = std::max(alpha, value);
                if (alpha >= beta) {
                    _history[turn.first]  += depth * depth;
                    _history[turn.second] += depth * depth;
                    break;
                }
            }
        } else {
            value = OG_INF_SCORE;
            for (const auto& turn : turns) {
                OgUndoStep steps[2];
                int n = _make_turn(turn, steps);
                double cv = _game_over
                    ? ((_winner == _player) ? OG_WIN_SCORE : -OG_WIN_SCORE)
                    : _minimax(depth - 1, alpha, beta);
                _undo_turn(steps, n);
                if (cv < value) { value = cv; best_move = turn; }
                beta = std::min(beta, value);
                if (alpha >= beta) {
                    _history[turn.first]  += depth * depth;
                    _history[turn.second] += depth * depth;
                    break;
                }
            }
        }

        int8_t flag;
        if      (value <= orig_alpha) flag = OG_TT_UPPER;
        else if (value >= orig_beta)  flag = OG_TT_LOWER;
        else                          flag = OG_TT_EXACT;
        _tt[ttk] = {depth, value, flag, best_move, true};
        return value;
    }
};

} // namespace og
