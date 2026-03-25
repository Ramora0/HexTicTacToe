/*
 * engine_og.h -- Pure C++ minimax engine (flat-array variant, namespace og).
 *
 * No pybind11, no Emscripten -- just the search engine.
 *
 * Board and window data stored in fixed 140x140 flat arrays for
 * cache-friendly O(1) access.  TT and history remain as hash maps.
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

// ── Alias flat hash containers (still used for TT + history) ──
template <typename K, typename V, typename H = ankerl::unordered_dense::hash<K>>
using og_flat_map = ankerl::unordered_dense::map<K, V, H>;

template <typename K, typename H = ankerl::unordered_dense::hash<K>>
using og_flat_set = ankerl::unordered_dense::set<K, H>;

// ═══════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════
static constexpr int    OG_CANDIDATE_CAP      = 15;
static constexpr int    OG_ROOT_CANDIDATE_CAP = 13;
static constexpr int    OG_NEIGHBOR_DIST      = 2;
static constexpr double OG_DELTA_WEIGHT       = 15;
static constexpr int    OG_MAX_QDEPTH         = 16;
static constexpr int    OG_WIN_LENGTH         = 6;
static constexpr double OG_WIN_SCORE          = 100000000.0;
static constexpr double OG_INF_SCORE          = std::numeric_limits<double>::infinity();

// Array dimensions -- covers coordinates [-70, 69] with padding for
// windows (+/-5) and neighbor candidates (+/-2).
static constexpr int OG_ARR = 140;
static constexpr int OG_OFF = 70;

// TT flags
static constexpr int8_t OG_TT_EXACT = 0;
static constexpr int8_t OG_TT_LOWER = 1;
static constexpr int8_t OG_TT_UPPER = 2;

// ═══════════════════════════════════════════════════════════════════════
//  Coordinate packing  (still used for Coord values in vectors/turns)
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
//  Helper structs for array-backed sets
// ═══════════════════════════════════════════════════════════════════════
struct OgHotEntry { int d, qi, ri; };

struct OgHotSet {
    bool bits[3][OG_ARR][OG_ARR];
    std::vector<OgHotEntry> vec;

    void clear() { std::memset(bits, 0, sizeof(bits)); vec.clear(); }

    void insert(int d, int qi, int ri) {
        if (!bits[d][qi][ri]) {
            bits[d][qi][ri] = true;
            vec.push_back({d, qi, ri});
        }
    }

    void erase(int d, int qi, int ri) {
        if (bits[d][qi][ri]) {
            bits[d][qi][ri] = false;
            for (size_t i = 0; i < vec.size(); i++) {
                if (vec[i].d == d && vec[i].qi == qi && vec[i].ri == ri) {
                    vec[i] = vec.back(); vec.pop_back(); break;
                }
            }
        }
    }
};

struct OgCandSet {
    bool bits[OG_ARR][OG_ARR];
    std::vector<OgCoord> vec;

    void clear() { std::memset(bits, 0, sizeof(bits)); vec.clear(); }
    bool empty() const { return vec.empty(); }
    size_t size() const { return vec.size(); }
    bool count(OgCoord c) const { return bits[og_pack_q(c) + OG_OFF][og_pack_r(c) + OG_OFF]; }

    void insert(OgCoord c) {
        int qi = og_pack_q(c) + OG_OFF, ri = og_pack_r(c) + OG_OFF;
        if (!bits[qi][ri]) { bits[qi][ri] = true; vec.push_back(c); }
    }

    void erase(OgCoord c) {
        int qi = og_pack_q(c) + OG_OFF, ri = og_pack_r(c) + OG_OFF;
        if (bits[qi][ri]) {
            bits[qi][ri] = false;
            for (size_t i = 0; i < vec.size(); i++) {
                if (vec[i] == c) { vec[i] = vec.back(); vec.pop_back(); break; }
            }
        }
    }

    auto begin() const { return vec.begin(); }
    auto end()   const { return vec.end(); }
};

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

// Zobrist tables -- flat arrays, deterministic per (q, r) via splitmix64.
static uint64_t og_g_zobrist_a[OG_ARR][OG_ARR];
static uint64_t og_g_zobrist_b[OG_ARR][OG_ARR];

static inline uint64_t og_splitmix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

static inline uint64_t og_get_zobrist(int q, int r, int8_t player) {
    return (player == P_A) ? og_g_zobrist_a[q + OG_OFF][r + OG_OFF]
                           : og_g_zobrist_b[q + OG_OFF][r + OG_OFF];
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
    for (int i = 0; i < OG_ARR; i++)
        for (int j = 0; j < OG_ARR; j++) {
            int q = i - OG_OFF, r = j - OG_OFF;
            uint64_t base = static_cast<uint64_t>(static_cast<uint32_t>(q)) << 32
                          | static_cast<uint64_t>(static_cast<uint32_t>(r));
            og_g_zobrist_a[i][j] = og_splitmix64(base ^ 0xa02bdbf7bb3c0195ULL);
            og_g_zobrist_b[i][j] = og_splitmix64(base ^ 0x3f84d5b5b5470917ULL);
        }
    og_g_tables_ready = true;
}

// ═══════════════════════════════════════════════════════════════════════
//  MinimaxBot  (namespace og -- flat-array variant)
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

    // ── Pattern loading (call from wrapper after construction) ──
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

        // ── Clear arrays ──
        std::memset(_board, 0, sizeof(_board));
        std::memset(_wc, 0, sizeof(_wc));
        std::memset(_wp, 0, sizeof(_wp));
        std::memset(_cand_rc, 0, sizeof(_cand_rc));
        _board_cells.clear();
        _hot_a.clear();
        _hot_b.clear();
        _cand_set.clear();
        _rc_stack.clear();

        // ── Populate board from GameState ──
        for (const auto& cell : gs.cells) {
            _board[cell.q + OG_OFF][cell.r + OG_OFF] = cell.player;
            _board_cells.push_back(og_pack(cell.q, cell.r));
        }

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
        for (OgCoord c : _board_cells)
            _hash ^= og_get_zobrist(og_pack_q(c), og_pack_r(c),
                                  _board[og_pack_q(c) + OG_OFF][og_pack_r(c) + OG_OFF]);

        // ── Cell value mapping ──
        if (_player == P_A) { _cell_a = 1; _cell_b = 2; }
        else                { _cell_a = 2; _cell_b = 1; }

        // ── Init 6-cell windows ──
        for (OgCoord c : _board_cells) {
            int bq = og_pack_q(c), br = og_pack_r(c);
            int bqi = bq + OG_OFF, bri = br + OG_OFF;
            for (const auto& wo : og_g_win_offsets) {
                int sqi = bqi - wo.oq, sri = bri - wo.or_;
                auto& counts = _wc[wo.d_idx][sqi][sri];
                if (counts.first != 0 || counts.second != 0) continue;
                int d = wo.d_idx;
                int sq = bq - wo.oq, sr = br - wo.or_;
                int ac = 0, bc = 0;
                for (int j = 0; j < OG_WIN_LENGTH; j++) {
                    int8_t v = _board[sq + j * OG_DIR_Q[d] + OG_OFF][sr + j * OG_DIR_R[d] + OG_OFF];
                    if (v == P_A) ac++;
                    else if (v == P_B) bc++;
                }
                if (ac || bc) {
                    counts = {static_cast<int8_t>(ac), static_cast<int8_t>(bc)};
                    if (ac >= 4) _hot_a.insert(wo.d_idx, sqi, sri);
                    if (bc >= 4) _hot_b.insert(wo.d_idx, sqi, sri);
                }
            }
        }

        // ── Init N-cell eval windows ──
        _eval_score = 0.0;
        {
            const double* pv = _pv.data();
            for (OgCoord c : _board_cells) {
                int bq = og_pack_q(c), br = og_pack_r(c);
                int bqi = bq + OG_OFF, bri = br + OG_OFF;
                for (const auto& eo : _eval_offsets) {
                    int sqi = bqi - eo.oq, sri = bri - eo.or_;
                    int& slot = _wp[eo.d_idx][sqi][sri];
                    if (slot != 0) continue;
                    int sq = bq - eo.oq, sr = br - eo.or_;
                    int d = eo.d_idx;
                    int pi = 0;
                    bool has = false;
                    for (int j = 0; j < _eval_length; j++) {
                        int8_t v = _board[sq + j * OG_DIR_Q[d] + OG_OFF][sr + j * OG_DIR_R[d] + OG_OFF];
                        if (v != 0) {
                            pi += ((v == P_A) ? _cell_a : _cell_b) * _pow3[j];
                            has = true;
                        }
                    }
                    if (has) { slot = pi; _eval_score += pv[pi]; }
                }
            }
        }

        // ── Init candidates ──
        for (OgCoord c : _board_cells) {
            int bq = og_pack_q(c), br = og_pack_r(c);
            for (const auto& nb : og_g_nb_offsets) {
                int nq = bq + nb.dq, nr = br + nb.dr;
                int nqi = nq + OG_OFF, nri = nr + OG_OFF;
                _cand_rc[nqi][nri]++;
                if (_board[nqi][nri] == 0)
                    _cand_set.insert(og_pack(nq, nr));
            }
        }

        if (_cand_set.empty())
            return {0, 0, 0, 0, 1};

        bool maximizing = (_cur_player == _player);
        auto turns = _generate_turns();
        if (turns.empty())
            return {0, 0, 0, 0, 1};

        OgTurn best_move = turns[0];

        // ── Save state for TimeUp rollback ──
        if (!_saved) _saved = std::make_unique<OgSavedArrays>();
        std::memcpy(_saved->board, _board, sizeof(_board));
        std::memcpy(_saved->wc, _wc, sizeof(_wc));
        std::memcpy(_saved->wp, _wp, sizeof(_wp));
        std::memcpy(_saved->cand_rc, _cand_rc, sizeof(_cand_rc));
        std::memcpy(_saved->cand_bits, _cand_set.bits, sizeof(_cand_set.bits));
        _saved->cand_vec = _cand_set.vec;
        std::memcpy(_saved->hot_a_bits, _hot_a.bits, sizeof(_hot_a.bits));
        _saved->hot_a_vec = _hot_a.vec;
        std::memcpy(_saved->hot_b_bits, _hot_b.bits, sizeof(_hot_b.bits));
        _saved->hot_b_vec = _hot_b.vec;
        _saved->board_cells = _board_cells;
        auto saved_st       = OgSavedState{_cur_player, _moves_left, _winner, _game_over};
        int  saved_mc       = _move_count;
        uint64_t saved_hash = _hash;
        double   saved_eval = _eval_score;

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
                std::memcpy(_board, _saved->board, sizeof(_board));
                std::memcpy(_wc, _saved->wc, sizeof(_wc));
                std::memcpy(_wp, _saved->wp, sizeof(_wp));
                std::memcpy(_cand_rc, _saved->cand_rc, sizeof(_cand_rc));
                std::memcpy(_cand_set.bits, _saved->cand_bits, sizeof(_cand_set.bits));
                _cand_set.vec = std::move(_saved->cand_vec);
                std::memcpy(_hot_a.bits, _saved->hot_a_bits, sizeof(_hot_a.bits));
                _hot_a.vec = std::move(_saved->hot_a_vec);
                std::memcpy(_hot_b.bits, _saved->hot_b_bits, sizeof(_hot_b.bits));
                _hot_b.vec = std::move(_saved->hot_b_vec);
                _board_cells = std::move(_saved->board_cells);
                _move_count = saved_mc;
                _cur_player = saved_st.cur_player;
                _moves_left = saved_st.moves_left;
                _winner     = saved_st.winner;
                _game_over  = saved_st.game_over;
                _hash       = saved_hash;
                _eval_score = saved_eval;
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

    // ── Board state (flat arrays) ──
    int8_t _board[OG_ARR][OG_ARR] = {};
    std::vector<OgCoord> _board_cells;

    int8_t _cur_player  = P_A;
    int8_t _moves_left  = 1;
    int8_t _winner      = P_NONE;
    bool   _game_over   = false;
    int    _move_count  = 0;

    // ── 6-cell window counts ──
    std::pair<int8_t,int8_t> _wc[3][OG_ARR][OG_ARR] = {};
    OgHotSet _hot_a, _hot_b;

    // ── N-cell eval window patterns ──
    int _wp[3][OG_ARR][OG_ARR] = {};

    // ── Candidates ──
    int8_t   _cand_rc[OG_ARR][OG_ARR] = {};
    OgCandSet _cand_set;
    std::vector<int> _rc_stack;

    // ── Search state ──
    using Clock = std::chrono::steady_clock;
    Clock::time_point _deadline;
    uint64_t _hash      = 0;
    int8_t   _player    = P_A;
    int8_t   _cell_a    = 1;
    int8_t   _cell_b    = 2;
    double   _eval_score = 0;

    // ── Transposition table & history (hash maps) ──
    og_flat_map<uint64_t, OgTTEntry> _tt;
    og_flat_map<OgCoord, int>        _history;

    // ── RNG ──
    std::mt19937 _rng;

    // ── Saved state for TimeUp rollback ──
    struct OgSavedArrays {
        int8_t board[OG_ARR][OG_ARR];
        std::pair<int8_t,int8_t> wc[3][OG_ARR][OG_ARR];
        int wp[3][OG_ARR][OG_ARR];
        int8_t cand_rc[OG_ARR][OG_ARR];
        bool cand_bits[OG_ARR][OG_ARR];
        std::vector<OgCoord> cand_vec;
        bool hot_a_bits[3][OG_ARR][OG_ARR];
        std::vector<OgHotEntry> hot_a_vec;
        bool hot_b_bits[3][OG_ARR][OG_ARR];
        std::vector<OgHotEntry> hot_b_vec;
        std::vector<OgCoord> board_cells;
    };
    std::unique_ptr<OgSavedArrays> _saved;

    // ────────────────────────────────────────────────────────────────
    //  Pattern table construction
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

    // ────────────────────────────────────────────────────────────────
    //  Time control
    // ────────────────────────────────────────────────────────────────
    inline void _check_time() {
        _nodes++;
        if ((_nodes & 1023) == 0 && Clock::now() >= _deadline)
            throw OgTimeUp{};
    }

    // ────────────────────────────────────────────────────────────────
    //  TT key
    // ────────────────────────────────────────────────────────────────
    inline uint64_t _tt_key() const {
        return _hash ^ (static_cast<uint64_t>(_cur_player) * 0x9e3779b97f4a7c15ULL)
                      ^ (static_cast<uint64_t>(_moves_left) * 0x517cc1b727220a95ULL);
    }

    // ────────────────────────────────────────────────────────────────
    //  Incremental make / undo
    // ────────────────────────────────────────────────────────────────
    void _make(int q, int r) {
        int8_t player = _cur_player;

        // Zobrist
        _hash ^= og_get_zobrist(q, r, player);

        int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;
        int qi = q + OG_OFF, ri = r + OG_OFF;

        // ── 6-cell windows ──
        bool won = false;
        if (player == P_A) {
            for (const auto& wo : og_g_win_offsets) {
                int sqi = qi - wo.oq, sri = ri - wo.or_;
                auto& counts = _wc[wo.d_idx][sqi][sri];
                counts.first++;
                if (counts.first >= 4) _hot_a.insert(wo.d_idx, sqi, sri);
                if (counts.first == OG_WIN_LENGTH && counts.second == 0) won = true;
            }
        } else {
            for (const auto& wo : og_g_win_offsets) {
                int sqi = qi - wo.oq, sri = ri - wo.or_;
                auto& counts = _wc[wo.d_idx][sqi][sri];
                counts.second++;
                if (counts.second >= 4) _hot_b.insert(wo.d_idx, sqi, sri);
                if (counts.second == OG_WIN_LENGTH && counts.first == 0) won = true;
            }
        }

        // ── N-cell eval windows ──
        const double* pv = _pv.data();
        for (const auto& eo : _eval_offsets) {
            int sqi = qi - eo.oq, sri = ri - eo.or_;
            int& slot = _wp[eo.d_idx][sqi][sri];
            int old_pi = slot;
            int new_pi = old_pi + cell_val * _pow3[eo.k];
            _eval_score += pv[new_pi] - pv[old_pi];
            slot = new_pi;
        }

        // ── Candidates ──
        OgCoord cell = og_pack(q, r);
        _cand_set.erase(cell);
        _rc_stack.push_back(_cand_rc[qi][ri]);
        _cand_rc[qi][ri] = 0;

        for (const auto& nb : og_g_nb_offsets) {
            int nq = q + nb.dq, nr = r + nb.dr;
            int nqi = nq + OG_OFF, nri = nr + OG_OFF;
            _cand_rc[nqi][nri]++;
            if (_board[nqi][nri] == 0)
                _cand_set.insert(og_pack(nq, nr));
        }

        // Place stone
        _board[qi][ri] = player;
        _board_cells.push_back(cell);
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
        int qi = q + OG_OFF, ri = r + OG_OFF;

        // Remove stone
        _board[qi][ri] = 0;
        _board_cells.pop_back();
        _move_count--;
        _cur_player = st.cur_player;
        _moves_left = st.moves_left;
        _winner     = st.winner;
        _game_over  = st.game_over;

        // Zobrist
        _hash ^= og_get_zobrist(q, r, player);

        int8_t cell_val = (player == P_A) ? _cell_a : _cell_b;

        // ── 6-cell windows ──
        if (player == P_A) {
            for (const auto& wo : og_g_win_offsets) {
                int sqi = qi - wo.oq, sri = ri - wo.or_;
                auto& counts = _wc[wo.d_idx][sqi][sri];
                counts.first--;
                if (counts.first < 4) _hot_a.erase(wo.d_idx, sqi, sri);
            }
        } else {
            for (const auto& wo : og_g_win_offsets) {
                int sqi = qi - wo.oq, sri = ri - wo.or_;
                auto& counts = _wc[wo.d_idx][sqi][sri];
                counts.second--;
                if (counts.second < 4) _hot_b.erase(wo.d_idx, sqi, sri);
            }
        }

        // ── N-cell eval windows ──
        const double* pv = _pv.data();
        for (const auto& eo : _eval_offsets) {
            int sqi = qi - eo.oq, sri = ri - eo.or_;
            int& slot = _wp[eo.d_idx][sqi][sri];
            int old_pi = slot;
            int new_pi = old_pi - cell_val * _pow3[eo.k];
            _eval_score += pv[new_pi] - pv[old_pi];
            slot = new_pi;
        }

        // ── Candidates ──
        for (const auto& nb : og_g_nb_offsets) {
            int nq = q + nb.dq, nr = r + nb.dr;
            int nqi = nq + OG_OFF, nri = nr + OG_OFF;
            _cand_rc[nqi][nri]--;
            if (_cand_rc[nqi][nri] == 0)
                _cand_set.erase(og_pack(nq, nr));
        }
        int saved_rc = _rc_stack.back();
        _rc_stack.pop_back();
        if (saved_rc > 0) {
            OgCoord cell = og_pack(q, r);
            _cand_rc[qi][ri] = saved_rc;
            _cand_set.insert(cell);
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Turn make / undo
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
    //  Move delta
    // ────────────────────────────────────────────────────────────────
    double _move_delta(int q, int r, bool is_a) const {
        int8_t cell_val = is_a ? _cell_a : _cell_b;
        const double* pv = _pv.data();
        int qi = q + OG_OFF, ri = r + OG_OFF;
        double delta = 0.0;
        for (const auto& eo : _eval_offsets) {
            int old_pi = _wp[eo.d_idx][qi - eo.oq][ri - eo.or_];
            int new_pi = old_pi + cell_val * _pow3[eo.k];
            delta += pv[new_pi] - pv[old_pi];
        }
        return delta;
    }

    // ────────────────────────────────────────────────────────────────
    //  Win / threat detection
    // ────────────────────────────────────────────────────────────────
    std::pair<bool, OgTurn> _find_instant_win(int8_t player) const {
        int p_idx = (player == P_A) ? 0 : 1;
        const auto& hot = (player == P_A) ? _hot_a : _hot_b;

        for (const auto& he : hot.vec) {
            auto& counts = _wc[he.d][he.qi][he.ri];
            int my_count  = (p_idx == 0) ? counts.first : counts.second;
            int opp_count = (p_idx == 0) ? counts.second : counts.first;

            if (my_count >= OG_WIN_LENGTH - 2 && opp_count == 0) {
                int sq = he.qi - OG_OFF, sr = he.ri - OG_OFF;
                int dq = OG_DIR_Q[he.d], dr = OG_DIR_R[he.d];

                OgCoord cells[OG_WIN_LENGTH];
                int n = 0;
                for (int j = 0; j < OG_WIN_LENGTH; j++) {
                    int cq = sq + j * dq, cr = sr + j * dr;
                    if (_board[cq + OG_OFF][cr + OG_OFF] == 0)
                        cells[n++] = og_pack(cq, cr);
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

        for (const auto& he : hot.vec) {
            auto& counts = _wc[he.d][he.qi][he.ri];
            int opp_count = (p_idx == 0) ? counts.second : counts.first;
            if (opp_count != 0) continue;

            int sq = he.qi - OG_OFF, sr = he.ri - OG_OFF;
            int dq = OG_DIR_Q[he.d], dr = OG_DIR_R[he.d];

            for (int j = 0; j < OG_WIN_LENGTH; j++) {
                int cq = sq + j * dq, cr = sr + j * dr;
                if (_board[cq + OG_OFF][cr + OG_OFF] == 0)
                    threats.insert(og_pack(cq, cr));
            }
        }
        return threats;
    }

    std::vector<OgTurn> _filter_turns_by_threats(const std::vector<OgTurn>& turns) const {
        int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
        int p_idx = (opponent == P_A) ? 0 : 1;
        const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;

        std::vector<og_flat_set<OgCoord>> must_hit;
        for (const auto& he : hot.vec) {
            auto& counts = _wc[he.d][he.qi][he.ri];
            int my_count  = (p_idx == 0) ? counts.first  : counts.second;
            int opp_count = (p_idx == 0) ? counts.second : counts.first;
            if (my_count < OG_WIN_LENGTH - 2 || opp_count != 0) continue;

            int sq = he.qi - OG_OFF, sr = he.ri - OG_OFF;
            int dq = OG_DIR_Q[he.d], dr = OG_DIR_R[he.d];

            og_flat_set<OgCoord> empties;
            for (int j = 0; j < OG_WIN_LENGTH; j++) {
                int cq = sq + j * dq, cr = sr + j * dr;
                if (_board[cq + OG_OFF][cr + OG_OFF] == 0)
                    empties.insert(og_pack(cq, cr));
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
    //  Turn generation
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
        if (!_board_cells.empty()) {
            int64_t sq = 0, sr = 0;
            for (OgCoord c : _board_cells) { sq += og_pack_q(c); sr += og_pack_r(c); }
            int cq = static_cast<int>(sq / static_cast<int64_t>(_board_cells.size()));
            int cr = static_cast<int>(sr / static_cast<int64_t>(_board_cells.size()));
            int max_r = 0;
            for (OgCoord c : _board_cells) {
                int d = og_hex_distance(og_pack_q(c) - cq, og_pack_r(c) - cr);
                if (d > max_r) max_r = d;
            }
            int cd = max_r + 3;
            std::uniform_int_distribution<int> dist(0, 5);
            int di = dist(_rng);
            int col_q = cq + OG_COLONY_DQ[di] * cd;
            int col_r = cr + OG_COLONY_DR[di] * cd;
            if (std::abs(col_q) < OG_OFF && std::abs(col_r) < OG_OFF &&
                _board[col_q + OG_OFF][col_r + OG_OFF] == 0)
                cands.push_back(og_pack(col_q, col_r));
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
    //  Quiescence search
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
    //  Root search
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
    //  Minimax
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

        // Instant win for current player
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

        // Opponent instant win -> check if blockable
        int8_t opponent = (_cur_player == P_A) ? P_B : P_A;
        {
            auto [opp_found, opp_wt] = _find_instant_win(opponent);
            if (opp_found) {
                int p_idx = (opponent == P_A) ? 0 : 1;
                const auto& hot = (opponent == P_A) ? _hot_a : _hot_b;
                std::vector<og_flat_set<OgCoord>> must_hit;
                for (const auto& he : hot.vec) {
                    auto& counts = _wc[he.d][he.qi][he.ri];
                    int mc = (p_idx == 0) ? counts.first  : counts.second;
                    int oc = (p_idx == 0) ? counts.second : counts.first;
                    if (mc < OG_WIN_LENGTH - 2 || oc != 0) continue;

                    int sq = he.qi - OG_OFF, sr = he.ri - OG_OFF;
                    int dq = OG_DIR_Q[he.d], dr = OG_DIR_R[he.d];
                    og_flat_set<OgCoord> empties;
                    for (int j = 0; j < OG_WIN_LENGTH; j++) {
                        int cq = sq + j * dq, cr = sr + j * dr;
                        if (_board[cq + OG_OFF][cr + OG_OFF] == 0)
                            empties.insert(og_pack(cq, cr));
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

        // Generate candidates and turns
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

        // TT move ordering
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
