/*
 * engine_wasm.cpp -- Emscripten/Embind wrapper for the opt engine.
 *
 * Build:
 *   em++ cpp/engine_wasm.cpp -O3 -std=c++17 -fexceptions \
 *        -s MODULARIZE=1 -s EXPORT_NAME="HexEngine" \
 *        -s ALLOW_MEMORY_GROWTH=1 --bind \
 *        -Icpp -DNDEBUG -o cpp/hex_engine.js
 *
 * Usage from JS:
 *   const engine = await HexEngine();
 *   const result = engine.getMove(
 *       [0,0, 1,-1, ...],   // movesA: flat [q,r,q,r,...] for player A
 *       [0,1, 2,-1, ...],   // movesB: flat [q,r,q,r,...] for player B
 *       1,                   // currentPlayer: 1=A, 2=B
 *       2,                   // movesLeft: 1 or 2
 *       0.5                  // timeLimit in seconds
 *   );
 *   // result is [q1, r1, q2, r2, depth, nodes, score]
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "engine.h"
#include "pattern_data.h"

// Persistent bot instance (keeps TT/history across calls).
static opt::MinimaxBot g_bot;
static bool g_initialized = false;

static void ensure_init() {
    if (!g_initialized) {
        g_bot.load_patterns(PATTERN_VALUES, PATTERN_COUNT, PATTERN_EVAL_LENGTH);
        g_initialized = true;
    }
}

// Returns [q1, r1, q2, r2, depth, nodes, score]
emscripten::val getMove(emscripten::val movesA, emscripten::val movesB,
                        int currentPlayer, int movesLeft, double timeLimit) {
    ensure_init();

    GameState gs;

    int lenA = movesA["length"].as<int>();
    for (int i = 0; i < lenA; i += 2)
        gs.cells.push_back({movesA[i].as<int>(), movesA[i + 1].as<int>(), P_A});

    int lenB = movesB["length"].as<int>();
    for (int i = 0; i < lenB; i += 2)
        gs.cells.push_back({movesB[i].as<int>(), movesB[i + 1].as<int>(), P_B});

    gs.cur_player = static_cast<int8_t>(currentPlayer);
    gs.moves_left = static_cast<int8_t>(movesLeft);
    gs.move_count = static_cast<int>(gs.cells.size());

    g_bot.time_limit = timeLimit;
    auto mr = g_bot.get_move(gs);

    auto result = emscripten::val::array();
    result.call<void>("push", mr.q1);
    result.call<void>("push", mr.r1);
    result.call<void>("push", mr.q2);
    result.call<void>("push", mr.r2);
    result.call<void>("push", g_bot.last_depth);
    result.call<void>("push", g_bot._nodes);
    result.call<void>("push", g_bot.last_score);
    return result;
}

EMSCRIPTEN_BINDINGS(hex_engine) {
    emscripten::function("getMove", &getMove);
}
