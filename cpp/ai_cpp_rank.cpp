/*
 * ai_cpp_rank.cpp -- pybind11 wrapper for the rank-tracking engine variant.
 *
 * Build:  python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine_rank.h"

namespace py = pybind11;

// ── Helper: load patterns via Python ai module ──
static void load_patterns_from_python(rank::MinimaxBot& engine, py::object pattern_path) {
    py::module_ ai_mod = py::module_::import("ai");
    std::string path;
    if (pattern_path.is_none())
        path = ai_mod.attr("_DEFAULT_PATTERN_PATH").cast<std::string>();
    else
        path = pattern_path.cast<std::string>();

    py::tuple result = ai_mod.attr("_load_pattern_values")(path).cast<py::tuple>();
    py::list  pv_list = result[0].cast<py::list>();
    int       eval_length = result[1].cast<int>();

    std::vector<double> pv(pv_list.size());
    for (size_t i = 0; i < pv_list.size(); i++)
        pv[i] = pv_list[i].cast<double>();

    engine.load_patterns(pv, eval_length, path);
}

// ── Helper: extract GameState from Python game object ──
static GameState extract_game_state(py::object game) {
    py::module_ game_mod = py::module_::import("game");
    py::object  PyA = game_mod.attr("Player").attr("A");

    py::dict py_board = game.attr("board").cast<py::dict>();

    GameState gs;
    gs.cells.reserve(py_board.size());
    for (auto item : py_board) {
        py::tuple key = item.first.cast<py::tuple>();
        int q = key[0].cast<int>(), r = key[1].cast<int>();
        int8_t p = item.second.is(PyA) ? P_A : P_B;
        gs.cells.push_back({q, r, p});
    }

    py::object py_cur = game.attr("current_player");
    gs.cur_player  = py_cur.is(PyA) ? P_A : P_B;
    gs.moves_left  = game.attr("moves_left_in_turn").cast<int8_t>();
    gs.move_count  = game.attr("move_count").cast<int>();
    return gs;
}

// ── Wrapper class ──
struct PyMinimaxBot {
    rank::MinimaxBot engine;

    PyMinimaxBot() { load_patterns_from_python(engine, py::none()); }

    PyMinimaxBot(double tl, py::object pattern_path = py::none())
        : engine(tl)
    {
        load_patterns_from_python(engine, pattern_path);
    }

    py::list get_move(py::object game) {
        auto gs = extract_game_state(game);
        if (gs.cells.empty()) {
            py::list res;
            res.append(py::make_tuple(0, 0));
            return res;
        }

        auto mr = engine.get_move(gs);

        py::list res;
        res.append(py::make_tuple(mr.q1, mr.r1));
        if (mr.num_moves > 1)
            res.append(py::make_tuple(mr.q2, mr.r2));
        return res;
    }
};

// ═══════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(ai_cpp_rank, m) {
    m.doc() = "C++ minimax engine with rank-pair tracking for analysis";

    py::class_<PyMinimaxBot>(m, "MinimaxBot")
        .def(py::init<double, py::object>(),
             py::arg("time_limit") = 0.05,
             py::arg("pattern_path") = py::none())
        .def("get_move", &PyMinimaxBot::get_move, py::arg("game"))
        .def_property("pair_moves",
            [](PyMinimaxBot& b) { return b.engine.pair_moves; },
            [](PyMinimaxBot& b, bool v) { b.engine.pair_moves = v; })
        .def_property("time_limit",
            [](PyMinimaxBot& b) { return b.engine.time_limit; },
            [](PyMinimaxBot& b, double v) { b.engine.time_limit = v; })
        .def_property("track_ranks",
            [](PyMinimaxBot& b) { return b.engine.track_ranks; },
            [](PyMinimaxBot& b, bool v) { b.engine.track_ranks = v; })
        .def("get_rank_data",
            [](PyMinimaxBot& b) { return b.engine.rank_tracker.data; })
        .def("get_scatter_data",
            [](PyMinimaxBot& b) { return b.engine.rank_tracker.scatter; })
        .def("clear_rank_data",
            [](PyMinimaxBot& b) { b.engine.rank_tracker.clear(); })
        .def("__str__", [](const PyMinimaxBot&) { return std::string("ai_cpp_rank"); });
}
