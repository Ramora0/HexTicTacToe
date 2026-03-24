/*
 * ai_cpp_og.cpp -- Thin pybind11 wrapper around engine_og.h (og variant).
 *
 * Build:  python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine_og.h"

namespace py = pybind11;

// ── Helper: load patterns via Python ai module ──
static void og_load_patterns_from_python(og::MinimaxBot& engine, py::object pattern_path) {
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
static GameState og_extract_game_state(py::object game) {
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
struct PyOgMinimaxBot {
    og::MinimaxBot engine;

    PyOgMinimaxBot() { og_load_patterns_from_python(engine, py::none()); }

    PyOgMinimaxBot(double tl, py::object pattern_path = py::none())
        : engine(tl)
    {
        og_load_patterns_from_python(engine, pattern_path);
    }

    py::list get_move(py::object game) {
        auto gs = og_extract_game_state(game);
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

    py::tuple getstate() const {
        auto es = engine.get_state();
        py::bytes pv_bytes(reinterpret_cast<const char*>(es.pv.data()),
                           es.pv.size() * sizeof(double));
        return py::make_tuple(es.time_limit, pv_bytes,
                              static_cast<int>(es.pv.size()),
                              es.eval_length, es.pattern_path_str);
    }

    void setstate(py::tuple t) {
        EngineState es;
        es.time_limit       = t[0].cast<double>();
        auto pv_str         = t[1].cast<std::string>();
        int  pv_size        = t[2].cast<int>();
        es.eval_length      = t[3].cast<int>();
        es.pattern_path_str = t[4].cast<std::string>();
        es.pv.resize(pv_size);
        std::memcpy(es.pv.data(), pv_str.data(), pv_size * sizeof(double));
        engine.set_state(es);
    }
};

// ═══════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(ai_cpp_og, m) {
    m.doc() = "C++ port of ai.py MinimaxBot (baseline)";

    py::class_<PyOgMinimaxBot>(m, "MinimaxBot")
        .def(py::init<double, py::object>(),
             py::arg("time_limit") = 0.05,
             py::arg("pattern_path") = py::none())
        .def("get_move", &PyOgMinimaxBot::get_move, py::arg("game"))
        .def_property("pair_moves",
            [](PyOgMinimaxBot& b) { return b.engine.pair_moves; },
            [](PyOgMinimaxBot& b, bool v) { b.engine.pair_moves = v; })
        .def_property("time_limit",
            [](PyOgMinimaxBot& b) { return b.engine.time_limit; },
            [](PyOgMinimaxBot& b, double v) { b.engine.time_limit = v; })
        .def_property("last_depth",
            [](PyOgMinimaxBot& b) { return b.engine.last_depth; },
            [](PyOgMinimaxBot& b, int v) { b.engine.last_depth = v; })
        .def_property("_nodes",
            [](PyOgMinimaxBot& b) { return b.engine._nodes; },
            [](PyOgMinimaxBot& b, int v) { b.engine._nodes = v; })
        .def_property("last_score",
            [](PyOgMinimaxBot& b) { return b.engine.last_score; },
            [](PyOgMinimaxBot& b, double v) { b.engine.last_score = v; })
        .def_property("last_ebf",
            [](PyOgMinimaxBot& b) { return b.engine.last_ebf; },
            [](PyOgMinimaxBot& b, double v) { b.engine.last_ebf = v; })
        .def_property("max_depth",
            [](PyOgMinimaxBot& b) { return b.engine.max_depth; },
            [](PyOgMinimaxBot& b, int v) { b.engine.max_depth = v; })
        .def("__str__", [](const PyOgMinimaxBot&) { return std::string("ai_cpp_og"); })
        .def(py::pickle(
            [](const PyOgMinimaxBot& bot) { return bot.getstate(); },
            [](py::tuple t) {
                PyOgMinimaxBot bot;
                bot.setstate(t);
                return bot;
            }
        ));
}
