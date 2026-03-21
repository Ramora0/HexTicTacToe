"""Interactive viewer for positions in learned_eval/data/positions.pkl.

Browse positions with arrow keys, see eval scores, current player, and game ID.
Controls:
  Left/Right or A/D: previous/next position
  Up/Down or W/S: jump 100 positions
  Home/End: first/last position
  G: go to position number (type in terminal)
  F: filter by game ID (type in terminal)
  ESC/C: clear filter
  Q: quit
"""

import math
import os
import pickle
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Player

# --- Layout ---
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
MAX_HEX_SIZE = 28
VISIBLE_DIST = 2  # padding around occupied cells

# --- Colors ---
BG_COLOR = (24, 24, 32)
EMPTY_FILL = (48, 48, 58)
GRID_LINE = (72, 72, 85)
PLAYER_A_COLOR = (220, 62, 62)
PLAYER_B_COLOR = (62, 120, 220)
WIN_BORDER = (255, 215, 0)
TEXT_COLOR = (220, 220, 230)
SUBTLE_TEXT = (130, 130, 150)
HIGHLIGHT_COLOR = (255, 200, 60)
PANEL_BG = (32, 32, 44)


def _hex_distance(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


def hex_corners(cx, cy, size):
    return [
        (cx + size * math.cos(math.radians(60 * i + 30)),
         cy + size * math.sin(math.radians(60 * i + 30)))
        for i in range(6)
    ]


def hex_to_pixel(q, r, size, ox, oy):
    x = size * math.sqrt(3) * (q + r * 0.5) + ox
    y = size * 1.5 * r + oy
    return x, y


def pixel_to_hex(mx, my, size, ox, oy):
    px = (mx - ox) / size
    py = (my - oy) / size
    r_frac = 2.0 / 3 * py
    q_frac = px / math.sqrt(3) - r_frac / 2
    s_frac = -q_frac - r_frac
    rq, rr, rs = round(q_frac), round(r_frac), round(s_frac)
    dq = abs(rq - q_frac)
    dr = abs(rr - r_frac)
    ds = abs(rs - s_frac)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return int(rq), int(rr)


def get_visible_cells(board):
    if not board:
        return set()
    cells = set()
    offsets = [
        (dq, dr)
        for dq in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
        for dr in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
        if _hex_distance(dq, dr) <= VISIBLE_DIST
    ]
    for q, r in board:
        for oq, or_ in offsets:
            cells.add((q + oq, r + or_))
    return cells


def compute_view(visible_cells):
    if not visible_cells:
        return MAX_HEX_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2

    S3 = math.sqrt(3)
    uxs = [S3 * (q + r * 0.5) for q, r in visible_cells]
    uys = [1.5 * r for q, r in visible_cells]

    min_ux, max_ux = min(uxs), max(uxs)
    min_uy, max_uy = min(uys), max(uys)

    ext_x = max_ux - min_ux + S3
    ext_y = max_uy - min_uy + 2

    avail_x = WINDOW_WIDTH - 60
    avail_y = WINDOW_HEIGHT - 220  # more room for info panel

    size = MAX_HEX_SIZE
    if ext_x > 0:
        size = min(size, avail_x / ext_x)
    if ext_y > 0:
        size = min(size, avail_y / ext_y)
    size = max(6.0, size)

    center_ux = (min_ux + max_ux) / 2
    center_uy = (min_uy + max_uy) / 2
    ox = WINDOW_WIDTH / 2 - center_ux * size
    oy = (WINDOW_HEIGHT - 120) / 2 - center_uy * size + 40

    return size, ox, oy


def draw_position(screen, board, current_player, eval_score, game_id,
                  idx, total, visible_cells, hex_size, ox, oy, fonts,
                  hover_hex, filter_gid):
    font_big, font_med, font_sm = fonts
    screen.fill(BG_COLOR)

    # Draw hex cells
    for (q, r) in visible_cells:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)

        player = board.get((q, r))
        if player == Player.A:
            fill = PLAYER_A_COLOR
        elif player == Player.B:
            fill = PLAYER_B_COLOR
        else:
            fill = EMPTY_FILL

        pygame.draw.polygon(screen, fill, corners)
        pygame.draw.polygon(screen, GRID_LINE, corners, 2)

    # Hover highlight
    if hover_hex and hover_hex in visible_cells:
        cx, cy = hex_to_pixel(*hover_hex, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)
        pygame.draw.polygon(screen, HIGHLIGHT_COLOR, corners, 2)

    # --- Info panel at bottom ---
    panel_y = WINDOW_HEIGHT - 160
    pygame.draw.rect(screen, PANEL_BG, (0, panel_y, WINDOW_WIDTH, 160))
    pygame.draw.line(screen, GRID_LINE, (0, panel_y), (WINDOW_WIDTH, panel_y), 2)

    # Position counter
    cp_name = "A (Red)" if current_player == Player.A else "B (Blue)"
    cp_color = PLAYER_A_COLOR if current_player == Player.A else PLAYER_B_COLOR

    a_count = sum(1 for v in board.values() if v == Player.A)
    b_count = sum(1 for v in board.values() if v == Player.B)

    title = font_big.render(f"Position {idx + 1} / {total}", True, TEXT_COLOR)
    screen.blit(title, title.get_rect(centerx=WINDOW_WIDTH // 2, y=panel_y + 8))

    # Eval score with color coding
    eval_color = (100, 220, 100) if eval_score > 0 else (220, 100, 100) if eval_score < 0 else TEXT_COLOR
    eval_text = font_med.render(f"Eval: {eval_score:+,.0f}", True, eval_color)
    screen.blit(eval_text, (30, panel_y + 45))

    cp_text = font_med.render(f"Current: {cp_name}", True, cp_color)
    screen.blit(cp_text, (30, panel_y + 70))

    stones_text = font_med.render(f"Stones: A={a_count} B={b_count} ({a_count + b_count} total)", True, SUBTLE_TEXT)
    screen.blit(stones_text, (30, panel_y + 95))

    gid_text = font_med.render(f"Game ID: {game_id}", True, SUBTLE_TEXT)
    screen.blit(gid_text, (WINDOW_WIDTH - 250, panel_y + 45))

    if filter_gid is not None:
        filt_text = font_med.render(f"Filter: game {filter_gid}", True, HIGHLIGHT_COLOR)
        screen.blit(filt_text, (WINDOW_WIDTH - 250, panel_y + 70))

    # Controls
    controls = font_sm.render(
        "Left/Right: prev/next  |  Up/Down: ±100  |  G: goto  |  F: filter game  |  Q: quit",
        True, SUBTLE_TEXT
    )
    screen.blit(controls, controls.get_rect(centerx=WINDOW_WIDTH // 2, y=panel_y + 130))

    # Hover info
    if hover_hex:
        q, r = hover_hex
        occupant = board.get((q, r))
        occ_str = "empty" if occupant is None else ("A" if occupant == Player.A else "B")
        hover_text = font_sm.render(f"({q}, {r}) {occ_str}", True, SUBTLE_TEXT)
        screen.blit(hover_text, (WINDOW_WIDTH - 150, 10))

    pygame.display.flip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="View training positions")
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "data", "positions.pkl"))
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        all_positions = pickle.load(f)
    print(f"Loaded {len(all_positions)} positions from {args.input}")

    if not all_positions:
        print("No positions to view!")
        return

    # Print some stats
    evals = [entry[2] for entry in all_positions]
    print(f"Eval range: [{min(evals):.0f}, {max(evals):.0f}], mean: {sum(evals)/len(evals):.0f}")
    game_ids = sorted({entry[3] for entry in all_positions}) if len(all_positions[0]) >= 4 else []
    if game_ids:
        print(f"Games: {len(game_ids)} unique game IDs")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Position Viewer")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("Arial", 24, bold=True),
        pygame.font.SysFont("Arial", 18),
        pygame.font.SysFont("Arial", 14),
    )

    # State
    positions = all_positions  # may be filtered
    idx = 0
    hover_hex = None
    filter_gid = None

    def apply_filter(gid):
        nonlocal positions, idx, filter_gid
        if gid is None:
            positions = all_positions
            filter_gid = None
        else:
            positions = [p for p in all_positions if len(p) >= 4 and p[3] == gid]
            filter_gid = gid
            if not positions:
                print(f"No positions for game {gid}, clearing filter")
                positions = all_positions
                filter_gid = None
        idx = 0

    running = True
    while running:
        entry = positions[idx]
        board, cp, eval_score = entry[0], entry[1], entry[2]
        game_id = entry[3] if len(entry) >= 4 else -1

        visible_cells = get_visible_cells(board)
        hex_size, ox, oy = compute_view(visible_cells)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEMOTION:
                q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                if (q, r) in visible_cells:
                    hover_hex = (q, r)
                else:
                    hover_hex = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    idx = min(idx + 1, len(positions) - 1)
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    idx = max(idx - 1, 0)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    idx = min(idx + 100, len(positions) - 1)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    idx = max(idx - 100, 0)
                elif event.key == pygame.K_HOME:
                    idx = 0
                elif event.key == pygame.K_END:
                    idx = len(positions) - 1
                elif event.key == pygame.K_g:
                    try:
                        n = int(input("Go to position (1-based): ")) - 1
                        idx = max(0, min(n, len(positions) - 1))
                    except (ValueError, EOFError):
                        pass
                elif event.key == pygame.K_f:
                    try:
                        gid_str = input("Filter by game ID (empty to clear): ").strip()
                        if gid_str:
                            apply_filter(int(gid_str))
                        else:
                            apply_filter(None)
                    except (ValueError, EOFError):
                        pass
                elif event.key in (pygame.K_ESCAPE, pygame.K_c):
                    apply_filter(None)

        draw_position(screen, board, cp, eval_score, game_id,
                      idx, len(positions), visible_cells, hex_size, ox, oy, fonts,
                      hover_hex, filter_gid)
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
