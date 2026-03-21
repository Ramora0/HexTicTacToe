"""Pygame interface for Hexagonal Tic-Tac-Toe (infinite grid).

Run this file to play against the AI (you are Player A, AI is Player B).
Controls: Click to place, R to restart, Q to quit.
"""

import sys
import math
import pygame
from game import HexGame, Player
from og_ai import MinimaxBot

# --- Layout ---
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800
MAX_HEX_SIZE = 28
VISIBLE_DIST = 8  # show empty cells up to this distance from occupied stones

# --- Colors ---
BG_COLOR = (24, 24, 32)
EMPTY_FILL = (48, 48, 58)
GRID_LINE = (72, 72, 85)
PLAYER_A_COLOR = (220, 62, 62)
PLAYER_B_COLOR = (62, 120, 220)
PLAYER_A_HOVER = (120, 40, 40)
WIN_BORDER = (255, 215, 0)
TEXT_COLOR = (220, 220, 230)
SUBTLE_TEXT = (130, 130, 150)

AI_MOVE_DELAY = 300  # ms between AI stone placements


def _hex_distance(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


# Precomputed offsets for visible cell generation
_VISIBLE_OFFSETS = tuple(
    (dq, dr)
    for dq in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    for dr in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    if _hex_distance(dq, dr) <= VISIBLE_DIST
)


def hex_corners(cx, cy, size):
    """Return 6 corner points for a pointy-top hexagon centered at (cx, cy)."""
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
    """Convert pixel position to nearest axial hex coordinate."""
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


def get_visible_cells(game):
    """Cells to render: occupied + empties within VISIBLE_DIST of occupied."""
    board = game.board
    if not board:
        return {(oq, or_) for oq, or_ in _VISIBLE_OFFSETS}
    cells = set()
    for q, r in board:
        for oq, or_ in _VISIBLE_OFFSETS:
            cells.add((q + oq, r + or_))
    return cells


def compute_view(visible_cells):
    """Compute (hex_size, ox, oy) to fit and center visible cells in the window."""
    if not visible_cells:
        return MAX_HEX_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2

    S3 = math.sqrt(3)
    uxs = [S3 * (q + r * 0.5) for q, r in visible_cells]
    uys = [1.5 * r for q, r in visible_cells]

    min_ux, max_ux = min(uxs), max(uxs)
    min_uy, max_uy = min(uys), max(uys)

    # Extent in unit coords + padding for hex edges
    ext_x = max_ux - min_ux + S3
    ext_y = max_uy - min_uy + 2

    # Available area (leave room for status text top/bottom)
    avail_x = WINDOW_WIDTH - 60
    avail_y = WINDOW_HEIGHT - 140

    size = MAX_HEX_SIZE
    if ext_x > 0:
        size = min(size, avail_x / ext_x)
    if ext_y > 0:
        size = min(size, avail_y / ext_y)
    size = max(8.0, size)

    center_ux = (min_ux + max_ux) / 2
    center_uy = (min_uy + max_uy) / 2
    ox = WINDOW_WIDTH / 2 - center_ux * size
    oy = WINDOW_HEIGHT / 2 - center_uy * size + 20

    return size, ox, oy


def draw_board(screen, game, visible_cells, hover_hex, hex_size, ox, oy, fonts, ai_stats=None):
    font_big, font_med, font_sm = fonts
    screen.fill(BG_COLOR)

    board = game.board

    # Hex cells
    for (q, r) in visible_cells:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)

        player = board.get((q, r))
        if player == Player.A:
            fill = PLAYER_A_COLOR
        elif player == Player.B:
            fill = PLAYER_B_COLOR
        elif hover_hex == (q, r) and not game.game_over:
            fill = PLAYER_A_HOVER
        else:
            fill = EMPTY_FILL

        pygame.draw.polygon(screen, fill, corners)
        pygame.draw.polygon(screen, GRID_LINE, corners, 2)

    # Winning cells highlight
    for (q, r) in game.winning_cells:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)
        pygame.draw.polygon(screen, WIN_BORDER, corners, 3)

    # Status text
    if game.winner != Player.NONE:
        name = "You win!" if game.winner == Player.A else "AI wins!"
        color = PLAYER_A_COLOR if game.winner == Player.A else PLAYER_B_COLOR
        status = font_big.render(name, True, color)
    elif game.game_over:
        status = font_big.render("Draw!", True, TEXT_COLOR)
    elif game.current_player == Player.B:
        status = font_big.render("AI is thinking...", True, PLAYER_B_COLOR)
    else:
        status = font_big.render("Your turn", True, PLAYER_A_COLOR)

    screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))

    if not game.game_over and game.current_player == Player.A:
        moves_surf = font_med.render(
            f"Moves left: {game.moves_left_in_turn}", True, SUBTLE_TEXT
        )
        screen.blit(moves_surf, moves_surf.get_rect(centerx=WINDOW_WIDTH // 2, y=55))

    if ai_stats:
        depth, nodes = ai_stats
        ai_info = font_sm.render(f"AI: depth {depth}, {nodes:,} nodes", True, SUBTLE_TEXT)
        screen.blit(ai_info, ai_info.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 50))

    instr = font_sm.render("Click to place  |  R = restart  |  Q = quit", True, SUBTLE_TEXT)
    screen.blit(instr, instr.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 30))

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hex Tic-Tac-Toe \u2014 6 in a Row")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("Arial", 28, bold=True),
        pygame.font.SysFont("Arial", 20),
        pygame.font.SysFont("Arial", 16),
    )

    game = HexGame(win_length=6)
    ai = MinimaxBot(time_limit=0.5)

    hover_hex = None
    last_ai_time = 0
    ai_stats = None

    while True:
        now = pygame.time.get_ticks()

        visible_cells = get_visible_cells(game)
        hex_size, ox, oy = compute_view(visible_cells)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEMOTION:
                if game.current_player == Player.A and not game.game_over:
                    q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                    if (q, r) in visible_cells and game.is_valid_move(q, r):
                        hover_hex = (q, r)
                    else:
                        hover_hex = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game.current_player == Player.A and not game.game_over:
                    q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                    if (q, r) in visible_cells and game.make_move(q, r):
                        hover_hex = None
                        last_ai_time = now  # start AI delay timer

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    hover_hex = None
                    ai_stats = None
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE and not game.game_over:
                    if not game.board:
                        # First move: place at center, skip AI entirely
                        game.make_move(0, 0)
                    elif game.current_player == Player.A:
                        # AI plays the human's remaining moves
                        draw_board(screen, game, visible_cells, None, hex_size, ox, oy, fonts, ai_stats)
                        result = ai.get_move(game)
                        if ai.pair_moves:
                            for q, r in result:
                                if not game.game_over:
                                    game.make_move(q, r)
                        else:
                            game.make_move(*result)
                        ai_stats = (ai.last_depth, ai._nodes)
                        last_ai_time = pygame.time.get_ticks()
                    hover_hex = None

        # AI turn — one move per delay tick
        if (game.current_player == Player.B and not game.game_over
                and now - last_ai_time >= AI_MOVE_DELAY):
            # Draw "thinking" frame before computing
            draw_board(screen, game, visible_cells, None, hex_size, ox, oy, fonts, ai_stats)
            result = ai.get_move(game)
            if ai.pair_moves:
                for q, r in result:
                    if not game.game_over:
                        game.make_move(q, r)
            else:
                game.make_move(*result)
            ai_stats = (ai.last_depth, ai._nodes)
            last_ai_time = pygame.time.get_ticks()

        draw_board(screen, game, visible_cells, hover_hex, hex_size, ox, oy, fonts, ai_stats)
        clock.tick(60)


if __name__ == "__main__":
    main()
