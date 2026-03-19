"""Pygame interface for Hexagonal Tic-Tac-Toe.

Run this file to play against the AI (you are Player A, AI is Player B).
Controls: Click to place, R to restart, Q to quit.
"""

import sys
import math
import pygame
from game import HexGame, Player
from ai import MinimaxBot

# --- Layout ---
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800
HEX_SIZE = 28
BOARD_RADIUS = 5

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


def draw_board(screen, game, hover_hex, ox, oy, fonts):
    font_big, font_med, font_sm = fonts
    screen.fill(BG_COLOR)

    # Hex cells
    for (q, r), player in game.board.items():
        cx, cy = hex_to_pixel(q, r, HEX_SIZE, ox, oy)
        corners = hex_corners(cx, cy, HEX_SIZE)

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
        cx, cy = hex_to_pixel(q, r, HEX_SIZE, ox, oy)
        corners = hex_corners(cx, cy, HEX_SIZE)
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

    instr = font_sm.render("Click to place  |  R = restart  |  Q = quit", True, SUBTLE_TEXT)
    screen.blit(instr, instr.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 30))

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hex Tic-Tac-Toe — 6 in a Row")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("Arial", 28, bold=True),
        pygame.font.SysFont("Arial", 20),
        pygame.font.SysFont("Arial", 16),
    )

    game = HexGame(radius=BOARD_RADIUS, win_length=6)
    ai = MinimaxBot(time_limit=0.5)
    ox = WINDOW_WIDTH // 2
    oy = WINDOW_HEIGHT // 2 + 20

    hover_hex = None
    last_ai_time = 0

    while True:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEMOTION:
                if game.current_player == Player.A and not game.game_over:
                    q, r = pixel_to_hex(*event.pos, HEX_SIZE, ox, oy)
                    hover_hex = (q, r) if game.is_valid_move(q, r) else None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game.current_player == Player.A and not game.game_over:
                    q, r = pixel_to_hex(*event.pos, HEX_SIZE, ox, oy)
                    if game.make_move(q, r):
                        hover_hex = None
                        last_ai_time = now  # start AI delay timer

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    hover_hex = None
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

        # AI turn — one move per delay tick
        if (game.current_player == Player.B and not game.game_over
                and now - last_ai_time >= AI_MOVE_DELAY):
            # Draw "thinking" frame before computing
            draw_board(screen, game, None, ox, oy, fonts)
            q, r = ai.get_move(game)
            game.make_move(q, r)
            last_ai_time = pygame.time.get_ticks()

        draw_board(screen, game, hover_hex, ox, oy, fonts)
        clock.tick(60)


if __name__ == "__main__":
    main()
