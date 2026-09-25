"""Render a slide-ready schematic of VAE genotype imputation.

The animation uses illustrative allele calls and predictions. It does not run
PG-SUI's VAE or report measured model performance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1600
HEIGHT = 900
BACKGROUND = "#091827"
CELL_BACKGROUND = "#142B3F"
CELL_BORDER = "#28465B"
WHITE = "#F2F7FA"
MUTED = "#A9BDCC"
FAINT = "#658298"
TEAL = "#51D4C0"
TEAL_DARK = "#174A4C"
AMBER = "#FFCE73"
AMBER_DARK = "#66441F"
RED = "#FF8178"
GREEN = "#87DDB0"

GRID_X = 258
GRID_Y = 270
CELL_WIDTH = 61
CELL_HEIGHT = 49
GAP = 2

REFERENCE = "ACGTGCACTGACGTACCTGA"
ORIGINAL_MISSING = frozenset(
    {
        (0, 4),
        (0, 14),
        (1, 8),
        (1, 17),
        (2, 2),
        (2, 12),
        (3, 6),
        (3, 18),
        (4, 1),
        (4, 11),
        (5, 5),
        (5, 16),
        (6, 9),
        (6, 19),
        (7, 3),
        (7, 13),
    }
)
SIMULATED_MASK = (
    (0, 9),
    (1, 3),
    (2, 16),
    (3, 1),
    (4, 15),
    (5, 10),
    (6, 6),
    (7, 18),
)
ALL_GAPS = tuple(
    sorted(ORIGINAL_MISSING | set(SIMULATED_MASK), key=lambda x: (x[1], x[0]))
)
BASES = "ACGT"


@dataclass(frozen=True)
class FrameState:
    """Animation state for one frame."""

    phase: str
    visible_masks: int = 0
    vae_step: int = -1
    filled: int = 0
    scored: int = 0


@lru_cache(maxsize=None)
def font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Load a readable font on macOS or Linux."""
    paths = (
        (
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
        if mono
        else (
            "/System/Library/Fonts/Avenir Next.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def allele_matrix() -> tuple[str, ...]:
    """Build a compact illustrative alignment with plausible variant sites."""
    rows: list[str] = []
    for row in range(8):
        calls = list(REFERENCE)
        for col in range(len(calls)):
            if (row * 7 + col * 3) % 13 == 0 or (row + col * 2) % 23 == 0:
                calls[col] = BASES[(BASES.index(calls[col]) + 1 + row % 2) % 4]
        rows.append("".join(calls))
    return tuple(rows)


ALLELES = allele_matrix()


def predicted_call(row: int, col: int) -> str:
    """Return a scripted prediction, including one masked mismatch."""
    observed = ALLELES[row][col]
    if (row, col) == (4, 15):
        return BASES[(BASES.index(observed) + 1) % 4]
    if (row, col) in ORIGINAL_MISSING and (row + col) % 7 == 0:
        return BASES[(BASES.index(observed) + 2) % 4]
    return observed


def label_center(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    fill: str,
    size: int,
    *,
    mono: bool = False,
) -> None:
    """Place a centered label."""
    draw.text(position, value, font=font(size, mono=mono), fill=fill, anchor="mm")


def stage_content(state: FrameState) -> tuple[str, str, int]:
    """Return the current stage subtitle, detail, and ordinal."""
    if state.phase == "input":
        return (
            "Original gaps in an alignment of allele calls",
            "Original gaps have no known truth",
            1,
        )
    if state.phase == "mask":
        return (
            "Hide observed calls to create a validation mask",
            "Masked calls retain known truth for scoring",
            2,
        )
    if state.phase == "vae":
        return (
            "VAE inference: encode, sample latent z, decode",
            "Predictions are illustrative; no model was run",
            3,
        )
    if state.phase == "fill":
        return (
            "Predict both original gaps and held-out calls",
            "Teal = original gaps   ·   Amber = validation mask",
            4,
        )
    if state.scored == len(SIMULATED_MASK):
        return (
            "Score only the held-out calls: 7 / 8 match",
            "Original gaps are filled but cannot be scored",
            5,
        )
    return (
        "Compare predictions with held-out truth",
        "Original gaps are filled but cannot be scored",
        5,
    )


def draw_grid(draw: ImageDraw.ImageDraw, state: FrameState) -> None:
    """Draw the alignment and progressively reveal masks, calls, and scores."""
    draw.text((97, 214), "SAMPLE", font=font(19), fill=FAINT)
    draw.text((258, 214), "LOCUS", font=font(19), fill=FAINT)

    for col in range(len(REFERENCE)):
        x = GRID_X + col * CELL_WIDTH + (CELL_WIDTH - GAP) // 2
        label_center(draw, (x, 238), f"{col + 1:02d}", FAINT, 17, mono=True)

    revealed = set(SIMULATED_MASK[: state.visible_masks])
    filled = set(ALL_GAPS[: state.filled])
    scored = set(SIMULATED_MASK[: state.scored])

    for row, calls in enumerate(ALLELES):
        y = GRID_Y + row * CELL_HEIGHT
        label_center(draw, (166, y + 22), f"S{row + 1:02d}", MUTED, 23, mono=True)
        for col, observed in enumerate(calls):
            location = (row, col)
            x = GRID_X + col * CELL_WIDTH
            box = (x, y, x + CELL_WIDTH - GAP, y + CELL_HEIGHT - GAP)
            is_original = location in ORIGINAL_MISSING
            is_masked = location in revealed
            is_filled = location in filled

            background = CELL_BACKGROUND
            outline = CELL_BORDER
            foreground = WHITE
            shown = observed

            if is_original:
                background, outline, foreground, shown = BACKGROUND, TEAL, TEAL, "—"
            elif is_masked:
                background, outline, foreground, shown = AMBER_DARK, AMBER, AMBER, "?"
            if is_filled and is_original:
                background, outline, foreground, shown = (
                    TEAL_DARK,
                    TEAL,
                    WHITE,
                    predicted_call(row, col),
                )
            elif is_filled and is_masked:
                background, outline, foreground, shown = (
                    AMBER_DARK,
                    AMBER,
                    WHITE,
                    predicted_call(row, col),
                )

            if location in scored:
                outline = GREEN if predicted_call(row, col) == observed else RED

            draw.rounded_rectangle(
                box, radius=8, fill=background, outline=outline, width=2
            )
            label_center(
                draw,
                (x + 29, y + 23),
                shown,
                foreground,
                26 if shown != "—" else 23,
                mono=True,
            )
            if location in scored:
                if predicted_call(row, col) == observed:
                    draw.line((x + 44, y + 12, x + 48, y + 16), fill=GREEN, width=3)
                    draw.line((x + 48, y + 16, x + 55, y + 7), fill=GREEN, width=3)
                else:
                    draw.line((x + 45, y + 8, x + 54, y + 17), fill=RED, width=3)
                    draw.line((x + 54, y + 8, x + 45, y + 17), fill=RED, width=3)


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    """Draw the visual key directly under the alignment."""
    items = [
        (296, WHITE, "A", "observed"),
        (555, TEAL, "—", "original gap"),
        (867, AMBER, "?", "held-out call"),
        (1193, TEAL, "A", "imputed"),
    ]
    for x, color, marker, description in items:
        label_center(draw, (x, 704), marker, color, 24, mono=True)
        draw.text((x + 24, 690), description, font=font(20), fill=MUTED)


def draw_vae_path(draw: ImageDraw.ImageDraw, state: FrameState) -> None:
    """Draw the VAE processing path and highlight the active operation."""
    y = 789
    nodes = (
        (255, "masked input"),
        (550, "encoder"),
        (824, "latent z"),
        (1098, "decoder"),
        (1376, "predictions"),
    )
    active_step = (
        state.vae_step
        if state.phase == "vae"
        else (-1 if state.phase in {"input", "mask"} else 4)
    )
    for idx, (x, label) in enumerate(nodes):
        if idx < len(nodes) - 1:
            next_x = nodes[idx + 1][0]
            draw.line((x + 22, y, next_x - 27, y), fill=CELL_BORDER, width=3)
            draw.polygon(
                [(next_x - 28, y - 6), (next_x - 16, y), (next_x - 28, y + 6)],
                fill=CELL_BORDER,
            )
        active = idx == active_step
        circle_color = TEAL if active else CELL_BORDER
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill=circle_color)
        label_center(draw, (x, y), str(idx + 1), BACKGROUND if active else MUTED, 17)
        label_center(draw, (x, y + 47), label, WHITE if active else MUTED, 20)


def draw_frame(state: FrameState) -> Image.Image:
    """Render one RGB animation frame."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    subtitle, detail, ordinal = stage_content(state)

    draw.text((95, 53), "VAE GENOTYPE IMPUTATION", font=font(52), fill=WHITE)
    draw.text(
        (98, 132),
        subtitle,
        font=font(28),
        fill=TEAL if ordinal in {1, 3, 4} else AMBER,
    )
    draw.text((98, 174), detail, font=font(20), fill=MUTED)
    draw.text(
        (1460, 74),
        f"{ordinal:02d} / 05",
        font=font(23, mono=True),
        fill=FAINT,
        anchor="ra",
    )
    draw.text(
        (1460, 141),
        "ILLUSTRATIVE · NO MODEL RUN",
        font=font(18),
        fill=FAINT,
        anchor="ra",
    )
    draw.line((96, 201, 1504, 201), fill=CELL_BORDER, width=2)

    draw_grid(draw, state)
    draw_legend(draw)
    draw.line((96, 751, 1504, 751), fill=CELL_BORDER, width=2)
    draw_vae_path(draw, state)
    draw.text(
        (98, 870),
        (
            "Illustrative allele calls and predictions  •  "
            "held-out sites have truth; original gaps do not"
        ),
        font=font(17),
        fill=FAINT,
    )
    return image


def frame_sequence() -> list[tuple[FrameState, int]]:
    """Define stages and their display durations in milliseconds."""
    frames = [(FrameState("input"), 2000)]
    frames.extend(
        (FrameState("mask", visible_masks=count), 500) for count in range(1, 9)
    )
    frames.extend(
        (FrameState("vae", visible_masks=8, vae_step=step), 320) for step in range(5)
    )
    frames.extend(
        (FrameState("fill", visible_masks=8, filled=count), 450)
        for count in range(2, len(ALL_GAPS) + 1, 2)
    )
    frames.extend(
        (FrameState("score", visible_masks=8, filled=len(ALL_GAPS), scored=count), 500)
        for count in range(1, 9)
    )
    frames.append(
        (FrameState("score", visible_masks=8, filled=len(ALL_GAPS), scored=8), 2000)
    )
    return frames


def main() -> None:
    """Write the animated GIF and a final-frame poster PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        gif_path = Path(__file__).parents[2] / "img" / "vae_imputation.gif"
        poster_path = Path(__file__).parent / "vae_imputation_poster.png"
    else:
        gif_path = args.output_dir / "vae_imputation.gif"
        poster_path = args.output_dir / "vae_imputation_poster.png"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)

    sequence = frame_sequence()
    rendered = [draw_frame(state) for state, _ in sequence]
    palette_source = rendered[-1].quantize(colors=224)
    indexed = [
        frame.quantize(palette=palette_source, dither=Image.Dither.NONE)
        for frame in rendered
    ]
    indexed[0].save(
        gif_path,
        save_all=True,
        append_images=indexed[1:],
        duration=[duration for _, duration in sequence],
        loop=0,
        optimize=True,
        disposal=2,
    )
    rendered[-1].save(poster_path, optimize=True)


if __name__ == "__main__":
    main()
