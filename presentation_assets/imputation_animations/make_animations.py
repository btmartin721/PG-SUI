"""Generate schematic animations of PG-SUI imputers and masking strategies.

The input calls and all non-deterministic predictions are illustrative. This
script does not fit an imputer, simulate missingness with PG-SUI, or report
measured performance. Output GIFs are intended for README and Sphinx pages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1280
HEIGHT = 720
BACKGROUND = "#091725"
SURFACE = "#102536"
SURFACE_LIGHT = "#173348"
BORDER = "#325168"
WHITE = "#F2F8FB"
MUTED = "#B1C7D4"
FAINT = "#7894A5"
TEAL = "#54DFC9"
AMBER = "#FFD178"
RED = "#F38F91"
BLUE = "#82B9FF"
PURPLE = "#C5A2FF"
GREEN = "#9CE5A9"

GENOTYPES: tuple[tuple[int, ...], ...] = (
    (0, 0, -1, 1, 1, 0, 2),
    (0, 1, 0, 1, 0, -1, 0),
    (0, -1, 0, 2, 0, 1, 0),
    (0, 1, 2, 1, -1, 0, 0),
    (2, 0, 0, 1, 0, 0, -1),
    (-1, 0, 0, 0, 2, 0, 1),
)
ORIGINAL_MISSING = frozenset(
    (row, col)
    for row, values in enumerate(GENOTYPES)
    for col, value in enumerate(values)
    if value < 0
)
MISSING_COORDS = tuple(sorted(ORIGINAL_MISSING))
MISSING_INDEX = {coordinate: index for index, coordinate in enumerate(MISSING_COORDS)}
VALIDATION_MASK = frozenset({(0, 4), (4, 2)})
PREDICTIONS = (1, 0, 2, 1, 0, 0)


def observed_counts(locus: int) -> tuple[int, int, int]:
    """Count synthetic training calls, excluding held-out validation sites."""
    return tuple(
        sum(
            row[locus] == genotype
            for sample, row in enumerate(GENOTYPES)
            if (sample, locus) not in VALIDATION_MASK
        )
        for genotype in (0, 1, 2)
    )


def locus_mode(locus: int) -> int:
    """Return the observed per-locus mode, breaking ties toward lower codes."""
    counts = observed_counts(locus)
    return max((0, 1, 2), key=lambda genotype: (counts[genotype], -genotype))


@dataclass(frozen=True)
class ModelStory:
    """A public imputer and its diagram labels."""

    slug: str
    title: str
    family: str
    accent: str
    subtitle: str
    stages: tuple[str, ...]


STORIES: tuple[ModelStory, ...] = (
    ModelStory(
        "ref_allele",
        "Reference allele",
        "Deterministic baseline",
        BLUE,
        "Replace each original gap with the REF genotype (0).",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Identify REF = 0",
            "Apply the REF rule",
            "Check held-out calls",
            "Fill original gaps",
        ),
    ),
    ModelStory(
        "most_frequent",
        "Most frequent genotype",
        "Deterministic baseline",
        GREEN,
        "Use the mode at each locus; population-specific modes are optional.",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Count observed calls",
            "Choose locus modes",
            "Check held-out calls",
            "Fill original gaps",
        ),
    ),
    ModelStory(
        "random_forest",
        "Random forest",
        "Supervised model",
        GREEN,
        "Iterative imputation predicts a locus using a forest of trees.",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Use other loci",
            "Combine tree votes",
            "Check held-out calls",
            "Fill original gaps",
        ),
    ),
    ModelStory(
        "hist_gradient_boosting",
        "Histogram gradient boosting",
        "Supervised model",
        AMBER,
        "Bin predictors, then add trees that refine genotype predictions.",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Bin observed values",
            "Add boosting trees",
            "Check held-out calls",
            "Fill original gaps",
        ),
    ),
    ModelStory(
        "autoencoder",
        "Autoencoder",
        "Neural imputer",
        TEAL,
        "Encoder to latent code to decoder; reconstruct observed calls.",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Encode genotypes",
            "Decode latent code",
            "Check held-out calls",
            "Fill original gaps",
        ),
    ),
    ModelStory(
        "vae",
        "Variational autoencoder",
        "Neural imputer",
        PURPLE,
        "Encoder estimates a latent distribution; decoder reconstructs calls.",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Estimate mean and variance",
            "Sample z; decode",
            "Check held-out calls",
            "Fill original gaps",
        ),
    ),
    ModelStory(
        "nlpca",
        "Nonlinear PCA",
        "Neural imputer",
        BLUE,
        "Initialize sample embeddings with PCA; optimize a neural decoder.",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Initialize sample z",
            "Train decoder and z",
            "Project; check calls",
            "Fill original gaps",
        ),
    ),
    ModelStory(
        "ubp",
        "Unsupervised backpropagation",
        "Neural imputer",
        RED,
        "PCA initialization, decoder refinement, joint training, projection.",
        (
            "Input SNP matrix",
            "Hold out known calls",
            "Initialize sample z",
            "Refine then jointly train",
            "Project; check calls",
            "Fill original gaps",
        ),
    ),
)


@lru_cache(maxsize=None)
def font(
    size: int, *, bold: bool = False, mono: bool = False
) -> ImageFont.FreeTypeFont:
    """Load a readable font with cross-platform fallbacks."""
    if mono:
        paths = (
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
    elif bold:
        paths = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
    else:
        paths = (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    size: int = 22,
    fill: str = WHITE,
    bold: bool = False,
    mono: bool = False,
    anchor: str | None = None,
) -> None:
    """Draw one accessible text label."""
    draw.text(
        xy, value, font=font(size, bold=bold, mono=mono), fill=fill, anchor=anchor
    )


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    """Draw a shared surface panel."""
    draw.rounded_rectangle(box, radius=22, fill=SURFACE, outline=BORDER, width=2)


def arrow(
    draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str
) -> None:
    """Draw a directed link."""
    draw.line((start, end), fill=color, width=4)
    draw.polygon(
        [(end[0], end[1]), (end[0] - 11, end[1] - 7), (end[0] - 11, end[1] + 7)],
        fill=color,
    )


def draw_matrix(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    phase: int,
    output: bool,
    accent: str,
    predictions: tuple[int, ...],
) -> None:
    """Draw the input or final imputed genotype matrix."""
    label(draw, (x, y - 35), "LOCUS", size=17, fill=FAINT, bold=True)
    for col in range(7):
        label(
            draw,
            (x + 48 + col * 29, y - 9),
            str(col + 1),
            size=15,
            fill=MUTED,
            mono=True,
            anchor="mm",
        )
    for row, values in enumerate(GENOTYPES):
        label(
            draw,
            (x + 4, y + 16 + row * 40),
            f"S{row + 1}",
            size=17,
            fill=MUTED,
            mono=True,
            anchor="lm",
        )
        for col, value in enumerate(values):
            cell = (
                x + 32 + col * 29,
                y + row * 40,
                x + 58 + col * 29,
                y + 32 + row * 40,
            )
            missing = (row, col) in ORIGINAL_MISSING
            held_out = (row, col) in VALIDATION_MASK and phase >= 1 and not output
            revealed = output and phase >= 5
            background = SURFACE_LIGHT
            foreground = WHITE
            shown = str(value)
            if missing:
                background = "#174B4A"
                foreground = TEAL
                shown = "–"
                if revealed:
                    background = "#237362"
                    foreground = WHITE
                    shown = str(predictions[MISSING_INDEX[(row, col)]])
            elif held_out:
                background = "#745326"
                foreground = AMBER
                shown = "?"
            draw.rounded_rectangle(
                cell, radius=5, fill=background, outline=BORDER, width=1
            )
            label(
                draw,
                ((cell[0] + cell[2]) // 2, (cell[1] + cell[3]) // 2),
                shown,
                size=18,
                fill=foreground,
                bold=True,
                mono=True,
                anchor="mm",
            )
    label(
        draw, (x + 7, y + 270), "0 REF   1 HET   2 ALT", size=16, fill=FAINT, mono=True
    )


def network(
    draw: ImageDraw.ImageDraw,
    layers: tuple[tuple[int, int], ...],
    accent: str,
    phase: int,
) -> None:
    """Draw layered decoder or encoder-decoder nodes and links."""
    node_columns: list[list[tuple[int, int]]] = []
    for x, count in layers:
        top = 317 - (count - 1) * 29
        node_columns.append([(x, top + i * 58) for i in range(count)])
    for left, right in zip(node_columns, node_columns[1:]):
        for index, source in enumerate(left):
            for target in right:
                draw.line(
                    (source, target),
                    fill=accent if phase >= 3 and index % 2 == 0 else BORDER,
                    width=2,
                )
    for column_index, column in enumerate(node_columns):
        for x, y in column:
            color = accent if phase >= 2 + min(column_index, 1) else SURFACE_LIGHT
            draw.ellipse(
                (x - 12, y - 12, x + 12, y + 12), fill=color, outline=WHITE, width=2
            )


def tree(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, active: bool) -> None:
    """Draw a small decision tree with two levels."""
    edge = color if active else BORDER
    for child_x, child_y in ((x - 32, y + 60), (x + 32, y + 60)):
        draw.line((x, y, child_x, child_y), fill=edge, width=3)
        for leaf_x in (child_x - 16, child_x + 16):
            draw.line((child_x, child_y, leaf_x, child_y + 47), fill=edge, width=3)
            draw.ellipse(
                (leaf_x - 7, child_y + 40, leaf_x + 7, child_y + 54),
                fill=color if active else SURFACE_LIGHT,
            )
        draw.ellipse(
            (child_x - 9, child_y - 9, child_x + 9, child_y + 9),
            fill=color if active else SURFACE_LIGHT,
        )
    draw.ellipse(
        (x - 11, y - 11, x + 11, y + 11), fill=color if active else SURFACE_LIGHT
    )


def model_diagram(draw: ImageDraw.ImageDraw, story: ModelStory, phase: int) -> None:
    """Draw a method-specific algorithm diagram in the center panel."""
    accent = story.accent
    if story.slug == "ref_allele":
        label(draw, (660, 287), "REF", size=47, bold=True, fill=accent, anchor="mm")
        label(draw, (660, 348), "genotype code 0", size=25, fill=WHITE, anchor="mm")
        for index, x in enumerate((540, 600, 660, 720, 780)):
            value = "0" if phase >= 3 else "?"
            draw.rounded_rectangle(
                (x - 22, 405, x + 22, 450),
                radius=9,
                fill=accent if phase >= 3 else SURFACE_LIGHT,
            )
            label(
                draw,
                (x, 428),
                value,
                size=27,
                bold=True,
                fill=BACKGROUND if phase >= 3 else MUTED,
                anchor="mm",
            )
        label(
            draw,
            (660, 482),
            "Same rule at every missing cell",
            size=21,
            fill=MUTED,
            anchor="mm",
        )
    elif story.slug == "most_frequent":
        for index, locus in enumerate((1, 4, 6)):
            counts = observed_counts(locus)
            x = 510 + index * 150
            label(
                draw,
                (x, 258),
                f"L{locus + 1}",
                size=26,
                bold=True,
                fill=accent,
                anchor="mm",
            )
            for genotype, count in enumerate(counts):
                yy = 320 + genotype * 50
                label(
                    draw, (x - 50, yy), str(genotype), size=19, mono=True, anchor="mm"
                )
                draw.rounded_rectangle(
                    (x - 25, yy - 10, x - 25 + count * 18, yy + 10),
                    radius=5,
                    fill=accent if phase >= 2 else BORDER,
                )
            label(
                draw,
                (x, 488),
                f"mode: {locus_mode(locus)}",
                size=20,
                fill=WHITE,
                anchor="mm",
            )
        label(
            draw,
            (660, 533),
            "Optional: compute modes within populations",
            size=19,
            fill=MUTED,
            anchor="mm",
        )
    elif story.slug in {"random_forest", "hist_gradient_boosting"}:
        if story.slug == "hist_gradient_boosting":
            label(
                draw,
                (660, 239),
                "Bin observed SNP features",
                size=22,
                fill=MUTED,
                anchor="mm",
            )
            for x, height in zip((480, 505, 530, 555, 580), (30, 55, 84, 48, 20)):
                draw.rounded_rectangle(
                    (x, 327 - height, x + 20, 327),
                    radius=4,
                    fill=accent if phase >= 2 else BORDER,
                )
            arrow(draw, (610, 310), (640, 310), accent)
            tree_x = (690, 780, 870)
            label(
                draw,
                (775, 507),
                "Add tree corrections",
                size=20,
                fill=MUTED,
                anchor="mm",
            )
        else:
            label(
                draw,
                (660, 239),
                "Other loci inform the target SNP",
                size=22,
                fill=MUTED,
                anchor="mm",
            )
            tree_x = (510, 660, 810)
            label(
                draw,
                (660, 507),
                "Combine forest votes",
                size=20,
                fill=MUTED,
                anchor="mm",
            )
        for index, x in enumerate(tree_x):
            tree(draw, x, 331, accent, phase >= 3 or (phase >= 2 and index == 0))
        if story.slug == "random_forest":
            label(
                draw,
                (660, 548),
                "IterativeImputer updates missing loci",
                size=19,
                fill=WHITE,
                anchor="mm",
            )
        else:
            label(
                draw,
                (660, 548),
                "IterativeImputer repeats by locus",
                size=19,
                fill=WHITE,
                anchor="mm",
            )
    elif story.slug == "autoencoder":
        network(draw, ((470, 5), (565, 4), (660, 2), (755, 4), (850, 5)), accent, phase)
        for x, text_value in ((470, "input"), (660, "latent z"), (850, "rebuild")):
            label(draw, (x, 495), text_value, size=20, fill=MUTED, anchor="mm")
        label(
            draw,
            (660, 537),
            "Loss uses known genotype calls",
            size=19,
            fill=WHITE,
            anchor="mm",
        )
    elif story.slug == "vae":
        network(draw, ((470, 5), (565, 3), (760, 3), (850, 5)), accent, phase)
        for x, name in ((642, "mean"), (642, "log variance")):
            yy = 280 if name == "mean" else 365
            draw.rounded_rectangle(
                (x - 67, yy - 22, x + 67, yy + 22),
                radius=11,
                fill="#403658",
                outline=accent,
            )
            label(draw, (x, yy), name, size=18, fill=WHITE, anchor="mm")
        draw.ellipse(
            (693, 302, 741, 350),
            fill=accent if phase >= 3 else SURFACE_LIGHT,
            outline=WHITE,
            width=2,
        )
        label(
            draw,
            (717, 326),
            "z",
            size=25,
            bold=True,
            fill=BACKGROUND if phase >= 3 else WHITE,
            anchor="mm",
        )
        label(
            draw,
            (660, 511),
            "Reconstruction loss + KL penalty",
            size=20,
            fill=WHITE,
            anchor="mm",
        )
    elif story.slug in {"nlpca", "ubp"}:
        draw.rounded_rectangle(
            (440, 300, 545, 360), radius=14, fill="#24435C", outline=accent, width=2
        )
        label(draw, (493, 329), "PCA", size=29, bold=True, fill=WHITE, anchor="mm")
        arrow(draw, (548, 330), (608, 330), accent)
        draw.rounded_rectangle(
            (615, 300, 675, 360),
            radius=13,
            fill=accent if phase >= 2 else SURFACE_LIGHT,
        )
        label(
            draw,
            (645, 329),
            "z",
            size=28,
            bold=True,
            fill=BACKGROUND if phase >= 2 else WHITE,
            anchor="mm",
        )
        arrow(draw, (682, 330), (729, 330), accent)
        network(draw, ((750, 3), (805, 4), (860, 5)), accent, phase)
        if story.slug == "nlpca":
            label(
                draw,
                (660, 480),
                "Optimize sample z + decoder weights",
                size=21,
                fill=WHITE,
                anchor="mm",
            )
            label(
                draw,
                (660, 515),
                "Refine originally missing inputs during training",
                size=18,
                fill=MUTED,
                anchor="mm",
            )
            label(
                draw,
                (660, 550),
                "Inference: project z with decoder fixed",
                size=18,
                fill=MUTED,
                anchor="mm",
            )
        else:
            label(
                draw,
                (660, 485),
                "1  PCA initialization",
                size=19,
                fill=WHITE,
                anchor="mm",
            )
            label(
                draw,
                (660, 518),
                "2  Decoder refinement  →  3  joint training",
                size=18,
                fill=WHITE,
                anchor="mm",
            )
            label(
                draw,
                (660, 551),
                "Inference: project z with decoder fixed",
                size=18,
                fill=MUTED,
                anchor="mm",
            )


def draw_model_frame(story: ModelStory, phase: int) -> Image.Image:
    """Render one stage of a model workflow."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 13), fill=story.accent)
    label(draw, (46, 41), story.title, size=39, bold=True)
    label(draw, (48, 93), story.family.upper(), size=17, bold=True, fill=story.accent)
    label(draw, (340, 98), story.subtitle, size=20, fill=MUTED)
    for index in range(6):
        x = 48 + index * 201
        draw.rounded_rectangle(
            (x, 133, x + 184, 142),
            radius=5,
            fill=story.accent if index <= phase else BORDER,
        )
    rounded_panel(draw, (32, 171, 336, 610))
    rounded_panel(draw, (353, 171, 963, 610))
    rounded_panel(draw, (980, 171, 1248, 610))
    label(draw, (55, 199), "INPUT", size=22, bold=True, fill=TEAL)
    label(draw, (376, 199), "METHOD", size=22, bold=True, fill=story.accent)
    label(draw, (1003, 199), "OUTPUT", size=22, bold=True, fill=TEAL)
    predictions = (
        tuple(0 for _ in MISSING_COORDS) if story.slug == "ref_allele" else PREDICTIONS
    )
    if story.slug == "most_frequent":
        predictions = tuple(locus_mode(col) for _, col in MISSING_COORDS)
    draw_matrix(
        draw,
        48,
        272,
        phase=phase,
        output=False,
        accent=story.accent,
        predictions=predictions,
    )
    draw_matrix(
        draw,
        982,
        272,
        phase=phase,
        output=True,
        accent=story.accent,
        predictions=predictions,
    )
    model_diagram(draw, story, phase)
    if phase >= 4:
        draw.rounded_rectangle(
            (493, 570, 823, 599),
            radius=10,
            fill="#244458",
            outline=story.accent,
            width=2,
        )
        note = (
            "Compare predictions with held-out truth"
            if phase == 4
            else "Fill original gaps; preserve known calls"
        )
        label(draw, (658, 584), note, size=17, fill=WHITE, anchor="mm")
    arrow(draw, (337, 388), (349, 388), story.accent)
    arrow(draw, (964, 388), (976, 388), story.accent)
    label(
        draw,
        (50, 645),
        f"{phase + 1:02d} / 06  {story.stages[phase]}",
        size=24,
        bold=True,
        fill=story.accent,
    )
    label(
        draw,
        (50, 681),
        "Schematic example • held-out calls have truth; original gaps do not • predictions are illustrative",
        size=17,
        fill=FAINT,
    )
    return image


MASK_STORIES: tuple[tuple[str, str, str, frozenset[tuple[int, int]], str], ...] = (
    (
        "random",
        "Uniform over eligible calls",
        BLUE,
        frozenset({(0, 1), (1, 4), (2, 3), (4, 0), (5, 6)}),
        "Scattered cells",
    ),
    (
        "random_weighted",
        "Common codes get more weight",
        GREEN,
        frozenset({(0, 0), (1, 0), (2, 2), (4, 4), (5, 5)}),
        "Frequency per locus",
    ),
    (
        "random_weighted_inv",
        "Rare codes get more weight",
        AMBER,
        frozenset({(0, 6), (2, 3), (3, 2), (4, 0), (5, 4)}),
        "Inverse frequency",
    ),
    (
        "nonrandom",
        "Select a tree clade",
        PURPLE,
        frozenset({(0, 3), (1, 3), (2, 3), (0, 4), (1, 4)}),
        "Clade × a few loci",
    ),
    (
        "nonrandom_weighted",
        "Longer branches get more weight",
        RED,
        frozenset({(3, 1), (4, 1), (5, 1), (3, 0), (4, 0)}),
        "Branch-length bias",
    ),
)


def draw_mask_panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    story: tuple[str, str, str, frozenset[tuple[int, int]], str],
    active: bool,
    show_mask: bool,
) -> None:
    """Draw one strategy's miniature genotype matrix and mechanism."""
    name, description, accent, mask, mechanism = story
    box = (x, 180, x + 236, 617)
    draw.rounded_rectangle(
        box,
        radius=18,
        fill=SURFACE if active else "#0D2030",
        outline=accent if active else BORDER,
        width=3 if active else 1,
    )
    parts = name.split("_")
    if len(parts) > 2:
        title_lines = ("_".join(parts[:2]), "_".join(parts[2:]))
    else:
        title_lines = (name,)
    for index, line in enumerate(title_lines):
        label(
            draw,
            (x + 12, 205 + index * 22),
            line,
            size=18 if len(line) < 18 else 16,
            bold=True,
            fill=accent,
        )
    if name.startswith("nonrandom"):
        line_width = 5 if name.endswith("weighted") else 2
        draw.line(
            (x + 20, 244, x + 47, 244, x + 47, 229, x + 81, 229),
            fill=accent,
            width=line_width,
        )
        draw.line((x + 47, 244, x + 47, 257, x + 81, 257), fill=accent, width=2)
        label(draw, (x + 91, 229), "clade", size=14, fill=accent, anchor="lm")
    elif name in {"random_weighted", "random_weighted_inv"}:
        weights = (6, 4, 2) if name == "random_weighted" else (2, 4, 6)
        for genotype, height in enumerate(weights):
            xx = x + 20 + genotype * 25
            draw.rounded_rectangle(
                (xx, 257 - height * 4, xx + 17, 257), radius=3, fill=accent
            )
            label(
                draw,
                (xx + 8, 263),
                str(genotype),
                size=12,
                fill=FAINT,
                mono=True,
                anchor="mt",
            )
    else:
        for dot_x, dot_y in ((24, 238), (45, 257), (69, 235), (87, 252)):
            draw.ellipse(
                (x + dot_x - 4, dot_y - 4, x + dot_x + 4, dot_y + 4), fill=accent
            )
    label(draw, (x + 12, 286), description, size=14, fill=MUTED)
    cell_w, cell_h = 27, 31
    for row, values in enumerate(GENOTYPES):
        label(
            draw,
            (x + 15, 327 + row * 36),
            f"S{row + 1}",
            size=13,
            fill=FAINT,
            mono=True,
            anchor="lm",
        )
        for col, value in enumerate(values):
            cell_x = x + 45 + col * cell_w
            cell_y = 311 + row * 36
            is_masked = (row, col) in mask and show_mask and value >= 0
            is_original = value < 0
            fill = (
                "#69502B" if is_masked else "#174B4A" if is_original else SURFACE_LIGHT
            )
            draw.rounded_rectangle(
                (cell_x, cell_y, cell_x + 23, cell_y + cell_h),
                radius=4,
                fill=fill,
                outline=BORDER,
            )
            label(
                draw,
                (cell_x + 11, cell_y + 15),
                "?" if is_masked else "–" if is_original else str(value),
                size=15,
                bold=True,
                mono=True,
                fill=AMBER if is_masked else TEAL if is_original else WHITE,
                anchor="mm",
            )
    draw.line((x + 12, 541, x + 224, 541), fill=BORDER, width=1)
    label(draw, (x + 14, 562), mechanism, size=17, bold=True, fill=accent)
    if name.startswith("nonrandom"):
        label(draw, (x + 14, 592), "Tree-informed", size=14, fill=MUTED)
    else:
        label(draw, (x + 14, 592), "Cell sampling", size=14, fill=MUTED)


def draw_mask_frame(stage: int) -> Image.Image:
    """Render a comparison frame for all five implemented mask strategies."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 13), fill=TEAL)
    label(draw, (40, 41), "How simulated missingness is selected", size=36, bold=True)
    label(
        draw,
        (42, 96),
        "Mask observed genotypes; retain their values as ground truth for evaluation.",
        size=21,
        fill=MUTED,
    )
    if stage == 0:
        label(
            draw,
            (42, 146),
            "Start with observed calls and pre-existing gaps",
            size=22,
            fill=TEAL,
            bold=True,
        )
    elif stage <= len(MASK_STORIES):
        label(
            draw,
            (42, 146),
            f"Strategy {stage} of 5: {MASK_STORIES[stage - 1][0]}",
            size=22,
            fill=MASK_STORIES[stage - 1][2],
            bold=True,
        )
    else:
        label(
            draw,
            (42, 146),
            "Compare mask patterns at the same loci and samples",
            size=22,
            fill=TEAL,
            bold=True,
        )
    for index, story in enumerate(MASK_STORIES):
        draw_mask_panel(
            draw,
            32 + index * 247,
            story,
            stage == index + 1 or stage == 6,
            stage > index or stage == 6,
        )
    label(
        draw,
        (41, 653),
        "Amber ? = newly masked observed call    Teal – = originally missing; never scored",
        size=20,
        fill=WHITE,
    )
    label(
        draw,
        (41, 684),
        "Schematic selections • actual draws are stochastic and constrained by eligible calls and target proportion",
        size=16,
        fill=FAINT,
    )
    return image


def save_gif(frames: list[Image.Image], path: Path, durations: list[int]) -> None:
    """Write an optimized looping GIF with a stable shared color palette."""
    palette = frames[-1].quantize(colors=192)
    indexed = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    """Generate all public-imputer and simulated-missingness GIFs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "img",
        help="Directory for the generated GIFs (default: repository img/).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for story in STORIES:
        frames = [draw_model_frame(story, phase) for phase in range(6)]
        save_gif(
            frames,
            args.output_dir / f"impute_{story.slug}_workflow.gif",
            [850, 850, 1050, 1050, 850, 1900],
        )
    mask_frames = [draw_mask_frame(stage) for stage in range(7)]
    save_gif(
        mask_frames,
        args.output_dir / "simulated_missingness_strategies.gif",
        [1200, 1500, 1500, 1500, 1500, 1500, 2400],
    )


if __name__ == "__main__":
    main()
