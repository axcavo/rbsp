from collections import defaultdict
from pathlib import Path
from PIL import Image
from numpy import dtype

from rbsp import write_rbsp, Header, SIGNATURE, Run, Block

uint16_le = dtype('<u2').type


def to_rbsp(file: Path | str, output: Path | str) -> dict[str, str]:
    """Convert image to rbsp format and return mapping of RGB hex to Block ID hex."""

    if isinstance(file, str):
        file = Path(file)

    image = Image.open(file).convert("RGB")
    width, height = image.size

    rgb_hex_to_rb_hex: dict[str, str] = {}

    color_to_id: dict[tuple[int, int, int], int] = {}
    id_to_runs = defaultdict(list)
    next_id = 0

    id_map = [[0] * width for _ in range(height)]

    for y in range(height):

        current_color = image.getpixel((0, y))
        if current_color not in color_to_id:
            color_to_id[current_color] = next_id

            rgb_hex = f"{current_color[0]:02X}{current_color[1]:02X}{current_color[2]:02X}"
            rb_hex = f"{next_id:04X}"
            rgb_hex_to_rb_hex[rgb_hex] = rb_hex

            next_id += 1

        current_id = color_to_id[current_color]
        x_start = 0

        for x in range(width):
            color = image.getpixel((x, y))

            if color not in color_to_id:
                if next_id > 65535:
                    raise ValueError("The source image contains too many different colors, max is 65535.")
                color_to_id[color] = next_id

                rgb_hex = f"{color[0]:02X}{color[1]:02X}{color[2]:02X}"
                rb_hex = f"{next_id:04X}"
                rgb_hex_to_rb_hex[rgb_hex] = rb_hex

                next_id += 1

            cid = color_to_id[color]
            id_map[y][x] = cid

            if cid != current_id:
                id_to_runs[current_id].append(
                    Run(y=uint16_le(y), x_start=uint16_le(x_start), x_end=uint16_le(x - 1))
                )
                current_id = cid
                x_start = x

        id_to_runs[current_id].append(
            Run(y=uint16_le(y), x_start=uint16_le(x_start), x_end=uint16_le(width - 1))
        )

    id_to_neighbors = defaultdict(set)
    for y in range(height):
        for x in range(width):
            cid = id_map[y][x]
            if x < width - 1:
                right_id = id_map[y][x + 1]
                if cid != right_id:
                    id_to_neighbors[cid].add(right_id)
                    id_to_neighbors[right_id].add(cid)
            if y < height - 1:
                bottom_id = id_map[y + 1][x]
                if cid != bottom_id:
                    id_to_neighbors[cid].add(bottom_id)
                    id_to_neighbors[bottom_id].add(cid)

    blocks = []
    for cid, runs in id_to_runs.items():
        min_x = min(r.x_start for r in runs)
        max_x = max(r.x_end for r in runs)
        min_y = runs[0].y
        max_y = runs[-1].y

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        neighbors = list(id_to_neighbors[cid])

        blocks.append(Block(
            block_id=uint16_le(cid),
            run_count=uint16_le(len(runs)),
            runs=runs,
            neighbour_count=uint16_le(len(neighbors)),
            neighbours=[uint16_le(n) for n in neighbors],
            min_x=uint16_le(min_x),
            max_x=uint16_le(max_x),
            min_y=uint16_le(min_y),
            max_y=uint16_le(max_y),
            center_x=uint16_le(center_x),
            center_y=uint16_le(center_y)
        ))

    header = Header(
        version=uint16_le(0),
        width=uint16_le(width),
        height=uint16_le(height),
        block_count=uint16_le(len(blocks)),
        data_offset=uint16_le(10 + len(SIGNATURE)),
    )

    if isinstance(output, str):
        output = Path(output)

    write_rbsp(header, blocks, output)

    return rgb_hex_to_rb_hex
