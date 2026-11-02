from collections import defaultdict
from pathlib import Path
from PIL import Image
from numpy import dtype
import math

from ._rbsp_io import write_rbsp

from ._data import Run, BlockGraph, BlockSpatial, BlockRender, ChunkCell, Header, MapFile, SIGNATURE

from ._compiler_math import extract_province_borders

uint16_le = dtype('<u2').type
uint32_le = dtype('<u4').type


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

    # --- 1. EXTRACT RUNS AND IDs ---
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
                    Run(
                        y=uint16_le(y),
                        x_start=uint16_le(x_start),
                        x_end=uint16_le(x - 1),
                        padding=uint16_le(0)
                    )
                )
                current_id = cid
                x_start = x

        id_to_runs[current_id].append(
            Run(
                y=uint16_le(y),
                x_start=uint16_le(x_start),
                x_end=uint16_le(width - 1),
                padding=uint16_le(0)
            )
        )

    # --- 2. EXTRACT NEIGHBORS ---
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

    # --- 3. BUILD DOD ARRAYS --- (UNINDENTED!)
    graph_blocks = []
    render_blocks = []
    spatial_blocks = []

    global_runs = []
    global_neighbours = []
    global_borders = []

    print("Tracing province borders...")
    province_borders = extract_province_borders(id_map, width, height)

    for cid, runs in sorted(id_to_runs.items()):
        min_x = min(r.x_start for r in runs)
        max_x = max(r.x_end for r in runs)
        min_y = runs[0].y
        max_y = runs[-1].y

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        neighbors = list(id_to_neighbors[cid])

        border_points = province_borders.get(cid, [])

        run_start_index = len(global_runs)
        neighbour_start_index = len(global_neighbours)
        border_start_index = len(global_borders)

        global_runs.extend(runs)
        global_neighbours.extend([uint16_le(n) for n in neighbors])
        global_borders.extend(border_points)

        graph_blocks.append(BlockGraph(
            block_id=uint16_le(cid),
            center_x=uint16_le(center_x),
            center_y=uint16_le(center_y),
            neighbour_count=uint16_le(len(neighbors)),
            neighbour_start_index=uint32_le(neighbour_start_index)
        ))

        render_blocks.append(BlockRender(
            block_id=uint16_le(cid),
            run_count=uint16_le(len(runs)),
            border_count=uint32_le(len(border_points)),
            run_start_index=uint32_le(run_start_index),
            border_start_index=uint32_le(border_start_index)
        ))

        spatial_blocks.append(BlockSpatial(
            block_id=uint16_le(cid),
            min_x=uint16_le(min_x),
            max_x=uint16_le(max_x),
            min_y=uint16_le(min_y),
            max_y=uint16_le(max_y),
            padding=uint16_le(0)
        ))

    # --- 4. BUILD THE CHUNK GRID (O(1) Spatial Hash) ---
    print("Building spatial hash grid...")
    CHUNK_SIZE = 256

    chunk_grid_width = math.ceil(width / CHUNK_SIZE)
    chunk_grid_height = math.ceil(height / CHUNK_SIZE)

    chunk_cells = []
    global_chunk_blocks = []

    for cy in range(chunk_grid_height):
        for cx in range(chunk_grid_width):
            start_y = cy * CHUNK_SIZE
            end_y = min(start_y + CHUNK_SIZE, height)
            start_x = cx * CHUNK_SIZE
            end_x = min(start_x + CHUNK_SIZE, width)

            unique_blocks_in_chunk = set()
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    unique_blocks_in_chunk.add(id_map[y][x])

            sorted_blocks = sorted(list(unique_blocks_in_chunk))

            block_start_index = len(global_chunk_blocks)
            global_chunk_blocks.extend([uint16_le(bid) for bid in sorted_blocks])

            chunk_cells.append(ChunkCell(
                block_count=uint16_le(len(sorted_blocks)),
                padding=uint16_le(0),
                block_start_index=uint32_le(block_start_index)
            ))

    header = Header(
        signature=SIGNATURE,
        version=uint16_le(0),
        width=uint16_le(width),
        height=uint16_le(height),
        block_count=uint16_le(len(graph_blocks)),
        chunk_grid_width=uint16_le(chunk_grid_width),  # <-- FIXED
        chunk_grid_height=uint16_le(chunk_grid_height),  # <-- FIXED
        data_offset=uint32_le(24),
    )

    map_file = MapFile(
        header=header,
        graph_blocks=graph_blocks,
        render_blocks=render_blocks,
        spatial_blocks=spatial_blocks,
        global_runs=global_runs,
        global_neighbours=global_neighbours,
        global_borders=global_borders,
        chunk_cells=chunk_cells,  # <-- FIXED
        global_chunk_blocks=global_chunk_blocks  # <-- FIXED
    )

    if isinstance(output, str):
        output = Path(output)

    write_rbsp(map_file, output)

    return rgb_hex_to_rb_hex


if __name__ == "__main__":
    print(to_rbsp('../template_1.png', 'o.rbsp'))