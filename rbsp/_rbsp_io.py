import mmap
from io import BufferedWriter
from pathlib import Path
from typing import Any

import numpy as np
from numpy import frombuffer, dtype

from rbsp._data import Header, BlockGraph, BlockRender, BlockSpatial, Run, Point, MapFile, SIGNATURE, ChunkCell

# ---------------------------------------------------------
# ZERO-COPY MEMORY MAP STRUCTS
# ---------------------------------------------------------
dt_graph = np.dtype([('block_id', '<u2'), ('center_x', '<u2'), ('center_y', '<u2'),
                     ('neighbour_count', '<u2'), ('neighbour_start_index', '<u4')])

dt_render = np.dtype([('block_id', '<u2'), ('run_count', '<u2'), ('border_count', '<u4'),
                      ('run_start_index', '<u4'), ('border_start_index', '<u4')])

dt_spatial = np.dtype([('block_id', '<u2'), ('min_x', '<u2'), ('max_x', '<u2'),
                       ('min_y', '<u2'), ('max_y', '<u2'), ('padding', '<u2')])

dt_run = np.dtype([('y', '<u2'), ('x_start', '<u2'), ('x_end', '<u2'), ('padding', '<u2')])
dt_point = np.dtype([('x', '<u2'), ('y', '<u2')])
dt_chunk = np.dtype([('block_count', '<u2'), ('padding', '<u2'), ('block_start_index', '<u4')])

uint8_le = dtype('<u1').type
uint16_le = dtype('<u2').type
uint32_le = dtype('<u4').type


def read_rbsp_mmap(file_path: Path) -> dict[str, Any]:
    """Zero-copy memory-mapped file reader for instant DOD loading."""
    f = open(file_path, "r+b")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # 1. Parse Header Manually (24 bytes)
    signature = mm[:8]
    if signature != SIGNATURE:
        raise ValueError(f"File: {file_path} has an invalid signature.")

    version = np.frombuffer(mm, dtype='<u2', count=1, offset=8)[0]
    width = np.frombuffer(mm, dtype='<u2', count=1, offset=10)[0]
    height = np.frombuffer(mm, dtype='<u2', count=1, offset=12)[0]
    block_count = np.frombuffer(mm, dtype='<u2', count=1, offset=14)[0]
    chunk_w = np.frombuffer(mm, dtype='<u2', count=1, offset=16)[0]
    chunk_h = np.frombuffer(mm, dtype='<u2', count=1, offset=18)[0]
    data_offset = np.frombuffer(mm, dtype='<u4', count=1, offset=20)[0]

    offset = data_offset

    # 2. Map Split Blocks (O(1) access)
    graph_blocks = np.frombuffer(mm, dtype=dt_graph, count=block_count, offset=offset)
    offset += graph_blocks.nbytes

    render_blocks = np.frombuffer(mm, dtype=dt_render, count=block_count, offset=offset)
    offset += render_blocks.nbytes

    spatial_blocks = np.frombuffer(mm, dtype=dt_spatial, count=block_count, offset=offset)
    offset += spatial_blocks.nbytes

    # Calculate counts for global arrays (Forced 64-bit to prevent overflow)
    total_runs = np.sum(render_blocks['run_count'], dtype=np.uint64)
    total_neighbours = np.sum(graph_blocks['neighbour_count'], dtype=np.uint64)
    total_borders = np.sum(render_blocks['border_count'], dtype=np.uint64)

    # 3. Map Massive Global Arrays (Zero-copy)
    global_runs = np.frombuffer(mm, dtype=dt_run, count=total_runs, offset=offset)
    offset += global_runs.nbytes

    global_neighbours = np.frombuffer(mm, dtype='<u2', count=total_neighbours, offset=offset)
    offset += global_neighbours.nbytes

    global_borders = np.frombuffer(mm, dtype=dt_point, count=total_borders, offset=offset)
    offset += global_borders.nbytes

    # 4. Map Spatial Hash Grid (Cast to int to prevent overflow)
    total_cells = int(chunk_w) * int(chunk_h)
    chunk_cells = np.frombuffer(mm, dtype=dt_chunk, count=total_cells, offset=offset)
    offset += chunk_cells.nbytes

    total_chunk_blocks = np.sum(chunk_cells['block_count'], dtype=np.uint64)
    global_chunk_blocks = np.frombuffer(mm, dtype='<u2', count=total_chunk_blocks, offset=offset)

    return {
        "graph_blocks": graph_blocks,
        "render_blocks": render_blocks,
        "spatial_blocks": spatial_blocks,
        "global_runs": global_runs,
        "global_neighbours": global_neighbours,
        "global_borders": global_borders,
        "chunk_cells": chunk_cells,
        "global_chunk_blocks": global_chunk_blocks,
        "_mmap_handle": mm,
        "_file_handle": f
    }


def unload_map(mapped_data: dict[str, Any]):
    """Safely closes the memory mapped file when the map is no longer needed."""

    # 1. Close the mmap interface first
    if "_mmap_handle" in mapped_data:
        mapped_data["_mmap_handle"].close()

    # 2. Close the underlying file descriptor
    if "_file_handle" in mapped_data:
        mapped_data["_file_handle"].close()

    # Clear the dictionary to prevent accidental access to dangling pointers
    mapped_data.clear()


def write_header(header: 'Header', file: BufferedWriter) -> None:
    file.write(header.signature)
    file.write(header.version.tobytes())
    file.write(header.width.tobytes())
    file.write(header.height.tobytes())
    file.write(header.block_count.tobytes())
    file.write(header.chunk_grid_width.tobytes())
    file.write(header.chunk_grid_height.tobytes())
    file.write(header.data_offset.tobytes())


def write_block_graph(block: 'BlockGraph', file: BufferedWriter) -> None:
    file.write(block.block_id.tobytes())
    file.write(block.center_x.tobytes())
    file.write(block.center_y.tobytes())
    file.write(block.neighbour_count.tobytes())
    file.write(block.neighbour_start_index.tobytes())


def write_block_render(block: 'BlockRender', file: BufferedWriter) -> None:
    file.write(block.block_id.tobytes())
    file.write(block.run_count.tobytes())
    file.write(block.border_count.tobytes())
    file.write(block.run_start_index.tobytes())
    file.write(block.border_start_index.tobytes())


def write_block_spatial(block: 'BlockSpatial', file: BufferedWriter) -> None:
    file.write(block.block_id.tobytes())
    file.write(block.min_x.tobytes())
    file.write(block.max_x.tobytes())
    file.write(block.min_y.tobytes())
    file.write(block.max_y.tobytes())
    file.write(block.padding.tobytes())


def write_run(run: 'Run', file: BufferedWriter) -> None:
    file.write(run.y.tobytes())
    file.write(run.x_start.tobytes())
    file.write(run.x_end.tobytes())
    file.write(run.padding.tobytes())


def write_point(point: 'Point', file: BufferedWriter) -> None:
    file.write(point.x.tobytes())
    file.write(point.y.tobytes())


def write_rbsp(map_file: 'MapFile', file_path: Path) -> None:
    with open(file_path, 'wb') as file:
        write_header(map_file.header, file)

        for block in map_file.graph_blocks:
            write_block_graph(block, file)

        for block in map_file.render_blocks:
            write_block_render(block, file)

        for block in map_file.spatial_blocks:
            write_block_spatial(block, file)

        for run in map_file.global_runs:
            write_run(run, file)

        for neighbour in map_file.global_neighbours:
            file.write(neighbour.tobytes())

        for point in map_file.global_borders:
            write_point(point, file)

        for cell in map_file.chunk_cells:
            file.write(cell.block_count.tobytes())
            file.write(cell.padding.tobytes())
            file.write(cell.block_start_index.tobytes())

        for block_id in map_file.global_chunk_blocks:
            file.write(block_id.tobytes())


def read_rbsp(file_path: Path) -> 'MapFile':
    with open(file_path, 'rb') as file:
        def read_uint16() -> uint16_le:
            return frombuffer(file.read(2), dtype=uint16_le)[0]

        def read_uint32() -> uint32_le:
            return frombuffer(file.read(4), dtype=uint32_le)[0]

        header = Header(
            signature=file.read(8),
            version=read_uint16(),
            width=read_uint16(),
            height=read_uint16(),
            block_count=read_uint16(),
            chunk_grid_width=read_uint16(),
            chunk_grid_height=read_uint16(),
            data_offset=read_uint32(),
        )

        if header.signature != SIGNATURE:
            raise ValueError(f"File: {file_path} has an invalid signature.")

        file.seek(int(header.data_offset))

        graph_blocks = []
        for _ in range(header.block_count):
            graph_blocks.append(BlockGraph(
                block_id=read_uint16(),
                center_x=read_uint16(),
                center_y=read_uint16(),
                neighbour_count=read_uint16(),
                neighbour_start_index=read_uint32()
            ))

        render_blocks = []
        for _ in range(header.block_count):
            render_blocks.append(BlockRender(
                block_id=read_uint16(),
                run_count=read_uint16(),
                border_count=read_uint32(),
                run_start_index=read_uint32(),
                border_start_index=read_uint32()
            ))

        spatial_blocks = []
        for _ in range(header.block_count):
            spatial_blocks.append(BlockSpatial(
                block_id=read_uint16(),
                min_x=read_uint16(),
                max_x=read_uint16(),
                min_y=read_uint16(),
                max_y=read_uint16(),
                padding=read_uint16()
            ))

        total_runs = sum(b.run_count for b in render_blocks)
        total_neighbours = sum(b.neighbour_count for b in graph_blocks)
        total_borders = sum(b.border_count for b in render_blocks)

        global_runs = []
        for _ in range(total_runs):
            global_runs.append(Run(
                y=read_uint16(),
                x_start=read_uint16(),
                x_end=read_uint16(),
                padding=read_uint16()
            ))

        global_neighbours = []
        for _ in range(total_neighbours):
            global_neighbours.append(read_uint16())

        global_borders = []
        for _ in range(total_borders):
            global_borders.append(Point(
                x=read_uint16(),
                y=read_uint16()
            ))

        total_cells = header.chunk_grid_width * header.chunk_grid_height

        chunk_cells = []
        for _ in range(total_cells):
            chunk_cells.append(ChunkCell(
                block_count=read_uint16(),
                padding=read_uint16(),
                block_start_index=read_uint32()
            ))

        total_chunk_blocks = sum(c.block_count for c in chunk_cells)
        global_chunk_blocks = []
        for _ in range(total_chunk_blocks):
            global_chunk_blocks.append(read_uint16())

        return MapFile(
            header=header,
            graph_blocks=graph_blocks,
            render_blocks=render_blocks,
            spatial_blocks=spatial_blocks,
            global_runs=global_runs,
            global_neighbours=global_neighbours,
            global_borders=global_borders,
            chunk_cells=chunk_cells,
            global_chunk_blocks=global_chunk_blocks
        )
