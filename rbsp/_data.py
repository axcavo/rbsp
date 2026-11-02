from dataclasses import dataclass
from numpy import dtype

uint8_le = dtype('<u1').type
uint16_le = dtype('<u2').type
uint32_le = dtype('<u4').type

SIGNATURE = b'MAGIC\0\0\0'


@dataclass(frozen=True)
class Header:
    """Total size: 16 bytes (excluding signature if handled separately, or adjust padding)"""
    version: uint16_le
    width: uint16_le
    height: uint16_le
    block_count: uint16_le

    chunk_grid_width: uint16_le
    chunk_grid_height: uint16_le

    data_offset: uint32_le
    signature: bytes = SIGNATURE


@dataclass(frozen=True)
class Point:
    x: uint16_le
    y: uint16_le


@dataclass(frozen=True)
class Run:
    y: uint16_le
    x_start: uint16_le
    x_end: uint16_le
    padding: uint16_le = uint16_le(0)  # Prevents misaligned memory reads


@dataclass(frozen=True)
class BlockGraph:
    block_id: uint16_le
    center_x: uint16_le
    center_y: uint16_le

    neighbour_count: uint16_le
    neighbour_start_index: uint32_le


@dataclass(frozen=True)
class BlockRender:
    """Hot data for Graphics. Size: 16 bytes (Aligned to 8)"""
    block_id: uint16_le
    run_count: uint16_le

    border_count: uint32_le  # <-- UPGRADED to 32-bit

    run_start_index: uint32_le
    border_start_index: uint32_le


@dataclass(frozen=True)
class BlockSpatial:
    block_id: uint16_le
    min_x: uint16_le
    max_x: uint16_le
    min_y: uint16_le
    max_y: uint16_le
    padding: uint16_le = uint16_le(0)


@dataclass(frozen=True)
class ChunkCell:
    """Size: 8 bytes (Perfectly aligned)"""
    block_count: uint16_le
    padding: uint16_le
    block_start_index: uint32_le  # Points to 'global_chunk_blocks'


@dataclass(frozen=True)
class MapFile:
    header: Header

    graph_blocks: list[BlockGraph]
    render_blocks: list[BlockRender]
    spatial_blocks: list[BlockSpatial]

    global_runs: list[Run]
    global_neighbours: list[uint16_le]
    global_borders: list[Point]

    # --- NEW: Spatial Hash Grid ---
    chunk_cells: list[ChunkCell]
    global_chunk_blocks: list[uint16_le]