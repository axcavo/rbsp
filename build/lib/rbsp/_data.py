from dataclasses import dataclass

from numpy import dtype

uint16_le = dtype('<u2').type
SIGNATURE = b'MAGIC'


@dataclass(frozen=True)
class Header:
    version: uint16_le
    width: uint16_le
    height: uint16_le
    block_count: uint16_le
    data_offset: uint16_le
    signature: bytes = SIGNATURE


@dataclass(frozen=True)
class Run:
    y: uint16_le
    x_start: uint16_le
    x_end: uint16_le


@dataclass(frozen=True)
class Point:
    x: uint16_le
    y: uint16_le


@dataclass(frozen=True)
class Block:
    block_id: uint16_le
    run_count: uint16_le
    runs: list[Run]

    # This could be another dataclass, BlockMetadata or smth.
    neighbour_count: uint16_le
    neighbours: list[uint16_le]

    min_x: uint16_le
    max_x: uint16_le
    min_y: uint16_le
    max_y: uint16_le

    center_x: uint16_le
    center_y: uint16_le
