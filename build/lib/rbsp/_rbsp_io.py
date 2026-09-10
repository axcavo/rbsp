from io import BufferedWriter
from pathlib import Path

from numpy import frombuffer, dtype

from rbsp._data import Header, Block, Run, SIGNATURE, Point

# Redeclare instead of importing cause pycharm sucks.
uint16_le = dtype('<u2').type


def write_header(header: Header, file: BufferedWriter) -> None:
    file.write(header.signature)
    file.write(header.version.tobytes())
    file.write(header.width.tobytes())
    file.write(header.height.tobytes())
    file.write(header.block_count.tobytes())
    file.write(header.data_offset.tobytes())


def write_block(block: Block, file: BufferedWriter) -> None:
    file.write(block.block_id.tobytes())
    file.write(block.run_count.tobytes())

    for run in block.runs:
        write_run(run, file)

    file.write(block.neighbour_count.tobytes())

    for neighbour in block.neighbours:
        file.write(neighbour.tobytes())

    file.write(block.min_x.tobytes())
    file.write(block.max_x.tobytes())
    file.write(block.min_y.tobytes())
    file.write(block.max_y.tobytes())
    file.write(block.center_x.tobytes())
    file.write(block.center_y.tobytes())


def write_run(run: Run, file: BufferedWriter) -> None:
    file.write(run.y.tobytes())
    file.write(run.x_start.tobytes())
    file.write(run.x_end.tobytes())


def write_point(point: Point, file: BufferedWriter) -> None:
    file.write(point.x.tobytes())
    file.write(point.y.tobytes())


def write_rbsp(header: Header, blocks: list[Block], file_path: Path) -> None:
    with open(file_path, 'wb') as file:
        write_header(header, file)

        for block in blocks:
            write_block(block, file)


def read_rbsp(file_path: Path) -> tuple[Header, list[Block]]:
    with open(file_path, 'rb') as file:

        def read_uint16() -> uint16_le:
            raw_bytes = file.read(2)
            return frombuffer(raw_bytes, dtype=uint16_le)[0]

        # Read header.
        header = Header(
            signature=(file.read(len(SIGNATURE))),
            version=read_uint16(),
            width=read_uint16(),
            height=read_uint16(),
            block_count=read_uint16(),
            data_offset=read_uint16(),
        )

        # Verify signature.
        if header.signature != SIGNATURE:
            raise ValueError(f"File: {file_path} has an invalid signature.")

        # Go to blocks start.
        file.seek(int(header.data_offset))

        # Read blocks.
        blocks = []
        for i in range(header.block_count):
            block_id = read_uint16()
            run_count = read_uint16()

            # Read runs.
            runs = []
            for j in range(run_count):
                runs.append(
                    Run(
                        y=read_uint16(),
                        x_start=read_uint16(),
                        x_end=read_uint16(),
                    )
                )

            neighbour_count = read_uint16()
            neighbours = []
            for j in range(neighbour_count):
                neighbours.append(read_uint16())

            metadata = {
                "min_x": read_uint16(),
                "max_x": read_uint16(),
                "min_y": read_uint16(),
                "max_y": read_uint16(),
                "center_x": read_uint16(),
                "center_y": read_uint16(),
            }

            blocks.append(
                Block(
                    block_id=block_id,
                    run_count=run_count,
                    runs=runs,
                    neighbour_count=neighbour_count,
                    neighbours=neighbours,
                    **metadata
                )
            )

        return header, blocks