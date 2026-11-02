from numpy import dtype
from rbsp import Point

uint16_le = dtype('<u2').type


def extract_province_borders(id_map: list[list[int]], width: int, height: int) -> dict[int, list[Point]]:
    """
    Uses Moore-Neighbor Tracing to extract ordered boundary points for every province.
    Handles disjoint regions (islands) by separating them with a restart marker.
    """
    # 8-way directional offsets for Moore Tracing (Clockwise)
    # N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]

    borders: dict[int, list[Point]] = {}
    visited_edges = set()  # Tracks (x, y, dir_index) to prevent infinite loops

    for y in range(height):
        for x in range(width):
            cid = id_map[y][x]

            # Check if this is a boundary pixel (left edge of the screen, or left pixel is different)
            if x == 0 or id_map[y][x - 1] != cid:
                # We found a starting edge!
                start_state = (x, y, 6)  # 6 is West (where we know the boundary is)

                if start_state in visited_edges:
                    continue  # Already traced this contour

                if cid not in borders:
                    borders[cid] = []
                elif len(borders[cid]) > 0:
                    # If this province already has points (e.g. an island),
                    # insert a primitive restart marker so the GPU doesn't draw a line across the ocean.
                    # We use 65535 (max uint16) as a universal break marker.
                    borders[cid].append(Point(uint16_le(65535), uint16_le(65535)))

                # --- BEGIN MOORE TRACE --- (Corregida la indentación aquí)
                curr_x, curr_y = x, y
                enter_dir = 6

                while True:
                    # NEW: Strict cycle detection
                    state = (curr_x, curr_y, enter_dir)
                    if state in visited_edges:
                        break  # We've completed the loop!

                    visited_edges.add(state)
                    borders[cid].append(Point(uint16_le(curr_x), uint16_le(curr_y)))

                    # Start looking from the direction we just came from, clockwise
                    search_start = (enter_dir + 2) % 8
                    found_next = False

                    for i in range(8):
                        dir_idx = (search_start + i) % 8
                        dx, dy = DIRECTIONS[dir_idx]
                        nx, ny = curr_x + dx, curr_y + dy

                        # Check bounds and ID
                        if 0 <= nx < width and 0 <= ny < height and id_map[ny][nx] == cid:
                            curr_x, curr_y = nx, ny
                            enter_dir = (dir_idx + 4) % 8  # Reverse direction to know where we entered
                            found_next = True
                            break

                    if not found_next:
                        break  # Isolated pixel, nowhere to go

    return borders