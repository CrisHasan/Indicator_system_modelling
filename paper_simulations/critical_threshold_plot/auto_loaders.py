import numpy as np


def load_time_S_blocks(path):
    """
    Read an AUTO solution file and parse its time and S solution blocks.

    Returns
    -------
    time_array : np.ndarray
        Shape (n_rows, n_solutions)
    S_array : np.ndarray
        Shape (n_rows, n_solutions)
    """
    with open(path, "r") as f:
        return parse_time_S_blocks(f)


def parse_time_S_blocks(lines):
    """Parse time and S values from solution blocks in an AUTO file."""
    time_blocks = []
    S_blocks = []
    current_time = []
    current_S = []
    in_block = False

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        # Solution blocks begin with a header line of 16 tokens.
        if len(parts) == 16:
            if in_block and current_time:
                time_blocks.append(current_time)
                S_blocks.append(current_S)
            current_time = []
            current_S = []
            in_block = True
            continue

        # Short integer-only lines separate blocks.
        if len(parts) == 3 and all(token.lstrip("+-").isdigit() for token in parts):
            if in_block and current_time:
                time_blocks.append(current_time)
                S_blocks.append(current_S)
                current_time = []
                current_S = []
                in_block = False
            continue

        if not in_block:
            continue

        try:
            t_val = float(parts[0])
            s_val = float(parts[1])
        except ValueError:
            continue

        current_time.append(t_val)
        current_S.append(s_val)

    if in_block and current_time:
        time_blocks.append(current_time)
        S_blocks.append(current_S)

    time_array = np.column_stack(time_blocks)
    S_array = np.column_stack(S_blocks)
    return time_array, S_array
