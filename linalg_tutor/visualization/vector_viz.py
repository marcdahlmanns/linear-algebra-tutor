"""ASCII visualization for vectors."""

import numpy as np
from typing import Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def visualize_vector_2d(vector: np.ndarray, label: str = "v", scale: int = 10, show_components: bool = True) -> str:
    """Create ASCII art visualization of a 2D vector.

    Args:
        vector: 2D numpy array [x, y]
        label: Label for the vector
        scale: Scale factor for display (default 10 = 1 unit)
        show_components: Whether to show component breakdown

    Returns:
        ASCII art string representation
    """
    if len(vector) != 2:
        return f"Error: Expected 2D vector, got {len(vector)}D"

    x, y = vector[0], vector[1]

    # Calculate grid dimensions
    max_coord = max(abs(x), abs(y), 1)
    grid_size = int(max_coord * scale) + 2

    # Create grid
    height = grid_size * 2 + 1
    width = grid_size * 2 + 1
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Center point
    cx, cy = grid_size, grid_size

    # Draw axes
    for i in range(width):
        grid[cy][i] = '─'
    for i in range(height):
        grid[i][cx] = '│'
    grid[cy][cx] = '┼'

    # Draw vector arrow
    end_x = int(x * scale)
    end_y = -int(y * scale)  # Flip y for screen coordinates

    # Bresenham's line algorithm
    x0, y0 = 0, 0
    x1, y1 = end_x, end_y

    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    curr_x, curr_y = x0, y0
    while True:
        points.append((curr_x, curr_y))
        if curr_x == x1 and curr_y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            curr_x += sx
        if e2 < dx:
            err += dx
            curr_y += sy

    # Draw the line
    for px, py in points[:-1]:
        gx, gy = cx + px, cy + py
        if 0 <= gx < width and 0 <= gy < height:
            if grid[gy][gx] in ['─', '│', '┼']:
                grid[gy][gx] = '┼'
            else:
                grid[gy][gx] = '*'

    # Draw arrowhead
    if end_x != 0 or end_y != 0:
        gx, gy = cx + end_x, cy + end_y
        if 0 <= gx < width and 0 <= gy < height:
            # Determine arrow direction
            if abs(x) > abs(y):
                grid[gy][gx] = '→' if x > 0 else '←'
            else:
                grid[gy][gx] = '↑' if y > 0 else '↓'

    # Convert grid to string
    result = '\n'.join([''.join(row) for row in grid])

    # Add labels
    header = f"{label} = [{x:.2f}, {y:.2f}]"
    if show_components:
        magnitude = np.linalg.norm(vector)
        header += f"  |{label}| = {magnitude:.2f}"

    return f"{header}\n{result}"


def visualize_vector_components(vector: np.ndarray, label: str = "v") -> str:
    """Show vector component breakdown with visual bars.

    Args:
        vector: numpy array of any dimension
        label: Label for the vector

    Returns:
        Component breakdown visualization
    """
    lines = [f"{label} = {vector}"]
    lines.append("")

    max_abs = max(abs(v) for v in vector)
    scale = 30 / max_abs if max_abs > 0 else 1

    for i, val in enumerate(vector):
        bar_length = int(abs(val) * scale)
        bar = '█' * bar_length
        if val >= 0:
            lines.append(f"{label}[{i}] = {val:7.2f}  {bar}")
        else:
            lines.append(f"{label}[{i}] = {val:7.2f}  -{bar}")

    lines.append("")
    magnitude = np.linalg.norm(vector)
    lines.append(f"Magnitude: |{label}| = {magnitude:.2f}")

    return '\n'.join(lines)


def visualize_vector_addition_2d(v1: np.ndarray, v2: np.ndarray, scale: int = 10) -> str:
    """Visualize vector addition in 2D (parallelogram rule).

    Args:
        v1: First 2D vector
        v2: Second 2D vector
        scale: Scale factor for display

    Returns:
        ASCII art showing v1 + v2
    """
    if len(v1) != 2 or len(v2) != 2:
        return "Error: Both vectors must be 2D"

    result = v1 + v2

    # Create larger grid to fit all vectors
    max_coord = max(abs(v1[0]), abs(v1[1]), abs(v2[0]), abs(v2[1]),
                    abs(result[0]), abs(result[1]), 1)
    grid_size = int(max_coord * scale) + 2

    height = grid_size * 2 + 1
    width = grid_size * 2 + 1
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Center point
    cx, cy = grid_size, grid_size

    # Draw axes
    for i in range(width):
        grid[cy][i] = '─'
    for i in range(height):
        grid[i][cx] = '│'
    grid[cy][cx] = '┼'

    def draw_vector(start_x, start_y, vec, marker='*', arrow='→'):
        """Helper to draw a vector from a starting point."""
        end_x = start_x + int(vec[0] * scale)
        end_y = start_y - int(vec[1] * scale)

        # Simple line drawing
        steps = max(abs(end_x - start_x), abs(end_y - start_y), 1)
        for step in range(steps):
            t = step / steps
            px = int(start_x + t * (end_x - start_x))
            py = int(start_y + t * (end_y - start_y))
            gx, gy = cx + px, cy + py
            if 0 <= gx < width and 0 <= gy < height:
                if grid[gy][gx] in [' ', '─', '│', '┼']:
                    grid[gy][gx] = marker

        # Arrow
        gx, gy = cx + end_x, cy + end_y
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = arrow

    # Draw v1 from origin
    draw_vector(0, 0, v1, marker='·', arrow='1')

    # Draw v2 from end of v1
    v1_end_x = int(v1[0] * scale)
    v1_end_y = -int(v1[1] * scale)
    draw_vector(v1_end_x, v1_end_y, v2, marker='·', arrow='2')

    # Draw result from origin
    draw_vector(0, 0, result, marker='*', arrow='R')

    result_str = '\n'.join([''.join(row) for row in grid])

    header = f"v₁ = {v1}  →  1\n"
    header += f"v₂ = {v2}  →  2\n"
    header += f"v₁ + v₂ = {result}  →  R\n"

    return f"{header}\n{result_str}"


def visualize_dot_product(v1: np.ndarray, v2: np.ndarray) -> str:
    """Visualize dot product calculation.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Step-by-step dot product visualization
    """
    if len(v1) != len(v2):
        return "Error: Vectors must have same dimension"

    lines = []
    lines.append(f"v₁ = {v1}")
    lines.append(f"v₂ = {v2}")
    lines.append("")
    lines.append("Dot Product: v₁ · v₂")
    lines.append("")

    # Component-wise multiplication
    products = []
    for i, (a, b) in enumerate(zip(v1, v2)):
        product = a * b
        products.append(product)
        lines.append(f"  v₁[{i}] × v₂[{i}] = {a:6.2f} × {b:6.2f} = {product:8.2f}")

    lines.append("  " + "─" * 40)
    total = sum(products)
    lines.append(f"  Sum = {total:.2f}")
    lines.append("")

    # Geometric interpretation
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    if mag_v1 > 0 and mag_v2 > 0:
        cos_theta = total / (mag_v1 * mag_v2)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        theta_deg = np.degrees(theta)

        lines.append(f"Geometric interpretation:")
        lines.append(f"  |v₁| = {mag_v1:.2f}")
        lines.append(f"  |v₂| = {mag_v2:.2f}")
        lines.append(f"  angle θ = {theta_deg:.1f}°")
        lines.append(f"  v₁ · v₂ = |v₁| |v₂| cos(θ) = {mag_v1:.2f} × {mag_v2:.2f} × {cos_theta:.2f}")

    return '\n'.join(lines)


def print_vector_visualization(vector: np.ndarray, label: str = "v", show_2d: bool = True):
    """Print vector visualization to console with Rich formatting.

    Args:
        vector: Vector to visualize
        label: Label for the vector
        show_2d: Whether to show 2D ASCII art (only for 2D vectors)
    """
    console = Console()

    # Component breakdown
    component_viz = visualize_vector_components(vector, label)
    console.print(Panel(component_viz, title=f"Vector {label}", border_style="cyan"))

    # 2D ASCII art
    if len(vector) == 2 and show_2d:
        art = visualize_vector_2d(vector, label)
        console.print("\n")
        console.print(Panel(art, title="2D Visualization", border_style="green"))
