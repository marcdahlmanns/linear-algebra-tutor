"""Rich visualization for matrices."""

import numpy as np
from typing import Optional, List, Tuple, Set
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


def create_matrix_table(
    matrix: np.ndarray,
    title: str = "Matrix",
    highlight_cells: Optional[Set[Tuple[int, int]]] = None,
    highlight_rows: Optional[List[int]] = None,
    highlight_cols: Optional[List[int]] = None,
    show_indices: bool = True,
) -> Table:
    """Create a Rich table for displaying a matrix.

    Args:
        matrix: 2D numpy array to display
        title: Title for the table
        highlight_cells: Set of (row, col) tuples to highlight
        highlight_rows: List of row indices to highlight
        highlight_cols: List of column indices to highlight
        show_indices: Whether to show row/column indices

    Returns:
        Rich Table object
    """
    if len(matrix.shape) != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    rows, cols = matrix.shape
    highlight_cells = highlight_cells or set()
    highlight_rows = highlight_rows or []
    highlight_cols = highlight_cols or []

    # Create table
    table = Table(
        title=title,
        show_header=show_indices,
        header_style="bold cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )

    # Add columns
    if show_indices:
        table.add_column("", style="dim", justify="right")
        for j in range(cols):
            table.add_column(f"[{j}]", justify="right")
    else:
        for _ in range(cols):
            table.add_column("", justify="right")

    # Add rows
    for i in range(rows):
        row_data = []
        if show_indices:
            row_data.append(f"[{i}]")

        for j in range(cols):
            value = matrix[i, j]

            # Format value
            if np.isclose(value, 0, atol=1e-10):
                formatted = "0"
            elif abs(value) >= 1000 or (abs(value) < 0.01 and value != 0):
                formatted = f"{value:.2e}"
            else:
                formatted = f"{value:.4g}"

            # Apply highlighting
            cell_style = ""
            if (i, j) in highlight_cells:
                cell_style = "bold yellow on blue"
            elif i in highlight_rows:
                cell_style = "bold green"
            elif j in highlight_cols:
                cell_style = "bold magenta"

            if cell_style:
                row_data.append(f"[{cell_style}]{formatted}[/]")
            else:
                row_data.append(formatted)

        table.add_row(*row_data)

    return table


def visualize_matrix_multiply_step(
    A: np.ndarray,
    B: np.ndarray,
    result_row: int,
    result_col: int,
) -> str:
    """Visualize a single element calculation in matrix multiplication.

    Args:
        A: First matrix (m × n)
        B: Second matrix (n × p)
        result_row: Row index in result matrix
        result_col: Column index in result matrix

    Returns:
        String showing the calculation
    """
    if A.shape[1] != B.shape[0]:
        return "Error: Matrix dimensions incompatible"

    # Get the row from A and column from B
    a_row = A[result_row, :]
    b_col = B[:, result_col]

    lines = []
    lines.append(f"Computing C[{result_row},{result_col}]:")
    lines.append("")

    # Show row × column
    lines.append(f"Row {result_row} of A: {a_row}")
    lines.append(f"Col {result_col} of B: {b_col}")
    lines.append("")

    # Show dot product calculation
    terms = []
    for k, (a_val, b_val) in enumerate(zip(a_row, b_col)):
        product = a_val * b_val
        terms.append(f"({a_val:.2f})({b_val:.2f})")
        lines.append(f"  A[{result_row},{k}] × B[{k},{result_col}] = {a_val:.4g} × {b_val:.4g} = {product:.4g}")

    result = np.dot(a_row, b_col)
    lines.append("  " + "─" * 50)
    lines.append(f"  C[{result_row},{result_col}] = {' + '.join(terms)} = {result:.4g}")

    return '\n'.join(lines)


def visualize_matrix_operation(
    operation: str,
    matrices: List[np.ndarray],
    result: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Panel:
    """Visualize a matrix operation with before/after display.

    Args:
        operation: Operation name (e.g., "Addition", "Multiplication")
        matrices: List of input matrices
        result: Result matrix
        labels: Labels for input matrices (default: A, B, C, ...)

    Returns:
        Rich Panel with visualization
    """
    console = Console()

    if labels is None:
        labels = [chr(65 + i) for i in range(len(matrices))]  # A, B, C, ...

    # Build content
    content = []

    # Show input matrices
    for i, (matrix, label) in enumerate(zip(matrices, labels)):
        if i > 0:
            content.append("\n")
        table = create_matrix_table(matrix, title=label, show_indices=False)
        # Convert table to string representation
        with console.capture() as capture:
            console.print(table)
        content.append(capture.get())

    # Operation symbol
    if operation.lower() == "addition":
        op_symbol = " + "
    elif operation.lower() == "subtraction":
        op_symbol = " - "
    elif operation.lower() == "multiplication":
        op_symbol = " × "
    else:
        op_symbol = f" {operation} "

    # Show result
    content.append(f"\n{op_symbol}\n")
    result_table = create_matrix_table(result, title="Result", show_indices=False)
    with console.capture() as capture:
        console.print(result_table)
    content.append(capture.get())

    return Panel(
        ''.join(content),
        title=f"Matrix {operation}",
        border_style="cyan",
    )


def visualize_matrix_transpose(matrix: np.ndarray) -> str:
    """Visualize matrix transpose operation.

    Args:
        matrix: Input matrix

    Returns:
        String showing original and transposed matrix
    """
    console = Console()
    lines = []

    lines.append("Original Matrix:")
    with console.capture() as capture:
        console.print(create_matrix_table(matrix, title="A", show_indices=True))
    lines.append(capture.get())

    lines.append("\nTranspose Operation: A^T")
    lines.append("(Rows become columns, columns become rows)")
    lines.append("")

    transposed = matrix.T
    with console.capture() as capture:
        console.print(create_matrix_table(transposed, title="A^T", show_indices=True))
    lines.append(capture.get())

    return '\n'.join(lines)


def visualize_determinant_2x2(matrix: np.ndarray) -> str:
    """Visualize 2×2 determinant calculation.

    Args:
        matrix: 2×2 matrix

    Returns:
        String showing determinant calculation
    """
    if matrix.shape != (2, 2):
        return "Error: Expected 2×2 matrix"

    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    lines = []
    lines.append("Determinant of 2×2 Matrix:")
    lines.append("")
    lines.append(f"│ {a:6.2f}  {b:6.2f} │")
    lines.append(f"│ {c:6.2f}  {d:6.2f} │")
    lines.append("")
    lines.append("Formula: det(A) = ad - bc")
    lines.append("")
    lines.append(f"  = ({a:.4g})({d:.4g}) - ({b:.4g})({c:.4g})")
    lines.append(f"  = {a*d:.4g} - {b*c:.4g}")
    lines.append(f"  = {a*d - b*c:.4g}")

    det = np.linalg.det(matrix)
    lines.append("")
    lines.append(f"det(A) = {det:.4g}")

    # Interpretation
    lines.append("")
    if abs(det) < 1e-10:
        lines.append("⚠️  Determinant is 0 → Matrix is singular (not invertible)")
    else:
        lines.append("✓ Determinant ≠ 0 → Matrix is invertible")

    return '\n'.join(lines)


def visualize_identity_matrix(size: int) -> str:
    """Visualize identity matrix.

    Args:
        size: Size of identity matrix

    Returns:
        String showing identity matrix
    """
    matrix = np.eye(size)
    console = Console()

    lines = []
    lines.append(f"Identity Matrix I_{size}")
    lines.append("")

    with console.capture() as capture:
        console.print(create_matrix_table(matrix, title=f"I_{size}", show_indices=True))
    lines.append(capture.get())

    lines.append("")
    lines.append("Properties:")
    lines.append("  • Diagonal elements = 1")
    lines.append("  • Off-diagonal elements = 0")
    lines.append("  • A × I = I × A = A")

    return '\n'.join(lines)


def visualize_matrix_properties(matrix: np.ndarray, label: str = "A") -> str:
    """Analyze and display matrix properties.

    Args:
        matrix: Matrix to analyze
        label: Label for the matrix

    Returns:
        String showing matrix properties
    """
    lines = []
    lines.append(f"Matrix {label} Properties:")
    lines.append("")

    rows, cols = matrix.shape
    lines.append(f"  Dimensions: {rows} × {cols}")

    # Square matrix
    is_square = rows == cols
    lines.append(f"  Square: {'Yes' if is_square else 'No'}")

    if is_square:
        # Symmetric
        is_symmetric = np.allclose(matrix, matrix.T)
        lines.append(f"  Symmetric: {'Yes' if is_symmetric else 'No'}")

        # Diagonal
        is_diagonal = np.allclose(matrix, np.diag(np.diag(matrix)))
        lines.append(f"  Diagonal: {'Yes' if is_diagonal else 'No'}")

        # Identity
        is_identity = np.allclose(matrix, np.eye(rows))
        lines.append(f"  Identity: {'Yes' if is_identity else 'No'}")

        # Determinant
        det = np.linalg.det(matrix)
        lines.append(f"  Determinant: {det:.4g}")

        # Invertible
        is_invertible = abs(det) > 1e-10
        lines.append(f"  Invertible: {'Yes' if is_invertible else 'No'}")

        # Trace
        trace = np.trace(matrix)
        lines.append(f"  Trace: {trace:.4g}")

    # Zero matrix
    is_zero = np.allclose(matrix, 0)
    lines.append(f"  Zero matrix: {'Yes' if is_zero else 'No'}")

    # Norm
    norm = np.linalg.norm(matrix, 'fro')
    lines.append(f"  Frobenius norm: {norm:.4g}")

    return '\n'.join(lines)


def print_matrix_visualization(matrix: np.ndarray, title: str = "Matrix", show_properties: bool = True):
    """Print matrix visualization to console with Rich formatting.

    Args:
        matrix: Matrix to visualize
        title: Title for the display
        show_properties: Whether to show matrix properties
    """
    console = Console()

    # Display matrix
    table = create_matrix_table(matrix, title=title, show_indices=True)
    console.print(table)

    # Display properties
    if show_properties:
        console.print("")
        properties = visualize_matrix_properties(matrix, title)
        console.print(Panel(properties, title="Properties", border_style="yellow"))


def visualize_determinant_3x3(A: np.ndarray) -> str:
    """Visualize 3×3 determinant calculation using cofactor expansion.

    Args:
        A: 3×3 matrix

    Returns:
        String showing the calculation steps
    """
    if A.shape != (3, 3):
        return f"This visualization requires a 3×3 matrix, got {A.shape}"

    output = []
    output.append("Determinant via Cofactor Expansion (first row):")
    output.append("")

    # Show matrix
    output.append("     ┌                 ┐")
    output.append(f"     │ {A[0,0]:6.3g}  {A[0,1]:6.3g}  {A[0,2]:6.3g} │")
    output.append(f"det  │ {A[1,0]:6.3g}  {A[1,1]:6.3g}  {A[1,2]:6.3g} │")
    output.append(f"     │ {A[2,0]:6.3g}  {A[2,1]:6.3g}  {A[2,2]:6.3g} │")
    output.append("     └                 ┘")
    output.append("")

    # Calculate minors
    M11 = A[1,1]*A[2,2] - A[1,2]*A[2,1]
    M12 = A[1,0]*A[2,2] - A[1,2]*A[2,0]
    M13 = A[1,0]*A[2,1] - A[1,1]*A[2,0]

    output.append(f"= {A[0,0]:g} × │{A[1,1]:g}  {A[1,2]:g}│ - {A[0,1]:g} × │{A[1,0]:g}  {A[1,2]:g}│ + {A[0,2]:g} × │{A[1,0]:g}  {A[1,1]:g}│")
    output.append(f"        │{A[2,1]:g}  {A[2,2]:g}│         │{A[2,0]:g}  {A[2,2]:g}│         │{A[2,0]:g}  {A[2,1]:g}│")
    output.append("")
    output.append(f"= {A[0,0]:g} × ({M11:g}) - {A[0,1]:g} × ({M12:g}) + {A[0,2]:g} × ({M13:g})")
    output.append("")

    det = A[0,0]*M11 - A[0,1]*M12 + A[0,2]*M13
    output.append(f"= {A[0,0]*M11:g} - {A[0,1]*M12:g} + {A[0,2]*M13:g}")
    output.append("")
    output.append(f"= {det:g}")

    return "\n".join(output)


def visualize_determinant_geometric(A: np.ndarray) -> str:
    """Explain geometric meaning of determinant.

    Args:
        A: 2×2 or 3×3 matrix

    Returns:
        String explaining geometric interpretation
    """
    det = np.linalg.det(A)

    output = []
    output.append("Geometric Meaning of Determinant:")
    output.append("")

    if A.shape == (2, 2):
        output.append("For a 2×2 matrix:")
        output.append(f"  det(A) = {det:.4g}")
        output.append("")
        output.append(f"  |det(A)| = {abs(det):.4g} = Area of parallelogram")
        output.append("  formed by the column vectors")
        output.append("")
        if det > 0:
            output.append("  det > 0 → Preserves orientation")
        elif det < 0:
            output.append("  det < 0 → Reverses orientation (reflection)")
        else:
            output.append("  det = 0 → Collapses to line (singular)")

    elif A.shape == (3, 3):
        output.append("For a 3×3 matrix:")
        output.append(f"  det(A) = {det:.4g}")
        output.append("")
        output.append(f"  |det(A)| = {abs(det):.4g} = Volume of parallelepiped")
        output.append("  formed by the column vectors")
        output.append("")
        if abs(det) < 1e-10:
            output.append("  det = 0 → Collapses to plane (singular)")
        elif det > 0:
            output.append("  det > 0 → Right-handed system")
        else:
            output.append("  det < 0 → Left-handed system")

    else:
        output.append(f"For an {A.shape[0]}×{A.shape[1]} matrix:")
        output.append(f"  det(A) = {det:.4g}")
        output.append("")
        output.append("  |det(A)| = n-dimensional volume scaling factor")

    output.append("")
    output.append("Key Properties:")
    output.append("  • det = 0 ⟺ matrix is singular (non-invertible)")
    output.append("  • det ≠ 0 ⟺ matrix is invertible")
    output.append("  • det(AB) = det(A) × det(B)")
    output.append("  • det(A^T) = det(A)")

    return "\n".join(output)


def visualize_eigenvalues_2x2(A: np.ndarray) -> str:
    """Visualize eigenvalue calculation for 2×2 matrix.

    Args:
        A: 2×2 matrix

    Returns:
        String showing the calculation
    """
    if A.shape != (2, 2):
        return f"This visualization requires a 2×2 matrix, got {A.shape}"

    output = []
    output.append("Finding Eigenvalues for 2×2 Matrix:")
    output.append("")

    # Show matrix
    a, b = A[0, :]
    c, d = A[1, :]
    output.append(f"A = [{a:g}  {b:g}]")
    output.append(f"    [{c:g}  {d:g}]")
    output.append("")

    # Characteristic equation: det(A - λI) = 0
    output.append("Characteristic equation: det(A - λI) = 0")
    output.append("")
    output.append(f"det([{a:g}-λ    {b:g}  ]) = 0")
    output.append(f"    [{c:g}      {d:g}-λ])")
    output.append("")

    # Expand
    trace = a + d
    det = a * d - b * c

    output.append(f"({a:g}-λ)({d:g}-λ) - ({b:g})({c:g}) = 0")
    output.append("")
    output.append(f"λ² - {trace:g}λ + {det:g} = 0")
    output.append("")

    # Solve using quadratic formula
    discriminant = trace**2 - 4*det
    output.append(f"Using quadratic formula:")
    output.append(f"  λ = ({trace:g} ± √({discriminant:g})) / 2")
    output.append("")

    if discriminant >= 0:
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
        output.append(f"  λ₁ = {lambda1:.4g}")
        output.append(f"  λ₂ = {lambda2:.4g}")

        if abs(discriminant) < 1e-10:
            output.append("")
            output.append("  (Repeated eigenvalue)")
    else:
        real_part = trace / 2
        imag_part = np.sqrt(-discriminant) / 2
        output.append(f"  λ₁ = {real_part:.4g} + {imag_part:.4g}i")
        output.append(f"  λ₂ = {real_part:.4g} - {imag_part:.4g}i")
        output.append("")
        output.append("  (Complex eigenvalues)")

    return "\n".join(output)


def explain_eigenvectors(A: np.ndarray, eigenvalue: float) -> str:
    """Explain eigenvectors for a given eigenvalue.

    Args:
        A: Matrix
        eigenvalue: An eigenvalue of A

    Returns:
        String explaining how to find eigenvectors
    """
    output = []
    output.append(f"Finding Eigenvector for λ = {eigenvalue:.4g}:")
    output.append("")
    output.append("An eigenvector v satisfies: Av = λv")
    output.append("")
    output.append("Rearranging: (A - λI)v = 0")
    output.append("")

    # Compute A - λI
    A_shifted = A - eigenvalue * np.eye(len(A))

    output.append("Solve the homogeneous system:")
    for i, row in enumerate(A_shifted):
        terms = []
        for j, val in enumerate(row):
            if abs(val) > 1e-10:
                if j == 0:
                    terms.append(f"{val:g}x₁")
                else:
                    sign = "+" if val >= 0 else "-"
                    terms.append(f"{sign} {abs(val):g}x₂" if len(row) == 2 else f"{sign} {abs(val):g}x_{j+1}")

        if terms:
            output.append("  " + " ".join(terms) + " = 0")

    output.append("")
    output.append("Any non-zero solution v is an eigenvector!")

    return "\n".join(output)
