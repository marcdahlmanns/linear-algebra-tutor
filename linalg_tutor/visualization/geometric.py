"""Geometric interpretations and visualizations."""

import numpy as np
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def explain_vector_magnitude(vector: np.ndarray, label: str = "v") -> str:
    """Explain the geometric meaning of vector magnitude.

    Args:
        vector: Input vector
        label: Label for the vector

    Returns:
        Explanation string
    """
    lines = []
    lines.append(f"Vector Magnitude (Length):")
    lines.append("")
    lines.append(f"{label} = {vector}")
    lines.append("")

    # Formula
    if len(vector) == 2:
        x, y = vector[0], vector[1]
        lines.append("For 2D vector: |v| = √(x² + y²)")
        lines.append("")
        lines.append(f"  |{label}| = √({x}² + {y}²)")
        lines.append(f"      = √({x**2:.4g} + {y**2:.4g})")
        lines.append(f"      = √{x**2 + y**2:.4g}")
    elif len(vector) == 3:
        x, y, z = vector[0], vector[1], vector[2]
        lines.append("For 3D vector: |v| = √(x² + y² + z²)")
        lines.append("")
        lines.append(f"  |{label}| = √({x}² + {y}² + {z}²)")
        lines.append(f"      = √({x**2:.4g} + {y**2:.4g} + {z**2:.4g})")
        lines.append(f"      = √{x**2 + y**2 + z**2:.4g}")
    else:
        lines.append("General formula: |v| = √(v₁² + v₂² + ... + vₙ²)")
        squares = [f"{val}²" for val in vector]
        lines.append(f"  |{label}| = √({' + '.join(squares)})")

    magnitude = np.linalg.norm(vector)
    lines.append(f"      = {magnitude:.4g}")
    lines.append("")
    lines.append("Geometric meaning: Distance from origin to point")

    return '\n'.join(lines)


def explain_dot_product_geometry(v1: np.ndarray, v2: np.ndarray) -> str:
    """Explain the geometric interpretation of dot product.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Explanation string
    """
    lines = []
    lines.append("Dot Product: Geometric Interpretation")
    lines.append("")

    dot = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    lines.append(f"v₁ = {v1}")
    lines.append(f"v₂ = {v2}")
    lines.append("")
    lines.append(f"v₁ · v₂ = {dot:.4g}")
    lines.append("")

    if mag_v1 > 0 and mag_v2 > 0:
        cos_theta = dot / (mag_v1 * mag_v2)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        theta_deg = np.degrees(theta)

        lines.append("Geometric formula:")
        lines.append("  v₁ · v₂ = |v₁| |v₂| cos(θ)")
        lines.append("")
        lines.append(f"  |v₁| = {mag_v1:.4g}")
        lines.append(f"  |v₂| = {mag_v2:.4g}")
        lines.append(f"  θ = {theta_deg:.2f}°")
        lines.append(f"  cos(θ) = {cos_theta:.4g}")
        lines.append("")

        # Interpretation
        lines.append("Interpretation:")
        if abs(theta_deg) < 1:
            lines.append("  ✓ Vectors point in the SAME direction (θ ≈ 0°)")
        elif abs(theta_deg - 180) < 1:
            lines.append("  ✓ Vectors point in OPPOSITE directions (θ ≈ 180°)")
        elif abs(theta_deg - 90) < 1:
            lines.append("  ✓ Vectors are PERPENDICULAR (θ ≈ 90°)")
        elif theta_deg < 90:
            lines.append(f"  ✓ Vectors form an ACUTE angle ({theta_deg:.1f}°)")
        else:
            lines.append(f"  ✓ Vectors form an OBTUSE angle ({theta_deg:.1f}°)")

        lines.append("")
        if abs(dot) < 1e-10:
            lines.append("  ⊥ Dot product is 0 → Vectors are orthogonal")
        elif dot > 0:
            lines.append("  → Dot product > 0 → Vectors point in similar direction")
        else:
            lines.append("  → Dot product < 0 → Vectors point in opposite directions")

    return '\n'.join(lines)


def explain_matrix_as_transformation(matrix: np.ndarray) -> str:
    """Explain matrix as a linear transformation.

    Args:
        matrix: Transformation matrix (must be square)

    Returns:
        Explanation string
    """
    if matrix.shape[0] != matrix.shape[1]:
        return "Error: Matrix must be square for transformation interpretation"

    lines = []
    lines.append("Matrix as Linear Transformation")
    lines.append("")

    n = matrix.shape[0]
    lines.append(f"This {n}×{n} matrix transforms vectors in ℝ^{n}")
    lines.append("")

    # Standard basis vectors
    lines.append("Effect on standard basis vectors:")
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        result = matrix @ e

        lines.append(f"  e_{i+1} = {e} → {result}")

    lines.append("")

    # Special transformations
    det = np.linalg.det(matrix)
    lines.append(f"Determinant: {det:.4g}")

    if abs(det) < 1e-10:
        lines.append("  ⚠️  det = 0 → Transformation COLLAPSES space (not invertible)")
    elif det < 0:
        lines.append("  ↻ det < 0 → Transformation includes REFLECTION")
    else:
        lines.append("  ✓ det > 0 → Transformation preserves orientation")

    lines.append("")
    lines.append(f"Volume scaling factor: {abs(det):.4g}")

    # Check for special types
    if np.allclose(matrix, np.eye(n)):
        lines.append("")
        lines.append("Type: IDENTITY transformation (no change)")
    elif np.allclose(matrix @ matrix.T, np.eye(n)):
        lines.append("")
        lines.append("Type: ORTHOGONAL transformation (rotation/reflection)")
    elif np.allclose(matrix, matrix.T):
        lines.append("")
        lines.append("Type: SYMMETRIC transformation")

    return '\n'.join(lines)


def explain_linear_combination(vectors: list[np.ndarray], coefficients: list[float]) -> str:
    """Explain linear combination of vectors.

    Args:
        vectors: List of vectors
        coefficients: List of scalar coefficients

    Returns:
        Explanation string
    """
    if len(vectors) != len(coefficients):
        return "Error: Number of vectors and coefficients must match"

    lines = []
    lines.append("Linear Combination")
    lines.append("")

    # Show the combination
    terms = []
    for i, (coef, vec) in enumerate(zip(coefficients, vectors)):
        label = f"v_{i+1}"
        lines.append(f"{label} = {vec}")
        terms.append(f"{coef:.4g}{label}")

    lines.append("")
    lines.append(f"Combination: {' + '.join(terms)}")
    lines.append("")

    # Calculate step by step
    lines.append("Step-by-step:")
    scaled_vectors = []
    for i, (coef, vec) in enumerate(zip(coefficients, vectors)):
        scaled = coef * vec
        scaled_vectors.append(scaled)
        lines.append(f"  {coef:.4g} × {vec} = {scaled}")

    # Sum
    result = sum(scaled_vectors)
    lines.append("")
    lines.append(f"Sum: {result}")
    lines.append("")
    lines.append("Geometric meaning: Scaled and added vectors")

    return '\n'.join(lines)


def explain_orthogonality(v1: np.ndarray, v2: np.ndarray) -> str:
    """Explain orthogonality between vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Explanation string
    """
    lines = []
    lines.append("Orthogonality Check")
    lines.append("")

    lines.append(f"v₁ = {v1}")
    lines.append(f"v₂ = {v2}")
    lines.append("")

    dot = np.dot(v1, v2)
    lines.append(f"v₁ · v₂ = {dot:.6g}")
    lines.append("")

    tolerance = 1e-10
    is_orthogonal = abs(dot) < tolerance

    if is_orthogonal:
        lines.append("✓ ORTHOGONAL: v₁ · v₂ = 0")
        lines.append("")
        lines.append("Geometric meaning:")
        lines.append("  • Vectors are perpendicular (90° angle)")
        lines.append("  • They point in completely independent directions")
        lines.append("  • One vector doesn't \"contribute\" to the other's direction")
    else:
        lines.append("✗ NOT ORTHOGONAL: v₁ · v₂ ≠ 0")
        lines.append("")

        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)

        if mag_v1 > 0 and mag_v2 > 0:
            cos_theta = dot / (mag_v1 * mag_v2)
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            theta_deg = np.degrees(theta)

            lines.append(f"Angle between vectors: {theta_deg:.2f}°")
            lines.append("")
            lines.append("For orthogonality, angle must be exactly 90°")

    return '\n'.join(lines)


def explain_projection(v: np.ndarray, onto: np.ndarray) -> str:
    """Explain vector projection.

    Args:
        v: Vector to project
        onto: Vector to project onto

    Returns:
        Explanation string with projection calculation
    """
    lines = []
    lines.append("Vector Projection")
    lines.append("")

    lines.append(f"Project v = {v}")
    lines.append(f"onto u = {onto}")
    lines.append("")

    # Calculate projection
    dot_product = np.dot(v, onto)
    onto_squared = np.dot(onto, onto)

    if onto_squared == 0:
        lines.append("Error: Cannot project onto zero vector")
        return '\n'.join(lines)

    scalar = dot_product / onto_squared
    projection = scalar * onto

    lines.append("Formula: proj_u(v) = ((v·u) / (u·u)) × u")
    lines.append("")
    lines.append(f"  v · u = {dot_product:.4g}")
    lines.append(f"  u · u = {onto_squared:.4g}")
    lines.append(f"  (v·u)/(u·u) = {scalar:.4g}")
    lines.append("")
    lines.append(f"  proj_u(v) = {scalar:.4g} × {onto}")
    lines.append(f"           = {projection}")
    lines.append("")
    lines.append("Geometric meaning:")
    lines.append("  • The 'shadow' of v cast onto u")
    lines.append("  • Component of v in the direction of u")

    # Calculate orthogonal component
    orthogonal = v - projection
    lines.append("")
    lines.append(f"Orthogonal component: v - proj_u(v) = {orthogonal}")
    lines.append("")
    lines.append("Note: v = proj_u(v) + orthogonal component")

    return '\n'.join(lines)


def create_transformation_table(matrix: np.ndarray, test_vectors: Optional[list[np.ndarray]] = None) -> Table:
    """Create a table showing how a matrix transforms vectors.

    Args:
        matrix: Transformation matrix
        test_vectors: Optional list of test vectors (default: standard basis)

    Returns:
        Rich Table showing transformations
    """
    n = matrix.shape[0]

    if test_vectors is None:
        # Use standard basis vectors
        test_vectors = [np.eye(n)[i] for i in range(n)]

    table = Table(title="Linear Transformation", show_header=True, header_style="bold cyan")
    table.add_column("Input Vector", style="green")
    table.add_column("→", style="dim")
    table.add_column("Output Vector", style="yellow")
    table.add_column("Change", style="magenta")

    for vec in test_vectors:
        result = matrix @ vec
        change = np.linalg.norm(result - vec)

        table.add_row(
            str(vec),
            "→",
            str(result),
            f"{change:.4g}"
        )

    return table
