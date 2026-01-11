"""Visualization commands for vectors and matrices."""

import typer
import numpy as np
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from ...visualization import (
    print_vector_visualization,
    print_matrix_visualization,
    visualize_vector_addition_2d,
    visualize_dot_product,
    explain_dot_product_geometry,
    visualize_matrix_multiply_step,
    visualize_determinant_2x2,
    visualize_identity_matrix,
    explain_vector_magnitude,
    explain_orthogonality,
    explain_projection,
    create_transformation_table,
)

app = typer.Typer(help="Visualize vectors and matrices")
console = Console()


@app.command()
def vector(
    components: Annotated[
        str, typer.Argument(help="Vector components, e.g., '1,2,3' or '4,-2'")
    ],
):
    """Visualize a vector with components and optional 2D plot."""
    try:
        # Parse components
        values = [float(x.strip()) for x in components.split(",")]
        vec = np.array(values)

        console.print(
            Panel.fit(
                f"[bold cyan]Vector Visualization[/bold cyan]",
                border_style="cyan",
            )
        )

        # Component visualization
        print_vector_visualization(vec, label="v", show_2d=(len(vec) == 2))

        # Magnitude explanation
        console.print("\n")
        magnitude_viz = explain_vector_magnitude(vec, label="v")
        console.print(Panel(magnitude_viz, title="Magnitude", border_style="green"))

    except ValueError as e:
        console.print(f"[red]Error: Invalid vector format. Use comma-separated numbers like '1,2,3'[/red]")
        console.print(f"[red]Details: {e}[/red]")


@app.command()
def matrix(
    rows: Annotated[
        str,
        typer.Argument(
            help="Matrix rows separated by semicolons, e.g., '1,2;3,4' for [[1,2],[3,4]]"
        ),
    ],
):
    """Visualize a matrix with properties."""
    try:
        # Parse matrix
        row_list = []
        for row_str in rows.split(";"):
            row_values = [float(x.strip()) for x in row_str.split(",")]
            row_list.append(row_values)

        mat = np.array(row_list)

        console.print(
            Panel.fit(
                f"[bold cyan]Matrix Visualization[/bold cyan]",
                border_style="cyan",
            )
        )

        # Matrix display with properties
        print_matrix_visualization(mat, title="Matrix A", show_properties=True)

    except ValueError as e:
        console.print(
            f"[red]Error: Invalid matrix format. Use 'row1;row2' with comma-separated values[/red]"
        )
        console.print(f"[red]Example: '1,2;3,4' for a 2×2 matrix[/red]")
        console.print(f"[red]Details: {e}[/red]")


@app.command()
def vector_add(
    v1: Annotated[str, typer.Argument(help="First vector, e.g., '1,2'")],
    v2: Annotated[str, typer.Argument(help="Second vector, e.g., '3,4'")],
):
    """Visualize vector addition (2D only for geometric plot)."""
    try:
        vec1 = np.array([float(x.strip()) for x in v1.split(",")])
        vec2 = np.array([float(x.strip()) for x in v2.split(",")])

        if len(vec1) != len(vec2):
            console.print("[red]Error: Vectors must have the same dimension[/red]")
            return

        console.print(
            Panel.fit(
                f"[bold cyan]Vector Addition: v₁ + v₂[/bold cyan]",
                border_style="cyan",
            )
        )

        # Show result
        result = vec1 + vec2
        console.print(f"\nv₁ = {vec1}")
        console.print(f"v₂ = {vec2}")
        console.print(f"v₁ + v₂ = {result}\n")

        # 2D visualization
        if len(vec1) == 2:
            viz = visualize_vector_addition_2d(vec1, vec2)
            console.print(Panel(viz, title="Geometric Visualization", border_style="green"))
        else:
            console.print("[yellow]Geometric plot only available for 2D vectors[/yellow]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def dot_product(
    v1: Annotated[str, typer.Argument(help="First vector, e.g., '1,2,3'")],
    v2: Annotated[str, typer.Argument(help="Second vector, e.g., '4,5,6'")],
):
    """Visualize dot product with geometric interpretation."""
    try:
        vec1 = np.array([float(x.strip()) for x in v1.split(",")])
        vec2 = np.array([float(x.strip()) for x in v2.split(",")])

        if len(vec1) != len(vec2):
            console.print("[red]Error: Vectors must have the same dimension[/red]")
            return

        console.print(
            Panel.fit(
                f"[bold cyan]Dot Product: v₁ · v₂[/bold cyan]",
                border_style="cyan",
            )
        )

        # Calculation
        viz = visualize_dot_product(vec1, vec2)
        console.print(Panel(viz, title="Calculation", border_style="yellow"))

        # Geometric meaning
        console.print("\n")
        geo_viz = explain_dot_product_geometry(vec1, vec2)
        console.print(Panel(geo_viz, title="Geometric Interpretation", border_style="green"))

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def matrix_multiply(
    A: Annotated[str, typer.Argument(help="First matrix, e.g., '1,2;3,4'")],
    B: Annotated[str, typer.Argument(help="Second matrix, e.g., '5,6;7,8'")],
):
    """Visualize matrix multiplication with step-by-step example."""
    try:
        # Parse matrices
        mat_a = np.array(
            [[float(x.strip()) for x in row.split(",")] for row in A.split(";")]
        )
        mat_b = np.array(
            [[float(x.strip()) for x in row.split(",")] for row in B.split(";")]
        )

        if mat_a.shape[1] != mat_b.shape[0]:
            console.print(
                f"[red]Error: Incompatible dimensions. A is {mat_a.shape}, B is {mat_b.shape}[/red]"
            )
            console.print(
                f"[red]For A×B, number of columns in A must equal number of rows in B[/red]"
            )
            return

        console.print(
            Panel.fit(
                f"[bold cyan]Matrix Multiplication: A × B[/bold cyan]",
                border_style="cyan",
            )
        )

        # Show matrices
        print_matrix_visualization(mat_a, title="Matrix A", show_properties=False)
        console.print("\n×\n")
        print_matrix_visualization(mat_b, title="Matrix B", show_properties=False)

        # Show result
        result = mat_a @ mat_b
        console.print("\n=\n")
        print_matrix_visualization(result, title="Result", show_properties=False)

        # Show step-by-step for first element
        console.print("\n")
        step_viz = visualize_matrix_multiply_step(mat_a, mat_b, 0, 0)
        console.print(
            Panel(step_viz, title="Example: Computing C[0,0]", border_style="yellow")
        )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def determinant(
    matrix: Annotated[
        str, typer.Argument(help="2×2 matrix, e.g., '1,2;3,4'")
    ],
):
    """Visualize determinant calculation for 2×2 matrix."""
    try:
        mat = np.array(
            [[float(x.strip()) for x in row.split(",")] for row in matrix.split(";")]
        )

        if mat.shape != (2, 2):
            console.print(
                f"[red]Error: Expected 2×2 matrix, got {mat.shape[0]}×{mat.shape[1]}[/red]"
            )
            return

        console.print(
            Panel.fit(
                f"[bold cyan]Determinant Calculation[/bold cyan]",
                border_style="cyan",
            )
        )

        viz = visualize_determinant_2x2(mat)
        console.print(Panel(viz, border_style="green"))

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def identity(
    size: Annotated[int, typer.Argument(help="Size of identity matrix")] = 3,
):
    """Visualize identity matrix."""
    if size < 1 or size > 10:
        console.print("[red]Error: Size must be between 1 and 10[/red]")
        return

    console.print(
        Panel.fit(
            f"[bold cyan]Identity Matrix I_{size}[/bold cyan]",
            border_style="cyan",
        )
    )

    viz = visualize_identity_matrix(size)
    console.print(viz)


@app.command()
def orthogonal(
    v1: Annotated[str, typer.Argument(help="First vector, e.g., '1,0'")],
    v2: Annotated[str, typer.Argument(help="Second vector, e.g., '0,1'")],
):
    """Check if two vectors are orthogonal."""
    try:
        vec1 = np.array([float(x.strip()) for x in v1.split(",")])
        vec2 = np.array([float(x.strip()) for x in v2.split(",")])

        if len(vec1) != len(vec2):
            console.print("[red]Error: Vectors must have the same dimension[/red]")
            return

        console.print(
            Panel.fit(
                f"[bold cyan]Orthogonality Check[/bold cyan]",
                border_style="cyan",
            )
        )

        viz = explain_orthogonality(vec1, vec2)
        console.print(Panel(viz, border_style="green"))

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def projection(
    v: Annotated[str, typer.Argument(help="Vector to project, e.g., '3,4'")],
    onto: Annotated[str, typer.Argument(help="Vector to project onto, e.g., '1,0'")],
):
    """Visualize vector projection."""
    try:
        vec = np.array([float(x.strip()) for x in v.split(",")])
        onto_vec = np.array([float(x.strip()) for x in onto.split(",")])

        if len(vec) != len(onto_vec):
            console.print("[red]Error: Vectors must have the same dimension[/red]")
            return

        console.print(
            Panel.fit(
                f"[bold cyan]Vector Projection[/bold cyan]",
                border_style="cyan",
            )
        )

        viz = explain_projection(vec, onto_vec)
        console.print(Panel(viz, border_style="green"))

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def demo():
    """Run a demonstration of various visualizations."""
    console.print(
        Panel.fit(
            "[bold cyan]Visualization Demo[/bold cyan]\n\n"
            "Demonstrating various visualization features",
            border_style="cyan",
        )
    )

    # Vector demo
    console.print("\n[bold]1. Vector Visualization[/bold]")
    vec = np.array([3, 4])
    print_vector_visualization(vec, label="v", show_2d=True)

    # Matrix demo
    console.print("\n[bold]2. Matrix Visualization[/bold]")
    mat = np.array([[1, 2], [3, 4]])
    print_matrix_visualization(mat, title="Matrix A", show_properties=True)

    # Dot product demo
    console.print("\n[bold]3. Dot Product[/bold]")
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    viz = visualize_dot_product(v1, v2)
    console.print(Panel(viz, border_style="cyan"))

    # Determinant demo
    console.print("\n[bold]4. Determinant[/bold]")
    mat2 = np.array([[2, 3], [1, 4]])
    det_viz = visualize_determinant_2x2(mat2)
    console.print(Panel(det_viz, border_style="green"))

    console.print("\n[bold green]✓ Demo complete![/bold green]")
    console.print("\nTry these commands:")
    console.print("  [cyan]linalg-tutor visualize vector 1,2,3[/cyan]")
    console.print("  [cyan]linalg-tutor visualize matrix '1,2;3,4'[/cyan]")
    console.print("  [cyan]linalg-tutor visualize dot-product 1,2 3,4[/cyan]")
