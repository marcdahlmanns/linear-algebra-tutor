"""Solve commands for advanced linear algebra operations."""

import typer
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from typing_extensions import Annotated

from ...core.solver import (
    GaussianEliminationSolver,
    RREFSolver,
    LinearSystemSolver,
    EigenvalueSolver,
    LUDecompositionSolver,
    QRDecompositionSolver,
    SVDSolver,
)

app = typer.Typer(help="Solve advanced linear algebra problems")
console = Console()


@app.command()
def gaussian(
    matrix: Annotated[
        str,
        typer.Argument(help="Matrix rows separated by semicolons, e.g., '1,2,3;4,5,6;7,8,9'"),
    ],
    augment: Annotated[
        str,
        typer.Option("--augment", "-b", help="Right-hand side vector for augmented matrix, e.g., '1,2,3'"),
    ] = None,
):
    """Perform Gaussian elimination to row echelon form."""
    try:
        # Parse matrix
        mat = _parse_matrix(matrix)

        # Parse augment if provided
        b_vec = None
        if augment:
            b_vec = np.array([float(x.strip()) for x in augment.split(",")])

        console.print(
            Panel.fit(
                "[bold cyan]Gaussian Elimination[/bold cyan]",
                border_style="cyan",
            )
        )

        # Solve
        solver = GaussianEliminationSolver()
        solution = solver.solve_with_steps(mat, b_vec)

        # Display steps
        for i, step in enumerate(solution.steps):
            console.print(f"\n[bold]Step {i+1}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")
            if step.intermediate_result:
                console.print(f"[green]{step.intermediate_result}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def rref(
    matrix: Annotated[
        str,
        typer.Argument(help="Matrix rows separated by semicolons"),
    ],
    augment: Annotated[
        str,
        typer.Option("--augment", "-b", help="Right-hand side vector"),
    ] = None,
):
    """Reduce matrix to reduced row echelon form (RREF)."""
    try:
        mat = _parse_matrix(matrix)

        b_vec = None
        if augment:
            b_vec = np.array([float(x.strip()) for x in augment.split(",")])

        console.print(
            Panel.fit(
                "[bold cyan]Row Reduction to RREF[/bold cyan]",
                border_style="cyan",
            )
        )

        solver = RREFSolver()
        solution = solver.solve_with_steps(mat, b_vec)

        for i, step in enumerate(solution.steps):
            console.print(f"\n[bold]Step {i+1}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")
            if step.intermediate_result:
                console.print(f"[green]{step.intermediate_result}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def linear_system(
    matrix: Annotated[
        str,
        typer.Argument(help="Coefficient matrix A"),
    ],
    vector: Annotated[
        str,
        typer.Argument(help="Right-hand side vector b"),
    ],
):
    """Solve linear system Ax = b."""
    try:
        A = _parse_matrix(matrix)
        b = np.array([float(x.strip()) for x in vector.split(",")])

        console.print(
            Panel.fit(
                "[bold cyan]Solve Linear System: Ax = b[/bold cyan]",
                border_style="cyan",
            )
        )

        solver = LinearSystemSolver()
        solution = solver.solve_with_steps(A, b)

        for i, step in enumerate(solution.steps):
            console.print(f"\n[bold]Step {i+1}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")

        if solution.final_answer is not None:
            console.print("\n[bold green]Solution:[/bold green]")
            console.print(f"x = {solution.final_answer}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def eigenvalues(
    matrix: Annotated[
        str,
        typer.Argument(help="Square matrix"),
    ],
):
    """Find eigenvalues and eigenvectors."""
    try:
        mat = _parse_matrix(matrix)

        if mat.shape[0] != mat.shape[1]:
            console.print(f"[red]Error: Matrix must be square, got {mat.shape}[/red]")
            return

        console.print(
            Panel.fit(
                "[bold cyan]Eigenvalue Decomposition[/bold cyan]",
                border_style="cyan",
            )
        )

        solver = EigenvalueSolver()
        solution = solver.solve_with_steps(mat)

        for i, step in enumerate(solution.steps):
            console.print(f"\n[bold]Step {i+1}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def lu(
    matrix: Annotated[
        str,
        typer.Argument(help="Square matrix to decompose"),
    ],
):
    """Compute LU decomposition: A = LU."""
    try:
        mat = _parse_matrix(matrix)

        if mat.shape[0] != mat.shape[1]:
            console.print(f"[red]Error: Matrix must be square, got {mat.shape}[/red]")
            return

        console.print(
            Panel.fit(
                "[bold cyan]LU Decomposition: A = LU[/bold cyan]",
                border_style="cyan",
            )
        )

        solver = LUDecompositionSolver()
        solution = solver.solve_with_steps(mat)

        for i, step in enumerate(solution.steps):
            console.print(f"\n[bold]Step {i+1}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")
            if step.intermediate_result:
                console.print(f"[green]{step.intermediate_result}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def qr(
    matrix: Annotated[
        str,
        typer.Argument(help="Matrix to decompose"),
    ],
):
    """Compute QR decomposition: A = QR."""
    try:
        mat = _parse_matrix(matrix)

        console.print(
            Panel.fit(
                "[bold cyan]QR Decomposition: A = QR[/bold cyan]",
                border_style="cyan",
            )
        )

        solver = QRDecompositionSolver()
        solution = solver.solve_with_steps(mat)

        for i, step in enumerate(solution.steps):
            console.print(f"\n[bold]Step {i+1}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def svd(
    matrix: Annotated[
        str,
        typer.Argument(help="Matrix to decompose"),
    ],
):
    """Compute Singular Value Decomposition: A = UΣVᵀ."""
    try:
        mat = _parse_matrix(matrix)

        console.print(
            Panel.fit(
                "[bold cyan]Singular Value Decomposition: A = UΣVᵀ[/bold cyan]",
                border_style="cyan",
            )
        )

        solver = SVDSolver()
        solution = solver.solve_with_steps(mat)

        for i, step in enumerate(solution.steps):
            console.print(f"\n[bold]Step {i+1}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def demo():
    """Run a demonstration of advanced solvers."""
    console.print(
        Panel.fit(
            "[bold cyan]Advanced Solvers Demo[/bold cyan]\n\n"
            "Demonstrating step-by-step solutions",
            border_style="cyan",
        )
    )

    # Demo 1: Gaussian Elimination
    console.print("\n[bold]1. Gaussian Elimination[/bold]")
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    console.print(f"Solve system with augmented matrix [A|b]")
    solver = GaussianEliminationSolver()
    solution = solver.solve_with_steps(A, b)

    console.print(f"\n[green]Final REF:[/green]")
    console.print(Panel(solution.steps[-1].mathematical_expression, border_style="green"))

    # Demo 2: Eigenvalues (2x2)
    console.print("\n[bold]2. Eigenvalues (2×2 Matrix)[/bold]")
    A2 = np.array([[4, -2], [1, 1]])

    console.print(f"Finding eigenvalues of:\n{A2}")
    eigen_solver = EigenvalueSolver()
    eigen_solution = eigen_solver.solve_with_steps(A2)

    eigenvalues = eigen_solution.final_answer["eigenvalues"]
    console.print(f"\n[green]Eigenvalues: {eigenvalues}[/green]")

    # Demo 3: LU Decomposition
    console.print("\n[bold]3. LU Decomposition[/bold]")
    A3 = np.array([[2, 3], [4, 9]], dtype=float)

    console.print(f"Decompose:\n{A3}")
    lu_solver = LUDecompositionSolver()
    lu_solution = lu_solver.solve_with_steps(A3)

    L = lu_solution.final_answer["L"]
    U = lu_solution.final_answer["U"]
    console.print(f"\n[green]L =\n{L}[/green]")
    console.print(f"\n[green]U =\n{U}[/green]")

    console.print("\n[bold green]✓ Demo complete![/bold green]")
    console.print("\nTry these commands:")
    console.print("  [cyan]linalg-tutor solve gaussian '2,1,-1;-3,-1,2;-2,1,2' -b '8,-11,-3'[/cyan]")
    console.print("  [cyan]linalg-tutor solve eigenvalues '4,-2;1,1'[/cyan]")
    console.print("  [cyan]linalg-tutor solve lu '2,3;4,9'[/cyan]")
    console.print("  [cyan]linalg-tutor solve qr '1,1;1,0;0,1'[/cyan]")


def _parse_matrix(matrix_str: str) -> np.ndarray:
    """Parse matrix from string format."""
    rows = []
    for row_str in matrix_str.split(";"):
        row_values = [float(x.strip()) for x in row_str.split(",")]
        rows.append(row_values)
    return np.array(rows)
