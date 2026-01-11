"""Main CLI application."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional

app = typer.Typer(
    name="linalg-tutor",
    help="Interactive Linear Algebra Teaching Tool",
    add_completion=False,
)

console = Console()

# Register command modules
from .commands import exercise, visualize, solve, generate

app.add_typer(exercise.app, name="exercise", help="Practice exercises")
app.add_typer(visualize.app, name="visualize", help="Visualize vectors and matrices")
app.add_typer(solve.app, name="solve", help="Solve advanced linear algebra problems")
app.add_typer(generate.app, name="generate", help="Generate random exercises for infinite practice")


@app.command()
def start():
    """Start an interactive learning session."""
    console.print(Panel.fit(
        "[bold cyan]Welcome to Linear Algebra Tutor![/bold cyan]\n\n"
        "Learn linear algebra through interactive exercises with\n"
        "step-by-step solutions and visual demonstrations.",
        title="Linear Algebra Tutor",
        border_style="cyan"
    ))

    console.print("\n[bold]Quick Start:[/bold]")
    console.print("  [green]linalg-tutor exercise practice vectors[/green]     - Practice vector exercises")
    console.print("  [green]linalg-tutor exercise practice vectors -n 10[/green] - Practice 10 exercises")
    console.print("  [green]linalg-tutor exercise list vectors[/green]        - List all vector exercises")
    console.print("\n[bold]Visualizations:[/bold]")
    console.print("  [green]linalg-tutor visualize vector 1,2,3[/green]       - Visualize a vector")
    console.print("  [green]linalg-tutor visualize matrix '1,2;3,4'[/green]   - Visualize a matrix")
    console.print("  [green]linalg-tutor visualize dot-product 1,2 3,4[/green] - Dot product visualization")
    console.print("\n[bold]Advanced Solvers:[/bold]")
    console.print("  [green]linalg-tutor solve gaussian '1,2;3,4' -b '5,6'[/green] - Gaussian elimination")
    console.print("  [green]linalg-tutor solve eigenvalues '1,2;3,4'[/green] - Find eigenvalues")
    console.print("  [green]linalg-tutor solve lu '2,3;4,9'[/green]         - LU decomposition")
    console.print("\n[bold]Random Generators:[/bold]")
    console.print("  [green]linalg-tutor generate practice vector_add[/green]  - Infinite vector addition practice")
    console.print("  [green]linalg-tutor generate list-generators[/green]     - Show all 14 generators")
    console.print("  [green]linalg-tutor generate all-demo[/green]            - Demo all generators")
    console.print("\n[bold]Other Commands:[/bold]")
    console.print("  [green]linalg-tutor topics[/green]  - List all topics")
    console.print("  [green]linalg-tutor demo[/green]    - Run a demo exercise")
    console.print("  [green]linalg-tutor --help[/green]  - Show all commands")


@app.command()
def topics():
    """List all available topics."""
    table = Table(title="Linear Algebra Topics", show_header=True, header_style="bold magenta")

    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Topic", style="green")
    table.add_column("Description", style="white")

    # Hardcoded topics for now - will be dynamic later
    topics_data = [
        ("01", "Vectors", "Vector operations, dot product, cross product"),
        ("02", "Matrices", "Matrix operations, multiplication, inverse"),
        ("03", "Linear Systems", "Gaussian elimination, row reduction"),
        ("04", "Vector Spaces", "Subspaces, basis, dimension"),
        ("05", "Orthogonality", "Orthogonal projections, Gram-Schmidt"),
        ("06", "Determinants", "Properties, cofactor expansion"),
        ("07", "Eigenvalues", "Characteristic equation, diagonalization"),
        ("08", "Transformations", "Linear transformations, kernel, range"),
        ("09", "Decompositions", "SVD, QR, LU decomposition"),
        ("10", "Applications", "PCA, computer graphics, optimization"),
    ]

    for topic_id, name, description in topics_data:
        table.add_row(topic_id, name, description)

    console.print(table)


@app.command()
def demo():
    """Run a demo exercise to test the system."""
    from ..core.exercises import ComputationalExercise, ExerciseDifficulty
    import numpy as np

    console.print(Panel.fit(
        "[bold cyan]Demo Exercise: Vector Addition[/bold cyan]",
        border_style="cyan"
    ))

    # Create a simple vector addition exercise
    exercise = ComputationalExercise(
        exercise_id="demo_vec_add_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add the vectors v = [2, 3] and w = [1, -1]",
        operation="vector_add",
        inputs={"v": np.array([2, 3]), "w": np.array([1, -1])},
        expected_output=np.array([3, 2]),
        hints=[
            "Add component-wise: [v₁+w₁, v₂+w₂]",
            "First component: 2 + 1 = ?",
            "Second component: 3 + (-1) = ?",
        ],
        tags=["vector_addition", "basic"],
    )

    console.print(f"\n[bold]{exercise.question}[/bold]\n")

    # Show hints
    console.print("[yellow]Hints available:[/yellow]")
    for i, hint in enumerate(exercise.hints, 1):
        console.print(f"  {i}. {hint}")

    # Get solution
    console.print("\n[green]Solution:[/green]")
    solution_text = exercise.get_solution(show_steps=True)
    console.print(Panel(solution_text, border_style="green"))

    # Test answer checking
    console.print("\n[cyan]Testing answer validation:[/cyan]")

    # Correct answer
    result1 = exercise.check_answer([3, 2])
    console.print(f"Answer [3, 2]: {'✓ Correct!' if result1.correct else '✗ Incorrect'}")

    # Wrong answer
    result2 = exercise.check_answer([4, 2])
    console.print(f"Answer [4, 2]: {'✓ Correct' if result2.correct else '✗ Incorrect'}")
    console.print(f"  Feedback: {result2.feedback}")


@app.command()
def version():
    """Show version information."""
    console.print("[bold cyan]Linear Algebra Tutor[/bold cyan] version [green]0.1.0[/green]")
    console.print("A comprehensive CLI tool for learning linear algebra")


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
