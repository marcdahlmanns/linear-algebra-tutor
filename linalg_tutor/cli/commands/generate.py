"""Commands for generating random exercises."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from ...core.generators import get_generator, GeneratorConfig, GENERATOR_REGISTRY
from ...core.exercises import ExerciseDifficulty
from ..ui.prompts import ExercisePrompt
from ...core.progress import ProgressTracker

app = typer.Typer(help="Generate random exercises for infinite practice")
console = Console()


@app.command()
def list_generators():
    """List all available exercise generators."""
    table = Table(title="Available Exercise Generators", show_header=True, header_style="bold cyan")

    table.add_column("Generator Name", style="green", no_wrap=True)
    table.add_column("Topic", style="yellow")
    table.add_column("Description", style="white")

    generators_info = [
        ("vector_add", "Vectors", "Vector addition (random dimensions)"),
        ("vector_scalar", "Vectors", "Scalar multiplication of vectors"),
        ("dot_product", "Vectors", "Dot product of two vectors"),
        ("vector_norm", "Vectors", "Vector magnitude/norm calculation"),
        ("cross_product", "Vectors", "Cross product (3D only)"),
        ("matrix_add", "Matrices", "Matrix addition (random sizes)"),
        ("matrix_scalar", "Matrices", "Scalar multiplication of matrices"),
        ("matrix_multiply", "Matrices", "Matrix multiplication (compatible sizes)"),
        ("matrix_transpose", "Matrices", "Matrix transpose"),
        ("determinant", "Matrices", "Determinant of 2×2 matrix"),
        ("matrix_inverse", "Matrices", "Matrix inverse (2×2)"),
        ("linear_system", "Linear Systems", "Solvable linear system Ax = b"),
        ("triangular_system", "Linear Systems", "Triangular system (easier)"),
        ("simple_2x2_system", "Linear Systems", "Simple 2×2 system"),
    ]

    for name, topic, description in generators_info:
        table.add_row(name, topic, description)

    console.print(table)
    console.print("\n[bold]Usage:[/bold]")
    console.print("  [green]linalg-tutor generate practice <generator-name>[/green]")
    console.print("  [green]linalg-tutor generate demo <generator-name>[/green]")


@app.command()
def practice(
    generator_name: Annotated[str, typer.Argument(help="Generator name (e.g., 'vector_add')")],
    count: Annotated[int, typer.Option("--count", "-n", help="Number of exercises")] = 5,
    difficulty: Annotated[str, typer.Option("--difficulty", "-d", help="Difficulty level")] = "practice",
    dimension: Annotated[int, typer.Option("--dim", help="Force specific dimension")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility")] = None,
):
    """Practice with randomly generated exercises."""
    # Validate generator
    if generator_name not in GENERATOR_REGISTRY:
        console.print(f"[red]Unknown generator: {generator_name}[/red]")
        console.print(f"[yellow]Available generators:[/yellow] {', '.join(GENERATOR_REGISTRY.keys())}")
        console.print("[cyan]Run 'linalg-tutor generate list-generators' to see all options[/cyan]")
        raise typer.Exit(1)

    # Parse difficulty
    try:
        diff_level = ExerciseDifficulty(difficulty.lower())
    except ValueError:
        console.print(f"[red]Invalid difficulty: {difficulty}[/red]")
        console.print("[yellow]Valid options: practice, application, challenge[/yellow]")
        raise typer.Exit(1)

    # Create generator config
    config = GeneratorConfig(difficulty=diff_level, seed=seed)
    if dimension is not None:
        config.min_dimension = dimension
        config.max_dimension = dimension

    # Get generator
    generator = get_generator(generator_name, config)

    # Display header
    console.print(
        Panel.fit(
            f"[bold cyan]Generated Practice: {generator_name}[/bold cyan]\n"
            f"Difficulty: {difficulty}\n"
            f"Exercises: {count}",
            border_style="cyan",
        )
    )

    # Initialize progress tracker
    tracker = ProgressTracker()

    # Practice session
    completed = 0
    correct = 0
    total_time = 0

    try:
        for i in range(count):
            # Generate exercise
            exercise = generator.generate()

            # Run interactive prompt
            prompt = ExercisePrompt(exercise, tracker, current_exercise=i+1, total_exercises=count)
            result = prompt.run()

            if result.correct:
                correct += 1
            completed += 1
            total_time += result.time_spent
    except KeyboardInterrupt:
        # User interrupted session
        console.print("\n[yellow]Session interrupted. Showing summary of completed exercises...[/yellow]\n")

    # Session summary (even if interrupted)
    if completed == 0:
        console.print("[yellow]No exercises completed.[/yellow]")
        return

    console.print("\n[bold]━━━ Session Summary ━━━[/bold]\n")

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Exercises Completed", str(completed))
    summary_table.add_row("Correct", str(correct))
    summary_table.add_row("Incorrect", str(completed - correct))
    summary_table.add_row("Accuracy", f"{(correct/completed*100) if completed > 0 else 0:.1f}%")
    summary_table.add_row("Total Time", f"{total_time:.1f}s")
    summary_table.add_row("Average Time", f"{total_time/completed if completed > 0 else 0:.1f}s per exercise")

    console.print(summary_table)


@app.command()
def demo(
    generator_name: Annotated[str, typer.Argument(help="Generator name")],
    count: Annotated[int, typer.Option("--count", "-n", help="Number of examples")] = 3,
):
    """Generate and display example exercises without practicing."""
    if generator_name not in GENERATOR_REGISTRY:
        console.print(f"[red]Unknown generator: {generator_name}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold cyan]Generator Demo: {generator_name}[/bold cyan]",
            border_style="cyan",
        )
    )

    # Get generator
    generator = get_generator(generator_name)

    # Generate and display examples
    for i in range(count):
        console.print(f"\n[bold]Example {i+1}:[/bold]")

        exercise = generator.generate()

        console.print(Panel(exercise.question, title="Question", border_style="yellow"))

        # Show answer
        answer = exercise.get_correct_answer()
        console.print(f"[green]Answer: {answer}[/green]")

        # Show first hint
        if exercise.hints:
            console.print(f"[dim]Hint: {exercise.hints[0]}[/dim]")

        console.print()


@app.command()
def batch(
    generator_name: Annotated[str, typer.Argument(help="Generator name")],
    count: Annotated[int, typer.Option("--count", "-n", help="Number to generate")] = 10,
    output: Annotated[str, typer.Option("--output", "-o", help="Output file (JSON)")] = None,
):
    """Generate a batch of exercises and optionally save to file."""
    if generator_name not in GENERATOR_REGISTRY:
        console.print(f"[red]Unknown generator: {generator_name}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Generating {count} exercises using '{generator_name}'...[/cyan]")

    generator = get_generator(generator_name)
    exercises = generator.generate_batch(count)

    console.print(f"[green]✓ Generated {len(exercises)} exercises[/green]")

    if output:
        # Save to file (would need JSON serialization)
        console.print(f"[yellow]Saving to file not yet implemented[/yellow]")
    else:
        # Display summary
        console.print("\n[bold]Generated Exercises:[/bold]")
        for i, ex in enumerate(exercises[:5], 1):  # Show first 5
            console.print(f"{i}. {ex.question[:60]}...")

        if len(exercises) > 5:
            console.print(f"... and {len(exercises) - 5} more")


@app.command()
def all_demo():
    """Run a quick demo of all generators."""
    console.print(
        Panel.fit(
            "[bold cyan]All Generators Demo[/bold cyan]\n\n"
            "Showing one example from each generator",
            border_style="cyan",
        )
    )

    for generator_name in GENERATOR_REGISTRY.keys():
        console.print(f"\n[bold yellow]═══ {generator_name} ═══[/bold yellow]")

        try:
            generator = get_generator(generator_name)
            exercise = generator.generate()

            console.print(f"[cyan]Q:[/cyan] {exercise.question}")
            console.print(f"[green]A:[/green] {exercise.get_correct_answer()}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[bold green]✓ Demo complete![/bold green]")
    console.print("\nTry: [cyan]linalg-tutor generate practice <generator-name>[/cyan]")
