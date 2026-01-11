"""Exercise practice commands."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...core.exercises import ExerciseDifficulty
from ...core.progress import ProgressTracker
from ...content.exercises_library import get_exercises_by_topic, get_exercises_by_difficulty
from ..ui.prompts import ExercisePrompt

app = typer.Typer()
console = Console()


@app.command()
def practice(
    topic: str = typer.Argument(..., help="Topic to practice (e.g., 'vectors')"),
    difficulty: str = typer.Option(
        "all",
        help="Difficulty level: practice, application, challenge, or all",
    ),
    count: int = typer.Option(
        5, help="Number of exercises to practice", min=1, max=50
    ),
    show_progress: bool = typer.Option(
        True, help="Show progress tracking"
    ),
):
    """Practice exercises for a specific topic.

    Examples:
        linalg-tutor exercise practice vectors
        linalg-tutor exercise practice vectors --difficulty practice --count 10
        linalg-tutor exercise practice vectors --difficulty challenge --count 3
    """
    # Load exercises
    if difficulty == "all":
        exercises = get_exercises_by_topic(topic)
    else:
        try:
            diff_level = ExerciseDifficulty(difficulty)
            exercises = get_exercises_by_difficulty(topic, diff_level)
        except ValueError:
            console.print(
                f"[red]Invalid difficulty: {difficulty}[/red]\n"
                "Valid options: practice, application, challenge, all"
            )
            raise typer.Exit(1)

    if not exercises:
        console.print(
            f"[yellow]No exercises found for topic '{topic}' with difficulty '{difficulty}'[/yellow]"
        )
        console.print("\nAvailable topics:")
        console.print("  • vectors")
        console.print("\nComing soon: matrices, linear-systems, eigenvalues, and more!")
        raise typer.Exit(0)

    # Limit to requested count
    exercises = exercises[:count]

    # Initialize progress tracker
    tracker = ProgressTracker() if show_progress else None

    # Welcome message
    console.print(
        Panel.fit(
            f"[bold cyan]Practice Session: {topic.title()}[/bold cyan]\n"
            f"Difficulty: {difficulty}\n"
            f"Exercises: {len(exercises)}",
            border_style="cyan",
        )
    )
    console.print()

    # Practice each exercise
    results = []
    try:
        for idx, exercise in enumerate(exercises, 1):
            prompt = ExercisePrompt(exercise, tracker, current_exercise=idx, total_exercises=len(exercises))
            result = prompt.run()
            results.append(result)
    except KeyboardInterrupt:
        # User interrupted session, show summary of what was completed
        console.print("\n[yellow]Session interrupted. Showing summary of completed exercises...[/yellow]\n")

    # Show summary (even if interrupted)
    if results:
        _show_session_summary(results, topic, tracker)
    else:
        console.print("[yellow]No exercises completed.[/yellow]")


@app.command()
def list(
    topic: str = typer.Argument(..., help="Topic to list exercises for"),
):
    """List all available exercises for a topic."""
    exercises = get_exercises_by_topic(topic)

    if not exercises:
        console.print(f"[yellow]No exercises found for topic '{topic}'[/yellow]")
        raise typer.Exit(0)

    # Create table
    table = Table(
        title=f"Available Exercises: {topic.title()}", show_header=True, header_style="bold magenta"
    )

    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Difficulty", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Question", style="white")

    for ex in exercises:
        # Get exercise type name
        ex_type = ex.__class__.__name__.replace("Exercise", "")

        # Truncate long questions
        question = ex.question
        if len(question) > 60:
            question = question[:57] + "..."

        table.add_row(ex.exercise_id, ex.difficulty.value, ex_type, question)

    console.print(table)
    console.print(f"\n[bold]Total: {len(exercises)} exercises[/bold]")


def _show_session_summary(results, topic: str, tracker: ProgressTracker):
    """Show summary of practice session.

    Args:
        results: List of ExerciseResult objects
        topic: Topic that was practiced
        tracker: Progress tracker (optional)
    """
    console.print("[bold cyan]━━━ Session Summary ━━━[/bold cyan]\n")

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    skipped = sum(1 for r in results if r.feedback in ("Skipped", "Solution shown"))
    total_time = sum(r.time_spent or 0 for r in results)
    avg_time = total_time / total if total > 0 else 0

    # Create summary table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Exercises Completed", f"{total}")
    table.add_row("Correct", f"[green]{correct}[/green]")
    table.add_row("Incorrect", f"[red]{total - correct - skipped}[/red]")
    if skipped > 0:
        table.add_row("Skipped/Solved", f"[yellow]{skipped}[/yellow]")
    table.add_row("Accuracy", f"{(correct/total)*100:.1f}%")
    table.add_row("Total Time", f"{total_time:.1f}s")
    table.add_row("Average Time", f"{avg_time:.1f}s per exercise")

    console.print(table)

    # Show mastery if tracker available
    if tracker:
        mastery = tracker.get_mastery_level(topic)
        mastery_pct = mastery * 100

        # Color based on mastery level
        if mastery_pct >= 80:
            color = "green"
            message = "Excellent! You've mastered this topic!"
        elif mastery_pct >= 60:
            color = "yellow"
            message = "Good progress! Keep practicing."
        else:
            color = "red"
            message = "Keep practicing to improve your mastery."

        console.print(f"\n[{color}]Mastery Level: {mastery_pct:.1f}%[/{color}]")
        console.print(f"[{color}]{message}[/{color}]")

    console.print()
