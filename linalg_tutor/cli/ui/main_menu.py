"""Main menu and guided learning interface."""

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

from ...core.progress.session_state import SessionState
from ...core.progress.tracker import ProgressTracker


console = Console()


def show_welcome():
    """Show welcome screen."""
    console.print(Panel.fit(
        "[bold cyan]Welcome to Linear Algebra Tutor![/bold cyan]\n"
        "An interactive learning application for mastering\n"
        "undergraduate linear algebra through practice.",
        title="Linear Algebra Tutor",
        border_style="cyan",
        padding=(0, 1)
    ))


def show_progress_overview(session: SessionState):
    """Show overview of learning progress.

    Args:
        session: Session state with learning progress
    """
    summary = session.get_progress_summary()
    current_chapter = session.get_current_chapter()

    console.print(Panel.fit(
        f"[bold]Learning Progress[/bold]\n"
        f"Current Chapter: [cyan]{current_chapter['name']}[/cyan]\n"
        f"Chapters Completed: [green]{summary['completed_chapters']}/{summary['total_chapters']}[/green]\n"
        f"Exercises Completed: [yellow]{summary['total_exercises']}[/yellow]\n"
        f"Progress: [cyan]{summary['progress_percent']:.0f}%[/cyan]",
        title="Your Progress",
        border_style="green",
        padding=(0, 1)
    ))


def show_chapter_list(session: SessionState):
    """Show list of all chapters with status.

    Args:
        session: Session state with learning progress
    """
    table = Table(title="Learning Path", show_header=True, header_style="bold cyan")

    table.add_column("Ch.", style="dim", width=3)
    table.add_column("Chapter", style="bold")
    table.add_column("Status", width=12)
    table.add_column("Description")

    for chapter in SessionState.CHAPTERS:
        topic = chapter["topic"]
        chapter_num = str(chapter["id"])

        # Determine status
        if topic in session.path.completed_topics:
            status = "[green]✓ Complete[/green]"
        elif topic in session.path.topics_in_progress:
            status = "[yellow]⚡ In Progress[/yellow]"
        elif chapter["id"] == session.path.current_chapter:
            status = "[cyan]→ Current[/cyan]"
        else:
            status = "[dim]○ Not Started[/dim]"

        table.add_row(
            chapter_num,
            chapter["name"],
            status,
            chapter["description"]
        )

    console.print(table)


def select_chapter_menu(session: SessionState) -> str:
    """Show chapter selection menu.

    Args:
        session: Session state

    Returns:
        Selected topic ID or 'back'
    """
    choices = []

    for chapter in SessionState.CHAPTERS:
        topic = chapter["topic"]
        status = ""

        if topic in session.path.completed_topics:
            status = "✓"
        elif topic in session.path.topics_in_progress:
            status = "⚡"
        elif chapter["id"] == session.path.current_chapter:
            status = "→"

        label = f"{status:2} Chapter {chapter['id']}: {chapter['name']}"
        choices.append({"name": label, "value": topic})

    choices.append({"name": "← Back to Main Menu", "value": "back"})

    answer = questionary.select(
        "Select a chapter:",
        choices=choices
    ).ask()

    return answer if answer else "back"


def chapter_menu(session: SessionState, topic: str) -> str:
    """Show chapter-specific menu.

    Args:
        session: Session state
        topic: Chapter topic ID

    Returns:
        Action to take
    """
    # Get chapter info
    chapter = None
    for ch in SessionState.CHAPTERS:
        if ch["topic"] == topic:
            chapter = ch
            break

    if not chapter:
        return "back"

    console.clear()
    console.print(Panel.fit(
        f"[bold cyan]Chapter {chapter['id']}: {chapter['name']}[/bold cyan]\n"
        f"{chapter['description']}",
        border_style="cyan",
        padding=(0, 1)
    ))

    choices = [
        {"name": "📚 Practice Curated Exercises", "value": "practice_curated"},
        {"name": "∞ Generate Practice Problems", "value": "practice_generated"},
        {"name": "👁 View Visualizations", "value": "visualizations"},
        {"name": "🔧 Advanced Solvers", "value": "solvers"},
        {"name": "← Back to Chapter Selection", "value": "back"},
    ]

    answer = questionary.select(
        "What would you like to do?",
        choices=choices
    ).ask()

    return answer if answer else "back"


def main_menu(session: SessionState) -> str:
    """Show main menu and return user choice.

    Args:
        session: Session state with learning progress

    Returns:
        User's menu choice
    """
    # Show progress if user has started
    if session.path.current_topic or session.path.completed_topics:
        show_progress_overview(session)

    # Build menu choices
    choices = []

    # Continue Learning - only show if there's progress
    if session.path.current_topic or session.path.completed_topics:
        current = session.get_current_chapter()
        choices.append({
            "name": f"→ Continue Learning: {current['name']}",
            "value": "continue"
        })

    # Start New Chapter / Select Chapter
    choices.extend([
        {"name": "📖 Select Chapter", "value": "select_chapter"},
        {"name": "📊 View Progress", "value": "view_progress"},
        {"name": "🎲 Quick Practice (Random)", "value": "quick_practice"},
        {"name": "❓ Help & Commands", "value": "help"},
        {"name": "⚙️  Settings", "value": "settings"},
        {"name": "🚪 Exit", "value": "exit"},
    ])

    answer = questionary.select(
        "Main Menu:",
        choices=choices
    ).ask()

    return answer if answer else "exit"


def show_help():
    """Show help screen with available commands."""
    console.print(Panel(
        "[bold]Linear Algebra Tutor - Help[/bold]\n"
        "[cyan]Guided Learning:[/cyan]\n"
        "  Just run 'linalg-tutor' and follow the menus!\n"
        "[cyan]Quick Commands (Optional):[/cyan]\n"
        "  linalg-tutor exercise practice vectors    - Practice vectors\n"
        "  linalg-tutor generate practice vector_add - Infinite practice\n"
        "  linalg-tutor visualize vector 3,4         - See visualizations\n"
        "  linalg-tutor solve eigenvalues '1,2;3,4'  - Step-by-step solver\n"
        "[cyan]Navigation:[/cyan]\n"
        "  • Use arrow keys to navigate menus\n"
        "  • Press Enter to select\n"
        "  • Press Ctrl+C to go back or exit\n"
        "[cyan]Progress:[/cyan]\n"
        "  Your progress is automatically saved!\n"
        "  Complete chapters to unlock new content.",
        title="Help",
        border_style="yellow",
        padding=(0, 1)
    ))
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def settings_menu(session: SessionState) -> str:
    """Show settings menu.

    Args:
        session: Session state

    Returns:
        Action to take
    """
    console.clear()

    choices = [
        {"name": "🔄 Reset All Progress", "value": "reset"},
        {"name": "📁 View Data Location", "value": "data_location"},
        {"name": "← Back to Main Menu", "value": "back"},
    ]

    answer = questionary.select(
        "Settings:",
        choices=choices
    ).ask()

    if answer == "reset":
        confirm = questionary.confirm(
            "Are you sure you want to reset all progress? This cannot be undone.",
            default=False
        ).ask()

        if confirm:
            session.reset()
            console.print("[yellow]✓ All progress has been reset.[/yellow]")
            questionary.press_any_key_to_continue("Press any key to continue...").ask()

    elif answer == "data_location":
        console.print(Panel(
            f"[bold]Data Storage Location:[/bold]\n"
            f"Session State: {session.state_file}\n"
            f"Progress Database: {session.data_dir / 'progress.db'}\n"
            f"[dim]You can backup or delete these files if needed.[/dim]",
            title="Data Location",
            border_style="cyan",
            padding=(0, 1)
        ))
        questionary.press_any_key_to_continue("Press any key to continue...").ask()

    return answer if answer else "back"
