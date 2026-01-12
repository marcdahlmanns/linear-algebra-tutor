"""Guided learning application controller."""

from typing import Optional
from rich.console import Console

from ..core.progress.session_state import SessionState
from ..core.progress.tracker import ProgressTracker
from ..content.exercises_library import get_exercises_by_topic
from .ui.main_menu import (
    show_welcome,
    main_menu,
    select_chapter_menu,
    chapter_menu,
    show_chapter_list,
    show_help,
    settings_menu,
    console
)
from .ui.prompts import ExercisePrompt
from ..core.generators import get_generator, GeneratorConfig


class GuidedLearningApp:
    """Main application controller for guided learning."""

    def __init__(self):
        """Initialize the guided learning application."""
        self.session = SessionState()
        self.tracker = ProgressTracker()
        self.running = True

    def run(self):
        """Run the main application loop."""
        # Show welcome on first run
        if not self.session.path.current_topic and not self.session.path.completed_topics:
            show_welcome()

        while self.running:
            try:
                console.clear()
                choice = main_menu(self.session)

                if choice == "exit":
                    self.running = False
                    console.print("[cyan]Thanks for learning with Linear Algebra Tutor![/cyan]")

                elif choice == "continue":
                    self._continue_learning()

                elif choice == "select_chapter":
                    self._select_chapter()

                elif choice == "view_progress":
                    self._view_progress()

                elif choice == "quick_practice":
                    self._quick_practice()

                elif choice == "help":
                    show_help()

                elif choice == "settings":
                    self._handle_settings()

            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                console.print("[yellow]Returning to main menu...[/yellow]")
                continue

    def _continue_learning(self):
        """Continue with current chapter."""
        current_chapter = self.session.get_current_chapter()
        if current_chapter:
            self._handle_chapter(current_chapter["topic"])

    def _select_chapter(self):
        """Show chapter selection and navigate."""
        while True:
            console.clear()
            show_chapter_list(self.session)

            topic = select_chapter_menu(self.session)

            if topic == "back":
                break

            self._handle_chapter(topic)

    def _handle_chapter(self, topic: str):
        """Handle chapter menu and actions.

        Args:
            topic: Chapter topic ID
        """
        while True:
            action = chapter_menu(self.session, topic)

            if action == "back":
                break

            elif action == "practice_curated":
                self._practice_curated_exercises(topic)

            elif action == "practice_generated":
                self._practice_generated_exercises(topic)

            elif action == "visualizations":
                self._show_visualizations(topic)

            elif action == "solvers":
                self._show_solvers(topic)

    def _practice_curated_exercises(self, topic: str):
        """Practice curated exercises for a topic.

        Args:
            topic: Topic to practice
        """
        console.clear()

        # Get exercises
        exercises = get_exercises_by_topic(topic)

        if not exercises:
            console.print(f"[yellow]No curated exercises available for {topic} yet.[/yellow]")
            console.print("[dim]Try generated practice instead![/dim]")
            import questionary
            questionary.press_any_key_to_continue("Press any key to continue...").ask()
            return

        # Ask how many
        import questionary
        count_answer = questionary.text(
            f"How many exercises? (1-{len(exercises)}, default: 5):",
            default="5"
        ).ask()

        try:
            count = int(count_answer)
            count = min(count, len(exercises))
        except (ValueError, TypeError):
            count = 5

        # Select exercises
        import random
        selected = random.sample(exercises, min(count, len(exercises)))

        console.print(f"[cyan]Starting practice session: {count} exercises[/cyan]")

        # Practice session
        results = []
        for idx, exercise in enumerate(selected, 1):
            try:
                prompt = ExercisePrompt(
                    exercise,
                    self.tracker,
                    current_exercise=idx,
                    total_exercises=len(selected)
                )
                result = prompt.run()
                results.append(result)
            except KeyboardInterrupt:
                console.print("[yellow]Practice interrupted.[/yellow]")
                break

        # Update session state
        completed = sum(1 for r in results if r.correct)
        time_spent = sum(r.time_spent or 0 for r in results)
        self.session.update_activity(topic, completed, time_spent)

        # Show summary
        if results:
            self._show_session_summary(results, topic)

    def _practice_generated_exercises(self, topic: str):
        """Practice with generated exercises.

        Args:
            topic: Topic to practice
        """
        console.clear()

        # Map topics to generators
        generator_map = {
            "vectors": ["vector_add", "vector_scalar", "dot_product", "vector_norm", "cross_product"],
            "matrices": ["matrix_add", "matrix_scalar", "matrix_multiply", "matrix_transpose", "determinant", "matrix_inverse"],
            "linear_systems": ["linear_system", "triangular_system", "simple_2x2_system"],
        }

        generators = generator_map.get(topic, [])

        if not generators:
            console.print(f"[yellow]No generators available for {topic} yet.[/yellow]")
            import questionary
            questionary.press_any_key_to_continue("Press any key to continue...").ask()
            return

        # Let user choose generator
        import questionary
        generator_choices = [
            {"name": gen.replace("_", " ").title(), "value": gen}
            for gen in generators
        ]
        generator_choices.append({"name": "← Back", "value": "back"})

        gen_name = questionary.select(
            "Choose practice type:",
            choices=generator_choices
        ).ask()

        if gen_name == "back" or not gen_name:
            return

        # Ask how many
        count_answer = questionary.text(
            "How many exercises? (default: 5):",
            default="5"
        ).ask()

        try:
            count = int(count_answer)
        except (ValueError, TypeError):
            count = 5

        # Generate and practice
        config = GeneratorConfig()
        generator = get_generator(gen_name, config)

        console.print(f"[cyan]Starting practice: {count} exercises[/cyan]")

        results = []
        for i in range(count):
            try:
                exercise = generator.generate()
                prompt = ExercisePrompt(
                    exercise,
                    self.tracker,
                    current_exercise=i + 1,
                    total_exercises=count
                )
                result = prompt.run()
                results.append(result)
            except KeyboardInterrupt:
                console.print("[yellow]Practice interrupted.[/yellow]")
                break

        # Update session state
        completed = sum(1 for r in results if r.correct)
        time_spent = sum(r.time_spent or 0 for r in results)
        self.session.update_activity(topic, completed, time_spent)

        # Show summary
        if results:
            self._show_session_summary(results, topic)

    def _show_visualizations(self, topic: str):
        """Show visualizations for a topic.

        Args:
            topic: Topic to visualize
        """
        console.print(f"[cyan]Visualizations for {topic}[/cyan]")
        console.print("[yellow]Use visualization commands:[/yellow]")
        console.print("  linalg-tutor visualize vector 3,4")
        console.print("  linalg-tutor visualize matrix '1,2;3,4'")
        console.print("  linalg-tutor visualize demo")

        import questionary
        questionary.press_any_key_to_continue("Press any key to continue...").ask()

    def _show_solvers(self, topic: str):
        """Show solvers for a topic.

        Args:
            topic: Topic for solvers
        """
        console.print(f"[cyan]Advanced Solvers for {topic}[/cyan]")
        console.print("[yellow]Use solver commands:[/yellow]")
        console.print("  linalg-tutor solve gaussian '2,1;3,4' -b '5,6'")
        console.print("  linalg-tutor solve eigenvalues '4,-2;1,1'")
        console.print("  linalg-tutor solve demo")

        import questionary
        questionary.press_any_key_to_continue("Press any key to continue...").ask()

    def _view_progress(self):
        """Show detailed progress view."""
        console.clear()
        show_chapter_list(self.session)

        summary = self.session.get_progress_summary()

        from rich.table import Table
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bold")

        stats_table.add_row("Total Exercises Completed", str(summary['total_exercises']))
        stats_table.add_row("Total Time Spent", f"{summary['total_time']:.1f}s")
        stats_table.add_row("Overall Progress", f"{summary['progress_percent']:.0f}%")

        console.print(stats_table)

        import questionary
        questionary.press_any_key_to_continue("Press any key to continue...").ask()

    def _quick_practice(self):
        """Quick random practice session."""
        console.clear()

        import questionary
        count_answer = questionary.text(
            "How many random exercises? (default: 5):",
            default="5"
        ).ask()

        try:
            count = int(count_answer)
        except (ValueError, TypeError):
            count = 5

        # Pick random generators
        import random
        all_generators = [
            "vector_add", "vector_scalar", "dot_product",
            "matrix_add", "matrix_scalar", "matrix_multiply",
        ]

        results = []
        for i in range(count):
            try:
                gen_name = random.choice(all_generators)
                generator = get_generator(gen_name)
                exercise = generator.generate()

                prompt = ExercisePrompt(
                    exercise,
                    self.tracker,
                    current_exercise=i + 1,
                    total_exercises=count
                )
                result = prompt.run()
                results.append(result)
            except KeyboardInterrupt:
                console.print("[yellow]Practice interrupted.[/yellow]")
                break

        # Show summary
        if results:
            self._show_session_summary(results, "Mixed Topics")

    def _handle_settings(self):
        """Handle settings menu."""
        while True:
            action = settings_menu(self.session)
            if action == "back":
                break

    def _show_session_summary(self, results, topic: str):
        """Show practice session summary.

        Args:
            results: List of ExerciseResult objects
            topic: Topic that was practiced
        """
        from rich.table import Table
        from rich.panel import Panel

        console.print(Panel.fit(
            f"[bold cyan]Session Complete: {topic.title()}[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        ))

        # Calculate stats
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        total_time = sum(r.time_spent or 0 for r in results)

        # Create summary table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Exercises Completed", str(total))
        table.add_row("Correct", f"[green]{correct}[/green]")
        table.add_row("Incorrect", f"[red]{total - correct}[/red]")
        table.add_row("Accuracy", f"{(correct/total)*100:.1f}%")
        table.add_row("Total Time", f"{total_time:.1f}s")

        console.print(table)

        import questionary
        questionary.press_any_key_to_continue("Press any key to continue...").ask()


def run_guided_app():
    """Entry point for guided learning application."""
    app = GuidedLearningApp()
    app.run()
