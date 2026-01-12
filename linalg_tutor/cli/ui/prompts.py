"""Interactive prompts and UI components for exercises."""

import ast
import os
import time
from typing import Any, Optional, List

import numpy as np
import questionary
from rich.align import Align
from rich.box import SIMPLE
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...core.exercises import (
    Exercise,
    ExerciseResult,
    ComputationalExercise,
    MultipleChoiceExercise,
    TrueFalseExercise,
    FillInExercise,
)
from ...core.progress import ProgressTracker
from ...visualization import (
    print_vector_visualization,
    print_matrix_visualization,
    visualize_dot_product,
    visualize_vector_addition_2d,
    explain_dot_product_geometry,
    visualize_matrix_multiply_step,
    visualize_determinant_2x2,
)



class ExercisePrompt:
    """Interactive prompt for completing an exercise with fixed screen updates."""

    def __init__(
        self,
        exercise: Exercise,
        tracker: Optional[ProgressTracker] = None,
        current_exercise: int = 1,
        total_exercises: int = 1,
    ):
        """Initialize exercise prompt.

        Args:
            exercise: Exercise to practice
            tracker: Optional progress tracker for recording attempts
            current_exercise: Current exercise number
            total_exercises: Total number of exercises in session
        """
        self.exercise = exercise
        self.tracker = tracker
        self.console = Console()
        self.start_time = None
        self.hints_shown = 0
        self.current_exercise = current_exercise
        self.total_exercises = total_exercises
        self.message = ""
        self.message_style = ""
        self.shown_hints: List[str] = []
        self.attempts = 0

    def _build_display(self):
        """Build the compact display - return Group instead of Layout."""
        # Get terminal size for dynamic sizing
        try:
            terminal_size = os.get_terminal_size()
            term_height = terminal_size.lines
        except OSError:
            term_height = 24  # Default fallback

        display_parts = []

        # Header: Progress indicator - compact single line
        difficulty_color = {
            "practice": "green",
            "application": "yellow",
            "challenge": "red",
        }
        color = difficulty_color.get(self.exercise.difficulty.value, "cyan")

        # Single line header without panel borders
        header_text = Text(
            f"[{color}]{self.exercise.topic.title()}[/{color}] │ "
            f"[bold]{self.current_exercise}/{self.total_exercises}[/bold] │ "
            f"[{color}]{self.exercise.difficulty.value.title()}[/{color}]",
            justify="center"
        )
        display_parts.append(header_text)

        # Question - compact panel
        question_text = Text(self.exercise.question, style="bold white")
        question_panel = Panel(
            Align.center(question_text, vertical="middle"),
            title=f"[{color}]Question[/{color}]",
            border_style=color,
        )
        display_parts.append(question_panel)

        # Show hints if any - compact format
        if self.shown_hints:
            hints_text = "\n".join([
                f"[yellow]💡 {i+1}:[/yellow] {hint}"
                for i, hint in enumerate(self.shown_hints)
            ])
            hints_panel = Panel(
                hints_text,
                title="Hints",
                border_style="yellow",
                expand=False,
                padding=(0, 1)
            )
            display_parts.append(hints_panel)

        # Show current message if any
        if self.message:
            display_parts.append(
                Text(self.message, style=self.message_style)
            )

        # Show stats - compact inline format
        if self.attempts > 0 or self.hints_shown > 0:
            stats_text = f"Attempts: {self.attempts} │ Hints: {self.hints_shown}/{len(self.exercise.hints)}"
            display_parts.append(Text(stats_text, style="dim"))

        return Group(*display_parts)

    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')

    def _display_screen(self):
        """Display the current state."""
        self._clear_screen()
        layout = self._build_display()
        self.console.print(layout)  # No extra spacing

    def run(self) -> ExerciseResult:
        """Run interactive exercise session with fixed screen updates.

        Returns:
            ExerciseResult with user's final answer
        """
        # Check terminal size (only if we're in a TTY)
        try:
            terminal_size = os.get_terminal_size()
            if terminal_size.columns < 80 or terminal_size.lines < 15:
                self.console.print(
                    Panel(
                        f"[bold red]Terminal too small![/bold red]\n\n"
                        f"Current size: {terminal_size.columns}×{terminal_size.lines}\n"
                        f"Required: 80×15 minimum\n\n"
                        f"Please resize your terminal and try again.",
                        title="Error",
                        border_style="red",
                    )
                )
                # Return a skip result
                return ExerciseResult(
                    correct=False,
                    user_answer=None,
                    correct_answer=self.exercise.get_correct_answer(),
                    feedback="Terminal too small",
                    hints_used=0,
                    time_spent=0,
                )
        except OSError:
            # Not a terminal (e.g., piped input), skip size check
            pass

        self.start_time = time.time()

        # Main interaction loop with Ctrl+C handling
        try:
            while True:
                # Display current state
                self._display_screen()

                # Build action choices
                choices = ["Submit answer", "Get a hint"]

                # Add visualize option for computational exercises
                if isinstance(self.exercise, ComputationalExercise):
                    choices.append("Visualize")

                choices.extend(["Show solution", "Skip this exercise"])

                action = questionary.select(
                    "What would you like to do?",
                    choices=choices,
                ).ask()

                if action == "Submit answer":
                    result = self._handle_submit()
                    if result:
                        # Show final result
                        self._display_screen()
                        return result

                elif action == "Get a hint":
                    self._show_hint()

                elif action == "Visualize":
                    self._show_visualization()

                elif action == "Show solution":
                    result = self._show_solution()
                    self._display_screen()
                    return result

                else:  # Skip
                    result = self._skip_exercise()
                    self._display_screen()
                    return result

        except KeyboardInterrupt:
            # User pressed Ctrl+C
            self._clear_screen()
            self.console.print("\n[yellow]Session interrupted by user (Ctrl+C)[/yellow]\n")

            # Record as interrupted attempt
            result = ExerciseResult(
                correct=False,
                user_answer=None,
                correct_answer=self.exercise.get_correct_answer(),
                feedback="Interrupted by user",
                hints_used=self.hints_shown,
                time_spent=time.time() - self.start_time,
            )

            if self.tracker:
                self.tracker.record_attempt(result, self.exercise, result.time_spent)
                self.console.print("[dim]Progress saved.[/dim]\n")

            return result


    def _handle_submit(self) -> Optional[ExerciseResult]:
        """Handle answer submission.

        Returns:
            ExerciseResult if answer is final, None to continue loop
        """
        # Get answer based on exercise type
        user_answer = self._get_answer_input()

        if user_answer is None:
            self.message = "Answer input cancelled"
            self.message_style = "yellow"
            return None  # User cancelled

        # Check answer
        self.attempts += 1
        result = self.exercise.check_answer(user_answer)
        result.hints_used = self.hints_shown
        result.time_spent = time.time() - self.start_time

        # Record attempt if tracker available
        if self.tracker:
            self.tracker.record_attempt(result, self.exercise, result.time_spent)

        # Show feedback
        if result.correct:
            self.message = "✓ Correct!"
            self.message_style = "bold green"
            return result
        else:
            self.message = f"✗ {result.feedback}"
            self.message_style = "bold red"

            # Redisplay with error message
            self._display_screen()

            # Ask if they want to try again
            retry = questionary.confirm("Try again?", default=True).ask()
            if not retry:
                self.message = f"The correct answer is: {result.correct_answer}"
                self.message_style = "yellow"
                return result

            return None  # Continue loop for retry

    def _get_answer_input(self) -> Optional[Any]:
        """Get answer input based on exercise type.

        Returns:
            User's answer, or None if cancelled
        """
        if isinstance(self.exercise, ComputationalExercise):
            return self._get_numerical_input()
        elif isinstance(self.exercise, MultipleChoiceExercise):
            return self._get_multiple_choice_input()
        elif isinstance(self.exercise, TrueFalseExercise):
            return self._get_true_false_input()
        elif isinstance(self.exercise, FillInExercise):
            if self.exercise.answer_type == "numerical":
                return self._get_numerical_input()
            else:
                return self._get_text_input()
        else:
            return self._get_text_input()

    def _get_numerical_input(self) -> Optional[Any]:
        """Get numerical input (vector, matrix, or scalar)."""
        self.console.print(
            "[dim]Format: [1,2,3] for vectors │ [[1,2],[3,4]] for matrices │ 42 for scalars[/dim]"
        )
        answer_str = questionary.text("Your answer:").ask()

        if not answer_str:
            return None

        try:
            return ast.literal_eval(answer_str)
        except (ValueError, SyntaxError):
            self.console.print(
                "[red]Invalid format. Please use Python list notation (e.g., [1, 2, 3])[/red]\n"
            )
            return None

    def _get_multiple_choice_input(self) -> Optional[int]:
        """Get multiple choice selection."""
        exercise = self.exercise  # Type hint helper
        assert isinstance(exercise, MultipleChoiceExercise)

        choice = questionary.select(
            "Your answer:",
            choices=[f"{i}. {choice}" for i, choice in enumerate(exercise.choices)],
        ).ask()

        if not choice:
            return None

        # Extract index from "0. choice text"
        return int(choice.split(".")[0])

    def _get_true_false_input(self) -> Optional[bool]:
        """Get true/false selection."""
        choice = questionary.select(
            "Your answer:", choices=["True", "False"]
        ).ask()

        if not choice:
            return None

        return choice == "True"

    def _get_text_input(self) -> Optional[str]:
        """Get text input."""
        return questionary.text("Your answer:").ask()

    def _show_hint(self):
        """Show the next available hint."""
        if self.hints_shown < len(self.exercise.hints):
            hint = self.exercise.hints[self.hints_shown]
            self.shown_hints.append(hint)
            self.hints_shown += 1
            self.message = f"Hint {self.hints_shown}/{len(self.exercise.hints)} revealed"
            self.message_style = "yellow"
        else:
            self.message = "No more hints available!"
            self.message_style = "yellow"

    def _show_visualization(self):
        """Show visualization for the exercise based on operation type."""
        if not isinstance(self.exercise, ComputationalExercise):
            self.message = "Visualization not available for this exercise type"
            self.message_style = "yellow"
            return

        # Clear screen and show full visualization
        self._clear_screen()

        exercise = self.exercise
        inputs = exercise.inputs
        operation = exercise.operation

        self.console.print(Panel.fit(
            "[bold cyan]Visualization[/bold cyan]",
            border_style="cyan"
        ))
        self.console.print()

        try:
            if operation == "vector_add" and "v" in inputs and "w" in inputs:
                v = inputs["v"]
                w = inputs["w"]
                if len(v) == 2 and len(w) == 2:
                    viz = visualize_vector_addition_2d(v, w)
                    self.console.print(Panel(viz, title="Vector Addition", border_style="cyan"))
                else:
                    self.console.print(f"v = {v}\nw = {w}\nv + w = {v + w}")

            elif operation == "vector_subtract" and "v" in inputs and "w" in inputs:
                v = inputs["v"]
                w = inputs["w"]
                self.console.print(f"v = {v}\nw = {w}\nv - w = {v - w}")

            elif operation == "scalar_mult" and "scalar" in inputs and "v" in inputs:
                scalar = inputs["scalar"]
                v = inputs["v"]
                print_vector_visualization(v, label="v", show_2d=(len(v) == 2))
                self.console.print(f"\nScalar: {scalar}\nResult: {scalar} × v = {scalar * v}")

            elif operation == "dot_product" and "v" in inputs and "w" in inputs:
                v = inputs["v"]
                w = inputs["w"]
                viz = visualize_dot_product(v, w)
                self.console.print(Panel(viz, title="Dot Product", border_style="cyan"))
                self.console.print()
                geo_viz = explain_dot_product_geometry(v, w)
                self.console.print(Panel(geo_viz, title="Geometric Meaning", border_style="green"))

            elif operation == "matrix_add" and "A" in inputs and "B" in inputs:
                A = inputs["A"]
                B = inputs["B"]
                print_matrix_visualization(A, title="Matrix A")
                self.console.print("\n+\n")
                print_matrix_visualization(B, title="Matrix B")
                self.console.print("\n=\n")
                print_matrix_visualization(A + B, title="Result")

            elif operation == "matrix_multiply" and "A" in inputs and "B" in inputs:
                A = inputs["A"]
                B = inputs["B"]
                print_matrix_visualization(A, title="Matrix A")
                self.console.print("\n×\n")
                print_matrix_visualization(B, title="Matrix B")
                self.console.print("\nExample calculation for first element:")
                step_viz = visualize_matrix_multiply_step(A, B, 0, 0)
                self.console.print(Panel(step_viz, border_style="yellow"))

            elif operation == "matrix_transpose" and "A" in inputs:
                A = inputs["A"]
                print_matrix_visualization(A, title="Original Matrix")
                self.console.print("\nTranspose:\n")
                print_matrix_visualization(A.T, title="A^T")

            elif operation == "determinant" and "A" in inputs:
                A = inputs["A"]
                if A.shape == (2, 2):
                    viz = visualize_determinant_2x2(A)
                    self.console.print(Panel(viz, title="Determinant Calculation", border_style="cyan"))
                else:
                    print_matrix_visualization(A, title="Matrix A")
                    det = np.linalg.det(A)
                    self.console.print(f"\ndet(A) = {det:.4g}")

            else:
                # Generic visualization
                self.console.print("[cyan]Input values:[/cyan]\n")
                for key, value in inputs.items():
                    if isinstance(value, np.ndarray):
                        if len(value.shape) == 1:
                            self.console.print(f"{key} = {value}")
                        else:
                            print_matrix_visualization(value, title=key)
                            self.console.print()
                    else:
                        self.console.print(f"{key} = {value}")

        except Exception as e:
            self.console.print(f"[yellow]Could not generate visualization: {e}[/yellow]")

        # Wait for user to continue
        self.console.print()
        questionary.press_any_key_to_continue("Press any key to continue...").ask()
        self.message = "Visualization shown"
        self.message_style = "cyan"

    def _show_solution(self) -> ExerciseResult:
        """Show the complete solution."""
        # Clear screen and show full solution
        self._clear_screen()

        self.console.print(Panel.fit(
            "[bold cyan]Solution[/bold cyan]",
            border_style="cyan"
        ))
        self.console.print()

        solution = self.exercise.get_solution(show_steps=True)
        self.console.print(Panel(solution, border_style="cyan"))

        # Wait for user to continue
        self.console.print()
        questionary.press_any_key_to_continue("Press any key to continue...").ask()

        # Record as incorrect attempt (showed solution)
        result = ExerciseResult(
            correct=False,
            user_answer=None,
            correct_answer=self.exercise.get_correct_answer(),
            feedback="Solution shown",
            hints_used=self.hints_shown,
            time_spent=time.time() - self.start_time,
        )

        if self.tracker:
            self.tracker.record_attempt(result, self.exercise, result.time_spent)

        self.message = "Solution shown"
        self.message_style = "yellow"
        return result

    def _skip_exercise(self) -> ExerciseResult:
        """Skip the exercise."""
        # Record as incorrect attempt (skipped)
        result = ExerciseResult(
            correct=False,
            user_answer=None,
            correct_answer=self.exercise.get_correct_answer(),
            feedback="Skipped",
            hints_used=self.hints_shown,
            time_spent=time.time() - self.start_time,
        )

        if self.tracker:
            self.tracker.record_attempt(result, self.exercise, result.time_spent)

        self.message = "Exercise skipped"
        self.message_style = "yellow"
        return result
