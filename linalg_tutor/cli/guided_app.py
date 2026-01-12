"""Guided learning application controller."""

from typing import Optional
from rich.console import Console
from rich.panel import Panel

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
        """Show interactive visualizations for a topic.

        Args:
            topic: Topic to visualize
        """
        import questionary
        import numpy as np
        from ..visualization import (
            print_vector_visualization,
            visualize_vector_addition_2d,
            visualize_dot_product,
            explain_dot_product_geometry,
            print_matrix_visualization,
            visualize_matrix_multiply_step,
            visualize_determinant_2x2,
            visualize_matrix_transpose,
        )

        # Build menu based on topic
        if topic == "vectors":
            choices = [
                {"name": "📐 Vector Visualization (2D)", "value": "vector_2d"},
                {"name": "➕ Vector Addition (2D)", "value": "vector_add"},
                {"name": "• Dot Product", "value": "dot_product"},
                {"name": "✖ Scalar Multiplication", "value": "scalar_mult"},
                {"name": "← Back", "value": "back"},
            ]
        elif topic == "matrices":
            choices = [
                {"name": "📊 Matrix Display", "value": "matrix_display"},
                {"name": "✖ Matrix Multiplication", "value": "matrix_mult"},
                {"name": "🔄 Matrix Transpose", "value": "transpose"},
                {"name": "🔢 Determinant (2×2)", "value": "determinant"},
                {"name": "← Back", "value": "back"},
            ]
        elif topic == "linear_systems":
            choices = [
                {"name": "📋 Augmented Matrix", "value": "augmented"},
                {"name": "← Back", "value": "back"},
            ]
        else:
            console.print(f"[yellow]No visualizations available for {topic} yet.[/yellow]")
            questionary.press_any_key_to_continue("Press any key to continue...").ask()
            return

        while True:
            console.clear()
            console.print(Panel.fit(
                f"[bold cyan]Visualizations: {topic.title()}[/bold cyan]",
                border_style="cyan",
                padding=(0, 1)
            ))

            selection = questionary.select(
                "Choose visualization:",
                choices=choices
            ).ask()

            if selection == "back" or not selection:
                break

            console.clear()

            # Show the selected visualization
            try:
                if selection == "vector_2d":
                    v = np.array([3, 4])
                    console.print(Panel.fit("[bold]Example: Vector v = [3, 4][/bold]", border_style="cyan"))
                    print_vector_visualization(v, label="v", show_2d=True)

                elif selection == "vector_add":
                    v = np.array([2, 3])
                    w = np.array([1, -1])
                    console.print(Panel.fit("[bold]Example: v = [2, 3] + w = [1, -1][/bold]", border_style="cyan"))
                    viz = visualize_vector_addition_2d(v, w)
                    console.print(Panel(viz, title="Vector Addition", border_style="cyan"))

                elif selection == "dot_product":
                    v = np.array([3, 4])
                    w = np.array([1, 2])
                    console.print(Panel.fit("[bold]Example: v = [3, 4] · w = [1, 2][/bold]", border_style="cyan"))
                    viz = visualize_dot_product(v, w)
                    console.print(Panel(viz, title="Dot Product", border_style="cyan"))
                    geo_viz = explain_dot_product_geometry(v, w)
                    console.print(Panel(geo_viz, title="Geometric Meaning", border_style="green"))

                elif selection == "scalar_mult":
                    v = np.array([2, 3])
                    scalar = 2
                    console.print(Panel.fit(f"[bold]Example: {scalar} × v where v = [2, 3][/bold]", border_style="cyan"))
                    print_vector_visualization(v, label="v", show_2d=True)
                    console.print(f"\n[cyan]Scalar:[/cyan] {scalar}")
                    console.print(f"[cyan]Result:[/cyan] {scalar} × v = {scalar * v}")

                elif selection == "matrix_display":
                    A = np.array([[1, 2], [3, 4]])
                    console.print(Panel.fit("[bold]Example: 2×2 Matrix[/bold]", border_style="cyan"))
                    print_matrix_visualization(A, title="Matrix A")

                elif selection == "matrix_mult":
                    A = np.array([[1, 2], [3, 4]])
                    B = np.array([[5, 6], [7, 8]])
                    console.print(Panel.fit("[bold]Example: Matrix Multiplication[/bold]", border_style="cyan"))
                    print_matrix_visualization(A, title="Matrix A")
                    console.print("\n[bold cyan]×[/bold cyan]\n")
                    print_matrix_visualization(B, title="Matrix B")
                    console.print("\n[bold]Example calculation for element (0,0):[/bold]")
                    step_viz = visualize_matrix_multiply_step(A, B, 0, 0)
                    console.print(Panel(step_viz, border_style="yellow"))

                elif selection == "transpose":
                    A = np.array([[1, 2, 3], [4, 5, 6]])
                    console.print(Panel.fit("[bold]Example: Matrix Transpose[/bold]", border_style="cyan"))
                    print_matrix_visualization(A, title="Original Matrix A")
                    console.print("\n[bold cyan]Transpose:[/bold cyan]\n")
                    viz = visualize_matrix_transpose(A)
                    console.print(viz)

                elif selection == "determinant":
                    A = np.array([[3, 2], [1, 4]])
                    console.print(Panel.fit("[bold]Example: 2×2 Determinant[/bold]", border_style="cyan"))
                    viz = visualize_determinant_2x2(A)
                    console.print(Panel(viz, title="Determinant Calculation", border_style="cyan"))

                elif selection == "augmented":
                    A = np.array([[2, 1], [3, 4]])
                    b = np.array([5, 11])
                    console.print(Panel.fit("[bold]Example: Augmented Matrix [A|b][/bold]", border_style="cyan"))
                    console.print("\n[bold]System: Ax = b[/bold]")
                    print_matrix_visualization(A, title="Coefficient Matrix A")
                    console.print(f"\n[bold]b =[/bold] {b}")
                    # Show augmented matrix
                    augmented = np.column_stack([A, b])
                    console.print("\n[bold cyan]Augmented Matrix [A|b]:[/bold cyan]")
                    print_matrix_visualization(augmented, title="[A|b]")

            except Exception as e:
                console.print(f"[red]Error showing visualization: {e}[/red]")

            console.print()
            questionary.press_any_key_to_continue("Press any key to continue...").ask()

    def _format_solution(self, solution):
        """Format a Solution or SimpleSolution object for display.

        Args:
            solution: Solution or SimpleSolution object with steps

        Returns:
            None (prints directly to console)
        """
        # Display problem statement or operation
        if hasattr(solution, 'problem_statement'):
            console.print(f"\n[bold cyan]Problem:[/bold cyan] {solution.problem_statement}\n")
        elif hasattr(solution, 'operation'):
            console.print(f"\n[bold cyan]Operation:[/bold cyan] {solution.operation}\n")

        # Display steps
        for i, step in enumerate(solution.steps, 1):
            # Handle both Solution (with step_number) and SimpleSolution (without)
            step_num = step.step_number if hasattr(step, 'step_number') else i
            console.print(f"[bold]Step {step_num}: {step.description}[/bold]")
            if step.mathematical_expression:
                console.print(Panel(step.mathematical_expression, border_style="yellow"))
            if step.explanation:
                console.print(f"[dim]{step.explanation}[/dim]")
            if hasattr(step, 'intermediate_result') and step.intermediate_result:
                console.print(f"[green]{step.intermediate_result}[/green]")
            console.print()

        # Display final answer
        if solution.final_answer is not None:
            console.print("[bold green]Final Answer:[/bold green]")
            console.print(f"{solution.final_answer}")

        # Display verification if present (only on Solution, not SimpleSolution)
        if hasattr(solution, 'verification') and solution.verification:
            console.print(f"\n[cyan]Verification:[/cyan] {solution.verification}")

    def _show_solvers(self, topic: str):
        """Show interactive solvers for a topic.

        Args:
            topic: Topic for solvers
        """
        import questionary
        import numpy as np
        from ..core.solver import get_solver

        # Build menu based on topic
        if topic == "vectors":
            choices = [
                {"name": "➕ Vector Addition", "value": "vector_add"},
                {"name": "• Dot Product", "value": "dot_product"},
                {"name": "← Back", "value": "back"},
            ]
        elif topic == "matrices":
            choices = [
                {"name": "✖ Matrix Multiplication", "value": "matrix_mult"},
                {"name": "🔄 Matrix Transpose", "value": "transpose"},
                {"name": "🔢 Determinant", "value": "determinant"},
                {"name": "← Back", "value": "back"},
            ]
        elif topic == "linear_systems":
            choices = [
                {"name": "📐 Gaussian Elimination", "value": "gaussian"},
                {"name": "🎯 RREF", "value": "rref"},
                {"name": "📊 Solve Linear System", "value": "solve_system"},
                {"name": "← Back", "value": "back"},
            ]
        elif topic == "eigenvalues":
            choices = [
                {"name": "λ Eigenvalues (2×2)", "value": "eigenvalues"},
                {"name": "← Back", "value": "back"},
            ]
        else:
            console.print(f"[yellow]No solvers available for {topic} yet.[/yellow]")
            questionary.press_any_key_to_continue("Press any key to continue...").ask()
            return

        while True:
            console.clear()
            console.print(Panel.fit(
                f"[bold cyan]Advanced Solvers: {topic.title()}[/bold cyan]",
                border_style="cyan",
                padding=(0, 1)
            ))

            selection = questionary.select(
                "Choose solver:",
                choices=choices
            ).ask()

            if selection == "back" or not selection:
                break

            console.clear()

            # Show the selected solver with example
            try:
                if selection == "vector_add":
                    v = np.array([1, 2, 3])
                    w = np.array([4, 5, 6])
                    expected = v + w
                    solver = get_solver("vector_add")
                    solution = solver.solve_with_steps({"v": v, "w": w}, expected)
                    console.print(Panel.fit("[bold]Example: Vector Addition[/bold]", border_style="cyan"))
                    self._format_solution(solution)

                elif selection == "dot_product":
                    v = np.array([1, 2, 3])
                    w = np.array([4, 5, 6])
                    expected = np.dot(v, w)
                    solver = get_solver("vector_dot")
                    solution = solver.solve_with_steps({"v": v, "w": w}, expected)
                    console.print(Panel.fit("[bold]Example: Dot Product[/bold]", border_style="cyan"))
                    self._format_solution(solution)

                elif selection == "matrix_mult":
                    A = np.array([[1, 2], [3, 4]])
                    B = np.array([[5, 6], [7, 8]])
                    expected = A @ B
                    solver = get_solver("matrix_multiply")
                    solution = solver.solve_with_steps({"A": A, "B": B}, expected)
                    console.print(Panel.fit("[bold]Example: Matrix Multiplication[/bold]", border_style="cyan"))
                    self._format_solution(solution)

                elif selection == "transpose":
                    A = np.array([[1, 2, 3], [4, 5, 6]])
                    expected = A.T
                    solver = get_solver("matrix_transpose")
                    solution = solver.solve_with_steps({"A": A}, expected)
                    console.print(Panel.fit("[bold]Example: Matrix Transpose[/bold]", border_style="cyan"))
                    self._format_solution(solution)

                elif selection == "determinant":
                    A = np.array([[3, 2], [1, 4]])
                    # Use visualization instead (no determinant solver exists)
                    from ..visualization import visualize_determinant_2x2, visualize_determinant_geometric
                    console.print(Panel.fit("[bold]Example: Determinant[/bold]", border_style="cyan"))
                    viz = visualize_determinant_2x2(A)
                    console.print(Panel(viz, border_style="cyan"))
                    console.print()
                    geo_viz = visualize_determinant_geometric(A)
                    console.print(Panel(geo_viz, title="Geometric Meaning", border_style="green"))

                elif selection == "gaussian":
                    A = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
                    b = np.array([8.0, -11.0, -3.0])
                    solver = get_solver("gaussian_elimination")
                    # Note: gaussian_elimination expects A and b as separate args, not dict
                    solution = solver.solve_with_steps(A, b)
                    console.print(Panel.fit("[bold]Example: Gaussian Elimination[/bold]", border_style="cyan"))
                    self._format_solution(solution)

                elif selection == "rref":
                    A = np.array([[2.0, 1.0], [4.0, 3.0]])
                    solver = get_solver("rref")
                    solution = solver.solve_with_steps(A, None)
                    console.print(Panel.fit("[bold]Example: RREF[/bold]", border_style="cyan"))
                    self._format_solution(solution)

                elif selection == "solve_system":
                    A = np.array([[1.0, 2.0], [3.0, 4.0]])
                    b = np.array([5.0, 11.0])
                    solver = get_solver("linear_system")
                    solution = solver.solve_with_steps(A, b)
                    console.print(Panel.fit("[bold]Example: Solve Linear System[/bold]", border_style="cyan"))
                    self._format_solution(solution)

                elif selection == "eigenvalues":
                    A = np.array([[4.0, -2.0], [1.0, 1.0]])
                    solver = get_solver("eigenvalue")
                    solution = solver.solve_with_steps(A)
                    console.print(Panel.fit("[bold]Example: Eigenvalues (2×2)[/bold]", border_style="cyan"))
                    self._format_solution(solution)

            except Exception as e:
                console.print(f"[red]Error running solver: {e}[/red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")

            console.print()
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
