# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Linear Algebra Tutor is an interactive CLI tool for teaching undergraduate linear algebra through exercises, step-by-step solutions, and progress tracking. Built with Python 3.10+, it uses Typer for CLI, Rich for terminal UI, and SQLAlchemy for progress persistence.

## Development Setup

**Recommended: Use uv for faster dependency management**

```bash
# Setup with uv
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -e .

# Verify installation
linalg-tutor --help
linalg-tutor demo
```

**Alternative: Standard pip**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt requirements-dev.txt
pip install -e .
```

## Common Commands

### Testing
```bash
# Run all tests with coverage (configured in pyproject.toml)
pytest

# Run specific test file
pytest tests/unit/test_exercises/test_computational.py

# Run specific test
pytest tests/unit/test_exercises/test_computational.py::test_computational_exercise_correct_answer

# Run tests without coverage
pytest --no-cov

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "progress"
```

### Code Quality
```bash
# Format code (line length: 100)
black linalg_tutor tests

# Lint code
ruff check linalg_tutor tests

# Type checking
mypy linalg_tutor
```

### Running the Application

**Practice Sessions:**
```bash
linalg-tutor exercise practice vectors          # Curated exercises
linalg-tutor exercise practice matrices -n 10   # Specify count
linalg-tutor generate practice vector_add       # Infinite practice
linalg-tutor generate practice matrix_multiply --count 10 --seed 42
```

**Visualizations:**
```bash
linalg-tutor visualize vector 3,4
linalg-tutor visualize matrix '1,2;3,4'
linalg-tutor visualize dot-product 1,2 3,4
linalg-tutor visualize demo                     # Show all
```

**Advanced Solvers:**
```bash
linalg-tutor solve gaussian '2,1,-1;-3,-1,2;-2,1,2' -b '8,-11,-3'
linalg-tutor solve eigenvalues '4,-2;1,1'
linalg-tutor solve lu '2,3;4,9'
linalg-tutor solve demo                         # Show all
```

**Generators:**
```bash
linalg-tutor generate list-generators           # Show all 14
linalg-tutor generate demo vector_add           # See examples
linalg-tutor generate all-demo                  # Demo all
```

**Other:**
```bash
linalg-tutor start                              # Welcome screen
linalg-tutor topics                             # List topics
linalg-tutor demo                               # Quick demo
```

## Architecture

### Three-Layer Design

1. **Exercise System** (`linalg_tutor/core/exercises/`)
   - Abstract base: `Exercise` class with `check_answer()`, `get_solution()`, `get_correct_answer()`
   - Four concrete types:
     - `ComputationalExercise`: Numerical problems with tolerance-based validation (rtol=1e-5, atol=1e-8)
     - `MultipleChoiceExercise`: Conceptual questions
     - `TrueFalseExercise`: Quick conceptual checks
     - `FillInExercise`: Complete partial solutions (supports text and numerical)
   - All exercises use Pydantic for validation and return `ExerciseResult` objects

2. **Solver System** (`linalg_tutor/core/solver/`)
   - Abstract base: `Solver` class with `solve_with_steps()` returning `Solution` objects
   - `Solution` contains list of `SolutionStep` (description, mathematical_expression, explanation)
   - Registry pattern: `get_solver(operation_name)` retrieves solver instances
   - Current solvers: vector operations (add, dot, scalar multiply), matrix operations (multiply, add, transpose)
   - When adding new solvers: implement `Solver`, register in `_SOLVER_REGISTRY` in `__init__.py`

3. **Progress Tracking** (`linalg_tutor/core/progress/`)
   - SQLAlchemy models: `ExerciseAttempt`, `TopicProgress`
   - `ProgressTracker` manages database (default: `~/.linalg_tutor/data/progress.db`)
   - Mastery calculation: `accuracy * (0.7 + 0.3 * recency_factor)` with 5% daily decay
   - Use context manager: `with ProgressTracker() as tracker:`

### Data Flow for Exercises

```
User Answer → Exercise.check_answer() → ExerciseResult
                                      ↓
Exercise.get_solution() → Solver.solve_with_steps() → Solution with steps
                                      ↓
ExerciseResult + Exercise → ProgressTracker.record_attempt() → Database
```

### Adding New Exercise Types

1. Inherit from `Exercise` in `linalg_tutor/core/exercises/`
2. Implement required abstract methods: `check_answer()`, `get_solution()`, `get_correct_answer()`
3. Add to `__init__.py` exports
4. Create tests in `tests/unit/test_exercises/`
5. Add fixtures in `tests/conftest.py`

### Adding New Solvers

1. Inherit from `Solver` in `linalg_tutor/core/solver/`
2. Implement `solve_with_steps(inputs, expected)` returning `Solution`
3. Register in `_SOLVER_REGISTRY` dict in `linalg_tutor/core/solver/__init__.py`
4. Create tests in `tests/unit/test_solver/`

### Adding New Generators

1. Inherit from `ExerciseGenerator` in `linalg_tutor/core/generators/`
2. Implement `generate()` returning a `ComputationalExercise` (or other exercise type)
3. Use helper methods: `self.random_vector()`, `self.random_matrix()`, `self.random_scalar()`
4. Generate dynamic hints based on actual random values
5. Register in `GENERATOR_REGISTRY` dict in `linalg_tutor/core/generators/__init__.py`
6. Add to info list in `linalg_tutor/cli/commands/generate.py`
7. Create tests (optional but recommended)

**Example:**
```python
class VectorAdditionGenerator(ExerciseGenerator):
    def generate(self) -> ComputationalExercise:
        dimension = self.random_dimension()
        v = self.random_vector(dimension)
        w = self.random_vector(dimension)
        result = v + w

        return ComputationalExercise(
            exercise_id=f"gen_vector_add_{uuid.uuid4().hex[:8]}",
            topic="vectors",
            difficulty=self.config.difficulty,
            question=f"Add the vectors v = {v.tolist()} and w = {w.tolist()}",
            operation="vector_add",
            inputs={"v": v, "w": w},
            expected_output=result,
            hints=[
                "Add component-wise: v + w = [v₁+w₁, v₂+w₂, ...]",
                f"First component: {v[0]:.4g} + {w[0]:.4g} = {result[0]:.4g}",
            ],
            tags=["generated", "vector_addition"],
        )
```

### Numerical Tolerance

All numerical comparisons use `numpy.allclose()`:
- Default relative tolerance: `1e-5`
- Default absolute tolerance: `1e-8`
- Configured per-exercise via `tolerance` and `atol` fields on `ComputationalExercise`

### Exercise Generator System (`linalg_tutor/core/generators/`)

**Path 5: Infinite Practice**

- **GeneratorConfig**: Dataclass controlling generation (difficulty, dimensions, value ranges, seed)
- **ExerciseGenerator**: Abstract base class with helper methods
  - `random_vector()`, `random_matrix()`, `random_scalar()` - generate random inputs
  - `generate()` - create one exercise
  - `generate_batch(count)` - create multiple exercises
- **14 Generator Types**:
  - Vector: `VectorAdditionGenerator`, `VectorScalarMultGenerator`, `DotProductGenerator`, `VectorNormGenerator`, `CrossProductGenerator`
  - Matrix: `MatrixAdditionGenerator`, `MatrixScalarMultGenerator`, `MatrixMultiplicationGenerator`, `MatrixTransposeGenerator`, `DeterminantGenerator`, `MatrixInverseGenerator`
  - Linear Systems: `LinearSystemGenerator`, `TriangularSystemGenerator`, `Simple2x2SystemGenerator`
- **Registry pattern**: `get_generator(name, config)` retrieves generator instances
- **Quality controls**: Singularity avoidance, non-zero constraints, numerical stability
- **Reproducibility**: Seed-based random generation for consistent problem sets

**Key Pattern - Reverse Construction** (for linear systems):
```python
# Generate solution first, then compute b = Ax to ensure solvability
x_solution = self.random_vector(n)
A = self.random_matrix(n, n)  # Non-singular
b = A @ x_solution
# Now give student Ax = b to solve, we know x is the answer
```

### Visualization System (`linalg_tutor/visualization/`)

**Path 3: Visual Learning**

- ASCII art for 2D vectors with coordinate plots
- Rich table formatting for matrices with properties
- Geometric interpretations (angles, projections, orthogonality)
- Step-by-step visualizations for operations
- 10 standalone visualization commands

### Advanced Solver System (`linalg_tutor/core/solver/`)

**Path 4: Step-by-Step Solutions**

Beyond basic operations, includes:
- Gaussian elimination with row operations
- RREF (Reduced Row Echelon Form)
- Eigenvalue calculation with characteristic polynomial (2×2)
- Matrix decompositions: LU, QR, SVD
- Linear system solver with solution analysis

### CLI Structure

- Entry point: `linalg_tutor/cli/app.py` using Typer
- Rich library for terminal formatting (tables, panels, syntax highlighting)
- Commands organized in `linalg_tutor/cli/commands/`:
  - `exercise.py` - Practice curated exercises
  - `visualize.py` - Visualization commands
  - `solve.py` - Advanced solver commands
  - `generate.py` - Generator practice commands
- Interactive UI: `linalg_tutor/cli/ui/prompts.py` with fixed-screen layout

## Key Constraints

- **Python 3.10+** required (uses modern type hints)
- **Line length: 100 characters** (black and ruff configured)
- **Pydantic models** used throughout for validation (note: using v2 with deprecated class-based Config)
- **NumPy arrays** are the standard for all matrix/vector computations
- **Test coverage** should remain above 60% (currently 66%)

## Testing Philosophy

- Fixtures in `tests/conftest.py` create sample exercises for reuse
- In-memory SQLite (`:memory:`) for progress tracker tests to avoid file I/O
- Computational exercises test: correct answers, wrong answers, shape mismatches, tolerance
- Solver tests verify: step generation, correct final answers, proper structure

## Current State (All 5 Paths Complete)

**Path 1: Interactive Sessions** ✅
- 46 curated exercises (16 vectors + 30 matrices)
- Interactive prompts with questionary
- Fixed-screen UI with Rich layouts
- Hints, visualizations, and solutions during practice
- Progress tracking with mastery calculation

**Path 2: Content Library** ✅
- 46 curated exercises across vectors and matrices
- Multiple difficulty levels (practice, application, challenge)
- 4 exercise types (computational, multiple choice, true/false, fill-in)
- Content registry system

**Path 3: Visualizations** ✅
- ASCII art for 2D vectors
- Rich matrix displays with properties
- Geometric interpretations (angles, projections, orthogonality)
- 10 standalone visualization commands
- Integrated into practice sessions

**Path 4: Advanced Solvers** ✅
- 7 solver types with step-by-step explanations
- Gaussian elimination, RREF, eigenvalues (2×2)
- Matrix decompositions (LU, QR, SVD)
- Linear system solver with solution analysis

**Path 5: Exercise Generators** ✅
- 14 configurable generators for infinite practice
- Quality controls (non-singular matrices, numerical stability)
- Reproducible with seeds
- Dynamic hint generation
- Registry pattern for easy access

**Statistics:**
- ~8,000+ lines of code
- 46 curated exercises
- 14 exercise generators
- 7 advanced solvers
- 10 visualization commands
- 40+ CLI commands
- 31 passing tests (66% coverage)

## Database Schema

SQLite tables (managed by SQLAlchemy):
- `exercise_attempts`: Individual attempt records (exercise_id, topic, correct, time_spent, hints_used)
- `topic_progress`: Aggregated topic statistics (topic, exercises_attempted, mastery_level, last_practiced)

User data stored in `~/.linalg_tutor/data/progress.db` by default.
