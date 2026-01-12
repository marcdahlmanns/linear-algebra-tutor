# Linear Algebra Tutor

A comprehensive command-line tool for learning undergraduate linear algebra through **interactive exercises**, **step-by-step solutions**, **visual demonstrations**, and **infinite practice**.

## 🎯 What Makes This Special

- **Menu-Driven Interface** - Just type `linalg-tutor` and navigate with arrow keys
- **Compact Professional UX** - Question always visible, no wasted space, no scrolling needed
- **46 Curated Exercises** across vectors and matrices with detailed explanations
- **14 Exercise Generators** for infinite randomized practice
- **Interactive Sessions** with hints, visualizations, and immediate feedback
- **Advanced Solvers** showing step-by-step solutions for complex operations
- **Beautiful Visualizations** including ASCII art and geometric interpretations
- **Progress Tracking** with mastery calculation and automatic session state

## ✨ Features

### 1. Guided Learning Interface
- **Menu-Driven Navigation**: No commands to memorize, just arrow keys
- **10-Chapter Learning Path**: Vectors → Matrices → Linear Systems → ... → Applications
- **Automatic Progress Saving**: Resume exactly where you left off
- **Chapter Status Indicators**: ✓ Complete, ⚡ In Progress, → Current, ○ Not Started
- **Session State Management**: Tracks exercises completed and time spent

### 2. Interactive Practice Sessions
- **4 Exercise Types**: Computational, Multiple Choice, True/False, Fill-in
- **Progressive Hints**: Up to 3 hints per exercise
- **Immediate Feedback**: Know instantly if you're correct
- **Visualizations**: See ASCII art and geometric interpretations during practice
- **Solutions**: Full step-by-step explanations when needed
- **Compact Fixed-Screen UI**: Question always visible, minimal scrolling (80×15 terminal minimum)

### 3. Infinite Practice with Generators
- **14 Generators**: Vector ops, matrix ops, linear systems
- **Configurable**: Control dimensions, difficulty, value ranges
- **Reproducible**: Use seeds for consistent problem sets
- **Quality Controlled**: Avoids singular matrices, degenerate cases

### 4. Visual Learning
- **ASCII Vector Art**: 2D vectors plotted with coordinates
- **Rich Matrix Tables**: Beautiful formatted displays
- **Geometric Interpretations**: Angles, projections, orthogonality
- **10 Visualization Commands**: Standalone tools for exploration

### 5. Advanced Computational Solvers
- **Gaussian Elimination**: Row operations to REF
- **RREF**: Complete row reduction
- **Eigenvalues**: With characteristic polynomial (2×2)
- **Matrix Decompositions**: LU, QR, SVD
- **Linear Systems**: Complete solution analysis

### 6. Progress Tracking
- **SQLite Database**: Persistent progress storage
- **Mastery Calculation**: Accuracy × recency with decay
- **Session Statistics**: Time, accuracy, improvement metrics
- **Automatic State Management**: Session progress saved to JSON

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd linearAlgebra

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### First Steps

**Easiest Way: Guided Learning Menu** (Recommended)

```bash
# Just run - no commands to memorize!
linalg-tutor

# Navigate with arrow keys through intuitive menus:
# → Continue Learning / Select Chapter / View Progress / Quick Practice
```

**Advanced: Direct Commands** (Optional for power users)

```bash
# Practice with curated exercises
linalg-tutor exercise practice vectors

# Generate infinite practice problems
linalg-tutor generate practice vector_add --count 10

# Visualize concepts
linalg-tutor visualize vector 3,4

# Solve with step-by-step explanations
linalg-tutor solve eigenvalues '4,-2;1,1'
```

## 📚 Command Reference

### Exercise Practice

```bash
# Practice curated exercises
linalg-tutor exercise practice vectors           # Default: 5 exercises
linalg-tutor exercise practice matrices -n 10    # Specify count
linalg-tutor exercise list vectors               # List all exercises

# Generate infinite practice
linalg-tutor generate list-generators            # Show all 14 generators
linalg-tutor generate practice vector_add        # Infinite vector addition
linalg-tutor generate practice matrix_multiply --count 10
linalg-tutor generate all-demo                   # Demo all generators
```

### Visualizations

```bash
# Vectors
linalg-tutor visualize vector 3,4                # 2D vector plot
linalg-tutor visualize dot-product 1,2 3,4       # Dot product with geometry
linalg-tutor visualize vector-add 2,1 1,3        # Vector addition

# Matrices
linalg-tutor visualize matrix '1,2;3,4'          # Matrix with properties
linalg-tutor visualize matrix-multiply '1,2;3,4' '5,6;7,8'
linalg-tutor visualize determinant '3,2;1,4'    # Determinant calculation

# Other
linalg-tutor visualize orthogonal 1,0 0,1        # Check orthogonality
linalg-tutor visualize projection 3,4 1,0        # Vector projection
linalg-tutor visualize demo                      # Demo all visualizations
```

### Advanced Solvers

```bash
# Linear systems
linalg-tutor solve gaussian '2,1,-1;-3,-1,2;-2,1,2' -b '8,-11,-3'
linalg-tutor solve rref '1,2;3,4'
linalg-tutor solve linear-system '1,2;3,4' '5,11'

# Eigenvalues
linalg-tutor solve eigenvalues '4,-2;1,1'       # Shows characteristic polynomial

# Decompositions
linalg-tutor solve lu '2,3;4,9'                 # LU decomposition
linalg-tutor solve qr '1,1;1,0;0,1'             # QR decomposition
linalg-tutor solve svd '1,2;3,4;5,6'            # SVD

# Demo
linalg-tutor solve demo                         # Demo all solvers
```

### Other Commands

```bash
linalg-tutor topics                             # List all topics
linalg-tutor demo                               # Run demo exercise
linalg-tutor version                            # Show version
linalg-tutor --help                             # Show all commands
```

## 📖 Topics Covered

### Currently Implemented
1. **Vectors**: Addition, scalar multiplication, dot product, norm, cross product
2. **Matrices**: Addition, multiplication, transpose, determinant, inverse
3. **Linear Systems**: Gaussian elimination, RREF, solution analysis

### Content Library
- **46 Curated Exercises**: 16 vector + 30 matrix exercises
- **14 Exercise Generators**: Infinite practice for all operations
- **7 Advanced Solvers**: Step-by-step for complex operations

### Planned Topics
4. Vector Spaces and Subspaces
5. Orthogonality and Projections
6. Eigenvalues and Diagonalization
7. Linear Transformations
8. Matrix Decompositions (complete)
9. Singular Value Decomposition
10. Applications (PCA, least squares, etc.)

## 🎓 Example Session

```bash
$ linalg-tutor exercise practice vectors -n 3

╭───────────────────────────╮
│ Practice Session: Vectors │
│ Difficulty: all           │
│ Exercises: 3              │
╰───────────────────────────╯

━━━ Exercise 1/3 ━━━

╭─────────────── Vectors - Practice ───────────────╮
│ Add the vectors v = [2, 3] and w = [1, -1]       │
╰──────────────────────────────────────────────────╯

? What would you like to do?
  > Submit answer
    Get a hint
    Visualize          ← NEW! See ASCII art
    Show solution
    Skip this exercise

? Answer: [3, 2]

✓ Correct!

━━━ Session Summary ━━━

Exercises Completed  3
Correct              3
Accuracy             100%
Mastery Level        85.2%
```

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=linalg_tutor

# Specific test file
pytest tests/test_exercises.py
```

### Code Quality

```bash
# Format code
black linalg_tutor tests

# Lint
ruff check linalg_tutor tests

# Type check
mypy linalg_tutor
```

## 📁 Project Structure

```
linearAlgebra/
├── linalg_tutor/                  # Main package
│   ├── cli/                       # Command-line interface
│   │   ├── app.py                # Main Typer app
│   │   ├── commands/             # Command modules
│   │   │   ├── exercise.py      # Exercise practice commands
│   │   │   ├── visualize.py     # Visualization commands
│   │   │   ├── solve.py         # Advanced solver commands
│   │   │   └── generate.py      # Generator commands
│   │   └── ui/                   # UI components
│   │       └── prompts.py        # Interactive prompts
│   ├── core/                      # Core business logic
│   │   ├── exercises/            # Exercise system
│   │   │   ├── base.py          # Base classes
│   │   │   ├── computational.py # Computational exercises
│   │   │   └── ...
│   │   ├── generators/           # Exercise generators
│   │   │   ├── base.py          # Generator base
│   │   │   ├── vector_generators.py
│   │   │   ├── matrix_generators.py
│   │   │   └── linear_systems.py
│   │   ├── solver/               # Step-by-step solvers
│   │   │   ├── gaussian_elimination.py
│   │   │   ├── eigenvalue.py
│   │   │   └── decomposition.py
│   │   ├── progress/             # Progress tracking
│   │   └── lessons/              # Lesson system
│   ├── content/                   # Content library
│   │   └── exercises_library.py  # 46 curated exercises
│   ├── visualization/             # Visualization tools
│   │   ├── vector_viz.py         # Vector visualizations
│   │   ├── matrix_viz.py         # Matrix visualizations
│   │   └── geometric.py          # Geometric interpretations
│   └── utils/                     # Utilities
├── tests/                         # Test suite (31 passing tests)
├── data/                          # User progress database
└── docs/                          # Documentation
    ├── INTERACTIVE_DEMO.md       # Interactive features guide
    ├── VISUALIZATIONS.md         # Visualization system guide
    ├── ADVANCED_SOLVERS.md       # Solver system guide
    ├── EXERCISE_GENERATORS.md    # Generator system guide
    └── CLAUDE.md                 # Developer guide
```

## 📊 Statistics

- **Total Lines of Code**: ~8,000+
- **Exercise Types**: 4 (Computational, Multiple Choice, True/False, Fill-in)
- **Curated Exercises**: 46 (16 vectors + 30 matrices)
- **Exercise Generators**: 14 (infinite practice)
- **Advanced Solvers**: 7 (Gaussian, RREF, eigenvalues, LU, QR, SVD, linear systems)
- **Visualization Commands**: 10
- **CLI Commands**: 40+
- **Test Coverage**: 66%
- **Python Files**: 60+

## 🎯 Use Cases

1. **Self-Study**: Learn linear algebra at your own pace
2. **Test Preparation**: Practice with infinite randomized problems
3. **Homework Helper**: Get step-by-step solutions
4. **Concept Visualization**: Understand geometry of linear algebra
5. **Skill Building**: Target specific operations with generators
6. **Teaching Aid**: Generate problem sets for students

## 🌟 Highlights

### Path 1: Interactive Sessions ✅
- 46 curated exercises with hints and solutions
- Interactive practice with immediate feedback
- Progress tracking and mastery calculation

### Path 2: Content Library ✅
- Comprehensive exercises for vectors and matrices
- Multiple difficulty levels
- Conceptual and computational problems

### Path 3: Visualizations ✅
- ASCII art for 2D vectors
- Rich matrix displays with properties
- Geometric interpretations (angles, projections)
- 10 standalone visualization commands

### Path 4: Advanced Solvers ✅
- 7 solver types with step-by-step explanations
- Gaussian elimination, RREF, eigenvalues
- Matrix decompositions (LU, QR, SVD)
- Linear system solver with solution analysis

### Path 5: Exercise Generators ✅
- 14 configurable generators
- Infinite practice with reproducible seeds
- Quality controls (non-singular matrices, etc.)
- Dynamic hint generation

## 📝 Documentation

- **[INTERACTIVE_DEMO.md](INTERACTIVE_DEMO.md)**: Interactive features and usage examples
- **[VISUALIZATIONS.md](VISUALIZATIONS.md)**: Complete visualization system guide
- **[ADVANCED_SOLVERS.md](ADVANCED_SOLVERS.md)**: Step-by-step solver documentation
- **[EXERCISE_GENERATORS.md](EXERCISE_GENERATORS.md)**: Generator system guide
- **[CLAUDE.md](CLAUDE.md)**: Developer guide for contributors

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional exercise generators
- More curated exercises
- New topics (vector spaces, transformations, etc.)
- Improved visualizations
- Bug fixes and optimizations

## 📄 License

MIT License

## 🙏 Acknowledgments

Built with:
- **Typer**: CLI framework
- **Rich**: Beautiful terminal UI
- **NumPy**: Numerical computations
- **Questionary**: Interactive prompts
- **SQLAlchemy**: Progress tracking
- **Pydantic**: Data validation

---

**Ready to master linear algebra?** Start with `linalg-tutor start` and explore the world of vectors, matrices, and linear transformations! 🚀
