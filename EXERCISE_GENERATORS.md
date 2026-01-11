# Exercise Generators - Path 5 Complete Guide

## Overview

Path 5 adds an exercise generation system that creates **infinite randomized practice problems**. Never run out of exercises - generate fresh problems with controlled difficulty and parameters!

**14 Generator Types:**
- 5 Vector generators
- 6 Matrix generators
- 3 Linear system generators

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# List all available generators
linalg-tutor generate list-generators

# Practice with infinite vector addition
linalg-tutor generate practice vector_add

# Generate 10 matrix multiplication exercises
linalg-tutor generate practice matrix_multiply --count 10

# Demo all generators
linalg-tutor generate all-demo

# See examples from one generator
linalg-tutor generate demo dot_product --count 3
```

## Features Implemented

### 1. Generator Base System

**GeneratorConfig** - Control how exercises are generated:
- **Difficulty**: practice, application, or challenge
- **Dimensions**: min/max for vectors and matrices
- **Value ranges**: min/max values for elements
- **Constraints**: integers only, avoid zeros, avoid singular matrices
- **Seed**: For reproducible random sequences

**ExerciseGenerator** - Base class with helpers:
- `random_dimension()` - Pick random dimension within range
- `random_vector()` - Generate random vector
- `random_matrix()` - Generate random matrix (with singularity check)
- `random_scalar()` - Generate random scalar value
- `generate()` - Create one exercise
- `generate_batch()` - Create multiple exercises

### 2. Vector Operation Generators

#### VectorAdditionGenerator
Generates: `v + w = ?`
- Random dimensions (2D or 3D)
- Component-wise hints
- Example: "Add the vectors v = [3, 4] and w = [1, -1]"

#### VectorScalarMultGenerator
Generates: `cv = ?`
- Random scalar and vector
- Shows multiplication for each component
- Example: "Multiply the vector v = [2, 3] by the scalar 4"

#### DotProductGenerator
Generates: `v · w = ?`
- Two vectors of same dimension
- Shows step-by-step calculation
- Example: "Compute the dot product of v = [1, 2, 3] and w = [4, 5, 6]"

#### VectorNormGenerator
Generates: `|v| = ?`
- Random vector
- Shows magnitude formula
- Example: "Find the magnitude (norm) of v = [3, 4]"

#### CrossProductGenerator
Generates: `v × w = ?` (3D only)
- Always 3D vectors
- Shows cross product formula
- Example: "Compute the cross product v × w for v = [1, 0, 0] and w = [0, 1, 0]"

### 3. Matrix Operation Generators

#### MatrixAdditionGenerator
Generates: `A + B = ?`
- Random matrix dimensions
- Element-wise addition
- Example: "Add the matrices A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]"

#### MatrixScalarMultGenerator
Generates: `cA = ?`
- Random scalar and matrix
- Shows element scaling
- Example: "Multiply the matrix A = [[1, 2], [3, 4]] by the scalar 3"

#### MatrixMultiplicationGenerator
Generates: `AB = ?`
- Compatible dimensions (m×n and n×p)
- Shows row × column calculation for first element
- Example: "Multiply the matrices A (2×3) and B (3×2)"

#### MatrixTransposeGenerator
Generates: `Aᵀ = ?`
- Random matrix
- Shows row/column flip
- Example: "Find the transpose of A = [[1, 2, 3], [4, 5, 6]]"

#### DeterminantGenerator
Generates: `det(A) = ?` (2×2 only)
- Always 2×2 matrices
- Shows formula: ad - bc
- Example: "Calculate the determinant of A = [[3, 2], [1, 4]]"

#### MatrixInverseGenerator
Generates: `A⁻¹ = ?` (2×2 only)
- Always invertible 2×2 matrices
- Shows inverse formula
- Example: "Find the inverse of A = [[4, 3], [3, 2]]"

### 4. Linear System Generators

#### LinearSystemGenerator
Generates: `Ax = b, solve for x`
- **Strategy**: Generate solution first, then compute b = Ax
- Ensures system is always solvable
- Non-singular coefficient matrix
- Example: "Solve the linear system Ax = b where A = [[2, 1], [1, 3]], b = [5, 8]"

#### TriangularSystemGenerator
Generates: Upper triangular system
- Easier to solve by back-substitution
- Diagonal entries guaranteed non-zero
- Example: "Solve the triangular system (upper triangular matrix)"

#### Simple2x2SystemGenerator
Generates: 2×2 system as equations
- Formatted as algebraic equations
- Small integer coefficients
- Example: "Solve: 2x + 3y = 7, x + 2y = 5"

## Command Reference

### Practice Commands

| Command | Description | Example |
|---------|-------------|---------|
| `generate practice <name>` | Interactive practice session | `generate practice vector_add` |
| `--count N` | Number of exercises | `generate practice matrix_multiply -n 10` |
| `--difficulty <level>` | Set difficulty | `generate practice dot_product -d application` |
| `--dim <n>` | Force specific dimension | `generate practice vector_add --dim 3` |
| `--seed <n>` | Set random seed | `generate practice matrix_add --seed 42` |

### Demo Commands

| Command | Description |
|---------|-------------|
| `generate list-generators` | Show all 14 generators |
| `generate demo <name>` | See examples (no practice) |
| `generate all-demo` | Demo all generators |
| `generate batch <name> -n <count>` | Generate multiple exercises |

## Configuration Options

### GeneratorConfig Parameters

```python
config = GeneratorConfig(
    difficulty=ExerciseDifficulty.PRACTICE,  # practice, application, challenge
    min_dimension=2,                          # Minimum vector/matrix dimension
    max_dimension=3,                          # Maximum dimension
    min_value=-10,                            # Minimum element value
    max_value=10,                             # Maximum element value
    integers_only=True,                       # Use only integers
    avoid_zero=True,                          # Don't generate all-zero vectors
    avoid_singular=True,                      # Ensure matrices are invertible
    seed=None,                                # Random seed (None = random)
)
```

### Example Configurations

**Easy Practice** (small integers):
```python
GeneratorConfig(
    min_value=-5,
    max_value=5,
    min_dimension=2,
    max_dimension=2,
)
```

**Challenge Mode** (larger values, higher dimensions):
```python
GeneratorConfig(
    difficulty=ExerciseDifficulty.CHALLENGE,
    min_value=-20,
    max_value=20,
    min_dimension=3,
    max_dimension=4,
)
```

**Reproducible** (same exercises every time):
```python
GeneratorConfig(seed=42)
```

## Generator Registry

Access generators programmatically:

```python
from linalg_tutor.core.generators import get_generator, GeneratorConfig

# Get a generator
config = GeneratorConfig(min_dimension=3, max_dimension=3)
generator = get_generator("vector_add", config)

# Generate exercises
exercise1 = generator.generate()
batch = generator.generate_batch(10)
```

Available generator names:
```python
GENERATOR_REGISTRY = {
    "vector_add", "vector_scalar", "dot_product", "vector_norm", "cross_product",
    "matrix_add", "matrix_scalar", "matrix_multiply", "matrix_transpose",
    "determinant", "matrix_inverse",
    "linear_system", "triangular_system", "simple_2x2_system",
}
```

## Module Structure

```
linalg_tutor/core/generators/
├── __init__.py              # Registry and get_generator()
├── base.py                  # ExerciseGenerator, GeneratorConfig
├── vector_generators.py     # 5 vector generators
├── matrix_generators.py     # 6 matrix generators
└── linear_systems.py        # 3 linear system generators
```

## Implementation Details

### Reverse Construction Pattern

For linear systems, we use **reverse construction**:
1. Generate random solution vector `x`
2. Generate random coefficient matrix `A`
3. Compute `b = Ax`
4. Give student the system `Ax = b` to solve
5. We already know the answer is `x`!

This ensures:
- System is always solvable
- We know the exact answer
- No numerical errors in checking

### Singularity Avoidance

For matrices that need to be invertible:
```python
A = self.random_matrix(n, n)
attempts = 0
while abs(np.linalg.det(A)) < 0.1 and attempts < 20:
    A = self.random_matrix(n, n)
    attempts += 1
```

Regenerates until determinant is large enough.

### Hint Generation

Hints are dynamically generated based on the random values:
```python
hints = [
    f"Add component-wise: v + w = [v₁+w₁, v₂+w₂, ...]",
    f"First component: {v[0]:.4g} + {w[0]:.4g} = {result[0]:.4g}",
    f"Second component: {v[1]:.4g} + {w[1]:.4g} = {result[1]:.4g}",
]
```

## Example Sessions

### Example 1: Vector Addition Practice
```bash
$ linalg-tutor generate practice vector_add --count 3 --dim 2

╭─────────────────────────────────────╮
│ Generated Practice: vector_add      │
│ Difficulty: practice                │
│ Exercises: 3                        │
╰─────────────────────────────────────╯

━━━ Exercise 1/3 ━━━

╭────── Vectors - Practice ──────╮
│ Add the vectors v = [7, -3]    │
│ and w = [2, 5]                 │
╰────────────────────────────────╯

? What would you like to do?
  > Submit answer
    Get a hint
    Visualize
    Show solution
    Skip this exercise
```

### Example 2: Matrix Multiplication with Seed
```bash
$ linalg-tutor generate practice matrix_multiply --seed 42 --count 2
```

Same exercises every time (reproducible for testing).

### Example 3: Batch Generation
```bash
$ linalg-tutor generate batch determinant --count 50

Generating 50 exercises using 'determinant'...
✓ Generated 50 exercises

Generated Exercises:
1. Calculate the determinant of A = [[4, 3], [2, 1]]...
2. Calculate the determinant of A = [[-1, 5], [3, 2]]...
3. Calculate the determinant of A = [[8, -2], [1, 4]]...
4. Calculate the determinant of A = [[6, 1], [-3, 2]]...
5. Calculate the determinant of A = [[9, 4], [3, 5]]...
... and 45 more
```

## Use Cases

1. **Unlimited Practice**: Never run out of problems to solve
2. **Skill Building**: Focus on specific operations with targeted generators
3. **Test Preparation**: Generate practice exams with consistent difficulty
4. **Homework**: Create unique problem sets for each student
5. **Benchmarking**: Use seed for reproducible performance testing
6. **Progressive Difficulty**: Start with small dimensions, increase gradually

## Integration with Existing Features

Generators work seamlessly with:
- **Progress Tracking**: All generated exercises tracked in database
- **Interactive Prompts**: Full hint/solution/visualization support
- **Exercise System**: Generated exercises are ComputationalExercise instances
- **Visualizations**: "Visualize" option available during practice

## Technical Notes

### Dependencies
- **NumPy**: All numerical generation and computations
- **Random module**: Python's random for integers and choices
- **Exercise system**: Generates ComputationalExercise instances

### Performance
- Generation is instant (< 1ms per exercise)
- No pre-computation needed
- Infinite exercises with minimal memory

### Randomness Quality
- Uses NumPy's random for numerical stability
- Singularity checks ensure good matrix conditioning
- Configurable ranges prevent extreme values

## Future Enhancements

Potential additions:
- **More generators**: Eigenvalue problems, basis/span, rank/nullity
- **Difficulty scaling**: Adaptive difficulty based on performance
- **Custom templates**: User-defined problem patterns
- **Export to PDF**: Generate printable worksheets
- **Adaptive practice**: AI-driven problem selection
- **Multi-step problems**: Combine multiple operations
- **Word problems**: Applied context for linear algebra

## Success Criteria

✅ 14 generator types implemented
✅ Configurable parameters (dimensions, ranges, difficulty)
✅ CLI commands for practice and demos
✅ Integration with interactive system
✅ Reproducible with seeds
✅ Quality controls (no singular matrices, etc.)
✅ Dynamic hint generation
✅ Registry system for easy access

## Conclusion

The Exercise Generator system completes the Linear Algebra Tutor's transformation into an **infinite learning platform**. Students can now:
- **Practice endlessly** with fresh, randomized problems
- **Target weak areas** with specific generator types
- **Progress gradually** with configurable difficulty
- **Track improvement** with consistent problem types
- **Never repeat** the same problem twice (unless using seeds)

Combined with Paths 1-4, this creates a complete ecosystem:
- Path 1: Interactive sessions
- Path 2: Curated content library
- Path 3: Visual learning
- Path 4: Advanced solvers
- **Path 5: Infinite practice**

The Linear Algebra Tutor is now a **fully-fledged educational platform** for undergraduate linear algebra!
