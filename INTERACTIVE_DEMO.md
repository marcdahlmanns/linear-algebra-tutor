# Interactive Exercise System - Demo Guide

## What We Built

### Path 1: Interactive Sessions ✅
You can now **actually practice** linear algebra with a fully interactive exercise system!

### Path 2: Content Library ✅
Comprehensive exercise library with 46 exercises across vectors and matrices!

### Path 3: Visualizations ✅
Beautiful ASCII art and geometric interpretations for visual learning!

## Quick Start

```bash
# Activate your environment
source .venv/bin/activate

# Practice 5 vector exercises (default)
linalg-tutor exercise practice vectors

# Practice 10 exercises
linalg-tutor exercise practice vectors --count 10

# Practice only "practice" difficulty
linalg-tutor exercise practice vectors --difficulty practice --count 5

# Practice challenge problems
linalg-tutor exercise practice vectors --difficulty challenge

# List all available exercises
linalg-tutor exercise list vectors

# See all commands
linalg-tutor --help
```

## Features Implemented

### 1. Interactive Exercise Flow
- **Submit answer**: Enter your solution (with smart parsing for vectors/matrices)
- **Get a hint**: Progressive hints (up to 3 per exercise)
- **Visualize** (NEW in Path 3): See ASCII art and geometric interpretations
- **Show solution**: See full step-by-step explanation
- **Skip exercise**: Move on if stuck

### 2. Four Exercise Types
- **Computational**: Numerical problems (vectors, dot products, matrix multiplication)
- **Multiple Choice**: Conceptual questions with explanations
- **True/False**: Quick property checks
- **Fill-in-the-blank**: Complete definitions or calculate single values

### 3. Rich Exercise Library (16 exercises for Vectors)
- Vector addition (2D and 3D)
- Vector subtraction
- Scalar multiplication
- Dot product
- Application problems (combining operations)
- Conceptual questions
- Property verification

### 4. Progress Tracking
- Automatic recording of attempts
- Mastery level calculation (accuracy × recency)
- Session summaries with statistics
- Smart topic recommendations

### 5. Beautiful Terminal UI
- Color-coded feedback (green=correct, red=incorrect, yellow=hints)
- Rich panels and tables
- Progress indicators
- Difficulty badges

## Example Session Flow

```
$ linalg-tutor exercise practice vectors --count 3

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
    Show solution
    Skip this exercise

? What would you like to do? Submit answer

Enter your answer:
  • For vectors: [1, 2, 3]
  • For matrices: [[1, 2], [3, 4]]
  • For scalars: 42 or 3.14

? Answer: [3, 2]

✓ Correct!

━━━ Exercise 2/3 ━━━
...

━━━ Session Summary ━━━

Exercises Completed  3
Correct              2
Incorrect            1
Accuracy             66.7%
Total Time           45.3s
Average Time         15.1s per exercise

Mastery Level: 72.5%
Good progress! Keep practicing.
```

## Input Formats

### Vectors
```python
[1, 2, 3]           # 3D vector
[4, -2]             # 2D vector
```

### Matrices
```python
[[1, 2], [3, 4]]    # 2×2 matrix
[[1, 2, 3],
 [4, 5, 6]]         # 2×3 matrix
```

### Scalars
```python
42                  # Integer
3.14                # Float
-5                  # Negative
```

### Multiple Choice
Just select from the list using arrow keys and Enter

### True/False
Select "True" or "False" from the list

## What Makes This Special

1. **Real Practice**: Not just reading - actually solving problems
2. **Immediate Feedback**: Know right away if you're correct
3. **Hints on Demand**: Get help when stuck without seeing the full solution
4. **Visual Learning**: ASCII art and geometric interpretations (Path 3)
5. **Progress Tracking**: See your improvement over time
6. **Step-by-Step Solutions**: Learn the proper method when you need it
7. **Variety**: Mix of computational, conceptual, and verification problems
8. **Interactive Exploration**: Visualize any vector or matrix on demand

## Exercise Library Summary

**Current topics: Vectors & Matrices (46 exercises)**

**Vectors** (16 exercises):
- 13 practice-level, 1 application-level, 2 challenge-level
- 9 Computational, 2 Multiple Choice, 3 True/False, 2 Fill-in

**Matrices** (30 exercises):
- 26 practice-level, 2 application-level, 2 challenge-level
- 16 Computational, 4 Multiple Choice, 5 True/False, 5 Fill-in

## Visualization Features (Path 3)

### Standalone Visualization Commands
```bash
# Visualize a vector with ASCII art
linalg-tutor visualize vector 3,4

# Visualize a matrix with properties
linalg-tutor visualize matrix '1,2;3,4'

# Dot product with geometric interpretation
linalg-tutor visualize dot-product 1,2 3,4

# Matrix multiplication with step-by-step
linalg-tutor visualize matrix-multiply '1,2;3,4' '5,6;7,8'

# Check if vectors are orthogonal
linalg-tutor visualize orthogonal 1,0 0,1

# See all visualizations
linalg-tutor visualize demo
```

### Integrated Exercise Visualizations
During exercises, select "Visualize" to see:
- **Vector addition**: ASCII parallelogram rule (2D)
- **Dot product**: Component calculation + angle between vectors
- **Matrix operations**: Step-by-step element calculations
- **Determinant**: Formula breakdown for 2×2 matrices

### What You Get
- **ASCII art plots** for 2D vectors (arrows with coordinates)
- **Rich table displays** for matrices with highlighting
- **Geometric interpretations** (angles, projections, orthogonality)
- **Step-by-step breakdowns** for complex operations
- **Property analysis** (determinant, trace, invertibility)

See `VISUALIZATIONS.md` for complete guide!

## Next Steps

Want to expand? Add:
- **Path 4**: Advanced solvers (Gaussian elimination, eigenvalues, SVD)
- **Path 5**: Exercise generators (infinite practice with randomized problems)
- More topics (linear systems, vector spaces, eigenvalues, transformations)
- More exercises per topic (500+ total goal)
- Challenge problems with detailed explanations
- Timed modes and competitive features
- Achievement badges and progress milestones

## Technical Notes

- Uses `questionary` for interactive prompts
- `rich` for beautiful terminal output
- Progress stored in SQLite at `~/.linalg_tutor/data/progress.db`
- All exercises have step-by-step solutions via solver system
