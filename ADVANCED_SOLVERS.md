# Advanced Solvers - Path 4 Complete Guide

## Overview

Path 4 adds sophisticated step-by-step solvers for advanced linear algebra operations:
- **Gaussian Elimination** to Row Echelon Form (REF)
- **Row Reduction** to Reduced Row Echelon Form (RREF)
- **Linear Systems** (Ax = b) with complete solution analysis
- **Eigenvalues & Eigenvectors** with characteristic polynomial
- **LU Decomposition** (A = LU)
- **QR Decomposition** (A = QR using Gram-Schmidt)
- **SVD** (Singular Value Decomposition: A = UΣVᵀ)

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run demonstration of all solvers
linalg-tutor solve demo

# Gaussian elimination
linalg-tutor solve gaussian '2,1,-1;-3,-1,2;-2,1,2' -b '8,-11,-3'

# Find eigenvalues and eigenvectors
linalg-tutor solve eigenvalues '4,-2;1,1'

# LU decomposition
linalg-tutor solve lu '2,3;4,9'

# QR decomposition
linalg-tutor solve qr '1,1;1,0;0,1'
```

## Features Implemented

### 1. Gaussian Elimination Solver

Performs row operations to achieve Row Echelon Form (REF):
- Partial pivoting for numerical stability
- Shows every row swap and elimination step
- Works with augmented matrices for linear systems
- Detailed explanation of each operation

**Example:**
```bash
linalg-tutor solve gaussian '1,2,3;4,5,6;7,8,9'
```

**Output includes:**
- Starting matrix
- Pivot selection and row swaps
- Elimination steps (R₂ → R₂ - kR₁)
- Final REF matrix

### 2. RREF Solver

Extends Gaussian elimination to Reduced Row Echelon Form:
- Phase 1: Forward elimination (uses Gaussian elimination)
- Phase 2: Back-substitution
- Scales rows to get leading 1s
- Eliminates above pivots

**Example:**
```bash
linalg-tutor solve rref '1,2,3;4,5,6;7,8,9' -b '1,2,3'
```

**Output includes:**
- All REF steps
- Row scaling operations
- Back-substitution steps
- Final RREF matrix

### 3. Linear System Solver

Solves Ax = b with complete analysis:
- Uses RREF to solve the system
- Detects inconsistent systems (no solution)
- Identifies free variables (infinitely many solutions)
- Finds unique solutions

**Example:**
```bash
linalg-tutor solve linear-system '1,2;3,4' '5,11'
```

**Analysis includes:**
- System type (consistent/inconsistent)
- Number of solutions (unique/infinite/none)
- Free variables if any
- Solution vector

### 4. Eigenvalue Solver

Computes eigenvalues and eigenvectors:
- Special handling for 2×2 matrices (shows characteristic polynomial explicitly)
- Uses numerical methods for larger matrices
- Verifies each eigenvalue-eigenvector pair
- Sorts eigenvalues by magnitude

**Example:**
```bash
linalg-tutor solve eigenvalues '4,-2;1,1'
```

**For 2×2 matrices shows:**
- Characteristic polynomial: det(A - λI) = 0
- Quadratic expansion
- Quadratic formula application
- Real vs complex eigenvalue analysis

**For all matrices:**
- Computed eigenvalues
- Corresponding eigenvectors
- Verification: Av = λv
- Error analysis

### 5. LU Decomposition Solver

Decomposes A into Lower × Upper triangular matrices:
- Records Gaussian elimination multipliers in L
- Builds U through row operations
- Verifies L × U = A

**Example:**
```bash
linalg-tutor solve lu '2,3;4,9'
```

**Shows:**
- Elimination multipliers
- L matrix construction
- U matrix (upper triangular)
- Verification step

### 6. QR Decomposition Solver

Uses Gram-Schmidt orthogonalization:
- Orthogonalizes column vectors
- Normalizes to get orthonormal basis
- Records projection coefficients in R
- Verifies Q × R = A and QᵀQ = I

**Example:**
```bash
linalg-tutor solve qr '1,1;1,0;0,1'
```

**Shows:**
- Gram-Schmidt process for each column
- Projection calculations
- Normalization steps
- Orthogonality verification

### 7. SVD Solver

Computes Singular Value Decomposition:
- Finds singular values (eigenvalues of AᵀA)
- Computes left and right singular vectors
- Builds Σ matrix
- Analyzes rank and condition number

**Example:**
```bash
linalg-tutor solve svd '1,2;3,4;5,6'
```

**Shows:**
- Singular values (sorted descending)
- U matrix (left singular vectors)
- Σ matrix (diagonal with singular values)
- Vᵀ matrix (right singular vectors transposed)
- Matrix rank and condition number

## Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `solve gaussian` | Gaussian elimination to REF | `solve gaussian '1,2;3,4' -b '5,6'` |
| `solve rref` | Row reduction to RREF | `solve rref '1,2;3,4'` |
| `solve linear-system` | Solve Ax = b | `solve linear-system '1,2;3,4' '5,11'` |
| `solve eigenvalues` | Find eigenvalues/vectors | `solve eigenvalues '4,-2;1,1'` |
| `solve lu` | LU decomposition | `solve lu '2,3;4,9'` |
| `solve qr` | QR decomposition | `solve qr '1,1;1,0;0,1'` |
| `solve svd` | Singular value decomposition | `solve svd '1,2;3,4;5,6'` |
| `solve demo` | Run all solver demos | `solve demo` |

## Input Formats

### Matrices
Semicolon-separated rows, comma-separated columns:
```bash
'1,2;3,4'              # 2×2 matrix
'1,2,3;4,5,6;7,8,9'    # 3×3 matrix
'1,2;3,4;5,6'          # 3×2 matrix
```

### Vectors
Comma-separated components:
```bash
'1,2,3'                # 3D vector
'5,11'                 # 2D vector
```

### Augmented Matrices
Use `-b` or `--augment` flag:
```bash
solve gaussian '1,2;3,4' -b '5,6'
# Creates augmented matrix [1 2 | 5]
#                          [3 4 | 6]
```

## Output Format

All solvers provide:
1. **Step number** and **description**
2. **Mathematical expression** in a formatted panel
3. **Explanation** of what the step achieves
4. **Intermediate results** where applicable

Example step:
```
Step 3: Eliminate entry at row 1, column 0
╭──────────────────────────────────────╮
│ R1 → R1 - (3)R0                      │
╰──────────────────────────────────────╯
Make matrix[1,0] = 0 using row 0
```

## Module Structure

```
linalg_tutor/core/solver/
├── simple_solution.py          # SimpleSolution and SimpleSolutionStep dataclasses
├── gaussian_elimination.py     # Gaussian, RREF, LinearSystem solvers
├── eigenvalue.py               # Eigenvalue and CharacteristicPolynomial solvers
├── decomposition.py            # LU, QR, SVD solvers
└── __init__.py                 # Solver registry
```

## Implementation Details

### SimpleSolution Structure
```python
@dataclass
class SimpleSolutionStep:
    description: str
    mathematical_expression: str = ""
    explanation: str = ""
    intermediate_result: Optional[Any] = None

@dataclass
class SimpleSolution:
    operation: str
    final_answer: Any
    steps: List[SimpleSolutionStep] = field(default_factory=list)
```

### Gaussian Elimination Algorithm
1. For each column (left to right):
   - Find pivot (largest absolute value in column)
   - Swap rows if needed
   - Eliminate below pivot
2. Clean up near-zero entries
3. Result: Row Echelon Form

### RREF Algorithm
1. Get to REF using Gaussian elimination
2. Scale pivot rows (make leading entries = 1)
3. Back-substitute (eliminate above pivots)
4. Result: Reduced Row Echelon Form

### Eigenvalue Solver
- **2×2 matrices**: Explicit characteristic polynomial
  - Formula: det(A - λI) = λ² - (trace)λ + det(A)
  - Quadratic formula for roots
- **Larger matrices**: Numerical methods (NumPy.linalg.eig)
- Verification: Check Av = λv for each pair

### Decomposition Tolerances
- **Near-zero threshold**: 1e-10
- **Verification error threshold**: 1e-6
- Uses partial pivoting for numerical stability

## Example Sessions

### Example 1: Solve 2×2 Linear System
```bash
$ linalg-tutor solve linear-system '3,2;1,2' '7,5'
```

Shows:
- RREF reduction of [3 2 | 7; 1 2 | 5]
- Solution: x = [1, 2]
- System type: Unique solution

### Example 2: Eigenvalues of 2×2 Matrix
```bash
$ linalg-tutor solve eigenvalues '1,2;2,1'
```

Shows:
- Characteristic polynomial: λ² - 2λ - 3 = 0
- Eigenvalues: λ₁ = 3, λ₂ = -1
- Eigenvectors with verification

### Example 3: LU Decomposition
```bash
$ linalg-tutor solve lu '4,3;6,3'
```

Shows:
- Multiplier: L[1,0] = 1.5
- L = [[1, 0], [1.5, 1]]
- U = [[4, 3], [0, -1.5]]
- Verification: L×U = A

## Technical Notes

### Dependencies
- **NumPy**: All numerical computations
- **Rich**: Terminal formatting and panels
- **Typer**: CLI framework
- **Python dataclasses**: SimpleSolution structure

### Performance
- Gaussian elimination: O(n³) for n×n matrix
- Eigenvalue computation: O(n³) numerical
- QR decomposition (Gram-Schmidt): O(mn²)
- SVD: O(min(m²n, mn²))

### Numerical Stability
- Uses partial pivoting in Gaussian elimination
- Cleans up near-zero entries (< 1e-10)
- Normalizes eigenvectors consistently
- Verifies all decompositions

## Future Enhancements

Potential additions:
- **Pivoting strategies**: Full pivoting, scaled partial pivoting
- **Iterative solvers**: Jacobi, Gauss-Seidel
- **Power method**: For dominant eigenvalue
- **Inverse iteration**: For specific eigenvalues
- **Cholesky decomposition**: For positive definite matrices
- **Hessenberg form**: For efficient eigenvalue computation
- **Jordan canonical form**: For generalized eigenvectors

## Success Criteria

✅ All 7 solver types implemented
✅ Step-by-step explanations for all operations
✅ Verification steps included
✅ CLI commands registered and working
✅ Demo command showcasing all features
✅ Matrix/vector input parsing
✅ Error handling for invalid inputs
✅ Numerical stability (pivoting, tolerance)

## Conclusion

The Advanced Solvers system transforms the Linear Algebra Tutor into a complete computational tool. Students can now:
- **Learn** algorithms through step-by-step breakdowns
- **Verify** hand calculations with detailed solutions
- **Understand** matrix decompositions visually
- **Explore** eigenvalue theory interactively

Combined with Path 3 (Visualizations), this creates a powerful learning environment where abstract algorithms become concrete and understandable!
