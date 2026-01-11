"""Sample exercise library for practice."""

import numpy as np

from ..core.exercises import (
    ComputationalExercise,
    MultipleChoiceExercise,
    TrueFalseExercise,
    FillInExercise,
    ExerciseDifficulty,
)

# ============================================================================
# VECTORS - ADDITION AND SUBTRACTION
# ============================================================================

VECTOR_EXERCISES = [
    # Basic vector addition
    ComputationalExercise(
        exercise_id="vec_add_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add the vectors v = [2, 3] and w = [1, -1]",
        operation="vector_add",
        inputs={"v": np.array([2, 3]), "w": np.array([1, -1])},
        expected_output=np.array([3, 2]),
        hints=[
            "Add corresponding components: [v₁+w₁, v₂+w₂]",
            "First component: 2 + 1 = ?",
            "Second component: 3 + (-1) = ?",
        ],
        tags=["vector_addition", "2d"],
    ),
    ComputationalExercise(
        exercise_id="vec_add_02",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add the vectors v = [5, -2] and w = [-3, 4]",
        operation="vector_add",
        inputs={"v": np.array([5, -2]), "w": np.array([-3, 4])},
        expected_output=np.array([2, 2]),
        hints=[
            "Add component-wise",
            "First: 5 + (-3) = ?",
            "Second: -2 + 4 = ?",
        ],
        tags=["vector_addition", "2d"],
    ),
    ComputationalExercise(
        exercise_id="vec_add_03",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add the vectors v = [1, 2, 3] and w = [4, 5, 6]",
        operation="vector_add",
        inputs={"v": np.array([1, 2, 3]), "w": np.array([4, 5, 6])},
        expected_output=np.array([5, 7, 9]),
        hints=[
            "Add each component separately",
            "Result will also be a 3D vector",
        ],
        tags=["vector_addition", "3d"],
    ),
    # Vector subtraction
    ComputationalExercise(
        exercise_id="vec_sub_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Subtract w from v where v = [5, 8] and w = [2, 3]",
        operation="vector_add",  # Implemented as addition with negative
        inputs={"v": np.array([5, 8]), "w": np.array([-2, -3])},
        expected_output=np.array([3, 5]),
        hints=[
            "v - w means subtract each component",
            "v - w = [5-2, 8-3]",
        ],
        tags=["vector_subtraction", "2d"],
    ),
    # Scalar multiplication
    ComputationalExercise(
        exercise_id="vec_scalar_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply the vector v = [2, -3] by the scalar 4",
        operation="vector_scalar_multiply",
        inputs={"scalar": 4, "vector": np.array([2, -3])},
        expected_output=np.array([8, -12]),
        hints=[
            "Multiply each component by the scalar",
            "4 × [2, -3] = [4×2, 4×(-3)]",
        ],
        tags=["scalar_multiplication", "2d"],
    ),
    ComputationalExercise(
        exercise_id="vec_scalar_02",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply the vector v = [1, 2, 3] by the scalar -2",
        operation="vector_scalar_multiply",
        inputs={"scalar": -2, "vector": np.array([1, 2, 3])},
        expected_output=np.array([-2, -4, -6]),
        hints=[
            "Multiply each component by -2",
            "Negative scalar reverses direction",
        ],
        tags=["scalar_multiplication", "3d"],
    ),
    # Dot product
    ComputationalExercise(
        exercise_id="vec_dot_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Compute the dot product of v = [3, 4] and w = [1, 2]",
        operation="vector_dot",
        inputs={"v": np.array([3, 4]), "w": np.array([1, 2])},
        expected_output=11.0,
        hints=[
            "Dot product: v·w = v₁w₁ + v₂w₂",
            "Multiply corresponding components and sum",
            "(3)(1) + (4)(2) = ?",
        ],
        tags=["dot_product", "2d"],
    ),
    ComputationalExercise(
        exercise_id="vec_dot_02",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Compute the dot product of v = [1, 2, 3] and w = [4, 5, 6]",
        operation="vector_dot",
        inputs={"v": np.array([1, 2, 3]), "w": np.array([4, 5, 6])},
        expected_output=32.0,
        hints=[
            "v·w = (1)(4) + (2)(5) + (3)(6)",
            "Calculate each product first, then sum",
        ],
        tags=["dot_product", "3d"],
    ),
    # Application level - combining operations
    ComputationalExercise(
        exercise_id="vec_app_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Compute 2v + 3w where v = [1, 0] and w = [0, 1]",
        operation="vector_add",
        inputs={"v": np.array([2, 0]), "w": np.array([0, 3])},
        expected_output=np.array([2, 3]),
        hints=[
            "First: Compute 2v and 3w separately",
            "2v = [2, 0] and 3w = [0, 3]",
            "Then add: [2, 0] + [0, 3]",
        ],
        tags=["vector_addition", "scalar_multiplication", "combination"],
    ),
    # Multiple choice questions
    MultipleChoiceExercise(
        exercise_id="vec_mc_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What is the geometric interpretation of the dot product v·w when both vectors are unit vectors?",
        choices=[
            "The angle between the vectors",
            "The cosine of the angle between the vectors",
            "The length of the projection",
            "The area of the parallelogram",
        ],
        correct_index=1,
        explanation="For unit vectors, v·w = ||v|| ||w|| cos(θ) = (1)(1) cos(θ) = cos(θ)",
        tags=["dot_product", "geometry", "conceptual"],
    ),
    MultipleChoiceExercise(
        exercise_id="vec_mc_02",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="If v·w = 0 and both v and w are non-zero, what can we conclude?",
        choices=[
            "The vectors are parallel",
            "The vectors are orthogonal (perpendicular)",
            "The vectors have equal magnitude",
            "One vector is the zero vector",
        ],
        correct_index=1,
        explanation="A dot product of zero means the vectors are orthogonal (perpendicular) because v·w = ||v|| ||w|| cos(90°) = 0",
        tags=["dot_product", "orthogonality", "conceptual"],
    ),
    # True/False questions
    TrueFalseExercise(
        exercise_id="vec_tf_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Vector addition is commutative: v + w = w + v for all vectors v and w",
        correct_answer=True,
        explanation="Vector addition is commutative because we add corresponding components, and scalar addition is commutative.",
        tags=["vector_addition", "properties"],
    ),
    TrueFalseExercise(
        exercise_id="vec_tf_02",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The dot product of a vector with itself is always zero",
        correct_answer=False,
        explanation="The dot product v·v = ||v||², which is the square of the vector's magnitude. It's only zero if v is the zero vector.",
        tags=["dot_product", "properties"],
    ),
    TrueFalseExercise(
        exercise_id="vec_tf_03",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiplying a vector by a negative scalar reverses its direction",
        correct_answer=True,
        explanation="A negative scalar flips the sign of all components, which reverses the vector's direction.",
        tags=["scalar_multiplication", "properties"],
    ),
    # Fill-in questions
    FillInExercise(
        exercise_id="vec_fill_01",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The magnitude of the vector [3, 4] is ___",
        answer_type="numerical",
        correct_answer=5.0,
        hints=[
            "Magnitude formula: ||v|| = √(v₁² + v₂²)",
            "Calculate √(3² + 4²)",
            "This is a 3-4-5 right triangle",
        ],
        tags=["magnitude", "pythagorean"],
    ),
    FillInExercise(
        exercise_id="vec_fill_02",
        topic="vectors",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="A vector with magnitude 1 is called a ___ vector",
        answer_type="text",
        correct_answer="unit",
        case_sensitive=False,
        hints=[
            "This type of vector has length 1",
            "Often denoted with a hat: û",
        ],
        tags=["terminology", "unit_vectors"],
    ),
]


# ============================================================================
# MATRICES - OPERATIONS AND PROPERTIES
# ============================================================================

MATRIX_EXERCISES = [
    # Matrix addition
    ComputationalExercise(
        exercise_id="mat_add_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add the matrices A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]]",
        operation="matrix_add",
        inputs={"A": np.array([[1, 2], [3, 4]]), "B": np.array([[5, 6], [7, 8]])},
        expected_output=np.array([[6, 8], [10, 12]]),
        hints=[
            "Add corresponding elements: A[i,j] + B[i,j]",
            "First row: [1+5, 2+6]",
            "Second row: [3+7, 4+8]",
        ],
        tags=["matrix_addition", "2x2"],
    ),
    ComputationalExercise(
        exercise_id="mat_add_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add A = [[2, -1], [0, 3]] and B = [[-1, 4], [2, -2]]",
        operation="matrix_add",
        inputs={"A": np.array([[2, -1], [0, 3]]), "B": np.array([[-1, 4], [2, -2]])},
        expected_output=np.array([[1, 3], [2, 1]]),
        hints=[
            "Add element-wise",
            "Handle negative numbers carefully",
        ],
        tags=["matrix_addition", "2x2", "negative"],
    ),
    ComputationalExercise(
        exercise_id="mat_add_03",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Add A = [[1, 2, 3], [4, 5, 6]] and B = [[0, -1, 2], [3, -2, 1]]",
        operation="matrix_add",
        inputs={"A": np.array([[1, 2, 3], [4, 5, 6]]), "B": np.array([[0, -1, 2], [3, -2, 1]])},
        expected_output=np.array([[1, 1, 5], [7, 3, 7]]),
        hints=[
            "This is a 2×3 matrix addition",
            "Add corresponding elements in each position",
        ],
        tags=["matrix_addition", "2x3"],
    ),
    ComputationalExercise(
        exercise_id="mat_sub_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Subtract B from A where A = [[5, 3], [2, 7]] and B = [[2, 1], [1, 4]]",
        operation="matrix_add",
        inputs={"A": np.array([[5, 3], [2, 7]]), "B": np.array([[-2, -1], [-1, -4]])},
        expected_output=np.array([[3, 2], [1, 3]]),
        hints=[
            "A - B means subtract each element",
            "A - B = [[5-2, 3-1], [2-1, 7-4]]",
        ],
        tags=["matrix_subtraction", "2x2"],
    ),
    ComputationalExercise(
        exercise_id="mat_scalar_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply the matrix A = [[1, 2], [3, 4]] by the scalar 3",
        operation="vector_scalar_multiply",  # Reusing for matrices
        inputs={"scalar": 3, "vector": np.array([[1, 2], [3, 4]])},
        expected_output=np.array([[3, 6], [9, 12]]),
        hints=[
            "Multiply each element by 3",
            "3 × [[1, 2], [3, 4]] = [[3×1, 3×2], [3×3, 3×4]]",
        ],
        tags=["scalar_multiplication", "2x2"],
    ),
    ComputationalExercise(
        exercise_id="mat_scalar_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply the matrix A = [[2, -1], [0, 3]] by the scalar -2",
        operation="vector_scalar_multiply",
        inputs={"scalar": -2, "vector": np.array([[2, -1], [0, 3]])},
        expected_output=np.array([[-4, 2], [0, -6]]),
        hints=[
            "Multiply each element by -2",
            "Pay attention to signs",
        ],
        tags=["scalar_multiplication", "2x2", "negative"],
    ),

    # Matrix multiplication
    ComputationalExercise(
        exercise_id="mat_mul_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply A = [[1, 2], [3, 4]] by B = [[2, 0], [1, 3]]",
        operation="matrix_multiply",
        inputs={"A": np.array([[1, 2], [3, 4]]), "B": np.array([[2, 0], [1, 3]])},
        expected_output=np.array([[4, 6], [10, 12]]),
        hints=[
            "C[i,j] = sum of (row i of A) × (column j of B)",
            "C[0,0] = 1×2 + 2×1 = 4",
            "C[0,1] = 1×0 + 2×3 = 6",
        ],
        tags=["matrix_multiplication", "2x2"],
    ),
    ComputationalExercise(
        exercise_id="mat_mul_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply A = [[2, 1], [0, 3]] by B = [[1, 4], [2, -1]]",
        operation="matrix_multiply",
        inputs={"A": np.array([[2, 1], [0, 3]]), "B": np.array([[1, 4], [2, -1]])},
        expected_output=np.array([[4, 7], [6, -3]]),
        hints=[
            "Row-column multiplication",
            "First row: [2×1 + 1×2, 2×4 + 1×(-1)]",
        ],
        tags=["matrix_multiplication", "2x2"],
    ),
    ComputationalExercise(
        exercise_id="mat_mul_03",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Multiply A = [[1, 0], [0, 1]] by B = [[5, 6], [7, 8]]",
        operation="matrix_multiply",
        inputs={"A": np.array([[1, 0], [0, 1]]), "B": np.array([[5, 6], [7, 8]])},
        expected_output=np.array([[5, 6], [7, 8]]),
        hints=[
            "A is the identity matrix",
            "What happens when you multiply by the identity?",
            "I × B = B",
        ],
        tags=["matrix_multiplication", "identity", "2x2"],
    ),
    ComputationalExercise(
        exercise_id="mat_mul_04",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Multiply A = [[1, 2, 3], [4, 5, 6]] by B = [[1, 4], [2, 5], [3, 6]]",
        operation="matrix_multiply",
        inputs={
            "A": np.array([[1, 2, 3], [4, 5, 6]]),
            "B": np.array([[1, 4], [2, 5], [3, 6]])
        },
        expected_output=np.array([[14, 32], [32, 77]]),
        hints=[
            "A is 2×3, B is 3×2, result will be 2×2",
            "C[0,0] = 1×1 + 2×2 + 3×3 = 14",
            "Each element requires 3 multiplications",
        ],
        tags=["matrix_multiplication", "rectangular"],
    ),
    ComputationalExercise(
        exercise_id="mat_mul_05",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Multiply A = [[2, 3], [1, 4]] by itself (compute A²)",
        operation="matrix_multiply",
        inputs={"A": np.array([[2, 3], [1, 4]]), "B": np.array([[2, 3], [1, 4]])},
        expected_output=np.array([[7, 18], [6, 19]]),
        hints=[
            "A² = A × A",
            "A[0,0] = 2×2 + 3×1 = 7",
        ],
        tags=["matrix_multiplication", "matrix_power", "2x2"],
    ),

    # Matrix transpose
    ComputationalExercise(
        exercise_id="mat_trans_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find the transpose of A = [[1, 2, 3], [4, 5, 6]]",
        operation="matrix_transpose",
        inputs={"A": np.array([[1, 2, 3], [4, 5, 6]])},
        expected_output=np.array([[1, 4], [2, 5], [3, 6]]),
        hints=[
            "Transpose swaps rows and columns",
            "Row i becomes column i",
            "A is 2×3, A^T will be 3×2",
        ],
        tags=["transpose", "rectangular"],
    ),
    ComputationalExercise(
        exercise_id="mat_trans_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find the transpose of A = [[1, 2], [3, 4]]",
        operation="matrix_transpose",
        inputs={"A": np.array([[1, 2], [3, 4]])},
        expected_output=np.array([[1, 3], [2, 4]]),
        hints=[
            "Swap rows and columns",
            "A^T[i,j] = A[j,i]",
        ],
        tags=["transpose", "2x2"],
    ),
    ComputationalExercise(
        exercise_id="mat_trans_03",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Find the transpose of A = [[1, 0, 2], [0, 3, 0], [2, 0, 4]]",
        operation="matrix_transpose",
        inputs={"A": np.array([[1, 0, 2], [0, 3, 0], [2, 0, 4]])},
        expected_output=np.array([[1, 0, 2], [0, 3, 0], [2, 0, 4]]),
        hints=[
            "What's special about this matrix?",
            "Check if A^T = A",
            "This is a symmetric matrix!",
        ],
        tags=["transpose", "symmetric", "3x3"],
    ),

    # Determinants (2x2 and simple 3x3)
    FillInExercise(
        exercise_id="mat_det_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find the determinant of A = [[3, 1], [2, 4]]",
        answer_type="numerical",
        correct_answer=10.0,
        hints=[
            "For 2×2: det(A) = ad - bc",
            "det([[a, b], [c, d]]) = ad - bc",
            "3×4 - 1×2 = ?",
        ],
        tags=["determinant", "2x2"],
    ),
    FillInExercise(
        exercise_id="mat_det_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find the determinant of A = [[2, 3], [4, 5]]",
        answer_type="numerical",
        correct_answer=-2.0,
        hints=[
            "det(A) = ad - bc",
            "2×5 - 3×4 = ?",
        ],
        tags=["determinant", "2x2"],
    ),
    FillInExercise(
        exercise_id="mat_det_03",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find the determinant of the identity matrix I = [[1, 0], [0, 1]]",
        answer_type="numerical",
        correct_answer=1.0,
        hints=[
            "Identity matrix has 1s on diagonal, 0s elsewhere",
            "1×1 - 0×0 = ?",
            "det(I) is always 1",
        ],
        tags=["determinant", "identity", "2x2"],
    ),
    FillInExercise(
        exercise_id="mat_det_04",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Find the determinant of A = [[4, 2], [2, 1]]",
        answer_type="numerical",
        correct_answer=0.0,
        hints=[
            "det(A) = ad - bc",
            "4×1 - 2×2 = ?",
            "A determinant of 0 means the matrix is singular (not invertible)",
        ],
        tags=["determinant", "singular", "2x2"],
    ),

    # Multiple choice questions
    MultipleChoiceExercise(
        exercise_id="mat_mc_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What is the result when you multiply any matrix A by the identity matrix I?",
        choices=[
            "The zero matrix",
            "The matrix A itself",
            "The transpose of A",
            "The inverse of A",
        ],
        correct_index=1,
        explanation="Multiplying by the identity matrix I leaves the matrix unchanged: A × I = A and I × A = A",
        tags=["identity", "properties"],
    ),
    MultipleChoiceExercise(
        exercise_id="mat_mc_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What must be true about matrices A and B for the product AB to be defined?",
        choices=[
            "They must be the same size",
            "Number of columns of A = number of rows of B",
            "They must both be square matrices",
            "Number of rows of A = number of columns of B",
        ],
        correct_index=1,
        explanation="For AB to exist, the number of columns in A must equal the number of rows in B. If A is m×n and B is n×p, then AB is m×p.",
        tags=["matrix_multiplication", "dimensions"],
    ),
    MultipleChoiceExercise(
        exercise_id="mat_mc_03",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What does it mean if det(A) = 0?",
        choices=[
            "A is the identity matrix",
            "A is invertible",
            "A is singular (not invertible)",
            "A is symmetric",
        ],
        correct_index=2,
        explanation="If det(A) = 0, the matrix is singular (non-invertible). It doesn't have full rank and its columns are linearly dependent.",
        tags=["determinant", "invertibility"],
    ),
    MultipleChoiceExercise(
        exercise_id="mat_mc_04",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If A is a 3×2 matrix and B is a 2×4 matrix, what is the size of AB?",
        choices=[
            "3×4",
            "2×2",
            "3×2",
            "4×3",
        ],
        correct_index=0,
        explanation="When multiplying matrices, if A is m×n and B is n×p, the result AB is m×p. So 3×2 multiplied by 2×4 gives 3×4.",
        tags=["matrix_multiplication", "dimensions"],
    ),

    # True/False questions
    TrueFalseExercise(
        exercise_id="mat_tf_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Matrix multiplication is commutative: AB = BA for all matrices A and B",
        correct_answer=False,
        explanation="Matrix multiplication is NOT commutative in general. AB ≠ BA for most matrices. In fact, even if AB exists, BA might not even be defined!",
        tags=["matrix_multiplication", "properties"],
    ),
    TrueFalseExercise(
        exercise_id="mat_tf_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Matrix addition is commutative: A + B = B + A",
        correct_answer=True,
        explanation="Matrix addition IS commutative because we add corresponding elements, and scalar addition is commutative.",
        tags=["matrix_addition", "properties"],
    ),
    TrueFalseExercise(
        exercise_id="mat_tf_03",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The transpose of a transpose returns the original matrix: (A^T)^T = A",
        correct_answer=True,
        explanation="Transposing twice returns you to the original matrix. Swapping rows/columns twice brings you back to where you started.",
        tags=["transpose", "properties"],
    ),
    TrueFalseExercise(
        exercise_id="mat_tf_04",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="For any matrix A, det(A) = det(A^T)",
        correct_answer=True,
        explanation="The determinant of a matrix equals the determinant of its transpose. This is a fundamental property of determinants.",
        tags=["determinant", "transpose", "properties"],
    ),
    TrueFalseExercise(
        exercise_id="mat_tf_05",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If A and B are both invertible, then (AB)^(-1) = A^(-1)B^(-1)",
        correct_answer=False,
        explanation="False! The correct formula is (AB)^(-1) = B^(-1)A^(-1). Note the order reverses. This is similar to how (fg)' = g'f' in composition.",
        tags=["inverse", "properties"],
    ),

    # Fill-in questions
    FillInExercise(
        exercise_id="mat_fill_01",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="A matrix with the same number of rows and columns is called a ___ matrix",
        answer_type="text",
        correct_answer="square",
        case_sensitive=False,
        hints=[
            "Examples: 2×2, 3×3, n×n",
            "All square matrices have equal dimensions",
        ],
        tags=["terminology"],
    ),
    FillInExercise(
        exercise_id="mat_fill_02",
        topic="matrices",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="A matrix where all elements are zero is called the ___ matrix",
        answer_type="text",
        correct_answer="zero",
        case_sensitive=False,
        hints=[
            "This is the additive identity for matrices",
            "A + 0 = A",
        ],
        tags=["terminology", "special_matrices"],
    ),
    FillInExercise(
        exercise_id="mat_fill_03",
        topic="matrices",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If A^T = A, the matrix A is called ___",
        answer_type="text",
        correct_answer="symmetric",
        case_sensitive=False,
        hints=[
            "Such matrices are symmetric across the main diagonal",
            "Examples: [[1,2],[2,3]] or [[1,0,0],[0,2,0],[0,0,3]]",
        ],
        tags=["terminology", "transpose", "symmetric"],
    ),
]


def get_exercises_by_topic(topic: str):
    """Get all exercises for a topic.

    Args:
        topic: Topic name (e.g., 'vectors', 'matrices')

    Returns:
        List of exercises for that topic
    """
    topic_lower = topic.lower()
    if topic_lower == "vectors":
        return VECTOR_EXERCISES
    elif topic_lower == "matrices":
        return MATRIX_EXERCISES
    else:
        return []


def get_exercises_by_difficulty(topic: str, difficulty: ExerciseDifficulty):
    """Get exercises filtered by difficulty.

    Args:
        topic: Topic name
        difficulty: Difficulty level

    Returns:
        List of exercises matching the criteria
    """
    all_exercises = get_exercises_by_topic(topic)
    return [ex for ex in all_exercises if ex.difficulty == difficulty]
