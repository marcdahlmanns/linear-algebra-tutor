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


# ============================================================================
# LINEAR SYSTEMS - SOLVING EQUATIONS
# ============================================================================

LINEAR_SYSTEMS_EXERCISES = [
    # Simple 2x2 systems
    ComputationalExercise(
        exercise_id="linsys_01",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Solve the system: x + y = 5, x - y = 1",
        operation="linear_system",
        inputs={"A": np.array([[1.0, 1.0], [1.0, -1.0]]), "b": np.array([5.0, 1.0])},
        expected_output=np.array([3.0, 2.0]),
        hints=[
            "Add the two equations to eliminate y",
            "2x = 6, so x = 3",
            "Substitute back: 3 + y = 5, so y = 2",
        ],
        tags=["2x2_system", "elimination"],
    ),
    ComputationalExercise(
        exercise_id="linsys_02",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Solve: 2x + 3y = 13, x + y = 5",
        operation="linear_system",
        inputs={"A": np.array([[2.0, 3.0], [1.0, 1.0]]), "b": np.array([13.0, 5.0])},
        expected_output=np.array([2.0, 3.0]),
        hints=[
            "Use substitution or elimination",
            "From second equation: x = 5 - y",
            "Substitute into first: 2(5-y) + 3y = 13",
        ],
        tags=["2x2_system", "substitution"],
    ),
    ComputationalExercise(
        exercise_id="linsys_03",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Solve: 3x - y = 7, x + 2y = 6",
        operation="linear_system",
        inputs={"A": np.array([[3.0, -1.0], [1.0, 2.0]]), "b": np.array([7.0, 6.0])},
        expected_output=np.array([2.8, 1.6]),
        hints=[
            "Multiply first equation by 2 to eliminate y",
            "6x - 2y = 14 and x + 2y = 6",
            "Add them: 7x = 20",
        ],
        tags=["2x2_system", "elimination"],
    ),

    # 3x3 systems
    ComputationalExercise(
        exercise_id="linsys_04",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Solve the system: x + y + z = 6, 2x + y - z = 1, x - y + z = 0",
        operation="linear_system",
        inputs={
            "A": np.array([[1.0, 1.0, 1.0], [2.0, 1.0, -1.0], [1.0, -1.0, 1.0]]),
            "b": np.array([6.0, 1.0, 0.0])
        },
        expected_output=np.array([1.0, 2.0, 3.0]),
        hints=[
            "Use Gaussian elimination",
            "Create augmented matrix [A|b]",
            "Row reduce to find x, y, z",
        ],
        tags=["3x3_system", "gaussian"],
    ),
    ComputationalExercise(
        exercise_id="linsys_05",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Solve: x + 2y + z = 7, 2x + y + 2z = 8, x + y + 3z = 9",
        operation="linear_system",
        inputs={
            "A": np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 1.0, 3.0]]),
            "b": np.array([7.0, 8.0, 9.0])
        },
        expected_output=np.array([1.0, 2.0, 2.0]),
        hints=[
            "Form augmented matrix",
            "Eliminate x from rows 2 and 3",
            "Continue with back substitution",
        ],
        tags=["3x3_system", "gaussian"],
    ),

    # Multiple choice
    MultipleChoiceExercise(
        exercise_id="linsys_mc_01",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="A system of linear equations can have how many solutions?",
        choices=[
            "Only one solution",
            "Exactly zero, one, or infinitely many solutions",
            "Any finite number of solutions",
            "Always infinitely many solutions",
        ],
        correct_index=1,
        explanation="By the fundamental theorem, a system has either no solution (inconsistent), exactly one solution (unique), or infinitely many solutions (dependent equations).",
        tags=["solutions", "theory"],
    ),
    MultipleChoiceExercise(
        exercise_id="linsys_mc_02",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What does it mean if a row in an augmented matrix reduces to [0 0 0 | 5]?",
        choices=[
            "The system has a unique solution",
            "The system has infinitely many solutions",
            "The system is inconsistent (no solution)",
            "The system is already solved",
        ],
        correct_index=2,
        explanation="A row like [0 0 0 | 5] represents 0 = 5, which is impossible. The system is inconsistent.",
        tags=["inconsistent", "augmented_matrix"],
    ),
    MultipleChoiceExercise(
        exercise_id="linsys_mc_03",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In Gaussian elimination, what is a pivot?",
        choices=[
            "Any non-zero entry in the matrix",
            "The first non-zero entry in a row",
            "The determinant of the matrix",
            "The last row of the matrix",
        ],
        correct_index=1,
        explanation="A pivot is the first non-zero entry in a row during row reduction. We use pivots to eliminate entries below them.",
        tags=["gaussian", "pivot"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="linsys_tf_01",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="If a system has more equations than unknowns, it must be inconsistent",
        correct_answer=False,
        explanation="False. An overdetermined system (more equations than unknowns) can still be consistent if the extra equations are compatible with the others.",
        tags=["overdetermined", "consistency"],
    ),
    TrueFalseExercise(
        exercise_id="linsys_tf_02",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Swapping two rows in an augmented matrix changes the solution set",
        correct_answer=False,
        explanation="False. Row operations (including row swaps) preserve the solution set. They're equivalent transformations.",
        tags=["row_operations", "equivalence"],
    ),
    TrueFalseExercise(
        exercise_id="linsys_tf_03",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="A homogeneous system (b = 0) always has at least one solution",
        correct_answer=True,
        explanation="True. The zero vector x = 0 is always a solution to Ax = 0. This is called the trivial solution.",
        tags=["homogeneous", "trivial_solution"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="linsys_fill_01",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The matrix form of a linear system Ax = b, where [A|b] is written together, is called the ___ matrix",
        answer_type="text",
        correct_answer="augmented",
        case_sensitive=False,
        hints=[
            "It augments the coefficient matrix with the constants",
            "Written as [A|b]",
        ],
        tags=["terminology", "augmented_matrix"],
    ),
    FillInExercise(
        exercise_id="linsys_fill_02",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The process of using row operations to transform a matrix to row echelon form is called ___ elimination",
        answer_type="text",
        correct_answer="Gaussian",
        case_sensitive=False,
        hints=[
            "Named after Carl Friedrich ___",
            "A fundamental algorithm in linear algebra",
        ],
        tags=["terminology", "gaussian"],
    ),
    FillInExercise(
        exercise_id="linsys_fill_03",
        topic="linear_systems",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="A system where the right-hand side is all zeros (Ax = 0) is called ___",
        answer_type="text",
        correct_answer="homogeneous",
        case_sensitive=False,
        hints=[
            "From Greek, meaning 'of the same kind'",
            "Always has the trivial solution x = 0",
        ],
        tags=["terminology", "homogeneous"],
    ),
]


# ============================================================================
# VECTOR SPACES - SUBSPACES, BASIS, DIMENSION
# ============================================================================

VECTOR_SPACES_EXERCISES = [
    # Conceptual questions
    MultipleChoiceExercise(
        exercise_id="vecspace_mc_01",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Which of the following is NOT required for a set to be a vector space?",
        choices=[
            "Closed under addition",
            "Closed under scalar multiplication",
            "Contains a zero vector",
            "Contains only vectors with positive components",
        ],
        correct_index=3,
        explanation="Vectors can have negative components! The three properties listed in options 1-3 are all required, plus associativity, commutativity, distributivity, etc.",
        tags=["vector_space", "axioms"],
    ),
    MultipleChoiceExercise(
        exercise_id="vecspace_mc_02",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What is a subspace?",
        choices=[
            "Any subset of a vector space",
            "A subset that is itself a vector space under the same operations",
            "A smaller-dimensional vector space",
            "The null space of a matrix",
        ],
        correct_index=1,
        explanation="A subspace must be closed under addition and scalar multiplication, and contain the zero vector. It's a vector space in its own right.",
        tags=["subspace", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="vecspace_mc_03",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The span of vectors {v₁, v₂, ..., vₙ} is:",
        choices=[
            "The set of all vectors perpendicular to them",
            "The set of all linear combinations of them",
            "The set {v₁, v₂, ..., vₙ} itself",
            "The null space they generate",
        ],
        correct_index=1,
        explanation="Span{v₁, ..., vₙ} = {c₁v₁ + c₂v₂ + ... + cₙvₙ : c₁,...,cₙ ∈ ℝ}. It's all possible linear combinations.",
        tags=["span", "linear_combination"],
    ),
    MultipleChoiceExercise(
        exercise_id="vecspace_mc_04",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What does it mean for vectors {v₁, v₂, v₃} to be linearly independent?",
        choices=[
            "They all have different lengths",
            "No vector can be written as a linear combination of the others",
            "They are mutually orthogonal",
            "They span the entire space",
        ],
        correct_index=1,
        explanation="Linear independence means c₁v₁ + c₂v₂ + c₃v₃ = 0 only when all cᵢ = 0. Equivalently, no vector is a combination of the others.",
        tags=["linear_independence", "definition"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="vecspace_tf_01",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The set of all 2×2 matrices forms a vector space",
        correct_answer=True,
        explanation="True! Matrices can be added and scalar multiplied, satisfying all vector space axioms. This is a 4-dimensional space.",
        tags=["vector_space", "matrices"],
    ),
    TrueFalseExercise(
        exercise_id="vecspace_tf_02",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Every vector space contains the zero vector",
        correct_answer=True,
        explanation="True. The zero vector is required by the axioms. It's the additive identity: v + 0 = v for all v.",
        tags=["zero_vector", "axioms"],
    ),
    TrueFalseExercise(
        exercise_id="vecspace_tf_03",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If {v₁, v₂, v₃} are linearly independent, then {v₁, v₂} must also be linearly independent",
        correct_answer=True,
        explanation="True. If v₁ and v₂ were dependent, then {v₁, v₂, v₃} couldn't be independent. Independence of a larger set implies independence of subsets.",
        tags=["linear_independence", "subsets"],
    ),
    TrueFalseExercise(
        exercise_id="vecspace_tf_04",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="A basis for ℝ³ must contain exactly 3 vectors",
        correct_answer=True,
        explanation="True. A basis must span the space and be linearly independent. For ℝⁿ, this requires exactly n vectors.",
        tags=["basis", "dimension"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="vecspace_fill_01",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="A set of vectors that spans a vector space and is linearly independent is called a ___",
        answer_type="text",
        correct_answer="basis",
        case_sensitive=False,
        hints=[
            "Provides a coordinate system for the space",
            "Minimal spanning set",
        ],
        tags=["basis", "terminology"],
    ),
    FillInExercise(
        exercise_id="vecspace_fill_02",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The number of vectors in any basis for a vector space is called its ___",
        answer_type="text",
        correct_answer="dimension",
        case_sensitive=False,
        hints=[
            "ℝ³ has ___ = 3",
            "All bases for the same space have this same value",
        ],
        tags=["dimension", "terminology"],
    ),
    FillInExercise(
        exercise_id="vecspace_fill_03",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The solution set to Ax = 0 is called the ___ space of A",
        answer_type="text",
        correct_answer="null",
        case_sensitive=False,
        hints=[
            "Also called the kernel",
            "It's a subspace!",
        ],
        tags=["null_space", "terminology"],
    ),
    FillInExercise(
        exercise_id="vecspace_fill_04",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The span of the columns of matrix A is called the ___ space of A",
        answer_type="text",
        correct_answer="column",
        case_sensitive=False,
        hints=[
            "Also called the range or image",
            "It's the set of all possible Ax",
        ],
        tags=["column_space", "terminology"],
    ),

    # Computational
    FillInExercise(
        exercise_id="vecspace_comp_01",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What is the dimension of the space of all 2×3 matrices?",
        answer_type="numerical",
        correct_answer=6.0,
        hints=[
            "Count the independent entries",
            "2 rows × 3 columns = ?",
            "Each entry can vary independently",
        ],
        tags=["dimension", "matrix_space"],
    ),
    FillInExercise(
        exercise_id="vecspace_comp_02",
        topic="vector_spaces",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The vectors [1,0,0], [0,1,0], [0,0,1] form a basis for ℝ³. This is called the ___ basis",
        answer_type="text",
        correct_answer="standard",
        case_sensitive=False,
        hints=[
            "The most common basis",
            "Also called the canonical basis",
        ],
        tags=["basis", "standard_basis"],
    ),
]


# ============================================================================
# ORTHOGONALITY - PROJECTIONS, GRAM-SCHMIDT, QR
# ============================================================================

ORTHOGONALITY_EXERCISES = [
    # Computational - orthogonality checks
    ComputationalExercise(
        exercise_id="ortho_01",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Compute the dot product of v = [1, 2] and w = [-2, 1] to check if they're orthogonal",
        operation="vector_dot",
        inputs={"v": np.array([1, 2]), "w": np.array([-2, 1])},
        expected_output=0.0,
        hints=[
            "Vectors are orthogonal if v·w = 0",
            "v·w = (1)(-2) + (2)(1)",
            "If you get 0, they're perpendicular!",
        ],
        tags=["dot_product", "orthogonality"],
    ),
    ComputationalExercise(
        exercise_id="ortho_02",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Compute the dot product of v = [3, 4] and w = [4, -3]",
        operation="vector_dot",
        inputs={"v": np.array([3, 4]), "w": np.array([4, -3])},
        expected_output=0.0,
        hints=[
            "Check: (3)(4) + (4)(-3)",
            "These vectors are orthogonal",
        ],
        tags=["dot_product", "orthogonality"],
    ),

    # Projections
    FillInExercise(
        exercise_id="ortho_proj_01",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The projection of vector [3, 4] onto [1, 0] has magnitude ___",
        answer_type="numerical",
        correct_answer=3.0,
        hints=[
            "Projection onto [1,0] extracts the x-component",
            "proj_u(v) = (v·u / u·u) × u",
            "The x-component of [3, 4] is 3",
        ],
        tags=["projection", "geometric"],
    ),
    FillInExercise(
        exercise_id="ortho_proj_02",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If v = [6, 8] and u = [1, 0], what is the first component of proj_u(v)?",
        answer_type="numerical",
        correct_answer=6.0,
        hints=[
            "proj_u(v) = ((v·u)/(u·u)) × u",
            "v·u = 6×1 + 8×0 = 6",
            "u·u = 1",
        ],
        tags=["projection", "computation"],
    ),

    # Multiple Choice
    MultipleChoiceExercise(
        exercise_id="ortho_mc_01",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What does it mean for two vectors to be orthogonal?",
        choices=[
            "They have the same length",
            "They are perpendicular (angle = 90°)",
            "They are parallel",
            "They point in opposite directions",
        ],
        correct_index=1,
        explanation="Orthogonal means perpendicular. Two vectors are orthogonal if their dot product is zero, which happens when the angle between them is 90°.",
        tags=["orthogonality", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="ortho_mc_02",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What is an orthonormal set of vectors?",
        choices=[
            "Vectors that are mutually perpendicular",
            "Vectors that all have length 1",
            "Vectors that are mutually perpendicular AND have length 1",
            "Vectors that form a right angle with the x-axis",
        ],
        correct_index=2,
        explanation="Orthonormal means both orthogonal (perpendicular to each other) AND normalized (each has length 1). It's ortho + normal.",
        tags=["orthonormal", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="ortho_mc_03",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The Gram-Schmidt process is used to:",
        choices=[
            "Find eigenvalues of a matrix",
            "Solve linear systems",
            "Convert a basis into an orthonormal basis",
            "Compute determinants",
        ],
        correct_index=2,
        explanation="Gram-Schmidt takes any linearly independent set and produces an orthonormal basis spanning the same space. It's an orthogonalization procedure.",
        tags=["gram_schmidt", "process"],
    ),
    MultipleChoiceExercise(
        exercise_id="ortho_mc_04",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In the QR decomposition A = QR, what is Q?",
        choices=[
            "A diagonal matrix",
            "An orthogonal matrix (columns are orthonormal)",
            "An upper triangular matrix",
            "The inverse of A",
        ],
        correct_index=1,
        explanation="In QR decomposition, Q is an orthogonal matrix (Q^T Q = I) and R is upper triangular. This is often computed using Gram-Schmidt.",
        tags=["qr_decomposition", "orthogonal_matrix"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="ortho_tf_01",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="If v·w = 0, then vectors v and w are orthogonal",
        correct_answer=True,
        explanation="True. Zero dot product is the definition of orthogonality. When v·w = ||v|| ||w|| cos(θ) = 0, then θ = 90°.",
        tags=["orthogonality", "dot_product"],
    ),
    TrueFalseExercise(
        exercise_id="ortho_tf_02",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The zero vector is orthogonal to every vector",
        correct_answer=True,
        explanation="True. 0·v = 0 for all vectors v, so technically the zero vector is orthogonal to everything (though it's a degenerate case).",
        tags=["zero_vector", "orthogonality"],
    ),
    TrueFalseExercise(
        exercise_id="ortho_tf_03",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If Q is an orthogonal matrix, then Q^(-1) = Q^T",
        correct_answer=True,
        explanation="True! This is a key property. For orthogonal matrices, the inverse equals the transpose. This makes computations much easier.",
        tags=["orthogonal_matrix", "inverse"],
    ),
    TrueFalseExercise(
        exercise_id="ortho_tf_04",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Orthogonal projection onto a subspace always decreases the length of the vector",
        correct_answer=False,
        explanation="False! If the vector is already in the subspace, projection doesn't change it. Projection decreases length only if the vector has a component perpendicular to the subspace.",
        tags=["projection", "length"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="ortho_fill_01",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Two vectors are ___ if their dot product is zero",
        answer_type="text",
        correct_answer="orthogonal",
        case_sensitive=False,
        hints=[
            "Another word for perpendicular",
            "From Greek ortho (straight) + gonia (angle)",
        ],
        tags=["terminology", "orthogonality"],
    ),
    FillInExercise(
        exercise_id="ortho_fill_02",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="A matrix whose columns are orthonormal is called an ___ matrix",
        answer_type="text",
        correct_answer="orthogonal",
        case_sensitive=False,
        hints=[
            "For such matrices, Q^T Q = I",
            "Same word as for perpendicular vectors",
        ],
        tags=["terminology", "orthogonal_matrix"],
    ),
    FillInExercise(
        exercise_id="ortho_fill_03",
        topic="orthogonality",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The ___ complement of a subspace W consists of all vectors orthogonal to W",
        answer_type="text",
        correct_answer="orthogonal",
        case_sensitive=False,
        hints=[
            "Denoted W⊥ (W perp)",
            "If v is in W⊥, then v·w = 0 for all w in W",
        ],
        tags=["terminology", "orthogonal_complement"],
    ),
]


# ============================================================================
# DETERMINANTS - PROPERTIES, COFACTOR EXPANSION, GEOMETRIC MEANING
# ============================================================================

DETERMINANTS_EXERCISES = [
    # Computational - 2x2 determinants
    FillInExercise(
        exercise_id="det_01",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find det([[4, 1], [2, 3]])",
        answer_type="numerical",
        correct_answer=10.0,
        hints=[
            "For 2×2: det([[a, b], [c, d]]) = ad - bc",
            "4×3 - 1×2 = ?",
        ],
        tags=["determinant", "2x2"],
    ),
    FillInExercise(
        exercise_id="det_02",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find det([[5, 2], [3, 1]])",
        answer_type="numerical",
        correct_answer=-1.0,
        hints=[
            "det = ad - bc",
            "5×1 - 2×3 = ?",
        ],
        tags=["determinant", "2x2"],
    ),
    FillInExercise(
        exercise_id="det_03",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Find det([[6, 3], [4, 2]])",
        answer_type="numerical",
        correct_answer=0.0,
        hints=[
            "6×2 - 3×4 = ?",
            "Zero determinant means singular matrix!",
        ],
        tags=["determinant", "singular", "2x2"],
    ),

    # 3x3 determinants
    FillInExercise(
        exercise_id="det_04",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Find det([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) for this diagonal matrix",
        answer_type="numerical",
        correct_answer=6.0,
        hints=[
            "For diagonal matrices, det = product of diagonal entries",
            "1 × 2 × 3 = ?",
        ],
        tags=["determinant", "diagonal", "3x3"],
    ),
    FillInExercise(
        exercise_id="det_05",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Find det([[2, 0, 0], [0, 2, 0], [0, 0, 2]])",
        answer_type="numerical",
        correct_answer=8.0,
        hints=[
            "Product of diagonal entries",
            "2 × 2 × 2 = ?",
        ],
        tags=["determinant", "diagonal", "3x3"],
    ),

    # Multiple Choice
    MultipleChoiceExercise(
        exercise_id="det_mc_01",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What is the geometric meaning of |det(A)| for a 2×2 matrix?",
        choices=[
            "The perimeter of a parallelogram",
            "The area of the parallelogram formed by the column vectors",
            "The angle between the column vectors",
            "The sum of the column vector lengths",
        ],
        correct_index=1,
        explanation="The absolute value of the determinant equals the area of the parallelogram formed by the column vectors. For 3×3, it's the volume of the parallelepiped.",
        tags=["determinant", "geometry"],
    ),
    MultipleChoiceExercise(
        exercise_id="det_mc_02",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="If det(A) = 0, what can we conclude?",
        choices=[
            "A is the zero matrix",
            "A is invertible",
            "A is singular (not invertible)",
            "A is symmetric",
        ],
        correct_index=2,
        explanation="det(A) = 0 means A is singular (non-invertible). The columns are linearly dependent and don't span the full space.",
        tags=["determinant", "invertibility"],
    ),
    MultipleChoiceExercise(
        exercise_id="det_mc_03",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What happens to the determinant if you swap two rows?",
        choices=[
            "It stays the same",
            "It changes sign",
            "It becomes zero",
            "It doubles",
        ],
        correct_index=1,
        explanation="Swapping two rows multiplies the determinant by -1. This is one of the fundamental properties of determinants.",
        tags=["determinant", "row_operations"],
    ),
    MultipleChoiceExercise(
        exercise_id="det_mc_04",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If det(A) = 5, what is det(2A)?",
        choices=[
            "5",
            "10",
            "25 (for 2×2) or 40 (for 3×3)",
            "Depends on the dimension n: 2^n × 5",
        ],
        correct_index=3,
        explanation="If A is n×n, then det(cA) = c^n det(A). For 2×2: det(2A) = 4×5 = 20. For 3×3: det(2A) = 8×5 = 40.",
        tags=["determinant", "scalar_multiplication"],
    ),
    MultipleChoiceExercise(
        exercise_id="det_mc_05",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What is the relationship between det(A) and det(A^T)?",
        choices=[
            "det(A^T) = -det(A)",
            "det(A^T) = det(A)",
            "det(A^T) = 1/det(A)",
            "No general relationship",
        ],
        correct_index=1,
        explanation="det(A) = det(A^T) always. The determinant is unchanged by transposition.",
        tags=["determinant", "transpose"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="det_tf_01",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The determinant of the identity matrix is always 1",
        correct_answer=True,
        explanation="True. det(I) = 1 for any size identity matrix. This follows from the product of diagonal entries.",
        tags=["determinant", "identity"],
    ),
    TrueFalseExercise(
        exercise_id="det_tf_02",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="det(AB) = det(A) × det(B)",
        correct_answer=True,
        explanation="True! The determinant of a product equals the product of determinants. This is a fundamental property.",
        tags=["determinant", "product"],
    ),
    TrueFalseExercise(
        exercise_id="det_tf_03",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="If A has a row of zeros, then det(A) = 0",
        correct_answer=True,
        explanation="True. A row of zeros means the rows are linearly dependent, so det(A) = 0 and A is singular.",
        tags=["determinant", "zero_row"],
    ),
    TrueFalseExercise(
        exercise_id="det_tf_04",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Adding a multiple of one row to another row changes the determinant",
        correct_answer=False,
        explanation="False! Row replacement (adding a multiple of one row to another) preserves the determinant. This is why Gaussian elimination is useful for computing determinants.",
        tags=["determinant", "row_operations"],
    ),
    TrueFalseExercise(
        exercise_id="det_tf_05",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If det(A) ≠ 0, then the columns of A are linearly independent",
        correct_answer=True,
        explanation="True. Non-zero determinant means A is invertible, which implies the columns are linearly independent and span the full space.",
        tags=["determinant", "linear_independence"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="det_fill_01",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The determinant of the identity matrix is ___",
        answer_type="numerical",
        correct_answer=1.0,
        hints=[
            "I = [[1,0,...], [0,1,...], ...]",
            "Product of diagonal entries",
        ],
        tags=["determinant", "identity"],
    ),
    FillInExercise(
        exercise_id="det_fill_02",
        topic="determinants",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="A matrix with determinant zero is called ___",
        answer_type="text",
        correct_answer="singular",
        case_sensitive=False,
        hints=[
            "It's not invertible",
            "Opposite of 'non-singular'",
        ],
        tags=["terminology", "singular"],
    ),
    FillInExercise(
        exercise_id="det_fill_03",
        topic="determinants",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The signed sum used to compute determinants by expanding along a row or column is called ___ expansion",
        answer_type="text",
        correct_answer="cofactor",
        case_sensitive=False,
        hints=[
            "Also called Laplace expansion",
            "Uses minors and alternating signs",
        ],
        tags=["terminology", "cofactor_expansion"],
    ),
]


# ============================================================================
# EIGENVALUES - CHARACTERISTIC POLYNOMIAL, EIGENVECTORS, DIAGONALIZATION
# ============================================================================

EIGENVALUES_EXERCISES = [
    # Computational - finding eigenvalues
    FillInExercise(
        exercise_id="eigen_01",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="One eigenvalue of A = [[3, 0], [0, 5]] is 3. What is the other eigenvalue?",
        answer_type="numerical",
        correct_answer=5.0,
        hints=[
            "Diagonal matrices have eigenvalues on the diagonal",
            "The other diagonal entry is 5",
        ],
        tags=["eigenvalues", "diagonal"],
    ),
    FillInExercise(
        exercise_id="eigen_02",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What is the trace (sum of eigenvalues) of A = [[2, 1], [0, 3]]?",
        answer_type="numerical",
        correct_answer=5.0,
        hints=[
            "For upper triangular, eigenvalues are diagonal entries",
            "2 + 3 = ?",
        ],
        tags=["eigenvalues", "trace", "triangular"],
    ),
    FillInExercise(
        exercise_id="eigen_03",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The matrix [[4, 0], [0, 4]] has a repeated eigenvalue. What is it?",
        answer_type="numerical",
        correct_answer=4.0,
        hints=[
            "This is 4I, a scaled identity",
            "Both diagonal entries are the same",
        ],
        tags=["eigenvalues", "repeated"],
    ),

    # Multiple Choice
    MultipleChoiceExercise(
        exercise_id="eigen_mc_01",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What is an eigenvector?",
        choices=[
            "A vector with length 1",
            "A vector that is orthogonal to all others",
            "A non-zero vector that satisfies Av = λv for some scalar λ",
            "The zero vector",
        ],
        correct_index=2,
        explanation="An eigenvector v satisfies Av = λv where λ is the corresponding eigenvalue. The transformation just scales v, doesn't change its direction.",
        tags=["eigenvector", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="eigen_mc_02",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="How do you find eigenvalues of a matrix A?",
        choices=[
            "Compute det(A)",
            "Solve det(A - λI) = 0",
            "Find the trace of A",
            "Compute A^2",
        ],
        correct_index=1,
        explanation="Eigenvalues satisfy det(A - λI) = 0. This is called the characteristic equation. The resulting polynomial is the characteristic polynomial.",
        tags=["eigenvalues", "characteristic_equation"],
    ),
    MultipleChoiceExercise(
        exercise_id="eigen_mc_03",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What does it mean for a matrix to be diagonalizable?",
        choices=[
            "It is already a diagonal matrix",
            "It can be written as A = PDP^(-1) where D is diagonal",
            "All its entries are non-zero",
            "Its determinant is non-zero",
        ],
        correct_index=1,
        explanation="A is diagonalizable if A = PDP^(-1) where D is diagonal (containing eigenvalues) and P contains eigenvectors. This requires n linearly independent eigenvectors.",
        tags=["diagonalization", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="eigen_mc_04",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If A is a 3×3 matrix, what is the degree of its characteristic polynomial?",
        choices=[
            "2",
            "3",
            "4",
            "Depends on the matrix",
        ],
        correct_index=1,
        explanation="For an n×n matrix, the characteristic polynomial det(A - λI) has degree n. For 3×3, it's a cubic polynomial.",
        tags=["characteristic_polynomial", "degree"],
    ),
    MultipleChoiceExercise(
        exercise_id="eigen_mc_05",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What is the geometric meaning of an eigenvalue λ?",
        choices=[
            "It's the angle of rotation",
            "It's the scaling factor for eigenvectors in that direction",
            "It's the determinant of the matrix",
            "It's the dimension of the eigenspace",
        ],
        correct_index=1,
        explanation="When A acts on eigenvector v, it just scales it by λ: Av = λv. The eigenvalue is the scaling factor in that direction.",
        tags=["eigenvalues", "geometry"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="eigen_tf_01",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The zero vector can be an eigenvector",
        correct_answer=False,
        explanation="False. Eigenvectors must be non-zero by definition. While 0 satisfies A(0) = λ(0), we exclude it because every λ would work.",
        tags=["eigenvector", "zero_vector"],
    ),
    TrueFalseExercise(
        exercise_id="eigen_tf_02",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Every square matrix has at least one real eigenvalue",
        correct_answer=False,
        explanation="False! Some matrices (like rotation matrices) have only complex eigenvalues. Example: [[0,-1],[1,0]] has eigenvalues ±i.",
        tags=["eigenvalues", "complex"],
    ),
    TrueFalseExercise(
        exercise_id="eigen_tf_03",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The eigenvalues of a triangular matrix are its diagonal entries",
        correct_answer=True,
        explanation="True! For upper or lower triangular matrices, the eigenvalues are exactly the diagonal entries. This makes them easy to find.",
        tags=["eigenvalues", "triangular"],
    ),
    TrueFalseExercise(
        exercise_id="eigen_tf_04",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If A is diagonalizable, then A has n distinct eigenvalues",
        correct_answer=False,
        explanation="False. A can be diagonalizable with repeated eigenvalues, as long as it has n linearly independent eigenvectors. Example: the identity matrix.",
        tags=["diagonalization", "repeated_eigenvalues"],
    ),
    TrueFalseExercise(
        exercise_id="eigen_tf_05",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.CHALLENGE,
        question="Similar matrices have the same eigenvalues",
        correct_answer=True,
        explanation="True. If B = P^(-1)AP, then A and B have the same eigenvalues (and same characteristic polynomial). They represent the same transformation in different bases.",
        tags=["eigenvalues", "similarity"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="eigen_fill_01",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="If Av = λv, then λ is called an ___ of A",
        answer_type="text",
        correct_answer="eigenvalue",
        case_sensitive=False,
        hints=[
            "From German 'eigen' meaning 'characteristic' or 'own'",
            "The scalar that scales the eigenvector",
        ],
        tags=["terminology", "eigenvalue"],
    ),
    FillInExercise(
        exercise_id="eigen_fill_02",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The polynomial det(A - λI) is called the ___ polynomial",
        answer_type="text",
        correct_answer="characteristic",
        case_sensitive=False,
        hints=[
            "Setting this equal to zero gives eigenvalues",
            "It characterizes the matrix",
        ],
        tags=["terminology", "characteristic_polynomial"],
    ),
    FillInExercise(
        exercise_id="eigen_fill_03",
        topic="eigenvalues",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The sum of the eigenvalues equals the ___ of the matrix",
        answer_type="text",
        correct_answer="trace",
        case_sensitive=False,
        hints=[
            "Also the sum of diagonal entries",
            "Denoted tr(A)",
        ],
        tags=["terminology", "trace"],
    ),
]


# ============================================================================
# TRANSFORMATIONS - LINEAR MAPS, KERNEL, RANGE, MATRIX REPRESENTATION
# ============================================================================

TRANSFORMATIONS_EXERCISES = [
    # Multiple Choice
    MultipleChoiceExercise(
        exercise_id="trans_mc_01",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="What defines a linear transformation T: V → W?",
        choices=[
            "T(0) = 0",
            "T preserves vector addition and scalar multiplication",
            "T is represented by a matrix",
            "T maps vectors to vectors",
        ],
        correct_index=1,
        explanation="T is linear if T(u+v) = T(u)+T(v) and T(cv) = cT(v) for all vectors u,v and scalars c. These two properties define linearity.",
        tags=["linear_transformation", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="trans_mc_02",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The kernel (null space) of a transformation T is:",
        choices=[
            "The set of all output vectors",
            "The set of all vectors that map to zero: {v : T(v) = 0}",
            "The set of eigenvectors",
            "The dimension of the domain",
        ],
        correct_index=1,
        explanation="ker(T) = {v : T(v) = 0}. It's all vectors that get mapped to the zero vector. For matrices, this is the null space.",
        tags=["kernel", "null_space"],
    ),
    MultipleChoiceExercise(
        exercise_id="trans_mc_03",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The range (image) of a transformation T is:",
        choices=[
            "The domain of T",
            "The set of all possible outputs: {T(v) : v in domain}",
            "The kernel of T",
            "The set of eigenvectors",
        ],
        correct_index=1,
        explanation="range(T) = {T(v) : v ∈ domain}. It's all possible outputs. For matrices, this is the column space.",
        tags=["range", "image"],
    ),
    MultipleChoiceExercise(
        exercise_id="trans_mc_04",
        topic="transformations",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The Rank-Nullity Theorem states that for T: ℝⁿ → ℝᵐ:",
        choices=[
            "rank(T) + nullity(T) = m",
            "rank(T) + nullity(T) = n",
            "rank(T) × nullity(T) = n",
            "rank(T) = nullity(T)",
        ],
        correct_index=1,
        explanation="dim(domain) = dim(range) + dim(kernel). Or: n = rank + nullity. The dimension of the domain splits between range and kernel.",
        tags=["rank_nullity", "theorem"],
    ),
    MultipleChoiceExercise(
        exercise_id="trans_mc_05",
        topic="transformations",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What does it mean for T to be one-to-one (injective)?",
        choices=[
            "Every output has exactly one input",
            "T maps everything to zero",
            "The kernel is {0} only",
            "Both A and C",
        ],
        correct_index=3,
        explanation="T is one-to-one if different inputs give different outputs. Equivalently, ker(T) = {0}. If T(u) = T(v), then u = v.",
        tags=["injective", "one_to_one"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="trans_tf_01",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="Every linear transformation maps the zero vector to the zero vector",
        correct_answer=True,
        explanation="True. T(0) = T(0·v) = 0·T(v) = 0 for any linear transformation. This follows from linearity.",
        tags=["linear_transformation", "zero_vector"],
    ),
    TrueFalseExercise(
        exercise_id="trans_tf_02",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The transformation T(x, y) = (x+1, y) is linear",
        correct_answer=False,
        explanation="False! T(0,0) = (1,0) ≠ (0,0). Linear transformations must map zero to zero. This is an affine transformation (linear + translation).",
        tags=["linear_transformation", "counterexample"],
    ),
    TrueFalseExercise(
        exercise_id="trans_tf_03",
        topic="transformations",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If T is onto (surjective), then the range of T equals the codomain",
        correct_answer=True,
        explanation="True. 'Onto' means every element in the codomain is hit. range(T) = codomain. For T: ℝⁿ → ℝᵐ to be onto, need rank = m.",
        tags=["surjective", "onto", "range"],
    ),
    TrueFalseExercise(
        exercise_id="trans_tf_04",
        topic="transformations",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The kernel of any transformation is a subspace",
        correct_answer=True,
        explanation="True. ker(T) is always a subspace: (1) contains 0, (2) closed under addition, (3) closed under scalar multiplication. All follow from linearity.",
        tags=["kernel", "subspace"],
    ),
    TrueFalseExercise(
        exercise_id="trans_tf_05",
        topic="transformations",
        difficulty=ExerciseDifficulty.CHALLENGE,
        question="If T: ℝⁿ → ℝⁿ is both one-to-one and onto, then T is invertible",
        correct_answer=True,
        explanation="True. One-to-one + onto = bijection = invertible. For matrices: A invertible ⟺ ker(A)={0} and rank(A)=n.",
        tags=["invertible", "bijection"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="trans_fill_01",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The set of vectors that map to zero under T is called the ___ of T",
        answer_type="text",
        correct_answer="kernel",
        case_sensitive=False,
        hints=[
            "Also called the null space",
            "ker(T) = {v : T(v) = 0}",
        ],
        tags=["terminology", "kernel"],
    ),
    FillInExercise(
        exercise_id="trans_fill_02",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The set of all possible outputs of T is called the ___ or image",
        answer_type="text",
        correct_answer="range",
        case_sensitive=False,
        hints=[
            "All vectors of the form T(v)",
            "For matrices, this is the column space",
        ],
        tags=["terminology", "range"],
    ),
    FillInExercise(
        exercise_id="trans_fill_03",
        topic="transformations",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="The dimension of the range is called the ___",
        answer_type="text",
        correct_answer="rank",
        case_sensitive=False,
        hints=[
            "For matrices, it's the number of pivot columns",
            "rank(T) = dim(range(T))",
        ],
        tags=["terminology", "rank"],
    ),
    FillInExercise(
        exercise_id="trans_fill_04",
        topic="transformations",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The dimension of the kernel is called the ___",
        answer_type="text",
        correct_answer="nullity",
        case_sensitive=False,
        hints=[
            "Number of free variables in Ax = 0",
            "nullity(T) = dim(ker(T))",
        ],
        tags=["terminology", "nullity"],
    ),

    # Computational
    FillInExercise(
        exercise_id="trans_comp_01",
        topic="transformations",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If T: ℝ⁵ → ℝ³ has rank 2, what is its nullity?",
        answer_type="numerical",
        correct_answer=3.0,
        hints=[
            "Use Rank-Nullity Theorem",
            "5 = rank + nullity = 2 + nullity",
        ],
        tags=["rank_nullity", "computation"],
    ),
    FillInExercise(
        exercise_id="trans_comp_02",
        topic="transformations",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="For a 4×6 matrix with rank 3, what is the nullity?",
        answer_type="numerical",
        correct_answer=3.0,
        hints=[
            "Domain dimension is 6 (number of columns)",
            "6 = 3 + nullity",
        ],
        tags=["rank_nullity", "matrices"],
    ),
]


# ============================================================================
# DECOMPOSITIONS - LU, QR, SVD
# ============================================================================

DECOMPOSITIONS_EXERCISES = [
    # Multiple Choice
    MultipleChoiceExercise(
        exercise_id="decomp_mc_01",
        topic="decompositions",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="In LU decomposition A = LU, what are L and U?",
        choices=[
            "L is lower triangular, U is upper triangular",
            "L is upper triangular, U is lower triangular",
            "Both are diagonal matrices",
            "Both are orthogonal matrices",
        ],
        correct_index=0,
        explanation="LU: L = lower triangular with 1s on diagonal, U = upper triangular. Essentially encodes Gaussian elimination.",
        tags=["lu_decomposition", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="decomp_mc_02",
        topic="decompositions",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="In QR decomposition A = QR, what are Q and R?",
        choices=[
            "Q is orthogonal, R is upper triangular",
            "Q is upper triangular, R is orthogonal",
            "Both are diagonal matrices",
            "Q is lower triangular, R is diagonal",
        ],
        correct_index=0,
        explanation="QR: Q = orthogonal matrix (Q^T Q = I), R = upper triangular. Often computed using Gram-Schmidt.",
        tags=["qr_decomposition", "definition"],
    ),
    MultipleChoiceExercise(
        exercise_id="decomp_mc_03",
        topic="decompositions",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="What is the main use of LU decomposition?",
        choices=[
            "Finding eigenvalues",
            "Computing projections",
            "Efficiently solving multiple linear systems with the same A",
            "Computing determinants only",
        ],
        correct_index=2,
        explanation="LU is great for solving Ax=b for multiple different b vectors. Once you have A=LU, solving is just forward and back substitution.",
        tags=["lu_decomposition", "applications"],
    ),
    MultipleChoiceExercise(
        exercise_id="decomp_mc_04",
        topic="decompositions",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In SVD A = UΣV^T, what is Σ?",
        choices=[
            "An orthogonal matrix",
            "A diagonal matrix with singular values",
            "An upper triangular matrix",
            "The inverse of A",
        ],
        correct_index=1,
        explanation="Singular Value Decomposition: U and V are orthogonal, Σ is diagonal with non-negative singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0).",
        tags=["svd", "singular_values"],
    ),
    MultipleChoiceExercise(
        exercise_id="decomp_mc_05",
        topic="decompositions",
        difficulty=ExerciseDifficulty.CHALLENGE,
        question="Which decomposition works for ANY matrix (even non-square, non-invertible)?",
        choices=[
            "LU decomposition",
            "Eigenvalue decomposition",
            "Cholesky decomposition",
            "SVD (Singular Value Decomposition)",
        ],
        correct_index=3,
        explanation="SVD exists for ANY matrix m×n. Others have restrictions: LU needs non-singular, eigenvalue needs square, Cholesky needs positive definite.",
        tags=["svd", "universality"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="decomp_tf_01",
        topic="decompositions",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="LU decomposition exists for every invertible matrix",
        correct_answer=False,
        explanation="False! LU without row pivoting may not exist (e.g., if a11 = 0). With partial pivoting (PA = LU), it exists for any invertible matrix.",
        tags=["lu_decomposition", "existence"],
    ),
    TrueFalseExercise(
        exercise_id="decomp_tf_02",
        topic="decompositions",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="QR decomposition can be used to find an orthonormal basis",
        correct_answer=True,
        explanation="True. The columns of Q form an orthonormal basis for the column space of A. QR is essentially Gram-Schmidt in matrix form.",
        tags=["qr_decomposition", "orthonormal_basis"],
    ),
    TrueFalseExercise(
        exercise_id="decomp_tf_03",
        topic="decompositions",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The singular values in SVD are the square roots of eigenvalues of A^T A",
        correct_answer=True,
        explanation="True. If A = UΣV^T, then A^T A = VΣ²V^T. The singular values σᵢ are √(eigenvalues of A^T A).",
        tags=["svd", "singular_values"],
    ),
    TrueFalseExercise(
        exercise_id="decomp_tf_04",
        topic="decompositions",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Computing det(A) using LU is more efficient than cofactor expansion for large matrices",
        correct_answer=True,
        explanation="True! With A = LU, det(A) = det(L)det(U) = product of U's diagonal. O(n³) vs O(n!) for cofactor expansion.",
        tags=["lu_decomposition", "determinant"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="decomp_fill_01",
        topic="decompositions",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="In A = LU, the L matrix stands for ___ triangular",
        answer_type="text",
        correct_answer="lower",
        case_sensitive=False,
        hints=[
            "L has non-zero entries below the diagonal",
            "Opposite of upper",
        ],
        tags=["terminology", "lu_decomposition"],
    ),
    FillInExercise(
        exercise_id="decomp_fill_02",
        topic="decompositions",
        difficulty=ExerciseDifficulty.PRACTICE,
        question="In A = QR, the Q matrix is ___",
        answer_type="text",
        correct_answer="orthogonal",
        case_sensitive=False,
        hints=[
            "Q^T Q = I",
            "Columns are orthonormal",
        ],
        tags=["terminology", "qr_decomposition"],
    ),
    FillInExercise(
        exercise_id="decomp_fill_03",
        topic="decompositions",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="SVD stands for ___ Value Decomposition",
        answer_type="text",
        correct_answer="Singular",
        case_sensitive=False,
        hints=[
            "A = UΣV^T",
            "The values in Σ are called ___ values",
        ],
        tags=["terminology", "svd"],
    ),
    FillInExercise(
        exercise_id="decomp_fill_04",
        topic="decompositions",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The rank of matrix A equals the number of non-zero ___ values in its SVD",
        answer_type="text",
        correct_answer="singular",
        case_sensitive=False,
        hints=[
            "Values on the diagonal of Σ",
            "If A = UΣV^T, count non-zero entries of Σ",
        ],
        tags=["svd", "rank"],
    ),

    # Computational
    FillInExercise(
        exercise_id="decomp_comp_01",
        topic="decompositions",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="If A = [[2, 0], [0, 3]] (diagonal), what is det(A) using the product of diagonal entries?",
        answer_type="numerical",
        correct_answer=6.0,
        hints=[
            "For diagonal or triangular matrices",
            "2 × 3 = ?",
        ],
        tags=["determinant", "diagonal"],
    ),
]


# ============================================================================
# APPLICATIONS - PCA, LEAST SQUARES, MARKOV CHAINS, GRAPHICS
# ============================================================================

APPLICATIONS_EXERCISES = [
    # Multiple Choice
    MultipleChoiceExercise(
        exercise_id="app_mc_01",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In least squares regression, we want to minimize:",
        choices=[
            "||Ax - b||, the distance from b to the column space of A",
            "||x||, the length of the solution vector",
            "det(A), the determinant",
            "rank(A), the rank of the matrix",
        ],
        correct_index=0,
        explanation="Least squares minimizes ||Ax - b||². We find the best approximation to Ax = b when no exact solution exists. The solution is x̂ = (A^T A)^(-1) A^T b.",
        tags=["least_squares", "optimization"],
    ),
    MultipleChoiceExercise(
        exercise_id="app_mc_02",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Principal Component Analysis (PCA) uses which decomposition?",
        choices=[
            "LU decomposition",
            "QR decomposition",
            "Eigenvalue decomposition (or SVD)",
            "Cholesky decomposition",
        ],
        correct_index=2,
        explanation="PCA finds principal components as eigenvectors of the covariance matrix (or via SVD). The largest eigenvalues indicate directions of maximum variance.",
        tags=["pca", "decomposition"],
    ),
    MultipleChoiceExercise(
        exercise_id="app_mc_03",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In computer graphics, what does a rotation matrix do?",
        choices=[
            "Changes the size of objects",
            "Rotates vectors around the origin while preserving length",
            "Translates objects to a new position",
            "Projects 3D objects to 2D",
        ],
        correct_index=1,
        explanation="Rotation matrices are orthogonal and preserve lengths and angles. They rotate vectors around the origin without stretching or shrinking.",
        tags=["computer_graphics", "rotation"],
    ),
    MultipleChoiceExercise(
        exercise_id="app_mc_04",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="A Markov chain transition matrix has what property?",
        choices=[
            "All entries are non-negative and each column sums to 1",
            "It's symmetric",
            "It's diagonal",
            "All entries equal 1",
        ],
        correct_index=0,
        explanation="Markov transition matrices are stochastic: entries ≥ 0 and columns sum to 1 (each column represents probabilities). The steady state is the eigenvector for λ = 1.",
        tags=["markov_chains", "stochastic_matrix"],
    ),
    MultipleChoiceExercise(
        exercise_id="app_mc_05",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Google's PageRank algorithm is based on:",
        choices=[
            "Gaussian elimination",
            "The eigenvector of a Markov chain transition matrix",
            "Least squares regression",
            "QR decomposition",
        ],
        correct_index=1,
        explanation="PageRank finds the steady-state vector of a Markov chain representing web surfing. It's the eigenvector corresponding to eigenvalue 1 of the transition matrix.",
        tags=["pagerank", "eigenvector"],
    ),

    # True/False
    TrueFalseExercise(
        exercise_id="app_tf_01",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="Least squares is used when we have more equations than unknowns (overdetermined system)",
        correct_answer=True,
        explanation="True. When Ax = b has no exact solution (inconsistent), least squares finds the best approximation by minimizing ||Ax - b||².",
        tags=["least_squares", "overdetermined"],
    ),
    TrueFalseExercise(
        exercise_id="app_tf_02",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="PCA is used to reduce dimensionality by finding directions of maximum variance",
        correct_answer=True,
        explanation="True. PCA projects high-dimensional data onto lower-dimensional subspace capturing most variance. First few principal components explain most of the data.",
        tags=["pca", "dimensionality_reduction"],
    ),
    TrueFalseExercise(
        exercise_id="app_tf_03",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In computer graphics, translation (moving objects) can be represented by matrix multiplication alone",
        correct_answer=False,
        explanation="False. Translation requires adding a vector (not linear). In homogeneous coordinates, we can represent it as matrix multiplication in 4×4 matrices.",
        tags=["computer_graphics", "translation"],
    ),
    TrueFalseExercise(
        exercise_id="app_tf_04",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="A Markov chain always converges to a unique steady state",
        correct_answer=False,
        explanation="False. Only for certain types (irreducible and aperiodic chains). Some chains have multiple steady states or oscillate.",
        tags=["markov_chains", "convergence"],
    ),

    # Fill-in
    FillInExercise(
        exercise_id="app_fill_01",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="The solution to the least squares problem is x̂ = (A^T A)^(-1) A^T b. The matrix (A^T A)^(-1) A^T is called the ___",
        answer_type="text",
        correct_answer="pseudoinverse",
        case_sensitive=False,
        hints=[
            "Also called Moore-Penrose inverse",
            "Denoted A^+",
        ],
        tags=["terminology", "pseudoinverse"],
    ),
    FillInExercise(
        exercise_id="app_fill_02",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="PCA stands for ___ Component Analysis",
        answer_type="text",
        correct_answer="Principal",
        case_sensitive=False,
        hints=[
            "Used in dimensionality reduction",
            "Finds ___ components (directions of max variance)",
        ],
        tags=["terminology", "pca"],
    ),
    FillInExercise(
        exercise_id="app_fill_03",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="A matrix where each column's entries are non-negative and sum to 1 is called a ___ matrix",
        answer_type="text",
        correct_answer="stochastic",
        case_sensitive=False,
        hints=[
            "Used in Markov chains",
            "Columns represent probability distributions",
        ],
        tags=["terminology", "stochastic"],
    ),
    FillInExercise(
        exercise_id="app_fill_04",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In a Markov chain, the ___ state is reached when the probability distribution no longer changes",
        answer_type="text",
        correct_answer="steady",
        case_sensitive=False,
        hints=[
            "Also called equilibrium or stationary state",
            "It's an eigenvector with eigenvalue 1",
        ],
        tags=["terminology", "steady_state"],
    ),

    # Conceptual computational
    FillInExercise(
        exercise_id="app_comp_01",
        topic="applications",
        difficulty=ExerciseDifficulty.APPLICATION,
        question="In 2D, rotation by 90° counterclockwise transforms [1, 0] to [0, 1]. What is the first component of where [0, 1] goes?",
        answer_type="numerical",
        correct_answer=-1.0,
        hints=[
            "90° rotation: [x, y] → [-y, x]",
            "[0, 1] → [-1, 0]",
            "The first component is -1",
        ],
        tags=["rotation", "transformation"],
    ),
    MultipleChoiceExercise(
        exercise_id="app_mc_06",
        topic="applications",
        difficulty=ExerciseDifficulty.CHALLENGE,
        question="Which application uses SVD to recommend movies based on user ratings?",
        choices=[
            "Computer vision",
            "Collaborative filtering (Netflix prize)",
            "Image compression",
            "Speech recognition",
        ],
        correct_index=1,
        explanation="Collaborative filtering uses SVD on the user-movie rating matrix to find latent factors. Similar users and movies cluster in lower-dimensional space.",
        tags=["svd", "machine_learning"],
    ),
]


def get_exercises_by_topic(topic: str):
    """Get all exercises for a topic.

    Args:
        topic: Topic name (e.g., 'vectors', 'matrices', 'linear_systems',
               'vector_spaces', 'orthogonality', 'determinants', 'eigenvalues',
               'transformations', 'decompositions', 'applications')

    Returns:
        List of exercises for that topic
    """
    topic_lower = topic.lower()
    if topic_lower == "vectors":
        return VECTOR_EXERCISES
    elif topic_lower == "matrices":
        return MATRIX_EXERCISES
    elif topic_lower == "linear_systems":
        return LINEAR_SYSTEMS_EXERCISES
    elif topic_lower == "vector_spaces":
        return VECTOR_SPACES_EXERCISES
    elif topic_lower == "orthogonality":
        return ORTHOGONALITY_EXERCISES
    elif topic_lower == "determinants":
        return DETERMINANTS_EXERCISES
    elif topic_lower == "eigenvalues":
        return EIGENVALUES_EXERCISES
    elif topic_lower == "transformations":
        return TRANSFORMATIONS_EXERCISES
    elif topic_lower == "decompositions":
        return DECOMPOSITIONS_EXERCISES
    elif topic_lower == "applications":
        return APPLICATIONS_EXERCISES
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
