"""Visualization module for vectors, matrices, and geometric interpretations."""

from .vector_viz import (
    visualize_vector_2d,
    visualize_vector_components,
    visualize_vector_addition_2d,
    visualize_dot_product,
    print_vector_visualization,
    visualize_vector_norm,
    visualize_cross_product,
    visualize_projection,
    visualize_vector_subtraction,
)

from .matrix_viz import (
    create_matrix_table,
    visualize_matrix_multiply_step,
    visualize_matrix_operation,
    visualize_matrix_transpose,
    visualize_determinant_2x2,
    visualize_identity_matrix,
    visualize_matrix_properties,
    print_matrix_visualization,
    visualize_determinant_3x3,
    visualize_determinant_geometric,
    visualize_eigenvalues_2x2,
    explain_eigenvectors,
)

from .geometric import (
    explain_vector_magnitude,
    explain_dot_product_geometry,
    explain_matrix_as_transformation,
    explain_linear_combination,
    explain_orthogonality,
    explain_projection,
    create_transformation_table,
)

from .linear_systems import (
    visualize_augmented_matrix,
    visualize_linear_system_2d,
    visualize_row_operation,
    visualize_solution_verification,
    visualize_system_classification,
    explain_gaussian_elimination,
    visualize_homogeneous_system,
)

__all__ = [
    # Vector visualizations
    "visualize_vector_2d",
    "visualize_vector_components",
    "visualize_vector_addition_2d",
    "visualize_dot_product",
    "print_vector_visualization",
    "visualize_vector_norm",
    "visualize_cross_product",
    "visualize_projection",
    "visualize_vector_subtraction",
    # Matrix visualizations
    "create_matrix_table",
    "visualize_matrix_multiply_step",
    "visualize_matrix_operation",
    "visualize_matrix_transpose",
    "visualize_determinant_2x2",
    "visualize_determinant_3x3",
    "visualize_determinant_geometric",
    "visualize_identity_matrix",
    "visualize_matrix_properties",
    "print_matrix_visualization",
    "visualize_eigenvalues_2x2",
    "explain_eigenvectors",
    # Geometric interpretations
    "explain_vector_magnitude",
    "explain_dot_product_geometry",
    "explain_matrix_as_transformation",
    "explain_linear_combination",
    "explain_orthogonality",
    "explain_projection",
    "create_transformation_table",
    # Linear systems
    "visualize_augmented_matrix",
    "visualize_linear_system_2d",
    "visualize_row_operation",
    "visualize_solution_verification",
    "visualize_system_classification",
    "explain_gaussian_elimination",
    "visualize_homogeneous_system",
]
