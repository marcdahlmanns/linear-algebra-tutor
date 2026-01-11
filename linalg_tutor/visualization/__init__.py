"""Visualization module for vectors, matrices, and geometric interpretations."""

from .vector_viz import (
    visualize_vector_2d,
    visualize_vector_components,
    visualize_vector_addition_2d,
    visualize_dot_product,
    print_vector_visualization,
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

__all__ = [
    # Vector visualizations
    "visualize_vector_2d",
    "visualize_vector_components",
    "visualize_vector_addition_2d",
    "visualize_dot_product",
    "print_vector_visualization",
    # Matrix visualizations
    "create_matrix_table",
    "visualize_matrix_multiply_step",
    "visualize_matrix_operation",
    "visualize_matrix_transpose",
    "visualize_determinant_2x2",
    "visualize_identity_matrix",
    "visualize_matrix_properties",
    "print_matrix_visualization",
    # Geometric interpretations
    "explain_vector_magnitude",
    "explain_dot_product_geometry",
    "explain_matrix_as_transformation",
    "explain_linear_combination",
    "explain_orthogonality",
    "explain_projection",
    "create_transformation_table",
]
