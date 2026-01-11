"""Solvers for vector operations."""

from typing import Any, Dict

import numpy as np

from .base import Solver, Solution, SolutionStep


class VectorAdditionSolver(Solver):
    """Step-by-step solver for vector addition."""

    def solve_with_steps(self, inputs: Dict[str, Any], expected: Any) -> Solution:
        """Generate step-by-step solution for vector addition.

        Args:
            inputs: Dict with keys 'v' and 'w' (vectors to add)
            expected: Expected result (optional)

        Returns:
            Solution with steps
        """
        v = np.array(inputs['v'])
        w = np.array(inputs['w'])

        steps = []

        # Step 1: State the problem
        steps.append(
            SolutionStep(
                step_number=1,
                description="Identify the vectors",
                mathematical_expression=f"v = {v.tolist()}, w = {w.tolist()}",
                explanation="We have two vectors that we need to add component-wise",
            )
        )

        # Step 2: Add component by component
        result = v + w

        component_additions = []
        for i in range(len(v)):
            component_additions.append(f"{v[i]} + {w[i]} = {result[i]}")

        steps.append(
            SolutionStep(
                step_number=2,
                description="Add corresponding components",
                mathematical_expression="v + w = [" + ", ".join(component_additions) + "]",
                explanation="Vector addition is performed by adding corresponding components",
                intermediate_result=result.tolist(),
            )
        )

        # Step 3: Write final result
        steps.append(
            SolutionStep(
                step_number=3,
                description="Final result",
                mathematical_expression=f"v + w = {result.tolist()}",
                explanation="This is our final answer",
            )
        )

        return Solution(
            problem_statement=f"Add vectors v = {v.tolist()} and w = {w.tolist()}",
            steps=steps,
            final_answer=result,
            verification=f"Check: Result has same dimension as input vectors ({len(result)})",
        )


class VectorDotProductSolver(Solver):
    """Step-by-step solver for vector dot product."""

    def solve_with_steps(self, inputs: Dict[str, Any], expected: Any) -> Solution:
        """Generate step-by-step solution for dot product.

        Args:
            inputs: Dict with keys 'v' and 'w' (vectors)
            expected: Expected result (optional)

        Returns:
            Solution with steps
        """
        v = np.array(inputs['v'])
        w = np.array(inputs['w'])

        steps = []

        # Step 1: State the problem
        steps.append(
            SolutionStep(
                step_number=1,
                description="Identify the vectors",
                mathematical_expression=f"v = {v.tolist()}, w = {w.tolist()}",
                explanation="We need to compute the dot product v · w",
            )
        )

        # Step 2: Multiply corresponding components
        products = v * w

        products_str = []
        for i in range(len(v)):
            products_str.append(f"({v[i]})({w[i]}) = {products[i]}")

        steps.append(
            SolutionStep(
                step_number=2,
                description="Multiply corresponding components",
                mathematical_expression=" + ".join(products_str),
                explanation="The dot product is the sum of products of corresponding components",
                intermediate_result=products.tolist(),
            )
        )

        # Step 3: Sum the products
        result = np.dot(v, w)

        steps.append(
            SolutionStep(
                step_number=3,
                description="Sum all products",
                mathematical_expression=f"{' + '.join(map(str, products))} = {result}",
                explanation="Add all the products together to get the final scalar result",
            )
        )

        return Solution(
            problem_statement=f"Compute v · w where v = {v.tolist()} and w = {w.tolist()}",
            steps=steps,
            final_answer=result,
            verification="The dot product is a scalar (single number)",
        )


class VectorScalarMultiplySolver(Solver):
    """Step-by-step solver for scalar multiplication of vectors."""

    def solve_with_steps(self, inputs: Dict[str, Any], expected: Any) -> Solution:
        """Generate step-by-step solution for scalar multiplication.

        Args:
            inputs: Dict with keys 'scalar' and 'vector'
            expected: Expected result (optional)

        Returns:
            Solution with steps
        """
        scalar = inputs['scalar']
        vector = np.array(inputs['vector'])

        steps = []

        # Step 1: State the problem
        steps.append(
            SolutionStep(
                step_number=1,
                description="Identify scalar and vector",
                mathematical_expression=f"scalar = {scalar}, vector = {vector.tolist()}",
                explanation="We multiply each component of the vector by the scalar",
            )
        )

        # Step 2: Multiply each component
        result = scalar * vector

        component_products = []
        for i in range(len(vector)):
            component_products.append(f"{scalar} × {vector[i]} = {result[i]}")

        steps.append(
            SolutionStep(
                step_number=2,
                description="Multiply each component by the scalar",
                mathematical_expression="[" + ", ".join(component_products) + "]",
                explanation="Scalar multiplication scales each component",
                intermediate_result=result.tolist(),
            )
        )

        # Step 3: Write final result
        steps.append(
            SolutionStep(
                step_number=3,
                description="Final result",
                mathematical_expression=f"{scalar} × {vector.tolist()} = {result.tolist()}",
                explanation="This is our final answer",
            )
        )

        return Solution(
            problem_statement=f"Multiply vector {vector.tolist()} by scalar {scalar}",
            steps=steps,
            final_answer=result,
            verification=f"Check: Result has same dimension as input vector ({len(result)})",
        )
