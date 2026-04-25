import math

import pytest

from data_harvesting.environment.data_collection.protocols import (
    coords_away_from_edge,
    direction_to_unit_vector,
    extend_unit_vector_to_edge,
    prevent_vector_escape,
    project_vector_onto_edge,
)


def assert_tuple_close(actual: tuple[float, float], expected: tuple[float, float], *, abs_tol: float = 1e-9) -> None:
    assert actual[0] == pytest.approx(expected[0], abs=abs_tol)
    assert actual[1] == pytest.approx(expected[1], abs=abs_tol)


class TestDirectionToUnitVector:
    def test_cardinal_directions(self) -> None:
        assert_tuple_close(direction_to_unit_vector(0.0), (1.0, 0.0))
        assert_tuple_close(direction_to_unit_vector(math.pi / 2), (0.0, 1.0), abs_tol=1e-8)
        assert_tuple_close(direction_to_unit_vector(math.pi), (-1.0, 0.0), abs_tol=1e-8)
        assert_tuple_close(direction_to_unit_vector(3 * math.pi / 2), (0.0, -1.0), abs_tol=1e-8)

    @pytest.mark.parametrize("angle", [0.0, 0.1, 1.234, 2 * math.pi - 1e-12, 2 * math.pi, -0.5])
    def test_result_is_unit_length(self, angle: float) -> None:
        x, y = direction_to_unit_vector(angle)
        assert (x * x + y * y) == pytest.approx(1.0, abs=1e-9)


class TestCoordsAwayFromEdge:
    def test_returns_true_when_within_margin_of_x_edge(self) -> None:
        # limit=10, margin=1 => x edge zone starts at |x|>=9
        assert coords_away_from_edge((9.0, 0.0), 10.0, 1.0) is True
        assert coords_away_from_edge((-9.0, 0.0), 10.0, 1.0) is True

    def test_returns_true_when_within_margin_of_y_edge(self) -> None:
        assert coords_away_from_edge((0.0, 9.0), 10.0, 1.0) is True
        assert coords_away_from_edge((0.0, -9.0), 10.0, 1.0) is True

    def test_returns_false_when_safely_inside_bounds(self) -> None:
        assert coords_away_from_edge((8.999, 0.0), 10.0, 1.0) is False
        assert coords_away_from_edge((0.0, 8.0), 10.0, 1.0) is False


_unit_vectors: list[tuple[float, float]] = [
    (1.0, 0.0), # Right
    (0.0, 1.0), # Up
    (-1.0, 0.0),# Left
    (0.0, -1.0),# Down
    (math.sqrt(2)/2, math.sqrt(2)/2), # Up-Right
    (-math.sqrt(2)/2, math.sqrt(2)/2),# Up-Left
    (-math.sqrt(2)/2, -math.sqrt(2)/2),# Down-Left
    (math.sqrt(2)/2, -math.sqrt(2)/2), # Down-Right
]

class TestExtendUnitVectorToEdge:
    def test_scales_to_nearest_edge_distance(self) -> None:
        # Starting at (2, 3) in a square [-10,10]^2.
        # Nearest edge distance is min(10-|2|, 10-|3|)=min(8,7)=7.
        # For unit vector (1,0), extension should be (7,0).
        assert_tuple_close(extend_unit_vector_to_edge((2.0, 3.0), (1.0, 0.0), 10.0), (8.0, 0.0))

    def test_respects_diagonal_unit_vector(self) -> None:
        # At origin, nearest edge dist=10. Vector (sqrt(2)/2, sqrt(2)/2) scaled by 10.
        u = (math.sqrt(2) / 2, math.sqrt(2) / 2)
        expected = (10.0, 10.0)
        assert_tuple_close(extend_unit_vector_to_edge((0.0, 0.0), u, 10.0), expected, abs_tol=1e-9)

    def test_zero_distance_when_already_at_an_edge_coordinate(self) -> None:
        # If already at x edge, distance_to_x_edge becomes 0 -> overall step should be 0.
        assert_tuple_close(extend_unit_vector_to_edge((10.0, 0.0), (1.0, 0.0), 10.0), (0.0, 0.0))

    def test_negative_unit_vector(self):
        assert_tuple_close(
            extend_unit_vector_to_edge((0, 0), (-1, 0), 5.0),
            (-5.0, 0.0)
        )

    def test_works_with_zero_unit_vector(self):
        assert_tuple_close(
            extend_unit_vector_to_edge((3.0, 4.0), (0.0, 0.0), 10.0),
            (0.0, 0.0)
        )

    @pytest.mark.parametrize("unit_vector", _unit_vectors)
    def test_vector_maintains_angle(self, unit_vector: tuple[float, float]) -> None:
        start_pos = (0, 0)
        result = extend_unit_vector_to_edge(start_pos, unit_vector, 10.0)

        angle_input = math.atan2(unit_vector[1], unit_vector[0])
        angle_result = math.atan2(result[1], result[0])
        assert angle_input == pytest.approx(angle_result, abs=1e-9)


_edge_tests: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((10.0, 0.0), (0.6, 0.8)),    # At right edge, moving up-right
    ((-10.0, 0.0), (-0.6, 0.8)),  # At left edge, moving up-left
    ((0.0, 10.0), (0.6, 0.8)),    # At top edge, moving up-right
    ((0.0, -10.0), (0.6, -0.8)),  # At bottom edge, moving down-right
    ((10.0, 10.0), (0.6, 0.8)),   # At top-right corner, moving up-right
    ((-10.0, 10.0), (-0.6, 0.8)), # At top-left corner, moving up-left
    ((-10.0, -10.0), (-0.6, -0.8)),# At bottom-left corner, moving down-left
    ((10.0, -10.0), (0.6, -0.8)), # At bottom-right corner, moving down-right
]
class TestProjectVectorOntoEdge:
    def test_projects_x_component_when_at_x_edge(self) -> None:
        assert_tuple_close(project_vector_onto_edge((10.0, 0.0), (0.6, 0.8), 10.0), (0.0, 10.0))
        assert_tuple_close(project_vector_onto_edge((-10.0, 0.0), (-0.6, 0.8), 10.0), (0.0, 10.0))

    def test_projects_y_component_when_at_y_edge(self) -> None:
        assert_tuple_close(project_vector_onto_edge((0.0, 10.0), (0.6, 0.8), 10.0), (10.0, 0.0))
        assert_tuple_close(project_vector_onto_edge((0.0, -10.0), (0.6, -0.8), 10.0), (10.0, 0.0))

    def test_projects_both_components_when_at_corner(self) -> None:
        assert_tuple_close(project_vector_onto_edge((10.0, 10.0), (0.6, 0.8), 10.0), (0.0, 0.0))

    def test_no_projection_when_inside_bounds(self) -> None:
        assert_tuple_close(project_vector_onto_edge((9.999, 0.0), (0.6, 0.8), 10.0), (0.001, 10))

    def test_gets_stuck_on_edges(self) -> None:
        position = (10, 10)
        unit_vector = (0.5, 0.5)
        projected = project_vector_onto_edge(position, unit_vector, 10.0)
        assert_tuple_close(projected, (0.0, 0.0))

    @pytest.mark.parametrize("position,unit_vector", _edge_tests)
    def test_correct_projection_at_edges(self, position, unit_vector) -> None:
        projected = project_vector_onto_edge(position, unit_vector, 10.0)
        # After projection, at least one component should be zero
        assert projected[0] == pytest.approx(0.0, abs=1e-9) or projected[1] == pytest.approx(0.0, abs=1e-9)

        # After projection, scalar product with original vector should be positive or zero
        scalar_product = projected[0] * unit_vector[0] + projected[1] * unit_vector[1]
        assert scalar_product >= -1e-9  # Allow small negative due to floating point

        # After projection, the destination point should be on the edge
        dest_x = position[0] + projected[0]
        dest_y = position[1] + projected[1]
        assert abs(dest_x) == pytest.approx(10.0, abs=1e-9) or abs(dest_y) == pytest.approx(10.0, abs=1e-9)

class TestPreventVectorEscape:
    def test_maintains_angle_when_inside_bounds(self) -> None:
        position = (0.0, 0.0)
        unit_vector = (0.6, 0.8)
        result = prevent_vector_escape(position, unit_vector, 10.0)
        angle_input = math.atan2(unit_vector[1], unit_vector[0])
        angle_result = math.atan2(result[1], result[0])
        assert angle_input == pytest.approx(angle_result, abs=1e-9)

