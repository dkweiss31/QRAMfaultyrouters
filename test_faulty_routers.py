import os

import numpy as np
import pytest
from faulty_routers import MonteCarloRouterInstances, QRAMRouter


@pytest.mark.parametrize("n", [5, 6, 7])
@pytest.mark.parametrize("eps", [0.02, 0.08])
@pytest.mark.parametrize("top_three_functioning", [True, False])
def test_memory_efficient(n, eps, top_three_functioning):
    num_instances = 1000
    rng_seed = 2545685
    filepath = "tmp.h5py"
    mc = MonteCarloRouterInstances(
        n, eps, num_instances, rng_seed, top_three_functioning, filepath,
    )
    result = mc.run()
    os.remove(filepath)


@pytest.mark.parametrize("n", [5, 6, 7])
@pytest.mark.parametrize("eps", [0.02, 0.08])
@pytest.mark.parametrize("top_three_functioning", [True, False])
def test_fabrication_instance(n, eps, top_three_functioning):
    num_instances = 20000
    rng_seed = 2545685 * n
    rng = np.random.default_rng(rng_seed)
    rando_faulty_counter = 0
    for instance in range(num_instances):
        tree = QRAMRouter()
        full_tree = tree.create_tree(n)
        full_tree.fabrication_instance(
            eps, rng, top_three_functioning=top_three_functioning
        )
        assert full_tree.tree_depth() == n
        if top_three_functioning:
            assert full_tree.functioning
            assert full_tree.right_child.functioning and full_tree.left_child.functioning
        if not full_tree.right_child.left_child.left_child.functioning:
            rando_faulty_counter += 1
    if not top_three_functioning:
        ideal_eps = eps + (1 - eps) * eps + (1 - eps)**2 * eps + (1 - eps)**3 * eps
    else:
        ideal_eps = eps + (1 - eps) * eps
    assert ideal_eps - 0.01 < rando_faulty_counter / num_instances < ideal_eps + 0.01
