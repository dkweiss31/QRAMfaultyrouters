import os

import numpy as np
import pytest
from faulty_routers import MonteCarloRouterInstances


@pytest.mark.parametrize("n", [5, 6, 7])
@pytest.mark.parametrize("eps", [0.02, 0.08])
@pytest.mark.parametrize("top_three_functioning", [True, False])
def test_memory_efficient(n, eps, top_three_functioning):
    num_instances = 1000
    rng_seed = 2545685
    filepath_true = "tmp_true.h5py"
    filepath_false = "tmp_false.h5py"

    mc_false = MonteCarloRouterInstances(
        n, eps, num_instances, rng_seed, top_three_functioning, filepath_false, memory_efficient=False
    )
    mc_true = MonteCarloRouterInstances(
        n, eps, num_instances, rng_seed, top_three_functioning, filepath_true, memory_efficient=True
    )

    result_false = mc_false.run()
    result_true = mc_true.run()
    os.remove(filepath_true)
    os.remove(filepath_false)
    for val_true, val_false in zip(result_true.values(), result_false.values()):
        assert np.allclose(val_true, val_false)
