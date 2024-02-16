import argparse
import sys

sys.path.append("/gpfs/gibbs/project/puri/dkw34/QRAMfaultyrouters/QRAMfaultyrouters")
sys.path.append("/Users/danielweiss/PycharmProjects/QRAMfaultyrouters")
from quantum_utils import generate_file_path
from faulty_routers import MonteCarloRouterInstances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchQOC sim of second order bin code")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--n", default=8, type=int, help="tree depth")
    parser.add_argument("--eps", default=0.01, type=float, help="failure rate")
    parser.add_argument("--num_instances", default=10000, type=int, help="number monte carlo instances")
    parser.add_argument("--rng_seed", default=1, type=int, help="rng seed")
    args = parser.parse_args()
    if args.idx == -1:
        filename = generate_file_path("h5py", "faulty_routers", "out")
    else:
        filename = f"out/{str(args.idx).zfill(5)}_faulty_routers.h5py"
    mcinstance = MonteCarloRouterInstances(
        args.n,
        args.eps,
        args.num_instances,
        args.rng_seed,
        filepath=filename,
    )
    mcinstance.run()
