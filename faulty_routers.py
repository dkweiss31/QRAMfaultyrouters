from typing import Callable

import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Style

from quantum_utils import write_to_h5

colorama_init()


class QRAMRouter:
    """
    location: str
        binary string representing the path to the router
    left_child and right_child: None or Router
        if None, this is a leaf node and accesses classical data. If
        Router, this specifies the left and right child routers of this internal router
    functioning: bool
        whether or not this router is functioning
    """
    def __init__(self, location="0b", left_child=None, right_child=None, functioning=True):
        self.location = location
        self.left_child = left_child
        self.right_child = right_child
        self.functioning = functioning

    def create_tree(self, tree_depth, current_location="0b"):
        """create a tree of depth tree_depth without deactivating any routers"""
        if tree_depth <= 0:
            raise ValueError("can't have a zero-bit QRAM silly")
        elif tree_depth == 1:
            return QRAMRouter(current_location + "", None, None)
        else:
            left_child = self.create_tree(tree_depth - 1, current_location + "0")
            right_child = self.create_tree(tree_depth - 1, current_location + "1")
            return QRAMRouter(current_location, left_child, right_child)

    def tree_depth(self):
        if self.right_child is None:
            return 1
        else:
            return self.right_child.tree_depth() + 1

    def print_tree(self, indent="", am_I_a_right_or_left_child="right"):
        if self.right_child:
            self.right_child.print_tree(indent + "     ", "right")
        print(indent, end="")
        if am_I_a_right_or_left_child == "right":
            print("┌───", end="")
        elif am_I_a_right_or_left_child == "left":
            print("└───", end="")
        else:
            print("    ", end="")
        if self.functioning:
            print(self.location)
        else:
            print(f"{Fore.RED}{self.location}{Style.RESET_ALL}")
        if self.left_child:
            self.left_child.print_tree(indent + "     ", "left")

    def deactivate_self_and_below_routers(self):
        """called if a parent router is not functioning. All below routers
        are then nonfunctioning"""
        self.functioning = False
        if self.right_child:
            self.right_child.deactivate_self_and_below_routers()
            self.left_child.deactivate_self_and_below_routers()

    def fabrication_instance(self, failure_rate=0.3, rn_list=None):
        """this function utilizes a list of random numbers (if passed) to decide
        if a given router is faulty. It strips off the first one and returns
        the remainder of the list for use for deciding if other routers are faulty"""
        if rn_list is None:
            rnd = np.random.random()
            new_rn_list = None
        else:
            rnd = rn_list[0]
            new_rn_list = rn_list[1:]
        if rnd < failure_rate:
            self.deactivate_self_and_below_routers()
            return new_rn_list
        else:
            self.functioning = True
            if self.right_child:
                new_rn_list = self.right_child.fabrication_instance(failure_rate, rn_list=new_rn_list)
                new_rn_list = self.left_child.fabrication_instance(failure_rate, rn_list=new_rn_list)
            return new_rn_list

    def count_number_faulty_addresses(self):
        if self.right_child is None:
            if self.functioning:
                return 0
            else:
                return 2
        else:
            if not self.functioning:
                return 2**self.tree_depth()
            else:
                return (self.right_child.count_number_faulty_addresses()
                        + self.left_child.count_number_faulty_addresses())

    def router_list_functioning_or_not(self, functioning=False, existing_router_list=None):
        """function that can compute a list of faulty routers and also functioning routers.
         If functioning is set to False, this creates a list of all non-functioning routers.
         If functioning is set to true, then it creates a list of functioning routers."""
        if existing_router_list is None:
            existing_router_list = []
        if self.right_child is None:
            if self.functioning == functioning:
                existing_router_list.append(self.location)
                return existing_router_list
            else:
                return existing_router_list
        else:
            existing_router_list = self.right_child.router_list_functioning_or_not(
                functioning=functioning, existing_router_list=existing_router_list
            )
            existing_router_list = self.left_child.router_list_functioning_or_not(
                functioning=functioning, existing_router_list=existing_router_list
            )
            return existing_router_list

    def assign_original_and_augmented(self):
        num_faulty_right = self.right_child.count_number_faulty_addresses()
        num_faulty_left = self.left_child.count_number_faulty_addresses()
        if num_faulty_right <= num_faulty_left:
            original_tree = self.right_child
            augmented_tree = self.left_child
        else:
            original_tree = self.left_child
            augmented_tree = self.right_child
        return original_tree, augmented_tree

    def bit_flip(self, ell, address):
        # flip first ell bits of router. Need to do some bin gymnastics
        # -1 is because we are looking at the routers as opposed to the addresses
        n = self.tree_depth()
        bit_flip_mask = bin((1 << ell) - 1) + (n-ell - 1) * "0"
        flipped_address = int(bit_flip_mask, 2) ^ int(address, 2)
        flipped_address_bin = format(flipped_address, f"#0{n + 2 - 1}b")
        return flipped_address_bin

    def router_repair(self):
        n = self.tree_depth()
        # each has depth n-1
        original_tree, augmented_tree = self.assign_original_and_augmented()
        faulty_router_list = original_tree.router_list_functioning_or_not(functioning=False)
        available_router_list = augmented_tree.router_list_functioning_or_not(functioning=True)
        num_faulty = len(faulty_router_list)
        if num_faulty == 0:
            # no repair necessary
            return [], 0
        # original faulty routers + augmented faulty routers
        total_faulty = num_faulty + 2**(n-1) - len(available_router_list)
        if total_faulty > 2**(n - 1):
            # QRAM unrepairable
            return [], np.inf
        fr_idx_list = list(range(num_faulty))
        # will contain mapping of faulty routers to repair routers
        repair_list = np.empty(num_faulty, dtype=object)
        fixed_faulty_list = []
        # only go up to ell = n-2, since no savings if we do l=n-1
        for ell in range(1, n-1):
            for faulty_router, fr_idx in zip(faulty_router_list, fr_idx_list):
                if faulty_router not in fixed_faulty_list:
                    # perform bitwise NOT on the first ell bits of faulty_router
                    flipped_address = self.bit_flip(ell, faulty_router)
                    if flipped_address in available_router_list:
                        repair_list[fr_idx] = flipped_address
                        fixed_faulty_list.append(faulty_router)
            if len(fixed_faulty_list) == num_faulty:
                # succeeded in repair with ell bit flips
                return repair_list, ell
        # failed repair, proceed with greedy assignment requiring n-1 bit flips
        return available_router_list[0: num_faulty], n - 1


class MonteCarloRouterInstances:
    def __init__(self, n, eps, num_instances, rng_seed, filepath="tmp.h5py", instantiate_trees=True):
        self.n = n
        self.eps = eps
        self.num_instances = num_instances
        self.rng_seed = rng_seed
        self.filepath = filepath
        self._init_attrs = set(self.__dict__.keys())
        self.rng = np.random.default_rng(rng_seed)
        self.rn_list = self.rng.random((num_instances, 2 ** n))
        self.instantiate_trees = instantiate_trees
        if instantiate_trees:
            self.trees = self.create_trees()
        else:
            self.trees = None

    def param_dict(self):
        return {k: v for k, v in self.__dict__.items() if k in self._init_attrs}

    def run(self):
        print("running faulty QRAM simulation")
        self.fab_instance_for_all_trees()
        max_num_bit_flips, avg_bit_flips = self.bit_flips_required_for_repair()
        avg_num_faulty, frac_repairable = self.avg_num_faulty_and_repairable()
        data_dict = {
            "max_num_bit_flips": max_num_bit_flips,
            "avg_bit_flips": avg_bit_flips,
            "avg_num_faulty": avg_num_faulty,
            "frac_repairable": frac_repairable,
        }
        print(f"writing results to {self.filepath}")
        write_to_h5(self.filepath, data_dict, self.param_dict())

    def create_trees(self):
        return list(map(lambda _: QRAMRouter().create_tree(self.n), range(self.num_instances)))

    def _fab_instance_for_one_tree(self, tree_rn_list):
        tree, rn_list = tree_rn_list
        _ = tree.fabrication_instance(self.eps, rn_list)
        return tree

    def fab_instance_for_all_trees(self):
        if self.trees is None:
            self.trees = self.create_trees()
        self.trees = list(map(self._fab_instance_for_one_tree, zip(self.trees, self.rn_list)))

    def map_over_trees(self, func):
        if self.trees is None:
            self.trees = self.create_trees()
        if type(func) is str:
            return list(map(lambda tree: getattr(tree, func)(), self.trees))
        elif type(func) is Callable:  # assume its a function that takes a single tree as argument
            return list(map(lambda tree: func(tree), self.trees))
        else:
            raise ValueError("func needs to be a string or a callable")

    def bit_flips_required_for_repair(self):
        repaired_routers = self.map_over_trees("router_repair")
        routers, num_bit_flips = list(zip(*repaired_routers))
        num_bit_flips = np.array(num_bit_flips)
        # exclude inf if present, indicating unrepairable
        num_bit_flips = num_bit_flips[num_bit_flips < np.inf]
        max_num_bit_flips = np.max(num_bit_flips)
        avg_bit_flips = np.average(num_bit_flips)
        return max_num_bit_flips, avg_bit_flips

    def avg_num_faulty_and_repairable(self):
        num_faulty = np.array(self.map_over_trees("count_number_faulty_addresses"))
        avg_num_faulty = np.average(num_faulty)
        frac_repairable = len(num_faulty[num_faulty <= 2 ** (self.n - 1)]) / self.num_instances
        return avg_num_faulty, frac_repairable
