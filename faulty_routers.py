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
    def __init__(
        self,
        location="0b",
        left_child=None,
        right_child=None,
        functioning=True,
        largest_functioning_subtree_depth=0,
        part_of_subtree=False,
    ):
        self.location = location
        self.left_child = left_child
        self.right_child = right_child
        self.functioning = functioning
        self.largest_functioning_subtree_depth_as_root = largest_functioning_subtree_depth
        self.part_of_subtree = part_of_subtree

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

    def print_tree(self, indent="", am_I_a_right_or_left_child="right", print_attr="location"):
        if self.right_child:
            self.right_child.print_tree(indent + "     ", am_I_a_right_or_left_child="right", print_attr=print_attr)
        print(indent, end="")
        if am_I_a_right_or_left_child == "right":
            print("┌───", end="")
        elif am_I_a_right_or_left_child == "left":
            print("└───", end="")
        else:
            print("    ", end="")
        if self.functioning:
            if self.part_of_subtree:
                print(f"{Fore.BLUE}{getattr(self, print_attr)}{Style.RESET_ALL}")
            else:
                print(getattr(self, print_attr))
        else:
            print(f"{Fore.RED}{getattr(self, print_attr)}{Style.RESET_ALL}")
        if self.left_child:
            self.left_child.print_tree(indent + "     ", am_I_a_right_or_left_child="left", print_attr=print_attr)

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

    def simple_reallocation(self, m, k):
        """m is the depth of the QRAM we are after, and k is the level of the QRAM this
        router should be acting as"""
        def _simple_reallocation(m_depth, k_depth, router, existing_subtree=None):
            if existing_subtree is None:
                existing_subtree = []
            router.part_of_subtree = True
            # we've reached the bottom
            if m_depth - k_depth == 1:
                existing_subtree.append(router.location)
                return existing_subtree
            right_avail_router_list = router.right_child.lowest_router_list_functioning_or_not(functioning=True)
            left_avail_router_list = router.left_child.lowest_router_list_functioning_or_not(functioning=True)
            num_right = len(right_avail_router_list)
            num_left = len(left_avail_router_list)
            # can succesfully split this router as enough addresses available on each side
            if num_right >= 2**(m_depth-k_depth-2) and num_left >= 2**(m_depth-k_depth-2):
                router.part_of_subtree = True
                existing_subtree = _simple_reallocation(
                    m_depth, k_depth + 1, router.right_child, existing_subtree=existing_subtree
                )
                existing_subtree = _simple_reallocation(
                    m_depth, k_depth + 1, router.left_child, existing_subtree=existing_subtree
                )
                return existing_subtree
            # check if right or left child router can serve as a level k router
            elif num_right >= 2**(m_depth - k_depth - 1):
                existing_subtree = _simple_reallocation(
                    m_depth, k_depth, router.right_child, existing_subtree=existing_subtree
                )
                return existing_subtree
            elif num_left >= 2**(m_depth - k_depth - 1):
                existing_subtree = _simple_reallocation(
                    m_depth, k_depth, router.left_child, existing_subtree=existing_subtree
                )
                return existing_subtree
            else:
                raise SimpleReallocationFailure
        try:
            subtree = _simple_reallocation(m, k, self)
            return subtree
        except SimpleReallocationFailure:
            return None

    def all_functioning_to_depth(self, depth):
        if depth == 0:  # indicates we were only asking about above routers, doesn't include self
            return True
        elif not self.functioning:
            return False
        elif self.right_child is None:
            if depth == 1:
                return True
            else:
                return False
        else:
            return (self.right_child.all_functioning_to_depth(depth - 1)
                    and self.left_child.all_functioning_to_depth(depth - 1))

    def calculate_my_largest_functioning_subtree(self):
        if not self.functioning:
            self.largest_functioning_subtree_depth_as_root = 0
        else:
            n = self.tree_depth()
            for depth in range(1, n + 1):  # go until not all functioning to a certain depth
                depth_functioning_subtree = self.all_functioning_to_depth(depth)
                if depth_functioning_subtree:
                    self.largest_functioning_subtree_depth_as_root = depth
                else:
                    break
            if n != 1:
                self.right_child.calculate_my_largest_functioning_subtree()
                self.left_child.calculate_my_largest_functioning_subtree()

    def largest_functioning_subtree(self):
        if self.right_child is None:
            return self.largest_functioning_subtree_depth_as_root
        else:
            return max(self.largest_functioning_subtree_depth_as_root,
                       self.right_child.largest_functioning_subtree(),
                       self.left_child.largest_functioning_subtree(),
                       )

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

    def lowest_router_list_functioning_or_not(self, functioning=False, existing_router_list=None):
        """function that can compute a list of faulty routers and also functioning routers
        among those at the bottom of the tree. If functioning is set to False,
        this creates a list of all non-functioning routers. If functioning is set to
        true, then it creates a list of functioning routers."""
        if existing_router_list is None:
            existing_router_list = []
        if self.right_child is None:
            if self.functioning == functioning:
                existing_router_list.append(self.location)
                return existing_router_list
            else:
                return existing_router_list
        else:
            existing_router_list = self.right_child.lowest_router_list_functioning_or_not(
                functioning=functioning, existing_router_list=existing_router_list
            )
            existing_router_list = self.left_child.lowest_router_list_functioning_or_not(
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
        faulty_router_list = original_tree.lowest_router_list_functioning_or_not(functioning=False)
        available_router_list = augmented_tree.lowest_router_list_functioning_or_not(functioning=True)
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
        repaired_routers = self.map_over_trees("router_repair")
        routers, num_bit_flips = list(zip(*repaired_routers))
        num_bit_flips = np.array(num_bit_flips)
        num_faulty = np.array(self.map_over_trees("count_number_faulty_addresses"))
        data_dict = {
            "num_bit_flips": num_bit_flips,
            "num_faulty": num_faulty,
        }
        print(f"writing results to {self.filepath}")
        write_to_h5(self.filepath, data_dict, self.param_dict())
        return 0

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


class AnalyticUnrepair:
    def __init__(self, eps):
        self.eps = eps
        self.saved_add_available = {
            "P_2_0": self.eps + (1 - self.eps) * eps ** 2,
            "P_2_1": 0.0,
            "P_2_2": 2.0 * self.eps * (1 - self.eps) ** 2,
            "P_2_3": 0.0,
            "P_2_4": (1 - self.eps) ** 3,
        }

    def prob_ell_available(self, n, ell):
        if ell < 0 or ell > 2**n:
            return 0.0
        if f"P_{n}_{ell}" in self.saved_add_available:
            return self.saved_add_available[f"P_{n}_{ell}"]
        P_n_ell = sum((1 - self.eps)
                      * self.prob_ell_available(n - 1, m)
                      * self.prob_ell_available(n - 1, ell - m)
                      for m in range(2 ** (n - 1) + 1))
        if ell == 0:
            P_n_ell += self.eps
        self.saved_add_available[f"P_{n}_{ell}"] = P_n_ell
        return P_n_ell

    def prob_unrepairable(self, n):
        return sum(self.prob_ell_available(n, ell) for ell in range(0, 2 ** (n - 1)))


class SimpleReallocationFailure(Exception):
    """Raised when the simple reallocaion fails at runtime"""
    pass


if __name__ == "__main__":
    NUM_INSTANCES = 10000
    n = 5
    EPS = 0.1
    # 6722222232 gives 5 and 3 configuration (not simply repairable)
    # 6243254322 gives 6 and 2
    # 22543268254 fails at the second level: starts off with 6, 4, but then drops to 3, 1 on the right.
    rng = np.random.default_rng(22543268585425543)  # 27585353
    RN_LIST = rng.random((NUM_INSTANCES, 2 ** n))
    MYTREE = QRAMRouter()
    FULLTREE = MYTREE.create_tree(n)
    FULLTREE.fabrication_instance(EPS, rn_list=RN_LIST[0])
    FULLTREE.calculate_my_largest_functioning_subtree()
    FULLTREE.print_tree()
    #FULLTREE.print_tree(print_attr="largest_functioning_subtree_depth_as_root")
    avail_router = FULLTREE.lowest_router_list_functioning_or_not(functioning=True)
    desired_depth_subQ = 4
    if len(avail_router) >= 2**(desired_depth_subQ - 1):
        subtree = FULLTREE.simple_reallocation(4, 0)
        FULLTREE.print_tree()
        print(subtree)
    #print(FULLTREE.largest_functioning_subtree())
