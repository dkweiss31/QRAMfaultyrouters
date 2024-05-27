import time
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
        part_of_subtree=False,
    ):
        self.location = location
        self.left_child = left_child
        self.right_child = right_child
        self.functioning = functioning
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

    def set_attr_self_and_below(self, attr, val):
        setattr(self, attr, val)
        if self.right_child:
            self.right_child.set_attr_self_and_below(attr, val)
            self.left_child.set_attr_self_and_below(attr, val)

    def fabrication_instance(
        self,
        failure_rate=0.3,
        rn_list=None,
        k_level=0,
        top_three_functioning=False
    ):
        """this function utilizes a list of random numbers (if passed) to decide
        if a given router is faulty. It strips off the first one and returns
        the remainder of the list for use for deciding if other routers are faulty"""
        if rn_list is None:
            rnd = np.random.random()
            new_rn_list = None
        else:
            rnd = rn_list[0]
            new_rn_list = rn_list[1:]
        if rnd > failure_rate or (top_three_functioning and k_level <= 1):
            self.functioning = True
            if self.right_child:
                new_rn_list = self.right_child.fabrication_instance(
                    failure_rate, rn_list=new_rn_list, k_level=k_level+1, top_three_functioning=top_three_functioning
                )
                new_rn_list = self.left_child.fabrication_instance(
                    failure_rate, rn_list=new_rn_list, k_level=k_level+1, top_three_functioning=top_three_functioning
                )
                return new_rn_list
            else:
                return new_rn_list
        else:
            self.set_attr_self_and_below("functioning", False)
            return new_rn_list

    def simple_reallocation(self, m, k):
        """m is the depth of the QRAM we are after, and k is the level of the QRAM this
        router should be acting as
        returns the routers at the bottom of the tree that are part of the m-bit
        QRAM as well as a flag: 1 if it succeeded, 0 if it failed to find an m-bit QRAM
        """
        num_avail = self.router_list_functioning_or_not(functioning=True)
        if len(num_avail) < 2**(m-2):
            return [], 0

        def _simple_reallocation(m_depth, k_depth, router, existing_subtree=None):
            if existing_subtree is None:
                existing_subtree = []
            router.part_of_subtree = True
            # we've reached the bottom
            if m_depth - k_depth == 1:
                existing_subtree.append(router.location)
                return existing_subtree
            right_avail_router_list = router.right_child.router_list_functioning_or_not(functioning=True)
            left_avail_router_list = router.left_child.router_list_functioning_or_not(functioning=True)
            num_right = len(right_avail_router_list)
            num_left = len(left_avail_router_list)
            # can succesfully split this router as enough addresses available on each side
            if num_right >= 2**(m_depth-k_depth-2) and num_left >= 2**(m_depth-k_depth-2):
                # try splitting this router. If we fail later on, we return here and enter the if
                # statements below
                try:
                    router.part_of_subtree = True
                    existing_subtree = _simple_reallocation(
                        m_depth, k_depth + 1, router.right_child, existing_subtree=existing_subtree
                    )
                    existing_subtree = _simple_reallocation(
                        m_depth, k_depth + 1, router.left_child, existing_subtree=existing_subtree
                    )
                    return existing_subtree
                except SimpleReallocationFailure:
                    router.set_attr_self_and_below("part_of_subtree", False)
                    pass
            # check if right or left child router can serve as a level k router
            if num_right >= 2**(m_depth - k_depth - 1):
                router.part_of_subtree = True
                existing_subtree = _simple_reallocation(
                    m_depth, k_depth, router.right_child, existing_subtree=existing_subtree
                )
                return existing_subtree
            if num_left >= 2**(m_depth - k_depth - 1):
                router.part_of_subtree = True
                existing_subtree = _simple_reallocation(
                    m_depth, k_depth, router.left_child, existing_subtree=existing_subtree
                )
                return existing_subtree
            raise SimpleReallocationFailure
        try:
            subtree = _simple_reallocation(m, k, self)
            return subtree, 1
        except SimpleReallocationFailure:
            return [], 0

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

    def router_list_functioning_or_not(self, functioning=False, existing_router_list=None, depth=None):
        """function that can compute a list of faulty routers and also functioning routers
        among those at the bottom of the tree. If functioning is set to False,
        this creates a list of all non-functioning routers. If functioning is set to
        true, then it creates a list of functioning routers."""
        if depth is None:
            depth = self.tree_depth() - 1
        if existing_router_list is None:
            existing_router_list = []
        if depth < 0:
            raise ValueError("depth should be zero or positive")
        if depth == 0 or self.right_child is None:
            if self.functioning == functioning:
                existing_router_list.append(self.location)
                return existing_router_list
            else:
                return existing_router_list
        else:
            existing_router_list = self.right_child.router_list_functioning_or_not(
                functioning=functioning, existing_router_list=existing_router_list, depth=depth-1
            )
            existing_router_list = self.left_child.router_list_functioning_or_not(
                functioning=functioning, existing_router_list=existing_router_list, depth=depth-1
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

    @staticmethod
    def compute_flag_qubits(repair_dict):
        """assumption is that the keys of repair_dict are faulty routers and
                the values are repair routers they have been assigned to"""
        return [format(int(faulty_address, 2) ^ int(repair_address, 2), "b")
                for (faulty_address, repair_address) in repair_dict.items()]

    def total_flag_cost(self, repair_dict):
        unique_flag_qubits = set(self.compute_flag_qubits(repair_dict))
        num_bit_flips = [sum(map(int, flag)) for flag in unique_flag_qubits]
        return sum(num_bit_flips)

    def _repair_as_you_go(
        self,
        inter_depth,
        original_tree,
        augmented_tree,
        prev_repair_dict,
    ):
        faulty_router_list = original_tree.router_list_functioning_or_not(
            functioning=False, depth=inter_depth
        )
        available_router_list = augmented_tree.router_list_functioning_or_not(
            functioning=True, depth=inter_depth
        )
        prev_repaired_routers = prev_repair_dict.keys()

        def _pop_and_queue_for_repair_if_necessary(child_router):
            # child_router could be a child on the augmented side,
            # hence not in faulty_router_list
            if child_router in faulty_router_list:
                child_index = faulty_router_list.index(child_router)
                faulty_router_list.pop(child_index)
            prev_reassigned_router = prev_repair_dict[child_router[:-1]]
            # new reassigned router at the next depth is right or left child
            new_reassigned_router = prev_reassigned_router + child_router[-1]
            # check if this router is dead or alive and add to faulty list if not
            if new_reassigned_router not in available_router_list:
                # TODO lose info here about which faulty router originally was assigned here?
                # in principle trackable though from previous repair dicts...
                faulty_router_list.append(new_reassigned_router)
            else:
                avail_index = available_router_list.index(new_reassigned_router)
                available_router_list.pop(avail_index)
                free_repair_dict[child_router] = new_reassigned_router

        free_repair_dict = {}
        for prev_repaired_router in prev_repaired_routers:
            _pop_and_queue_for_repair_if_necessary(prev_repaired_router + "0")
            _pop_and_queue_for_repair_if_necessary(prev_repaired_router + "1")

        if len(faulty_router_list) == 0:
            inter_repair_dict = {}
        else:
            inter_repair_dict = self.router_repair_global(faulty_router_list, available_router_list)
        return inter_repair_dict, free_repair_dict

    def repair_as_you_go(self):
        original_tree, augmented_tree = self.assign_original_and_augmented()
        full_depth = self.tree_depth()
        repair_dict_2, _ = self._repair_as_you_go(2, original_tree, augmented_tree, {})
        repair_dict_list = [repair_dict_2, ]
        free_repair_dict_list = [{}, ]
        for inter_depth in range(3, full_depth - 1):
            prev_repair_dict = repair_dict_list[-1] | free_repair_dict_list[-1]
            new_repair_dict, free_repair_dict = self._repair_as_you_go(
                inter_depth, original_tree, augmented_tree, prev_repair_dict
            )
            repair_dict_list.append(new_repair_dict)
            free_repair_dict_list.append(free_repair_dict)
        return repair_dict_list, free_repair_dict_list

    @staticmethod
    @np.vectorize
    def bit_flip_pattern_int(faulty_address, repair_address):
        """returns the number of bit flips required to map faulty to repair"""
        return int(faulty_address, 2) ^ int(repair_address, 2)

    @staticmethod
    def construct_as_you_go_mapping(repair_dict_list, faulty_router_list):
        final_repair_dict = {}
        reordered_repair_dict_list = repair_dict_list[::-1]
        last_repair = reordered_repair_dict_list[0]
        # this will be 0 or 1 depending on which side of the tree is "original"
        which_side = faulty_router_list[0][2]

        for faulty_router in last_repair.keys():
            if faulty_router in faulty_router_list:
                final_repair_dict[faulty_router] = last_repair[faulty_router]
            else:
                prev_faulty_router = faulty_router
                address_append = ""
                for k_idx, k_router_dict in enumerate(reordered_repair_dict_list[1:]):
                    # we've gone up a level, so strip off the last address
                    k_faulty_router = prev_faulty_router[0: -1]
                    # this address is more significant than all previous
                    address_append = prev_faulty_router[-1] + address_append
                    # k_faulty_router in k_router_dict indicates a free repair, we just strip
                    # the least significant bit (above) and move on
                    if k_faulty_router not in k_router_dict:
                        prev_router_dict_flipped = {val: key for key, val in k_router_dict.items()}
                        prev_faulty_router = prev_router_dict_flipped[k_faulty_router]
                        if prev_faulty_router[2] == which_side:
                            final_repair_dict[prev_faulty_router + address_append] = last_repair[faulty_router]
                            break
        return final_repair_dict

    def router_repair(self, method="global"):
        """method can be 'global' or 'as_you_go' """
        t_depth = self.tree_depth()
        # each has depth t_depth-1
        original_tree, augmented_tree = self.assign_original_and_augmented()
        faulty_router_list = original_tree.router_list_functioning_or_not(functioning=False)
        available_router_list = augmented_tree.router_list_functioning_or_not(functioning=True)
        num_faulty = len(faulty_router_list)
        if num_faulty == 0:
            # no repair necessary
            return {}, 0
        # original faulty routers + augmented faulty routers
        total_faulty = num_faulty + 2**(t_depth-1) - len(available_router_list)
        if total_faulty > 2**(t_depth - 1):
            # QRAM unrepairable
            return {}, np.inf
        if method == "global":
            repair_dict_or_list = self.router_repair_global(faulty_router_list, available_router_list)
            self.verify_allocation(repair_dict_or_list)
            flag_qubits = set(self.compute_flag_qubits(repair_dict_or_list))
            return repair_dict_or_list, len(flag_qubits)
        elif method == "as_you_go":
            repair_dict_or_list, free_repair_dict_list = self.repair_as_you_go()
            repair_dicts = [repair_dict | free_repair_dict for repair_dict, free_repair_dict
                            in zip(repair_dict_or_list, free_repair_dict_list)]
            final_repair = self.construct_as_you_go_mapping(repair_dicts, faulty_router_list)
            self.verify_allocation(final_repair)
            flag_qubits = [len(set(self.compute_flag_qubits(repair_dict))) for repair_dict in repair_dict_or_list]
            return final_repair, sum(flag_qubits)
        else:
            raise RuntimeError("method not supported")

    def router_repair_global(self, faulty_router_list, available_router_list):
        faulty_router_list = np.array(faulty_router_list)
        available_router_list = np.array(available_router_list)
        repair_dict = {}
        while True:
            frl, arl = np.meshgrid(faulty_router_list, available_router_list, indexing="ij")
            bit_flip_pattern_mat = self.bit_flip_pattern_int(frl, arl)
            # this is likely the slowest step
            # can change since we only take the largest?
            bit_flip_patterns, bit_flip_pattern_occurences = np.unique(bit_flip_pattern_mat, return_counts=True)
            sorted_idxs = np.argsort(bit_flip_pattern_occurences)
            bit_flip_pattern = bit_flip_patterns[sorted_idxs[-1]]
            assignment_idxs = np.argwhere(bit_flip_pattern_mat == bit_flip_pattern)
            for assignment_idx in assignment_idxs:
                faulty_idx, avail_idx = assignment_idx
                faulty_router = faulty_router_list[faulty_idx]
                avail_router = available_router_list[avail_idx]
                repair_dict[faulty_router] = avail_router
            # don't need to worry anymore about reassigning these
            faulty_router_list = np.delete(faulty_router_list, assignment_idxs[:, 0])
            available_router_list = np.delete(available_router_list, assignment_idxs[:, 1])
            if len(faulty_router_list) == 0:
                break
        return repair_dict

    @staticmethod
    def _check_router_in_list(compare_router, router_list):
        compare_router_height = len(compare_router)
        new_router_list = [router[:compare_router_height] for router in router_list]
        assert compare_router in new_router_list

    def verify_allocation(self, repair_dict):
        original_tree, augmented_tree = self.assign_original_and_augmented()
        faulty_router_list = original_tree.router_list_functioning_or_not(functioning=False)
        available_router_list = augmented_tree.router_list_functioning_or_not(functioning=True)
        # all routers assigned (taking into account that we assign "further up")
        for repaired_router in repair_dict.keys():
            self._check_router_in_list(repaired_router, faulty_router_list)
        # no repeated assignments
        assert len(repair_dict.values()) == len(set(repair_dict.values()))
        # only assigned to live routers
        for available_router in repair_dict.values():
            self._check_router_in_list(available_router, available_router_list)


class MonteCarloRouterInstances:
    def __init__(self, n, eps, num_instances, rng_seed,
                 top_three_functioning=False, filepath="tmp.h5py", instantiate_trees=True):
        self.n = n
        self.eps = eps
        self.num_instances = num_instances
        self.rng_seed = rng_seed
        self.top_three_functioning = top_three_functioning
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
        repaired_routers_global = self.map_over_trees("router_repair", "global")
        _, num_flags_global, together_global = list(zip(*repaired_routers_global))
        num_flags_global = np.array(num_flags_global, dtype=float)
        together_global = np.array(together_global, dtype=bool)
        num_faulty = np.array(self.map_over_trees("count_number_faulty_addresses"))
        data_dict = {
            "num_flags_global": num_flags_global,
            "together_global": together_global,
            "num_faulty": num_faulty,
        }
        for n in range(3, self.n):
            simple_repair_n = self.map_over_trees("simple_reallocation", n, 0)
            (_, n_success) = list(zip(*simple_repair_n))
            print(n, len(np.argwhere(n_success)))
            data_dict[f"n{n}_success"] = n_success
        print(f"writing results to {self.filepath}")
        write_to_h5(self.filepath, data_dict, self.param_dict())
        return 0

    def create_trees(self):
        return list(map(lambda _: QRAMRouter().create_tree(self.n), range(self.num_instances)))

    def _fab_instance_for_one_tree(self, tree_rn_list):
        tree, rn_list = tree_rn_list
        _ = tree.fabrication_instance(
            self.eps, rn_list, top_three_functioning=self.top_three_functioning
        )
        return tree

    def fab_instance_for_all_trees(self):
        if self.trees is None:
            self.trees = self.create_trees()
        self.trees = list(map(self._fab_instance_for_one_tree, zip(self.trees, self.rn_list)))

    def map_over_trees(self, func, *args):
        if self.trees is None:
            self.trees = self.create_trees()
        if type(func) is str:
            return list(map(lambda tree: getattr(tree, func)(*args), self.trees))
        elif type(func) is Callable:  # assume its a function that takes a single tree as argument
            return list(map(lambda tree: func(tree, *args), self.trees))
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
    TREE_DEPTH = 8
    EPS = 0.1  # 0.375
    # 6722222232 gives 5 and 3 configuration (not simply repairable)
    # 6243254322 gives 6 and 2
    # 22543268254 fails at the second level: starts off with 6, 4, but then drops to 3, 1 on the right.
    # tree_Depth = 10, eps = 0.05 rng 2543567243234588543 2770 below
    RNG_SEED = 2545672423485  # 254567242348543
    rng = np.random.default_rng(RNG_SEED)  # 27585353
    RN_LIST = rng.random((NUM_INSTANCES, 2 ** TREE_DEPTH))

    MYTREE = QRAMRouter()
    FULLTREE = MYTREE.create_tree(TREE_DEPTH)
    FULLTREE.fabrication_instance(EPS, rn_list=RN_LIST[27], top_three_functioning=True)  # half full n=6 #255
    FULLTREE.print_tree()
    assignment_global, flag_qubits_global = FULLTREE.router_repair(method="global")
    _final_repair, _flag_qubits = FULLTREE.router_repair(method="as_you_go")
    total_global = FULLTREE.total_flag_cost(assignment_global)
    print(0)
