import copy
import functools
import itertools
import time
from functools import reduce

import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Style

from quantum_utils import write_to_h5, param_map, unpack_param_map, parallel_map

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
        """find the depth of a tree by traversing down the right children"""
        if self.right_child is None:
            return 1
        else:
            return self.right_child.tree_depth() + 1

    def print_tree(self, indent="", am_I_a_right_or_left_child="right", print_attr="location"):
        """print a QRAM tree. This functionality is incredibly useful for visualizing the structure of a
        given faulty tree instance. It plots in white all of the functioning routers, in red all of the
        faulty routers and in blue all of the routers that were chosen as part of the simply-repaired subtree.
        You can also choose to print any attribute of a QRAM tree, the default is the location"""
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
        """set attribute attr of a given router to val. Do the same for all of its children"""
        setattr(self, attr, val)
        if self.right_child:
            self.right_child.set_attr_self_and_below(attr, val)
            self.left_child.set_attr_self_and_below(attr, val)

    def fabrication_instance(
        self,
        failure_rate=0.3,
        rng=None,
        k_level=0,
        top_three_functioning=False
    ):
        """this function utilizes a list of random numbers (if passed) to decide
        if a given router is faulty. It strips off the first one and returns
        the remainder of the list for use for deciding if other routers are faulty"""
        if rng is None:
            rng = np.random.default_rng(42)
        rnd = rng.random()
        if rnd > failure_rate or (top_three_functioning and k_level <= 1):
            self.functioning = True
            if self.right_child:
                rng = self.right_child.fabrication_instance(
                    failure_rate, rng=rng, k_level=k_level+1, top_three_functioning=top_three_functioning
                )
                rng = self.left_child.fabrication_instance(
                    failure_rate, rng=rng, k_level=k_level+1, top_three_functioning=top_three_functioning
                )
                return rng
            else:
                return rng
        else:
            self.set_attr_self_and_below("functioning", False)
            return rng

    def simple_reallocation(self, m, k=1):
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
            if m_depth == k_depth:
                existing_subtree.append(router.location)
                return existing_subtree
            right_avail_router_list = router.right_child.router_list_functioning_or_not(functioning=True)
            left_avail_router_list = router.left_child.router_list_functioning_or_not(functioning=True)
            num_right = len(right_avail_router_list)
            num_left = len(left_avail_router_list)
            # can succesfully split this router as enough addresses available on each side
            if num_right >= 2**(m_depth-k_depth-1) and num_left >= 2**(m_depth-k_depth-1):
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
            if num_right >= 2**(m_depth - k_depth):
                router.part_of_subtree = True
                existing_subtree = _simple_reallocation(
                    m_depth, k_depth, router.right_child, existing_subtree=existing_subtree
                )
                return existing_subtree
            if num_left >= 2**(m_depth - k_depth):
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

    def count_available_addresses_all_levels(self):
        # we're at the bottom of the tree, and we only reach the bottom if
        # the router is functioning
        if self.right_child is None:
            return 2
        # if we've reached here, we're at an interior node
        if self.right_child.functioning:
            count = self.right_child.count_available_addresses_all_levels()
        else:
            count = 1
        if self.left_child.functioning:
            count += self.left_child.count_available_addresses_all_levels()
        else:
            count += 1
        return count

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
            depth = self.tree_depth()
        if existing_router_list is None:
            existing_router_list = []
        if depth < 1:
            raise ValueError("depth should be greater than or equal to 1")
        if depth == 1 or self.right_child is None:
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
        prev_overall_mapping_dict,
    ):
        new_overall_mapping_dict = {}
        routers_needing_reassignment = []
        faulty_router_list = original_tree.router_list_functioning_or_not(
            functioning=False, depth=inter_depth
        )
        original_faulty_router_list = copy.deepcopy(faulty_router_list)
        available_router_list = augmented_tree.router_list_functioning_or_not(
            functioning=True, depth=inter_depth
        )

        for (prev_repaired_router, prev_available_router) in prev_overall_mapping_dict.items():
            for idx in ["0", "1"]:
                # pop prev repaired faulty routers from the list, since its their
                # mapped friends who might need repair plus new guys at the lowest level
                child_idx = faulty_router_list.index(prev_repaired_router + idx)
                faulty_router_list.pop(child_idx)
                if prev_available_router + idx not in available_router_list:
                    # router is newly unavailable at this level of the tree.
                    # needs reassignment and tracking of where it got mapped from
                    routers_needing_reassignment.append(prev_repaired_router + idx)
                    faulty_router_list.append(prev_available_router + idx)
                else:
                    # router is still available, get reassignment for free
                    avail_index = available_router_list.index(prev_available_router + idx)
                    available_router_list.pop(avail_index)
                    new_overall_mapping_dict[prev_repaired_router + idx] = prev_available_router + idx

        if len(faulty_router_list) == 0:
            inter_repair_dict = {}
            bfp = [0, ]
        else:
            inter_repair_dict, bfp = self.router_repair_global(faulty_router_list, available_router_list)

        # had to wait until we ran the global algorithm to see where these faulty
        # augmented routers got assigned
        for purgatory_router in routers_needing_reassignment:
            prev_available_router = prev_overall_mapping_dict[purgatory_router[0: -1]]
            repaired_router = prev_available_router + purgatory_router[-1]
            new_overall_mapping_dict[purgatory_router] = inter_repair_dict[repaired_router]

        # for newly faulty routers on the original side of the tree, add them to the mapping
        for newly_faulty_router in inter_repair_dict.keys():
            if newly_faulty_router in original_faulty_router_list:
                new_overall_mapping_dict[newly_faulty_router] = inter_repair_dict[newly_faulty_router]
        return inter_repair_dict, new_overall_mapping_dict, bfp

    def repair_as_you_go(self, start="two"):
        """start can either be 'two' or 'simple_success'. Question is do we start with an n=2 bit QRAM
         or do we start with the largest possible simple-repair QRAM"""
        largest_simple_repair = 2
        if start == "simple_success":
            # go up to n-2 only, don't want simple repair to steal the show
            for tree_depth in range(2, self.tree_depth() - 1):
                _, simple_success = self.simple_reallocation(tree_depth)
                if simple_success:
                    largest_simple_repair = tree_depth
                else:
                    break
        original_tree, augmented_tree = self.assign_original_and_augmented()
        full_depth = self.tree_depth()
        repair_dict_init, _, bfp_init = self._repair_as_you_go(largest_simple_repair + 1, original_tree, augmented_tree, {})
        repair_dict_list = [repair_dict_init, ]
        bfp_list = [bfp_init, ]
        overall_mapping_dict = copy.deepcopy(repair_dict_init)
        for inter_depth in range(largest_simple_repair + 2, full_depth):
            new_repair_dict, overall_mapping_dict, bfp = self._repair_as_you_go(
                inter_depth, original_tree, augmented_tree, overall_mapping_dict
            )
            repair_dict_list.append(new_repair_dict)
            bfp_list.append(bfp)
        return repair_dict_list, overall_mapping_dict, bfp_list

    @staticmethod
    @np.vectorize
    def bit_flip_pattern_int(faulty_address, repair_address):
        """returns the number of bit flips required to map faulty to repair"""
        if isinstance(faulty_address, (int, np.int32, np.int64)):
            return faulty_address ^ repair_address
        elif isinstance(faulty_address, (str, np.str_)):
            return int(faulty_address, 2) ^ int(repair_address, 2)
        else:
            raise ValueError

    def router_repair(self, method, **kwargs):
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
            repair_dict_or_list, bfp = self.router_repair_global(
                faulty_router_list, available_router_list
            )
            self.verify_allocation(repair_dict_or_list)
            # -1 to take care of 0 bfp, which doesn't cost a flag qubit
            return repair_dict_or_list, len(bfp) - 1
        elif method == "_global":
            repair_dict_or_list, bfp = self._router_repair_global(faulty_router_list, available_router_list)
            self.verify_allocation(repair_dict_or_list)
            flag_qubits = set(self.compute_flag_qubits(repair_dict_or_list))
            return repair_dict_or_list, len(flag_qubits)
        elif method == "as_you_go":
            start = kwargs.get("start", "two")
            repair_dict_or_list, overall_mapping_dict, bfp_list = self.repair_as_you_go(start=start)
            self.verify_allocation(overall_mapping_dict)
            return overall_mapping_dict, len(bfp_list[-1]) - 1
        elif method == "enumerate":
            repair_dict_or_list = dict(zip(faulty_router_list, available_router_list))
            # + 1 is to take care of the no error case which has to be included
            return repair_dict_or_list, int(np.ceil(np.log2(len(faulty_router_list) + 1)))
        else:
            raise RuntimeError("method not supported")

    def _router_repair_global(self, faulty_router_list, available_router_list):
        faulty_router_list = np.array(faulty_router_list)
        available_router_list = np.array(available_router_list)
        repair_dict = {}
        picked_bit_flip_patterns = []
        while True:
            frl, arl = np.meshgrid(faulty_router_list, available_router_list, indexing="ij")
            bit_flip_pattern_mat = self.bit_flip_pattern_int(frl, arl)
            # this is likely the slowest step
            bit_flip_patterns, bit_flip_pattern_occurences = np.unique(bit_flip_pattern_mat, return_counts=True)
            sorted_by_occurences_idxs = np.argsort(bit_flip_pattern_occurences)
            bit_flip_pattern = bit_flip_patterns[sorted_by_occurences_idxs[-1]]
            picked_bit_flip_patterns.append(bit_flip_pattern)
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
        return repair_dict, picked_bit_flip_patterns

    def router_repair_global(self, faulty_router_list, available_router_list):
        faulty_router_list = np.array(faulty_router_list)
        available_router_list = np.array(available_router_list)
        repair_dict = {}
        picked_bit_flip_patterns = [0, ]
        full_basis = [2**idx for idx in range(len(faulty_router_list[0]) - 2)]
        full_power_set = self.construct_power_set(full_basis)
        while True:
            picked_power_set = self.construct_power_set(picked_bit_flip_patterns)
            frl, arl = np.meshgrid(
                faulty_router_list, available_router_list, indexing="ij"
            )
            bit_flip_pattern_mat = self.bit_flip_pattern_int(frl, arl)
            bit_flip_patterns, bit_flip_pattern_occurences = np.unique(
                bit_flip_pattern_mat, return_counts=True
            )
            possible_bfps = [bfp for bfp in full_power_set if bfp not in picked_power_set]

            def num_fixed_for_bfp(possible_bfp):
                possible_new_power_set = self.construct_power_set(picked_bit_flip_patterns + [possible_bfp, ])
                # first index is the new power set index, second is the unique bit-flip patterns we want
                matches = bit_flip_patterns[None, :] == possible_new_power_set[:, None]
                matches = functools.reduce(
                    lambda x, y: np.logical_or(x, y), matches, len(bit_flip_patterns) * [False, ]
                )
                return sum(np.where(
                    matches, bit_flip_pattern_occurences, 0
                ))
            num_fixed = list(map(num_fixed_for_bfp, possible_bfps))
            idx_max_num_fixed = np.argmax(num_fixed)
            max_bfp = possible_bfps[idx_max_num_fixed]
            max_power_set = self.construct_power_set(picked_bit_flip_patterns + [max_bfp, ])
            match_matrix = bit_flip_pattern_mat[None, :, :] == max_power_set[:, None, None]
            match_matrix = functools.reduce(
                lambda x, y: np.logical_or(x, y), match_matrix, np.full_like(bit_flip_pattern_mat, False)
            )
            assignment_idxs = np.argwhere(match_matrix)
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
        return repair_dict, picked_bit_flip_patterns

    @staticmethod
    def _check_router_in_list(compare_router, router_list):
        compare_router_height = len(compare_router)
        new_router_list = [router[:compare_router_height] for router in router_list]
        assert compare_router in new_router_list

    def construct_power_set(self, bit_flip_patterns):
        num_bit_flip_patterns = len(bit_flip_patterns)
        power_set = []
        for num_patterns in range(num_bit_flip_patterns + 1):
            bit_flip_ind_combs = itertools.combinations(bit_flip_patterns, num_patterns)
            for bit_flip_ind_comb in bit_flip_ind_combs:
                bit_flip_pattern = reduce(
                    self.bit_flip_pattern_int, bit_flip_ind_comb, 0
                )
                power_set.append(bit_flip_pattern)
        return np.unique(power_set)

    def verify_allocation(self, repair_dict):
        original_tree, augmented_tree = self.assign_original_and_augmented()
        faulty_router_list = original_tree.router_list_functioning_or_not(
            functioning=False
        )
        available_router_list = augmented_tree.router_list_functioning_or_not(
            functioning=True
        )
        # no repeated assignments
        assert len(repair_dict.values()) == len(set(repair_dict.values()))
        # all faulty routers assigned
        assert len(repair_dict.values()) == len(faulty_router_list)
        for repaired_router, available_router in repair_dict.items():
            # all routers assigned (taking into account that we assign "further up")
            self._check_router_in_list(repaired_router, faulty_router_list)
            # only assigned to live routers
            self._check_router_in_list(available_router, available_router_list)


class MonteCarloRouterInstances:
    def __init__(self, n, eps, num_instances, rng_seed,
                 top_three_functioning=False, filepath="tmp.h5py",
                 ):
        self.n = n
        self.eps = eps
        self.num_instances = num_instances
        self.rng_seed = rng_seed
        self.top_three_functioning = top_three_functioning
        self.filepath = filepath
        self._init_attrs = set(self.__dict__.keys())

    def param_dict(self):
        return {k: v for k, v in self.__dict__.items() if k in self._init_attrs}

    def run_for_one_tree(self, idx_and_rng):
        idx, rng = idx_and_rng[0]
        tree = QRAMRouter().create_tree(self.n)
        _ = tree.fabrication_instance(
            self.eps, rng, top_three_functioning=self.top_three_functioning
        )
        start_time = time.time()
        _, num_flags_global = tree.router_repair(method="global")
        global_time = time.time() - start_time
        _, num_flags_as_you_go = tree.router_repair(method="as_you_go", start="simple_success")
        _, num_flags_enumerate = tree.router_repair(method="enumerate")
        num_faulty = tree.count_number_faulty_addresses()
        num_avail_all = tree.count_available_addresses_all_levels()
        simple_success = []
        for n in range(3, self.n + 1):
            _, n_success = tree.simple_reallocation(n)
            simple_success.append(int(n_success))
        return np.concatenate(
            ([num_flags_global,
              num_flags_as_you_go,
              num_flags_enumerate,
              num_faulty,
              num_avail_all,
              global_time],
             simple_success)
        )

    def run(self, num_cpus=1):
        print(f"running faulty QRAM simulation with {self.__dict__}")
        parent_rng = np.random.default_rng(self.rng_seed)
        streams = parent_rng.spawn(self.num_instances)
        idxs_and_streams = list(zip(np.arange(self.num_instances), streams))
        map_fun = functools.partial(parallel_map, num_cpus)
        result = unpack_param_map(param_map(
            self.run_for_one_tree, [idxs_and_streams, ], map_fun=map_fun
        )).astype(float)
        num_flags_global = result[..., 0]
        num_flags_as_you_go = result[..., 1]
        num_flags_enumerate = result[..., 2]
        num_faulty = result[..., 3]
        num_avail_all = result[..., 4]
        global_time = result[..., 5]
        n_n_success = result[..., 6:]
        data_dict = {
            "num_flags_global": num_flags_global,
            "num_flags_as_you_go": num_flags_as_you_go,
            "num_flags_enumerate": num_flags_enumerate,
            "num_faulty": num_faulty,
            "num_avail_all": num_avail_all,
            "global_time": global_time,
        }
        for idx, n in enumerate(range(3, self.n + 1)):
            data_dict[f"n{n}_success"] = n_n_success[..., idx]
        print(f"writing results to {self.filepath}")
        write_to_h5(self.filepath, data_dict, self.param_dict())
        return data_dict


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
