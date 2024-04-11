import copy
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
            self.set_attr_self_and_below("functioning", False)
            return new_rn_list
        else:
            self.functioning = True
            if self.right_child:
                new_rn_list = self.right_child.fabrication_instance(failure_rate, rn_list=new_rn_list)
                new_rn_list = self.left_child.fabrication_instance(failure_rate, rn_list=new_rn_list)
            return new_rn_list

    def simple_reallocation(self, m, k):
        """m is the depth of the QRAM we are after, and k is the level of the QRAM this
        router should be acting as
        returns the routers at the bottom of the tree that are part of the m-bit
        QRAM as well as a flag: 1 if it succeeded, 0 if it failed to find an m-bit QRAM
        """
        num_avail = self.lowest_router_list_functioning_or_not(functioning=True)
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
            right_avail_router_list = router.right_child.lowest_router_list_functioning_or_not(functioning=True)
            left_avail_router_list = router.left_child.lowest_router_list_functioning_or_not(functioning=True)
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

    def lowest_router_list_functioning_or_not(self, functioning=False, existing_router_list=None):
        # TODO doesn't have to be at the bottom! Fix me!!
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

    def collect_routers_functioning_together(self, routers_available_for_assignment=None):
        if routers_available_for_assignment is None:
            routers_available_for_assignment = []
        if self.right_child is None or not self.functioning:
            return routers_available_for_assignment
        # I'm not on the bottom and I'm functioning. Possible subtree available here
        else:
            t_depth = self.tree_depth()
            functioning_below = self.lowest_router_list_functioning_or_not(functioning=True)
            if 2**(t_depth-1) == len(functioning_below):
                routers_available_for_assignment += [functioning_below, ]
                return routers_available_for_assignment
            else:
                routers_available_for_assignment = self.right_child.collect_routers_functioning_together(
                    routers_available_for_assignment=routers_available_for_assignment
                )
                routers_available_for_assignment = self.left_child.collect_routers_functioning_together(
                    routers_available_for_assignment=routers_available_for_assignment
                )
                return routers_available_for_assignment

    def collect_routers_to_assign_together(self, routers_to_assign_together=None):
        if routers_to_assign_together is None:
            routers_to_assign_together = []
        # don't need to worry about this for routers at the bottom: addresses always assigned together
        if self.right_child is None:
            return routers_to_assign_together
        # otherwise we aren't at the bottom.  If we have a faulty router
        # then everybody below wants to be reassigned together
        if not self.functioning:
            routers_to_assign_together += [self.lowest_router_list_functioning_or_not(functioning=False), ]
            return routers_to_assign_together
        # If you're functioning, then continue with below routers
        else:
            routers_to_assign_together = self.right_child.collect_routers_to_assign_together(
                routers_to_assign_together=routers_to_assign_together
            )
            routers_to_assign_together = self.left_child.collect_routers_to_assign_together(
                routers_to_assign_together=routers_to_assign_together
            )
            return routers_to_assign_together

    def router_repair_together(self, original_tree, augmented_tree):
        faulty_routers_repair_together = original_tree.collect_routers_to_assign_together()
        routers_available_together = augmented_tree.collect_routers_functioning_together()
        all_faulty_router_list = original_tree.lowest_router_list_functioning_or_not(functioning=False)
        all_available_router_list = augmented_tree.lowest_router_list_functioning_or_not(functioning=True)
        faulty_sizes = np.zeros(len(faulty_routers_repair_together))
        available_sizes = np.zeros(len(routers_available_together))
        for repair_idx, faulty_list in enumerate(faulty_routers_repair_together):
            faulty_sizes[repair_idx] = len(faulty_list)
        for avail_idx, avalable_list in enumerate(routers_available_together):
            available_sizes[avail_idx] = len(avalable_list)
        # want to go through these from largest to smallest
        faulty_sorted_idxs = np.argsort(faulty_sizes)[::-1]
        router_repair_dict = {}
        for faulty_idx in faulty_sorted_idxs:
            faulty_size = faulty_sizes[faulty_idx]
            if all(faulty_size > available_sizes):
                return [], np.inf
            available_sorted_idxs = np.argsort(available_sizes)
            available_sorted_sizes = available_sizes[available_sorted_idxs]
            idx_to_insert_left = np.searchsorted(available_sorted_sizes, faulty_size, side="left")
            # pair faulty routers to these available ones, signified by shrinking
            # the size of the available router collection
            available_sorted_sizes[idx_to_insert_left] -= faulty_size
            available_sizes = available_sorted_sizes[available_sorted_idxs]
            # need to make sure that this is accessing the appropriate router list
            faulty_list = faulty_routers_repair_together[faulty_idx]
            paired_list = routers_available_together[available_sorted_idxs[idx_to_insert_left]]
            for sub_faulty_idx, faulty_router in enumerate(faulty_list):
                router_repair_dict[faulty_router] = paired_list[sub_faulty_idx]
                all_available_router_list.remove(paired_list[sub_faulty_idx])
                all_faulty_router_list.remove(faulty_router)
        # now need to assign the remaining routers
        single_repair_dict, num_bit_flips = self.router_repair_auction(all_faulty_router_list, all_available_router_list)
        return router_repair_dict | single_repair_dict, num_bit_flips

    def bit_flip(self, ell, address):
        # flip first ell bits of router. Need to do some bin gymnastics
        # -1 is because we are looking at the routers as opposed to the addresses
        t_depth = self.tree_depth()
        bit_flip_mask = bin((1 << ell) - 1) + (t_depth-ell - 1) * "0"
        flipped_address = int(bit_flip_mask, 2) ^ int(address, 2)
        flipped_address_bin = format(flipped_address, f"#0{t_depth + 2 - 1}b")
        return flipped_address_bin

    @staticmethod
    @np.vectorize
    def num_bit_flips(faulty_address, repair_address):
        """returns the number of bit flips required to map faulty to repair"""
        diff_address = int(faulty_address, 2) ^ int(repair_address, 2)
        return sum(map(int, format(diff_address, "b")))

    def router_repair(self, method="auction"):
        """method can be 'auction' or 'bit_flip'"""
        t_depth = self.tree_depth()
        # each has depth t_depth-1
        original_tree, augmented_tree = self.assign_original_and_augmented()
        faulty_router_list = original_tree.lowest_router_list_functioning_or_not(functioning=False)
        available_router_list = augmented_tree.lowest_router_list_functioning_or_not(functioning=True)
        num_faulty = len(faulty_router_list)
        if num_faulty == 0:
            # no repair necessary
            return [], 0
        # original faulty routers + augmented faulty routers
        total_faulty = num_faulty + 2**(t_depth-1) - len(available_router_list)
        if total_faulty > 2**(t_depth - 1):
            # QRAM unrepairable
            return [], np.inf
        if method == "auction":
            return self.router_repair_auction(faulty_router_list, available_router_list)
        elif method == "bit_flip":
            return self.router_repair_bit_flip(faulty_router_list, available_router_list)
        elif method == "together":
            return self.router_repair_together(original_tree, augmented_tree)
        else:
            raise RuntimeError("method not supported")

    def router_repair_auction(self, faulty_router_list, available_router_list, eps=None):
        num_faulty = len(faulty_router_list)
        # see Bertsekas https://link.springer.com/article/10.1007/BF00247653 this eps
        # is introduced to avoid the situation where there are only a few available routers
        # that everyone is competing for, but each available router is just as good as the next.
        # In this case no one raises their prices and we end up in a never-ending loop
        if eps is None:
            eps = 1. / num_faulty
        frl, arl = np.meshgrid(faulty_router_list, available_router_list, indexing="ij")
        # cost is defined as the number of required bit flips
        num_bit_flips_mat = self.num_bit_flips(frl, arl)
        # the algorithm is usually described in terms of maximizing benefit,
        # so we take the negative of the cost
        benefit_matrix = - num_bit_flips_mat
        # initialize with zero prices, which Bertsekas notes is the best way to
        # attack the asymmetric problem (where the number of faulty routers and
        # available routers is not the same)
        prices = np.zeros(len(available_router_list))
        max_num_bit_flips = 0
        unassigned = [router for router in faulty_router_list]
        assigned = {}

        while True:
            # bid
            benefit_less_price = benefit_matrix - prices[..., :]
            best_avail_router_idxs = np.argmax(benefit_less_price, axis=1)
            best_benefit_values = benefit_less_price[np.arange(num_faulty), best_avail_router_idxs]
            # extract second best item to calculate bid
            benefit_less_price[np.arange(num_faulty), best_avail_router_idxs] = -np.inf
            second_best_values = np.max(benefit_less_price, axis=1)
            bid_increments = best_benefit_values - second_best_values
            # assign
            # below only want to loop over those addresses who are unassigned. We
            # use inf as a placeholder and filter it out later
            mask = np.isin(faulty_router_list, unassigned).astype(int)
            best_avail_for_unassigned = [
                best_avail if unassigned else np.inf
                for unassigned, best_avail in zip(mask, best_avail_router_idxs)
            ]
            unique_avail_best_idxs = set(best_avail_for_unassigned)
            if np.inf in unique_avail_best_idxs:
                unique_avail_best_idxs.remove(np.inf)
            for best_idx in unique_avail_best_idxs:
                # indices of all routers that want to be assigned to best_idx
                faulty_routers_wanting = np.squeeze(
                    np.argwhere(best_avail_for_unassigned == best_idx), axis=1
                )
                # take the highest bidder from among that group
                _highest_bidder_idx = np.argmax(bid_increments[faulty_routers_wanting])
                # _highest_bidder_idx is the index of bid_increments[faulty_routers_wanting]
                # with the highest bid. So we need to take that index of faulty_routers_wanting
                # to extract the actual index (in terms of entries to faulty_router_list)
                highest_bidder_idx = faulty_routers_wanting[_highest_bidder_idx]
                winning_faulty_router = faulty_router_list[highest_bidder_idx]
                winning_bid = bid_increments[highest_bidder_idx]
                prices[best_idx] += winning_bid + eps
                if available_router_list[best_idx] in assigned:
                    unassigned.append(assigned[available_router_list[best_idx]])
                assigned[available_router_list[best_idx]] = winning_faulty_router
                unassigned.remove(winning_faulty_router)
                num_bit_flips = num_bit_flips_mat[highest_bidder_idx, best_idx]
                if num_bit_flips > max_num_bit_flips:
                    max_num_bit_flips = num_bit_flips
            if len(unassigned) == 0:
                break
        repair_dict = {val: key for key, val in assigned.items()}
        return repair_dict, max_num_bit_flips

    def router_repair_bit_flip(self, faulty_router_list, available_router_list):
        t_depth = self.tree_depth()
        num_faulty = len(faulty_router_list)
        available_router_list_copy = copy.deepcopy(available_router_list)
        # will contain mapping of faulty routers to repair routers
        repair_dict = {}
        fixed_faulty_list = []
        # only go up to ell = t_depth-2, since no savings if we do l=t_depth-1
        for ell in range(1, t_depth-1):
            for fr_idx, faulty_router in enumerate(faulty_router_list):
                if faulty_router not in fixed_faulty_list:
                    # perform bitwise NOT on the first ell bits of faulty_router
                    flipped_address = self.bit_flip(ell, faulty_router)
                    if flipped_address in available_router_list_copy:
                        repair_dict[faulty_router] = flipped_address
                        fixed_faulty_list.append(faulty_router)
                        available_router_list_copy.remove(flipped_address)
            if len(fixed_faulty_list) == num_faulty:
                # succeeded in repair with ell bit flips
                return repair_dict, ell
        # failed repair, proceed with greedy assignment requiring t_depth-1 bit flips
        repair_dict = dict(zip(faulty_router_list, available_router_list))
        return repair_dict, t_depth - 1


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
        simple_repair_n1 = self.map_over_trees("simple_reallocation", self.n - 1, 0)
        simple_repair_n2 = self.map_over_trees("simple_reallocation", self.n - 2, 0)
        (_, n1_success) = list(zip(*simple_repair_n1))
        (_, n2_success) = list(zip(*simple_repair_n2))
        data_dict = {
            "num_bit_flips": num_bit_flips,
            "num_faulty": num_faulty,
            "n1_success": n1_success,
            "n2_success": n2_success,
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
    TREE_DEPTH = 6
    EPS = 0.2
    # 6722222232 gives 5 and 3 configuration (not simply repairable)
    # 6243254322 gives 6 and 2
    # 22543268254 fails at the second level: starts off with 6, 4, but then drops to 3, 1 on the right.
    RNG_SEED = 2543567243234588543
    rng = np.random.default_rng(RNG_SEED)  # 27585353
    RN_LIST = rng.random((NUM_INSTANCES, 2 ** TREE_DEPTH))
    MYTREE = QRAMRouter()
    FULLTREE = MYTREE.create_tree(TREE_DEPTH)
    FULLTREE.fabrication_instance(EPS, rn_list=RN_LIST[934])
    FULLTREE.print_tree()
    print("-------------------------------")
    avail_router = FULLTREE.lowest_router_list_functioning_or_not(functioning=True)
    desired_depth_subQ = TREE_DEPTH - 2
    if len(avail_router) >= 2**(desired_depth_subQ - 1):
        SUBTREE = FULLTREE.simple_reallocation(desired_depth_subQ, 0)
        FULLTREE.print_tree()
        print(SUBTREE)

    mcinstance = MonteCarloRouterInstances(
        TREE_DEPTH,
        EPS,
        NUM_INSTANCES,
        RNG_SEED,
    )
    mcinstance.run()
