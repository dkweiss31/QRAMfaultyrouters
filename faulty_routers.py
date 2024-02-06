import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Style

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


class RouterRepair:
    def __init__(self, original_tree: QRAMRouter, augmented_tree: QRAMRouter):
        self.original_tree = original_tree
        self.augmented_tree = augmented_tree
        self.faulty_router_list = self.original_tree.router_list_functioning_or_not(functioning=False)
        self.available_router_list = self.augmented_tree.router_list_functioning_or_not(functioning=True)
        if len(self.available_router_list) < len(self.faulty_router_list):
            print("this QRAM must be discarded")
        # since the whole tree has depth tree_depth if each of the child trees
        # have depth tree_depth - 1
        self.n = original_tree.tree_depth() + 1

    def bit_flip(self, ell, address):
        # flip first ell bits of router. Need to do some bin gymnastics
        # -1 is because we are looking at the routers as opposed to the addresses
        bit_flip_mask = bin((1 << ell) - 1) + (self.n-ell - 1) * "0"
        flipped_address = int(bit_flip_mask, 2) ^ int(address, 2)
        flipped_address_bin = format(flipped_address, f"#0{self.n + 2 - 1}b")
        return flipped_address_bin

    def router_repair(self):
        num_faulty = len(self.faulty_router_list)
        fa_idx_list = list(range(num_faulty))
        # will contain mapping of faulty routers to repair routers
        repair_list = np.empty(num_faulty, dtype=object)
        fixed_faulty_list = []
        # only go up to ell = tree_depth-2, since no savings if we do l=tree_depth-1
        for ell in range(1, self.n-1):
            for fr, fr_idx in zip(self.faulty_router_list, fa_idx_list):
                if fr not in fixed_faulty_list:
                    # perform bitwise not on the first ell bits of fr
                    flipped_address = self.bit_flip(ell, fr)
                    if flipped_address in self.available_router_list:
                        repair_list[fr_idx] = flipped_address
                        fixed_faulty_list.append(fr)
            if len(fixed_faulty_list) == num_faulty:
                print(f"succeeded in repair with ell={ell}")
                return repair_list, ell
        print("failed repair, proceed with greedy assignment")
        return self.available_router_list[0: num_faulty], self.n - 1


if __name__ == "__main__":
    NUM_INSTANCES = 10000
    n = 5
    EPS = 0.1
    rng = np.random.default_rng(22563452243)  # 27585353
    RN_LIST = rng.random((NUM_INSTANCES, 2 ** n))

    # def faulty_tree(eps, tree_depth, rns):
    #     mytree = QRAMRouter()
    #     fulltree = mytree.create_tree(tree_depth)
    #     fulltree.fabrication_instance(eps, RN_LIST=rns)
    #     return fulltree
    #
    # alltrees = map(lambda rs: faulty_tree(eps, tree_depth, rs), RN_LIST)
    # alltrees_num_faulty = list(map(lambda tree: tree.count_number_faulty_addresses(), alltrees))
    # avgfaulty = np.average(alltrees_num_faulty)
    # analytic_faulty = 2**tree_depth - 2**tree_depth * (1-eps)**tree_depth
    # print(avgfaulty, analytic_faulty)

    MYTREE = QRAMRouter()
    FULLTREE = MYTREE.create_tree(n)
    FULLTREE.fabrication_instance(EPS, rn_list=RN_LIST[0])
    FULLTREE.print_tree()
    NUMFAULTY = FULLTREE.count_number_faulty_addresses()
    FAULTY_ROUTER_LIST = FULLTREE.router_list_functioning_or_not(functioning=False)
    FUNCTIONING_ROUTER_LIST = FULLTREE.router_list_functioning_or_not(functioning=True)
    NUM_FAULTY_RIGHT = FULLTREE.right_child.count_number_faulty_addresses()
    NUM_FAULTY_LEFT = FULLTREE.left_child.count_number_faulty_addresses()
    if NUM_FAULTY_RIGHT <= NUM_FAULTY_LEFT:
        ORIG_TREE = FULLTREE.right_child
        AUG_TREE = FULLTREE.left_child
    else:
        ORIG_TREE = FULLTREE.left_child
        AUG_TREE = FULLTREE.right_child

    ADD_REPAIR = RouterRepair(ORIG_TREE, AUG_TREE)
    REPAIR_LIST = ADD_REPAIR.router_repair()
    print(NUMFAULTY, "faulty: ", FAULTY_ROUTER_LIST)
    print("repair ", REPAIR_LIST)
