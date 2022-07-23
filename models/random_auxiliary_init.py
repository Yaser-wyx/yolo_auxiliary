import random

import torch.nn.init as init
from models.common import *

DISTRIBUTION_LIST = [init.normal_, init.xavier_normal_, init.kaiming_normal_, init.xavier_uniform_,
                     init.trunc_normal_, init.kaiming_uniform_, init.orthogonal_, ]


class RandomAuxiliaryInit(object):

    def __str__(self) -> str:
        return f"RandomAuxiliaryInit:\n" \
               f"distribution_num: {self.distribution_num}\n" \
               f"init_way: {self.init_way}\n" \
               f"distribution_fixed: {self.distribution_fixed}\n" \
               f"chosen_distributions: {self.chosen_distributions}\n"

    def __init__(self, distribution_num, init_way, distribution_fixed=False):
        self.distribution_num = distribution_num
        self.init_way = init_way
        self.distribution_fixed = distribution_fixed
        self.chosen_distributions = self.get_random_distributions() if distribution_fixed else None
        self.init_func = None

    def get_random_distributions(self):
        return np.random.choice(DISTRIBUTION_LIST, self.distribution_num, replace=False)

    def random_choice_init_distribution_func(self):
        self.init_func = random.choice(self.chosen_distributions)

    def apply_random_init(self, model: nn.Module):
        LOGGER.info(str(_RANDOM_AUXILIARY_INIT))

        if not self.distribution_fixed:
            self.chosen_distributions = self.get_random_distributions()

        if self.init_way == "whole-net":
            # in advance to get one distribution for initial
            self.random_choice_init_distribution_func()
        model.apply(random_weights_init)


_RANDOM_AUXILIARY_INIT: RandomAuxiliaryInit = None


def set_random_auxiliary_init(distribution_num, init_way, distribution_fixed=False):
    global _RANDOM_AUXILIARY_INIT
    _RANDOM_AUXILIARY_INIT = RandomAuxiliaryInit(distribution_num, init_way, distribution_fixed)
    LOGGER.info(str(_RANDOM_AUXILIARY_INIT))
    return _RANDOM_AUXILIARY_INIT


def get_random_auxiliary_init() -> RandomAuxiliaryInit:
    global _RANDOM_AUXILIARY_INIT
    assert _RANDOM_AUXILIARY_INIT is not None
    return _RANDOM_AUXILIARY_INIT


def random_weights_init(m):
    if type(m) in {nn.Conv2d, nn.Linear}:
        random_auxiliary_init = get_random_auxiliary_init()
        data = m.weight.data
        try:
            if random_auxiliary_init.init_way == "per-layer":
                random_auxiliary_init.random_choice_init_distribution_func()
            random_auxiliary_init.init_func(data)
        except:
            # it will run when data.dim() < 2 and random_auxiliary_init.init_func is not normal_ or uniform_
            init_func = random.choice([init.normal_, init.uniform_, ])
            init_func(data)
