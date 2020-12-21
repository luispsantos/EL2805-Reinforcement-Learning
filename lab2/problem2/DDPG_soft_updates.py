# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#

import torch.nn as nn

def soft_updates(network: nn.Module,
                 target_network: nn.Module,
                 tau: float) -> nn.Module:
    """ Performs a soft copy of the network's parameters to the target
        network's parameter

        Args:
            network (nn.Module): neural network from which we want to copy the
                parameters
            target_network (nn.Module): network that is being updated
            tau (float): time constant that defines the update speed in (0,1)

        Returns:
            target_network (nn.Module): the target network

    """
    tgt_state = target_network.state_dict()
    for k, v in network.state_dict().items():
        tgt_state[k] = (1 - tau)  * tgt_state[k]  + tau * v
    target_network.load_state_dict(tgt_state)
    return target_network
