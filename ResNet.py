import torch.nn.functional as F
import numpy as np
import torch.nn as nn

C_PUCT = 2.0

class ResBlock(nn.Module):
  def __init__(self, num_channels):
    super(ResBlock, self).__init__()
    self.conv_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    self.bn_1 = nn.BatchNorm2d(num_channels)
    self.conv_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    self.bn_2 = nn.BatchNorm2d(num_channels)

  def forward(self, x):
    r = x
    x = self.conv_1(x)
    x = self.bn_1(x)
    x = F.relu(x)
    x = self.conv_2(x)
    x = self.bn_2(x)
    x += r
    x = F.relu(x)
    return x

class ResNet(nn.Module):
  def __init__(self, num_blocks, num_channels):
    super(ResNet, self).__init__()
    state_size = 42 #7*6
    action_size = 7 #7*6
    self.start_block = nn.Sequential(
      nn.Conv2d(2, num_channels, kernel_size=3, padding=1), 
      nn.BatchNorm2d(num_channels),
      nn.ReLU()
    )
    self.res_blocks = nn.ModuleList([ResBlock(num_channels) for i in range(num_blocks)])
    self.policy_head = nn.Sequential(
      nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(32 * state_size, action_size)
    )
    self.value_head = nn.Sequential(
      nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
      nn.BatchNorm2d(3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(3 * state_size, 1),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.start_block(x)
    for res_block in self.res_blocks:
      x = res_block(x)
    policy = self.policy_head(x)
    value = self.value_head(x)
    return policy, value

class MCTSNode:
  def __init__(self, env, parent=None, prev_action=None, prior=0):
    self.env = env
    self.is_finish = None
    self.win = 0
    self.parent = parent
    self.prev_action = prev_action
    self.prior = prior
    self.children = []
    self.value_sum = 0.0
    self.visit_count = 0

  def select(self):
    return max(self.children, key=lambda child: child.ucb())

  def ucb(self):
    exploit = 0.0 if self.visit_count == 0 else  -1 * (self.value_sum / self.visit_count) 
    explore = C_PUCT * self.prior * np.sqrt(self.parent.visit_count) / (self.visit_count + 1)
    return exploit + explore

  def expand(self, policy):
    haveChild = False
    for i in range(len(policy)):
      if policy[i] > 0.0:
        child_env = self.env.clone()
        child_env.play_action(i)
        child_env.switch_player()

        child = MCTSNode(env=child_env,parent=self,prev_action=i,prior= policy[i])
        self.children.append(child)
        haveChild = True

    if not haveChild:
        print('warning, on a pas extand les enfants d une leaf')

  def backprop(self, value):
    self.value_sum += value
    self.visit_count += 1
    if self.parent is not None:
      self.parent.backprop(-value)
  
  def deepcopy(self):
    new_node = MCTSNode(env=self.env.clone(), parent=None, prev_action=self.prev_action, prior=self.prior)
    new_node.is_finish = self.is_finish
    new_node.win = self.win
    new_node.value_sum = self.value_sum
    new_node.visit_count = self.visit_count
    new_node.children = [child.deepcopy() for child in self.children]
    for child in new_node.children:
        child.parent = new_node
    return new_node
