from torch import nn


class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """
  def __init__(self, device):
    super(MessageFunction, self).__init__()
    self.device = device

  def compute_message(self, raw_messages):
    return None


class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension, device):
    super(MLPMessageFunction, self).__init__(device)
    self.device = device
    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
    )

  def compute_message(self, raw_messages):
    messages = self.mlp(raw_messages)

    return messages


class IdentityMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension, device):
    super(IdentityMessageFunction, self).__init__(device)
    self.device = device

  def compute_message(self, raw_messages):

    return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension, device):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension, device)
  elif module_type == "identity":
    return IdentityMessageFunction(raw_message_dimension, message_dimension, device)
