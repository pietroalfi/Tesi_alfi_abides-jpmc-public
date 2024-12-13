import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

class CustomDQNTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        input_size = obs_space.shape[0]  # state dim (10)
        print(f"Obs space shape: {obs_space.shape}")

        hidden_sizes = model_config.get("fcnet_hiddens", [50,20]) #[128, 64, 32]

        # Model construction
        layers = []
        prev_size = input_size

        # Add Layer Normalization
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LayerNorm(size))  # normalization layer 
            layers.append(nn.ReLU())          # non linear activation
            prev_size = size

        # Last layer and model construction (output)
        layers.append(nn.Linear(prev_size, num_outputs))
        self.model = nn.Sequential(*layers)

    def forward(self, input_dict, state, seq_lens):
        # function to evaluate a step 
        obs = input_dict["obs"]  
        obs = obs.view(obs.size(0), -1) 
        return self.model(obs), state

"""
NORMALIZZAZIONE SOLO PRIMO LAYER
class CustomDQNTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        input_size = obs_space.shape[0]  # Dimensione dello stato (12)
        hidden_sizes = model_config.get("fcnet_hiddens", [128, 64, 32])

        layers = []
        prev_size = input_size

        # Primo hidden layer con normalizzazione
        layers.append(nn.Linear(prev_size, hidden_sizes[0]))
        layers.append(nn.LayerNorm(hidden_sizes[0]))  # Normalizzazione del primo layer
        layers.append(nn.ReLU())
        prev_size = hidden_sizes[0]

        # Altri hidden layer senza normalizzazione
        for size in hidden_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        # Ultimo layer lineare (output)
        layers.append(nn.Linear(prev_size, num_outputs))
        self.model = nn.Sequential(*layers)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]  # Ottieni osservazioni dal dizionario di input
        obs = obs.view(obs.size(0), -1) 
        return self.model(obs), state
"""