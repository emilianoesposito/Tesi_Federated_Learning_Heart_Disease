# utils/enhanced_shamir_privacy.py
import numpy as np

class ShamirConfig:
    def __init__(self, threshold=3, num_participants=5):
        self.threshold = threshold
        self.num_participants = num_participants

class DifferentialPrivacyConfig:
    def __init__(self, epsilon_total=1.0, delta=1e-5):
        self.epsilon_total = epsilon_total
        self.delta = delta

class SecureAggregationProtocol:
    def __init__(self, shamir_cfg, dp_cfg):
        self.shamir_cfg = shamir_cfg
        self.dp_cfg = dp_cfg

    def aggregate(self, weighted_params_list):
        aggregated_params = []
        for i in range(len(weighted_params_list[0])):
            # Media pesata (FedAvg)
            layer_sum = sum([node[i] for node in weighted_params_list])
            
            # Differential Privacy: Rumore di Laplace basato su Epsilon
            noise_scale = 0.01 / self.dp_cfg.epsilon_total
            noise = np.random.laplace(0, noise_scale, layer_sum.shape)
            
            aggregated_params.append(layer_sum + noise)
        return aggregated_params