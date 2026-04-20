# utils/enhanced_shamir_privacy.py
import numpy as np

class ShamirConfig:
    def __init__(self, threshold=3, num_participants=5):
        self.threshold = threshold
        self.num_participants = num_participants

class DifferentialPrivacyConfig:
    def __init__(self, epsilon_total=1.0, delta=1e-5):
        # Evita divisioni per zero se epsilon è None o troppo piccolo
        self.epsilon_total = max(float(epsilon_total), 1e-10)
        self.delta = delta

class SecureAggregationProtocol:
    def __init__(self, shamir_cfg, dp_cfg):
        self.shamir_cfg = shamir_cfg
        self.dp_cfg = dp_cfg

    def aggregate(self, weighted_params_list):
        """
        weighted_params_list deve essere una LISTA dove ogni elemento 
        è la LISTA dei pesi (matrici numpy) di un ospedale.
        """
        aggregated_params = []
        
        # Determina il numero di layer dalla prima entry
        first_node_params = weighted_params_list[0]
        num_layers = len(first_node_params)
        
        for i in range(num_layers):
            # Estrae l'i-esimo layer da ogni partecipante e lo somma
            # Usiamo np.array() per sicurezza
            layers_to_sum = [np.array(node[i]) for node in weighted_params_list]
            layer_sum = sum(layers_to_sum)
            
            # Media (FedAvg)
            layer_avg = layer_sum / len(weighted_params_list)
            
            # Differential Privacy: Rumore di Laplace
            noise_scale = 0.01 / self.dp_cfg.epsilon_total
            noise = np.random.laplace(0, noise_scale, layer_avg.shape)
            
            aggregated_params.append(layer_avg + noise)
            
        return aggregated_params