# utils/blockchain_data_anchoring.py
import hashlib
import json

class ClinicalBlockchainAnchor:
    @staticmethod
    def create_record_hash(record_dict):
        canonical_json = json.dumps(record_dict, sort_keys=True).encode('utf-8')
        return hashlib.sha256(canonical_json).hexdigest()

    @staticmethod
    def build_merkle_root(hashes_list):
        if not hashes_list: return None
        combined = "".join(sorted(hashes_list)).encode('utf-8')
        return hashlib.sha256(combined).hexdigest()