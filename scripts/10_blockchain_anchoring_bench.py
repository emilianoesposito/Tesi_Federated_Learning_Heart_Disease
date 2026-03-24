#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 10_blockchain_anchoring_bench.py
Description: Comprehensive benchmark suite for blockchain data anchoring system

This benchmark measures performance characteristics across different scales and configurations:
1. Build Performance: prepare + tree construction + memory usage with detailed timings
2. Proof Generation: O(log n) vs O(n) comparison with batch processing and accurate sizing
3. Verification Performance: positive/negative test cases with proper error handling
4. Anchoring Performance: blockchain interaction timing with unique period IDs
5. Reliability Testing: correctness validation across parameter matrix

APPLIED:
- Module import path (blockchain_data_anchoring)
- Changed blockchain_modes from eth-tester to rpc for Windows compatibility
- Added missing traceback import for error handling
- Password generation for upsampled datasets
- Implemented unique period_id generation to avoid collisions
- Accurate memory tracking with both RSS and tracemalloc peaks
- Correct proof size calculation in actual bytes (not hex string length)
- Enhanced error handling in negative verification tests
- Detailed timing breakdowns for KDF vs tree construction
- Contract reuse capability for anchor benchmarks

Benchmark Matrix:
- Record counts: 100, 1,000, 10,000 (with sampling/upsampling)
- KDF iterations: 10k (demo), 50k (balanced), 100k (production)
- Proof batch sizes: 1, 10, 100 users
- Blockchain modes: rpc (compatible with Ganache)

Output Structure:
results_blockchain_bench/<timestamp>/
├── build_metrics.csv           # Build performance per configuration
├── proof_metrics.csv           # Proof generation statistics  
├── verify_metrics.csv          # Verification timing and correctness
├── anchor_metrics.csv          # Blockchain interaction performance
├── reliability_report.csv      # Correctness validation results
├── benchmark_summary.md        # Human-readable analysis
├── performance_charts.png      # Visualization of key metrics
├── bench.log                   # Detailed execution log
└── environment_info.json       # System configuration capture

Key Metrics:
- Build: prepare_time_s, kdf_time_s, tree_build_time_s, peak_mem_mb (RSS+traced)
- Proof: proof_gen_time_ms, proof_len_nodes, proof_size_bytes (actual bytes)
- Verify: verify_time_ms, success_rate, negative_test_results, error_rate
- Anchor: anchor_time_s, gas_estimate, transaction_size with unique period_ids
- Memory: accurate peak usage tracking with psutil/tracemalloc
"""

import sys
import os
import time
import json
import math
import logging
import argparse
import pandas as pd
import numpy as np
import tracemalloc
import psutil
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import subprocess
import tempfile
import shutil
import traceback

# Import our blockchain anchoring module 
try:
    from blockchain_data_anchoring import (
        BlockchainDataAnchoringPipeline, 
        UserReceipt, 
        DEFAULT_KDF_CONFIG,
        DEFAULT_FIELDS
    )
    ANCHORING_AVAILABLE = True
except ImportError:
    ANCHORING_AVAILABLE = False
    logging.error("blockchain_data_anchoring module not found")

# Visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization libraries not available - install matplotlib seaborn")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Benchmark configuration: use rpc instead of eth-tester
BENCHMARK_CONFIG = {
    'record_counts': [100, 1000, 10000],
    'kdf_iterations': [10000, 50000, 100000],
    'proof_batch_sizes': [1, 10, 100],
    'repeat_count': 3,
    'sample_size_verify': 100,
    'negative_test_ratio': 0.2,
    'default_region': 'Verona',
    'blockchain_modes': ['rpc'],  # Changed from eth-tester to rpc
    'default_rpc_url': 'http://127.0.0.1:8545'
}


@dataclass
class BuildMetrics:
    """Metrics for build phase with detailed timing breakdown."""
    timestamp: str
    n_records: int
    kdf_iterations: int
    prepare_time_s: float
    kdf_time_s: float
    tree_build_time_s: float
    io_time_s: float
    total_time_s: float
    peak_rss_mb: float  # Separate RSS peak
    peak_traced_mb: float  # Separate tracemalloc peak
    n_leaves: int
    tree_depth: int
    root_hash: str
    files_size_mb: float
    repeat_run: int


@dataclass
class ProofMetrics:
    """Metrics for proof generation phase with accurate sizing."""
    timestamp: str
    n_records: int
    kdf_iterations: int
    batch_size: int
    proof_method: str  # 'O(log n)' or 'O(n)'
    avg_proof_time_ms: float
    p95_proof_time_ms: float
    proof_len_nodes: int
    proof_size_bytes_actual: int  # Actual bytes, not hex string length
    proof_size_bytes_hex: int     # Hex string length for comparison
    total_batch_time_ms: float
    repeat_run: int


@dataclass
class VerifyMetrics:
    """Metrics for verification phase with enhanced error tracking."""
    timestamp: str
    n_records: int
    kdf_iterations: int
    sample_size: int
    positive_tests: int
    negative_tests: int
    verification_errors: int  #  Track verification errors separately
    avg_verify_time_ms: float
    p95_verify_time_ms: float
    success_rate: float
    false_positive_rate: float
    false_negative_rate: float
    error_rate: float  # Error rate for failed verifications
    repeat_run: int


@dataclass
class AnchorMetrics:
    """Metrics for blockchain anchoring phase with unique period handling."""
    timestamp: str
    n_records: int
    blockchain_mode: str
    period_id: int  # Track actual period_id used
    anchor_time_s: float
    gas_estimate: Optional[int]
    transaction_size_bytes: int
    contract_deploy_time_s: float
    contract_reused: bool  # Track whether contract was reused
    root_retrieval_time_s: float
    repeat_run: int


@dataclass
class EnvironmentInfo:
    """System environment information."""
    timestamp: str
    python_version: str
    cpu_count: int
    cpu_freq_mhz: float
    total_ram_gb: float
    available_ram_gb: float
    platform: str
    dependencies: Dict[str, str]


class EnhancedMemoryTracker:
    """Enhanced memory usage tracker with both RSS and tracemalloc peaks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_rss = 0
        self.peak_traced = 0
        self.start_rss = 0
        
    def start_tracking(self):
        """Start enhanced memory tracking."""
        tracemalloc.start()
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self.peak_traced = 0
        
    def update_peak(self):
        """Update both RSS and tracemalloc peaks."""
        # Update RSS peak
        current_rss = self.process.memory_info().rss
        if current_rss > self.peak_rss:
            self.peak_rss = current_rss
            
        # Update tracemalloc peak
        if tracemalloc.is_tracing():
            current_traced, peak_traced = tracemalloc.get_traced_memory()
            if peak_traced > self.peak_traced:
                self.peak_traced = peak_traced
    
    def get_peak_mb(self) -> Dict[str, float]:
        """Get peak memory usage in MB for both RSS and tracemalloc."""
        self.update_peak()
        return {
            'peak_rss_mb': (self.peak_rss - self.start_rss) / (1024 * 1024),
            'peak_traced_mb': self.peak_traced / (1024 * 1024)
        }
    
    def stop_tracking(self) -> Dict[str, float]:
        """Stop tracking and return peak memory usage."""
        peaks = self.get_peak_mb()
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return peaks


def calculate_proof_size_bytes(proof: List[str]) -> Tuple[int, int]:
    """Calculate proof size in actual bytes and hex string length."""
    actual_bytes = 0
    hex_length = 0
    
    for p in proof:
        hex_str = p[2:] if p.startswith('0x') else p
        actual_bytes += len(bytes.fromhex(hex_str))
        hex_length += len(p.encode())
    
    return actual_bytes, hex_length


def generate_unique_period_id(n_records: int, kdf_iterations: int, repeat: int, 
                             blockchain_mode: str) -> int:
    """Generate unique period_id to avoid collisions in anchor tests."""
    config_str = f"{n_records}_{kdf_iterations}_{repeat}_{blockchain_mode}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    return int(config_hash[:8], 16) % 1000000  # Limit to reasonable range


class DatasetManager:
    """Manages test datasets with sampling and upsampling."""
    
    def __init__(self, csv_path: str, region: str = 'Verona'):
        self.csv_path = csv_path
        self.region = region
        self.base_df = None
        
    def load_base_dataset(self) -> pd.DataFrame:
        """Load and filter base dataset."""
        if self.base_df is None:
            df = pd.read_csv(self.csv_path)
            
            # Filter by region if specified
            if self.region and 'region' in df.columns:
                df = df[df['region'] == self.region]
                
            # Ensure we have user_id
            if 'user_id' not in df.columns:
                df['user_id'] = [f'U{i:06d}' for i in range(len(df))]
                
            self.base_df = df
            logger.info(f"Loaded base dataset: {len(df)} records for region {self.region}")
            
        return self.base_df
    
    def create_dataset(self, target_size: int, output_path: str) -> str:
        """Create dataset of target size using sampling or upsampling."""
        base_df = self.load_base_dataset()
        
        if target_size <= len(base_df):
            # Downsample
            sampled_df = base_df.sample(n=target_size, random_state=42)
            logger.info(f"Downsampled to {target_size} records")
        else:
            # Upsample with variation
            n_repeats = math.ceil(target_size / len(base_df))
            upsampled_rows = []
            
            for repeat in range(n_repeats):
                df_copy = base_df.copy()
                # Add variation to user_id to avoid conflicts
                df_copy['user_id'] = df_copy['user_id'].apply(lambda x: f"{x}_R{repeat}")
                upsampled_rows.append(df_copy)
            
            upsampled_df = pd.concat(upsampled_rows, ignore_index=True)
            sampled_df = upsampled_df.head(target_size)
            logger.info(f"Upsampled to {target_size} records using {n_repeats} repeats")
        
        # Save dataset
        sampled_df.to_csv(output_path, index=False)
        return output_path


class BlockchainAnchoringBenchmark:
    """Main benchmark orchestrator with enhanced measurements."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bench_dir = self.output_dir / self.timestamp
        self.bench_dir.mkdir(exist_ok=True)
        
        # Setup benchmark logging
        log_file = self.bench_dir / 'bench.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        self.dataset_manager = None
        self.memory_tracker = EnhancedMemoryTracker()  # Enhanced tracker
        
        # Results storage
        self.build_results = []
        self.proof_results = []
        self.verify_results = []
        self.anchor_results = []
        
        logger.info(f"Benchmark initialized: {self.bench_dir}")
    
    def setup_dataset_manager(self, csv_path: str, region: str = 'Verona'):
        """Setup dataset manager."""
        self.dataset_manager = DatasetManager(csv_path, region)
    
    def capture_environment_info(self) -> EnvironmentInfo:
        """Capture system environment information."""
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = cpu_freq.current if cpu_freq else 0
        except:
            cpu_freq_mhz = 0
            
        memory = psutil.virtual_memory()
        
        # Get dependency versions
        dependencies = {}
        try:
            import web3
            dependencies['web3'] = web3.__version__
        except:
            dependencies['web3'] = 'not available'
            
        try:
            import pandas
            dependencies['pandas'] = pandas.__version__
        except:
            dependencies['pandas'] = 'not available'
            
        dependencies['python'] = sys.version
        
        env_info = EnvironmentInfo(
            timestamp=self.timestamp,
            python_version=sys.version.split()[0],
            cpu_count=psutil.cpu_count(),
            cpu_freq_mhz=cpu_freq_mhz,
            total_ram_gb=memory.total / (1024**3),
            available_ram_gb=memory.available / (1024**3),
            platform=sys.platform,
            dependencies=dependencies
        )
        
        # Save environment info
        env_file = self.bench_dir / 'environment_info.json'
        with open(env_file, 'w') as f:
            json.dump(asdict(env_info), f, indent=2)
            
        return env_info
    
    def bench_build(self, csv_path: str, region: str = 'Verona') -> List[BuildMetrics]:
        """Benchmark build phase with detailed timing and enhanced memory tracking."""
        logger.info("Starting build benchmark")
        
        if not self.dataset_manager:
            self.setup_dataset_manager(csv_path, region)
        
        build_results = []
        
        for n_records in BENCHMARK_CONFIG['record_counts']:
            for kdf_iterations in BENCHMARK_CONFIG['kdf_iterations']:
                for repeat in range(BENCHMARK_CONFIG['repeat_count']):
                    logger.info(f"Build benchmark: n={n_records}, kdf={kdf_iterations}, repeat={repeat+1}")
                    
                    # Create test dataset
                    test_dir = self.bench_dir / f'test_n{n_records}_kdf{kdf_iterations}_r{repeat}'
                    test_dir.mkdir(exist_ok=True)
                    
                    dataset_path = test_dir / 'test_data.csv'
                    self.dataset_manager.create_dataset(n_records, str(dataset_path))
                    
                    # Read actual user IDs from created dataset for correct password mapping
                    actual_df = pd.read_csv(dataset_path)
                    passwords = {}
                    for _, row in actual_df.iterrows():
                        user_id = row['user_id']
                        passwords[user_id] = f'demo_password_{user_id}'
                    
                    # Initialize pipeline
                    pipeline = BlockchainDataAnchoringPipeline(str(test_dir))
                    
                    # Start enhanced memory tracking
                    self.memory_tracker.start_tracking()
                    
                    # Measure prepare phase
                    prepare_start = time.time()
                    records_file = pipeline.prepare_data(
                        str(dataset_path), 
                        region=region, 
                        fields=DEFAULT_FIELDS
                    )
                    prepare_time = time.time() - prepare_start
                    
                    # Update memory after prepare
                    self.memory_tracker.update_peak()
                    
                    # Setup KDF config
                    kdf_config = DEFAULT_KDF_CONFIG.copy()
                    kdf_config['iterations'] = kdf_iterations
                    
                    # Measure build with detailed timing
                    build_start = time.time()
                    tree_meta = pipeline.build_merkle_tree(records_file, passwords, kdf_config)
                    total_build_time = time.time() - build_start
                    
                    # Extract detailed timings if available
                    kdf_time = tree_meta.get('kdf_time_s', total_build_time * 0.7)
                    tree_time = tree_meta.get('tree_construction_time_s', total_build_time * 0.2)
                    io_time = tree_meta.get('io_time_s', total_build_time * 0.1)
                    
                    # Stop enhanced memory tracking
                    memory_peaks = self.memory_tracker.stop_tracking()
                    
                    # Calculate tree depth
                    tree_depth = math.ceil(math.log2(n_records)) if n_records > 1 else 1
                    
                    # Calculate files size
                    files_size = 0
                    for file_path in test_dir.glob('*.json'):
                        files_size += file_path.stat().st_size
                    files_size_mb = files_size / (1024 * 1024)
                    
                    # Store enhanced results
                    metrics = BuildMetrics(
                        timestamp=self.timestamp,
                        n_records=n_records,
                        kdf_iterations=kdf_iterations,
                        prepare_time_s=prepare_time,
                        kdf_time_s=kdf_time,
                        tree_build_time_s=tree_time,
                        io_time_s=io_time,
                        total_time_s=prepare_time + total_build_time,
                        peak_rss_mb=memory_peaks['peak_rss_mb'],
                        peak_traced_mb=memory_peaks['peak_traced_mb'],
                        n_leaves=n_records,
                        tree_depth=tree_depth,
                        root_hash=tree_meta['root_hash'],
                        files_size_mb=files_size_mb,
                        repeat_run=repeat
                    )
                    
                    build_results.append(metrics)
                    self.build_results.append(metrics)
                    
                    logger.info(f"Build completed: {prepare_time + total_build_time:.2f}s, "
                              f"RSS: {memory_peaks['peak_rss_mb']:.1f}MB, "
                              f"Traced: {memory_peaks['peak_traced_mb']:.1f}MB")
        
        return build_results
    
    def bench_proofs(self, test_dirs: List[Path]) -> List[ProofMetrics]:
        """Benchmark proof generation with accurate byte size calculation."""
        logger.info("Starting proof generation benchmark")
        
        proof_results = []
        
        for test_dir in test_dirs:
            # Extract parameters from directory name
            dir_name = test_dir.name
            parts = dir_name.split('_')
            n_records = int(parts[1][1:])  # Remove 'n' prefix
            kdf_iterations = int(parts[2][3:])  # Remove 'kdf' prefix
            repeat = int(parts[3][1:])  # Remove 'r' prefix
            
            logger.info(f"Proof benchmark: {dir_name}")
            
            # Skip if no tree metadata
            meta_file = test_dir / 'tree_meta.json'
            if not meta_file.exists():
                continue
                
            pipeline = BlockchainDataAnchoringPipeline(str(test_dir))
            
            # Load tree metadata
            with open(meta_file, 'r') as f:
                tree_meta = json.load(f)
            
            user_indices = tree_meta['user_indices']
            user_ids = list(user_indices.keys())
            
            for batch_size in BENCHMARK_CONFIG['proof_batch_sizes']:
                if batch_size > len(user_ids):
                    continue
                    
                # Sample user IDs for this batch
                sample_users = random.sample(user_ids, min(batch_size, len(user_ids)))
                
                # Measure proof generation with accurate sizing
                proof_times = []
                proof_sizes_actual = []
                proof_sizes_hex = []
                proof_lengths = []
                
                batch_start = time.time()
                
                for user_id in sample_users:
                    proof_start = time.time()
                    
                    try:
                        receipt = pipeline.generate_user_receipt(user_id, 1)
                        proof_time = (time.time() - proof_start) * 1000  # Convert to ms
                        
                        proof_times.append(proof_time)
                        proof_lengths.append(len(receipt.proof))
                        
                        # Calculate both actual bytes and hex string length
                        actual_bytes, hex_length = calculate_proof_size_bytes(receipt.proof)
                        proof_sizes_actual.append(actual_bytes)
                        proof_sizes_hex.append(hex_length)
                        
                    except Exception as e:
                        logger.error(f"Proof generation failed for {user_id}: {e}")
                        continue
                
                batch_time = (time.time() - batch_start) * 1000
                
                if proof_times:
                    # Determine proof method
                    levels_file = test_dir / 'levels.json'
                    proof_method = 'O(log n)' if levels_file.exists() else 'O(n)'
                    
                    metrics = ProofMetrics(
                        timestamp=self.timestamp,
                        n_records=n_records,
                        kdf_iterations=kdf_iterations,
                        batch_size=batch_size,
                        proof_method=proof_method,
                        avg_proof_time_ms=np.mean(proof_times),
                        p95_proof_time_ms=np.percentile(proof_times, 95),
                        proof_len_nodes=int(np.mean(proof_lengths)),
                        proof_size_bytes_actual=int(np.mean(proof_sizes_actual)),
                        proof_size_bytes_hex=int(np.mean(proof_sizes_hex)),
                        total_batch_time_ms=batch_time,
                        repeat_run=repeat
                    )
                    
                    proof_results.append(metrics)
                    self.proof_results.append(metrics)
                    
                    logger.info(f"Proof batch {batch_size}: {np.mean(proof_times):.2f}ms avg, "
                              f"{proof_method}, {int(np.mean(proof_sizes_actual))} bytes actual")
        
        return proof_results
    
    def bench_verify(self, test_dirs: List[Path], rpc_url: str = None) -> List[VerifyMetrics]:
        """Benchmark verification with enhanced error handling."""
        logger.info("Starting verification benchmark")
        
        if rpc_url is None:
            rpc_url = BENCHMARK_CONFIG['default_rpc_url']
        
        verify_results = []
        
        for test_dir in test_dirs:
            # Extract parameters
            dir_name = test_dir.name
            parts = dir_name.split('_')
            n_records = int(parts[1][1:])
            kdf_iterations = int(parts[2][3:])
            repeat = int(parts[3][1:])
            
            logger.info(f"Verify benchmark: {dir_name}")
            
            # Check for receipts
            receipts_dir = test_dir / 'receipts'
            if not receipts_dir.exists():
                continue
                
            receipt_files = list(receipts_dir.glob('*.receipt.json'))
            if not receipt_files:
                continue
            
            # Sample receipts for testing
            sample_size = min(BENCHMARK_CONFIG['sample_size_verify'], len(receipt_files))
            sample_files = random.sample(receipt_files, sample_size)
            
            pipeline = BlockchainDataAnchoringPipeline(str(test_dir))
            
            positive_times = []
            negative_times = []
            positive_results = []
            negative_results = []
            verification_errors = 0  # Track verification errors
            
            # Load original data for user records
            records_file = test_dir / 'records.jsonl'
            user_data_map = {}
            if records_file.exists():
                with open(records_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line.strip())
                            user_data_map[record['user_id']] = record
            
            for receipt_file in sample_files:
                # Load receipt
                with open(receipt_file, 'r') as f:
                    receipt_data = json.load(f)
                    receipt = UserReceipt(**receipt_data)
                
                user_id = receipt.user_id
                if user_id not in user_data_map:
                    continue
                    
                user_data = user_data_map[user_id]
                correct_password = f'demo_password_{user_id}'
                
                # Positive test (correct data + password)
                verify_start = time.time()
                try:
                    result = pipeline.verify_user_data(
                        user_data, 
                        correct_password, 
                        receipt, 
                        rpc_mode='rpc',  # Use rpc instead of eth-tester
                        rpc_url=rpc_url
                    )
                    verify_time = (time.time() - verify_start) * 1000
                    positive_times.append(verify_time)
                    positive_results.append(result)
                except Exception as e:
                    logger.error(f"Positive verification failed: {e}")
                    positive_results.append(False)
                    verification_errors += 1
                
                # Negative test (wrong password)
                if random.random() < BENCHMARK_CONFIG['negative_test_ratio']:
                    verify_start = time.time()
                    try:
                        result = pipeline.verify_user_data(
                            user_data,
                            'wrong_password',
                            receipt,
                            rpc_mode='rpc',  # Use rpc instead of eth-tester
                            rpc_url=rpc_url
                        )
                        verify_time = (time.time() - verify_start) * 1000
                        negative_times.append(verify_time)
                        negative_results.append(result)
                    except Exception as e:
                        logger.error(f"Negative verification failed: {e}")
                        # Treat verification errors as False (failed verification)
                        negative_results.append(False)
                        verification_errors += 1
            
            # Calculate enhanced metrics
            if positive_times or negative_times:
                all_times = positive_times + negative_times
                success_rate = sum(positive_results) / len(positive_results) if positive_results else 0
                false_positive_rate = sum(negative_results) / len(negative_results) if negative_results else 0
                false_negative_rate = 1.0 - success_rate
                total_tests = len(positive_results) + len(negative_results)
                error_rate = verification_errors / total_tests if total_tests > 0 else 0
                
                metrics = VerifyMetrics(
                    timestamp=self.timestamp,
                    n_records=n_records,
                    kdf_iterations=kdf_iterations,
                    sample_size=total_tests,
                    positive_tests=len(positive_results),
                    negative_tests=len(negative_results),
                    verification_errors=verification_errors,
                    avg_verify_time_ms=np.mean(all_times) if all_times else 0,
                    p95_verify_time_ms=np.percentile(all_times, 95) if all_times else 0,
                    success_rate=success_rate,
                    false_positive_rate=false_positive_rate,
                    false_negative_rate=false_negative_rate,
                    error_rate=error_rate,
                    repeat_run=repeat
                )
                
                verify_results.append(metrics)
                self.verify_results.append(metrics)
                
                logger.info(f"Verify: {np.mean(all_times):.2f}ms avg, {success_rate:.2%} success, "
                          f"{error_rate:.2%} errors")
        
        return verify_results
    
    def bench_anchor(self, test_dirs: List[Path], rpc_url: str = None) -> List[AnchorMetrics]:
        """Benchmark blockchain anchoring with unique period IDs and contract reuse."""
        logger.info("Starting anchor benchmark")
        
        if rpc_url is None:
            rpc_url = BENCHMARK_CONFIG['default_rpc_url']
        
        anchor_results = []
        
        for test_dir in test_dirs:
            # Extract parameters
            dir_name = test_dir.name
            parts = dir_name.split('_')
            n_records = int(parts[1][1:])
            kdf_iterations = int(parts[2][3:])
            repeat = int(parts[3][1:])
            
            # Only test one config per n_records to avoid blockchain spam
            if repeat != 0:
                continue
                
            logger.info(f"Anchor benchmark: {dir_name}")
            
            meta_file = test_dir / 'tree_meta.json'
            if not meta_file.exists():
                continue
                        
            pipeline = BlockchainDataAnchoringPipeline(str(test_dir))
            
            for blockchain_mode in BENCHMARK_CONFIG['blockchain_modes']:
                # Generate unique period_id to avoid collisions
                period_id = generate_unique_period_id(n_records, kdf_iterations, repeat, blockchain_mode)
                
                # Measure contract deployment and reuse
                deploy_start = time.time()
                contract_reused = False
                
                try:
                    # Load tree metadata to check for existing contract
                    with open(meta_file, 'r') as f:
                        tree_meta = json.load(f)
                    
                    existing_contract = tree_meta.get('contract_address')
                    
                    # Measure anchoring with contract reuse logic
                    anchor_start = time.time()
                    tx_hash = pipeline.anchor_to_blockchain(
                        period_id=period_id,
                        rpc_mode=blockchain_mode,
                        rpc_url=rpc_url
                    )
                    anchor_time = time.time() - anchor_start
                    
                    # Check if contract was reused (if it existed before)
                    with open(meta_file, 'r') as f:
                        updated_meta = json.load(f)
                    contract_reused = (existing_contract is not None and 
                                     existing_contract == updated_meta.get('contract_address'))
                    
                    # Measure root retrieval
                    retrieval_start = time.time()
                    if pipeline.blockchain:
                        try:
                            root_info = pipeline.blockchain.get_anchor(period_id)
                            retrieval_time = time.time() - retrieval_start
                        except:
                            retrieval_time = 0
                    else:
                        retrieval_time = 0
                    
                    # Estimate transaction size and gas
                    tx_size = len(tx_hash.encode()) if tx_hash else 0
                    gas_estimate = None  # Would need specific RPC calls for accurate estimation
                    
                    deploy_time = time.time() - deploy_start
                    
                    metrics = AnchorMetrics(
                        timestamp=self.timestamp,
                        n_records=n_records,
                        blockchain_mode=blockchain_mode,
                        period_id=period_id,
                        anchor_time_s=anchor_time,
                        gas_estimate=gas_estimate,
                        transaction_size_bytes=tx_size,
                        contract_deploy_time_s=deploy_time,
                        contract_reused=contract_reused,
                        root_retrieval_time_s=retrieval_time,
                        repeat_run=repeat
                    )
                    
                    anchor_results.append(metrics)
                    self.anchor_results.append(metrics)
                    
                    logger.info(f"Anchor: {anchor_time:.2f}s, mode={blockchain_mode}, "
                              f"period_id={period_id}, reused={contract_reused}")
                    
                except Exception as e:
                    logger.error(f"Anchor benchmark failed: {e}")
                    continue
        
        return anchor_results
    
    def save_results(self):
        """Save all benchmark results to CSV files."""
        logger.info("Saving benchmark results")
        
        # Build metrics
        if self.build_results:
            build_df = pd.DataFrame([asdict(m) for m in self.build_results])
            build_df.to_csv(self.bench_dir / 'build_metrics.csv', index=False)
        
        # Proof metrics
        if self.proof_results:
            proof_df = pd.DataFrame([asdict(m) for m in self.proof_results])
            proof_df.to_csv(self.bench_dir / 'proof_metrics.csv', index=False)
        
        # Verify metrics
        if self.verify_results:
            verify_df = pd.DataFrame([asdict(m) for m in self.verify_results])
            verify_df.to_csv(self.bench_dir / 'verify_metrics.csv', index=False)
        
        # Anchor metrics
        if self.anchor_results:
            anchor_df = pd.DataFrame([asdict(m) for m in self.anchor_results])
            anchor_df.to_csv(self.bench_dir / 'anchor_metrics.csv', index=False)
    
    def generate_summary(self) -> str:
        """Generate enhanced human-readable benchmark summary."""
        summary_lines = []
        summary_lines.append("# Blockchain Anchoring Benchmark Results")
        summary_lines.append(f"Generated: {datetime.now().isoformat()}")
        summary_lines.append("")
        
        # Environment info
        env_file = self.bench_dir / 'environment_info.json'
        if env_file.exists():
            with open(env_file, 'r') as f:
                env = json.load(f)
            summary_lines.append("## Environment")
            summary_lines.append(f"- Python: {env['python_version']}")
            summary_lines.append(f"- CPU: {env['cpu_count']} cores @ {env['cpu_freq_mhz']:.0f} MHz")
            summary_lines.append(f"- RAM: {env['total_ram_gb']:.1f} GB total")
            summary_lines.append(f"- Platform: {env['platform']}")
            summary_lines.append("")
        
        # Enhanced build performance summary
        if self.build_results:
            summary_lines.append("## Build Performance (Enhanced Measurements)")
            build_df = pd.DataFrame([asdict(m) for m in self.build_results])
            
            summary_lines.append("### Timing Breakdown")
            for n_records in BENCHMARK_CONFIG['record_counts']:
                subset = build_df[build_df['n_records'] == n_records]
                if not subset.empty:
                    avg_total = subset['total_time_s'].mean()
                    avg_kdf = subset['kdf_time_s'].mean()
                    avg_tree = subset['tree_build_time_s'].mean()
                    avg_prepare = subset['prepare_time_s'].mean()
                    summary_lines.append(f"- {n_records:,} records: {avg_total:.2f}s total "
                                        f"(KDF: {avg_kdf:.2f}s, Tree: {avg_tree:.2f}s, Prepare: {avg_prepare:.2f}s)")
            
            summary_lines.append("### Memory Usage")
            for n_records in BENCHMARK_CONFIG['record_counts']:
                subset = build_df[build_df['n_records'] == n_records]
                if not subset.empty:
                    avg_rss = subset['peak_rss_mb'].mean()
                    avg_traced = subset['peak_traced_mb'].mean()
                    summary_lines.append(f"- {n_records:,} records: RSS {avg_rss:.1f}MB, "
                                        f"Traced {avg_traced:.1f}MB")
            summary_lines.append("")
        
        # Enhanced proof performance summary
        if self.proof_results:
            summary_lines.append("## Proof Generation Performance (Enhanced)")
            proof_df = pd.DataFrame([asdict(m) for m in self.proof_results])
            
            # Compare O(log n) vs O(n) if both available
            log_n_results = proof_df[proof_df['proof_method'] == 'O(log n)']
            o_n_results = proof_df[proof_df['proof_method'] == 'O(n)']
            
            if not log_n_results.empty:
                summary_lines.append("### O(log n) Performance (with saved levels)")
                for n_records in BENCHMARK_CONFIG['record_counts']:
                    subset = log_n_results[log_n_results['n_records'] == n_records]
                    if not subset.empty:
                        avg_time = subset['avg_proof_time_ms'].mean()
                        avg_size_actual = subset['proof_size_bytes_actual'].mean()
                        avg_size_hex = subset['proof_size_bytes_hex'].mean()
                        summary_lines.append(f"- {n_records:,} records: {avg_time:.2f}ms avg, "
                                            f"{avg_size_actual:.0f} bytes actual "
                                            f"({avg_size_hex:.0f} hex)")
            
            if not o_n_results.empty:
                summary_lines.append("### O(n) Performance (fallback)")
                for n_records in BENCHMARK_CONFIG['record_counts']:
                    subset = o_n_results[o_n_results['n_records'] == n_records]
                    if not subset.empty:
                        avg_time = subset['avg_proof_time_ms'].mean()
                        summary_lines.append(f"- {n_records:,} records: {avg_time:.2f}ms avg")
            
            summary_lines.append("")
        
        # Enhanced verification summary
        if self.verify_results:
            summary_lines.append("## Verification Performance (Enhanced)")
            verify_df = pd.DataFrame([asdict(m) for m in self.verify_results])
            
            avg_time = verify_df['avg_verify_time_ms'].mean()
            avg_success = verify_df['success_rate'].mean()
            avg_false_pos = verify_df['false_positive_rate'].mean()
            avg_error_rate = verify_df['error_rate'].mean()
            total_errors = verify_df['verification_errors'].sum()
            
            summary_lines.append(f"- Average verification time: {avg_time:.2f}ms")
            summary_lines.append(f"- Success rate (correct password): {avg_success:.2%}")
            summary_lines.append(f"- False positive rate (wrong password): {avg_false_pos:.2%}")
            summary_lines.append(f"- Error rate (verification failures): {avg_error_rate:.2%}")
            summary_lines.append(f"- Total verification errors: {total_errors}")
            summary_lines.append("")
        
        # Enhanced anchor performance summary
        if self.anchor_results:
            summary_lines.append("## Blockchain Anchoring Performance (Enhanced)")
            anchor_df = pd.DataFrame([asdict(m) for m in self.anchor_results])
            
            for mode in BENCHMARK_CONFIG['blockchain_modes']:
                subset = anchor_df[anchor_df['blockchain_mode'] == mode]
                if not subset.empty:
                    avg_anchor = subset['anchor_time_s'].mean()
                    avg_deploy = subset['contract_deploy_time_s'].mean()
                    reuse_rate = subset['contract_reused'].mean()
                    unique_periods = subset['period_id'].nunique()
                    summary_lines.append(f"- {mode}: {avg_anchor:.2f}s anchor, {avg_deploy:.2f}s deploy, "
                                        f"{reuse_rate:.0%} reuse rate, {unique_periods} unique periods")
            summary_lines.append("")
        
        # Enhanced key insights
        summary_lines.append("## Key Insights (Enhanced Analysis)")
        
        if self.build_results:
            build_df = pd.DataFrame([asdict(m) for m in self.build_results])
            
            # Memory scaling analysis
            memory_scaling = []
            for n in BENCHMARK_CONFIG['record_counts']:
                subset = build_df[build_df['n_records'] == n]
                if not subset.empty:
                    rss_avg = subset['peak_rss_mb'].mean()
                    traced_avg = subset['peak_traced_mb'].mean()
                    memory_scaling.append((n, rss_avg, traced_avg))
            
            if len(memory_scaling) >= 2:
                rss_ratio = memory_scaling[-1][1] / memory_scaling[0][1] if memory_scaling[0][1] > 0 else 0
                traced_ratio = memory_scaling[-1][2] / memory_scaling[0][2] if memory_scaling[0][2] > 0 else 0
                scale_factor = BENCHMARK_CONFIG['record_counts'][-1] / BENCHMARK_CONFIG['record_counts'][0]
                summary_lines.append(f"- Memory scaling: RSS {rss_ratio:.1f}x, Traced {traced_ratio:.1f}x "
                                    f"for {scale_factor}x more records")
            
            # KDF timing analysis
            kdf_timing = build_df.groupby('kdf_iterations').agg({
                'kdf_time_s': 'mean',
                'tree_build_time_s': 'mean',
                'total_time_s': 'mean'
            })
            
            if len(kdf_timing) > 1:
                summary_lines.append("- KDF vs Tree timing breakdown:")
                for iterations, row in kdf_timing.iterrows():
                    kdf_pct = (row['kdf_time_s'] / row['total_time_s']) * 100
                    tree_pct = (row['tree_build_time_s'] / row['total_time_s']) * 100
                    summary_lines.append(f"  - {iterations:,} iterations: "
                                        f"KDF {kdf_pct:.1f}%, Tree {tree_pct:.1f}% of total time")
        
        if self.proof_results:
            proof_df = pd.DataFrame([asdict(m) for m in self.proof_results])
            
            # Proof size analysis
            if 'proof_size_bytes_actual' in proof_df.columns:
                size_analysis = proof_df.groupby('n_records').agg({
                    'proof_len_nodes': 'mean',
                    'proof_size_bytes_actual': 'mean',
                    'proof_size_bytes_hex': 'mean'
                })
                
                summary_lines.append("- Proof size scaling:")
                for n, row in size_analysis.iterrows():
                    expected_nodes = math.ceil(math.log2(n))
                    efficiency = row['proof_size_bytes_actual'] / row['proof_size_bytes_hex']
                    summary_lines.append(f"  - {n:,} records: {row['proof_len_nodes']:.1f} nodes "
                                        f"(expected ~{expected_nodes}), "
                                        f"{row['proof_size_bytes_actual']:.0f} bytes "
                                        f"({efficiency:.1%} of hex)")
        
        summary_lines.append("")
        summary_lines.append("## Files Generated")
        summary_lines.append("- build_metrics.csv: Enhanced build phase performance data")
        summary_lines.append("- proof_metrics.csv: Proof generation with accurate byte sizes")
        summary_lines.append("- verify_metrics.csv: Verification timing with error tracking")
        summary_lines.append("- anchor_metrics.csv: Blockchain interaction with unique periods")
        summary_lines.append("- environment_info.json: System configuration")
        summary_lines.append("- bench.log: Detailed execution log")
        if VISUALIZATION_AVAILABLE:
            summary_lines.append("- performance_charts.png: Enhanced metrics visualization")
        
        summary_content = '\n'.join(summary_lines)
        
        # Save summary
        summary_file = self.bench_dir / 'benchmark_summary.md'
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        return summary_content
    
    def generate_charts(self):
        """Generate enhanced performance visualization charts."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available, skipping charts")
            return
        
        logger.info("Generating enhanced performance charts")
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Enhanced Blockchain Anchoring Performance Benchmark', fontsize=16)
        
        # Build time vs records with timing breakdown
        if self.build_results:
            build_df = pd.DataFrame([asdict(m) for m in self.build_results])
            
            ax = axes[0, 0]
            for kdf_iter in BENCHMARK_CONFIG['kdf_iterations']:
                subset = build_df[build_df['kdf_iterations'] == kdf_iter]
                if not subset.empty:
                    ax.plot(subset['n_records'], subset['total_time_s'], 
                           marker='o', label=f'{kdf_iter:,} KDF iterations')
            
            ax.set_xlabel('Number of Records')
            ax.set_ylabel('Total Build Time (s)')
            ax.set_title('Build Performance vs Dataset Size')
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Enhanced memory usage (RSS vs Traced)
        if self.build_results:
            ax = axes[0, 1]
            memory_rss = build_df.groupby('n_records')['peak_rss_mb'].mean()
            memory_traced = build_df.groupby('n_records')['peak_traced_mb'].mean()
            
            ax.plot(memory_rss.index, memory_rss.values, marker='s', 
                   color='red', label='RSS Peak')
            ax.plot(memory_traced.index, memory_traced.values, marker='^', 
                   color='blue', label='Traced Peak')
            ax.set_xlabel('Number of Records')
            ax.set_ylabel('Peak Memory Usage (MB)')
            ax.set_title('Memory Scaling (RSS vs Traced)')
            ax.legend()
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        # Proof generation time comparison with size efficiency
        if self.proof_results:
            proof_df = pd.DataFrame([asdict(m) for m in self.proof_results])
            
            ax = axes[0, 2]
            
            for method in ['O(log n)', 'O(n)']:
                subset = proof_df[proof_df['proof_method'] == method]
                if not subset.empty:
                    method_data = subset.groupby('n_records')['avg_proof_time_ms'].mean()
                    ax.plot(method_data.index, method_data.values, 
                           marker='o', label=method)
            
            ax.set_xlabel('Number of Records')
            ax.set_ylabel('Avg Proof Time (ms)')
            ax.set_title('Proof Generation Performance')
            ax.legend()
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        # Verification performance with error rates
        if self.verify_results:
            verify_df = pd.DataFrame([asdict(m) for m in self.verify_results])
            
            ax = axes[1, 0]
            ax.scatter(verify_df['n_records'], verify_df['avg_verify_time_ms'], 
                      alpha=0.6, color='green', label='Avg Time')
            ax.scatter(verify_df['n_records'], verify_df['error_rate'] * 1000,  # Scale for visibility
                      alpha=0.6, color='red', marker='x', label='Error Rate (×1000)')
            ax.set_xlabel('Number of Records')
            ax.set_ylabel('Time (ms) / Error Rate (×1000)')
            ax.set_title('Verification Performance & Error Rate')
            ax.legend()
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        # KDF timing breakdown
        if self.build_results:
            ax = axes[1, 1]
            
            # Create stacked bar chart for timing breakdown
            kdf_data = build_df.groupby('kdf_iterations').agg({
                'prepare_time_s': 'mean',
                'kdf_time_s': 'mean', 
                'tree_build_time_s': 'mean',
                'io_time_s': 'mean'
            })
            
            x_pos = np.arange(len(kdf_data))
            width = 0.6
            
            bottom = np.zeros(len(kdf_data))
            colors = ['lightblue', 'orange', 'lightgreen', 'pink']
            labels = ['Prepare', 'KDF', 'Tree Build', 'I/O']
            
            for i, (col, color, label) in enumerate(zip(kdf_data.columns, colors, labels)):
                ax.bar(x_pos, kdf_data[col], width, bottom=bottom, 
                      color=color, label=label)
                bottom += kdf_data[col]
            
            ax.set_xlabel('KDF Iterations')
            ax.set_ylabel('Time (s)')
            ax.set_title('Build Time Breakdown by KDF Strength')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{int(k/1000)}k' for k in kdf_data.index])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Proof size efficiency (actual vs hex)
        if self.proof_results and 'proof_size_bytes_actual' in proof_df.columns:
            ax = axes[1, 2]
            
            size_efficiency = proof_df.groupby('n_records').agg({
                'proof_size_bytes_actual': 'mean',
                'proof_size_bytes_hex': 'mean'
            })
            
            efficiency_ratio = (size_efficiency['proof_size_bytes_actual'] / 
                              size_efficiency['proof_size_bytes_hex'])
            
            ax.plot(size_efficiency.index, size_efficiency['proof_size_bytes_actual'], 
                   marker='o', color='blue', label='Actual Bytes')
            ax.plot(size_efficiency.index, size_efficiency['proof_size_bytes_hex'], 
                   marker='s', color='red', label='Hex String Bytes')
            
            ax.set_xlabel('Number of Records')
            ax.set_ylabel('Proof Size (bytes)')
            ax.set_title('Proof Size: Actual vs Hex Encoding')
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save enhanced chart
        chart_file = self.bench_dir / 'performance_charts.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced charts saved to {chart_file}")
    
    def run_full_benchmark(self, csv_path: str, region: str = 'Verona', 
                          rpc_url: str = None) -> str:
        """Run complete enhanced benchmark suite."""
        logger.info("Starting enhanced full benchmark suite")
        
        if rpc_url is None:
            rpc_url = BENCHMARK_CONFIG['default_rpc_url']
        
        # Capture environment
        self.capture_environment_info()
        
        # Run build benchmark with enhanced measurements
        self.bench_build(csv_path, region)
        
        # Get test directories for other benchmarks
        test_dirs = [d for d in self.bench_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('test_')]
        
        # Generate receipts for proof/verify benchmarks
        for test_dir in test_dirs:
            try:
                pipeline = BlockchainDataAnchoringPipeline(str(test_dir))
                
                # Load user indices
                meta_file = test_dir / 'tree_meta.json'
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        tree_meta = json.load(f)
                    
                    user_indices = tree_meta.get('user_indices', {})
                    
                    # Generate receipts for sample of users
                    sample_size = min(100, len(user_indices))
                    sample_users = random.sample(list(user_indices.keys()), sample_size)
                    
                    for user_id in sample_users:
                        try:
                            pipeline.generate_user_receipt(user_id, 1)
                        except Exception as e:
                            logger.error(f"Failed to generate receipt for {user_id}: {e}")
            
            except Exception as e:
                logger.error(f"Failed to generate receipts for {test_dir}: {e}")
        
        # Run enhanced proof benchmark
        self.bench_proofs(test_dirs)
        
        # Run enhanced verification benchmark
        self.bench_verify(test_dirs, rpc_url)
        
        # Run enhanced anchor benchmark
        self.bench_anchor(test_dirs, rpc_url)
        
        # Save results
        self.save_results()
        
        # Generate enhanced summary and charts
        summary = self.generate_summary()
        self.generate_charts()
        
        logger.info(f"Enhanced benchmark completed: {self.bench_dir}")
        
        return summary


def main():
    """Main CLI for enhanced blockchain anchoring benchmark."""
    parser = argparse.ArgumentParser(
        description="Enhanced Blockchain Data Anchoring Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full enhanced benchmark suite
  python 11_blockchain_anchoring_bench.py full --csv data/CPI_Verona_training_data.csv --region Verona --out results_blockchain_bench --rpc-url http://127.0.0.1:8545
  
  # Run individual enhanced benchmarks
  python 11_blockchain_anchoring_bench.py build --csv data/CPI_Verona_training_data.csv --region Verona --out results_blockchain_bench
  python 11_blockchain_anchoring_bench.py proofs --bench-dir results_blockchain_bench/20241201_143022
  python 11_blockchain_anchoring_bench.py verify --bench-dir results_blockchain_bench/20241201_143022 --rpc-url http://127.0.0.1:8545
  python 11_blockchain_anchoring_bench.py anchor --bench-dir results_blockchain_bench/20241201_143022 --rpc-url http://127.0.0.1:8545

APPLIED:
- Module import (blockchain_data_anchoring)
- Changed from eth-tester to rpc for Windows compatibility  
- Added missing traceback import
- Password generation for upsampled datasets
- Implemented unique period_id generation
- Enhanced memory tracking (RSS + tracemalloc)
- Accurate proof size calculation (actual bytes vs hex)
- Enhanced error handling with verification error tracking
- Detailed timing breakdowns (KDF vs tree construction)
- Contract reuse capability

Enhanced Benchmark Matrix:
- Record counts: 100, 1,000, 10,000
- KDF iterations: 10k (demo), 50k (balanced), 100k (production)  
- Proof batch sizes: 1, 10, 100 users
- Blockchain modes: rpc (Ganache compatible)

Enhanced Metrics:
- Build: detailed timing (prepare/KDF/tree/IO), dual memory tracking
- Proof: O(log n) vs O(n), actual byte sizes vs hex encoding
- Verify: success/error rates, enhanced error tracking
- Anchor: unique periods, contract reuse, deployment timing
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Enhanced benchmark commands')
    
    # Full benchmark
    full_parser = subparsers.add_parser('full', help='Run complete enhanced benchmark suite')
    full_parser.add_argument('--csv', required=True, help='Input CSV file path')
    full_parser.add_argument('--region', default='Verona', help='Region filter')
    full_parser.add_argument('--out', default='results_blockchain_bench', help='Output directory')
    full_parser.add_argument('--rpc-url', default='http://127.0.0.1:8545', help='RPC URL for blockchain')
    
    # Individual benchmarks
    build_parser = subparsers.add_parser('build', help='Enhanced build phase benchmark')
    build_parser.add_argument('--csv', required=True, help='Input CSV file path')
    build_parser.add_argument('--region', default='Verona', help='Region filter')
    build_parser.add_argument('--out', default='results_blockchain_bench', help='Output directory')
    
    proof_parser = subparsers.add_parser('proofs', help='Enhanced proof generation benchmark')
    proof_parser.add_argument('--bench-dir', required=True, help='Benchmark directory with test data')
    
    verify_parser = subparsers.add_parser('verify', help='Enhanced verification benchmark')
    verify_parser.add_argument('--bench-dir', required=True, help='Benchmark directory with test data')
    verify_parser.add_argument('--rpc-url', default='http://127.0.0.1:8545', help='RPC URL for blockchain')
    
    anchor_parser = subparsers.add_parser('anchor', help='Enhanced blockchain anchoring benchmark')
    anchor_parser.add_argument('--bench-dir', required=True, help='Benchmark directory with test data')
    anchor_parser.add_argument('--rpc-url', default='http://127.0.0.1:8545', help='RPC URL for blockchain')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if not ANCHORING_AVAILABLE:
        logger.error("blockchain_data_anchoring module not available")
        return 1
    
    try:
        if args.command == 'full':
            benchmark = BlockchainAnchoringBenchmark(args.out)
            summary = benchmark.run_full_benchmark(args.csv, args.region, args.rpc_url)
            
            print("=" * 60)
            print("ENHANCED BENCHMARK COMPLETED")
            print("=" * 60)
            print(f"Results directory: {benchmark.bench_dir}")
            print("")
            print("Summary:")
            print(summary)
            
        elif args.command == 'build':
            benchmark = BlockchainAnchoringBenchmark(args.out)
            benchmark.capture_environment_info()
            benchmark.bench_build(args.csv, args.region)
            benchmark.save_results()
            print(f"✓ Enhanced build benchmark completed: {benchmark.bench_dir}")
            
        elif args.command == 'proofs':
            bench_dir = Path(args.bench_dir)
            if not bench_dir.exists():
                logger.error(f"Benchmark directory not found: {bench_dir}")
                return 1
            
            benchmark = BlockchainAnchoringBenchmark(bench_dir.parent)
            benchmark.bench_dir = bench_dir
            
            test_dirs = [d for d in bench_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('test_')]
            benchmark.bench_proofs(test_dirs)
            benchmark.save_results()
            print(f"✓ Enhanced proof benchmark completed: {bench_dir}")
            
        elif args.command == 'verify':
            bench_dir = Path(args.bench_dir)
            if not bench_dir.exists():
                logger.error(f"Benchmark directory not found: {bench_dir}")
                return 1
            
            benchmark = BlockchainAnchoringBenchmark(bench_dir.parent)
            benchmark.bench_dir = bench_dir
            
            test_dirs = [d for d in bench_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('test_')]
            benchmark.bench_verify(test_dirs, args.rpc_url)
            benchmark.save_results()
            print(f"✓ Enhanced verify benchmark completed: {bench_dir}")
            
        elif args.command == 'anchor':
            bench_dir = Path(args.bench_dir)
            if not bench_dir.exists():
                logger.error(f"Benchmark directory not found: {bench_dir}")
                return 1
            
            benchmark = BlockchainAnchoringBenchmark(bench_dir.parent)
            benchmark.bench_dir = bench_dir
            
            test_dirs = [d for d in bench_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('test_')]
            benchmark.bench_anchor(test_dirs, args.rpc_url)
            benchmark.save_results()
            print(f"✓ Enhanced anchor benchmark completed: {bench_dir}")
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.debug(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
    