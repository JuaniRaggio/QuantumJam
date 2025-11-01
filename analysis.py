"""
Advanced analysis and visualization utilities for BB84 protocol.

This module provides tools for:
- Statistical analysis of protocol performance
- Security analysis and theoretical bounds
- Visualization of results
- Comparative analysis between protocols
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class StatisticalResults:
    """Container for statistical analysis results."""
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    confidence_interval_95: Tuple[float, float]

    def __str__(self) -> str:
        return (
            f"Mean: {self.mean:.6f} ± {self.std:.6f}\n"
            f"Median: {self.median:.6f}\n"
            f"Range: [{self.min_val:.6f}, {self.max_val:.6f}]\n"
            f"95% CI: [{self.confidence_interval_95[0]:.6f}, {self.confidence_interval_95[1]:.6f}]"
        )


class ProtocolAnalyzer:
    """
    Advanced analyzer for BB84 protocol performance and security.

    This class provides comprehensive statistical and security analysis
    tools for evaluating BB84 protocol execution.
    """

    @staticmethod
    def binary_entropy(p: float) -> float:
        """
        Calculate binary entropy function.

        Args:
            p: Probability (0 to 1)

        Returns:
            Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
        """
        if p == 0 or p == 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    @staticmethod
    def mutual_information(qber: float) -> float:
        """
        Calculate mutual information between Alice and Bob.

        Args:
            qber: Quantum bit error rate

        Returns:
            Mutual information I(A:B) = 1 - H(QBER)
        """
        return 1.0 - ProtocolAnalyzer.binary_entropy(qber)

    @staticmethod
    def channel_capacity(qber: float) -> float:
        """
        Calculate effective channel capacity.

        Args:
            qber: Quantum bit error rate

        Returns:
            Channel capacity in bits per transmission
        """
        if qber >= 0.5:
            return 0.0
        return 1.0 - ProtocolAnalyzer.binary_entropy(qber)

    @staticmethod
    def secret_key_rate_shor_preskill(qber: float) -> float:
        """
        Calculate secret key rate using Shor-Preskill bound.

        Args:
            qber: Quantum bit error rate

        Returns:
            Secret key rate (0 if QBER > 11%)
        """
        if qber >= 0.11:
            return 0.0
        return 1.0 - 2.0 * ProtocolAnalyzer.binary_entropy(qber)

    @staticmethod
    def secret_key_rate_lossy_channel(qber: float, eta: float) -> float:
        """
        Calculate secret key rate for lossy channel (GLLP bound).

        Args:
            qber: Quantum bit error rate
            eta: Channel transmission efficiency (0 to 1)

        Returns:
            Secret key rate accounting for losses
        """
        if qber >= 0.11 or eta <= 0:
            return 0.0

        h_qber = ProtocolAnalyzer.binary_entropy(qber)
        return -np.log2(1 - eta) * (1 - h_qber) - h_qber

    @staticmethod
    def calculate_statistics(data: np.ndarray) -> StatisticalResults:
        """
        Calculate comprehensive statistics for a dataset.

        Args:
            data: Numpy array of values

        Returns:
            StatisticalResults object
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        median = np.median(data)
        min_val = np.min(data)
        max_val = np.max(data)

        # 95% confidence interval
        confidence_level = 0.95
        degrees_freedom = len(data) - 1
        confidence_interval = stats.t.interval(
            confidence_level,
            degrees_freedom,
            loc=mean,
            scale=stats.sem(data)
        )

        return StatisticalResults(
            mean=mean,
            std=std,
            median=median,
            min_val=min_val,
            max_val=max_val,
            confidence_interval_95=confidence_interval
        )

    @staticmethod
    def estimate_eve_information(qber: float) -> float:
        """
        Estimate maximum information Eve could have obtained.

        Args:
            qber: Measured quantum bit error rate

        Returns:
            Upper bound on Eve's information (bits)
        """
        # Conservative estimate: Eve has full info on error-causing qubits
        return ProtocolAnalyzer.binary_entropy(qber)

    @staticmethod
    def required_privacy_amplification(qber: float, security_parameter: float = 1e-10) -> float:
        """
        Calculate required privacy amplification compression.

        Args:
            qber: Quantum bit error rate
            security_parameter: Desired security level (epsilon)

        Returns:
            Compression ratio needed for privacy amplification
        """
        if qber >= 0.11:
            return 0.0

        eve_info = ProtocolAnalyzer.estimate_eve_information(qber)
        security_overhead = -np.log2(security_parameter) / 100  # Simplified

        compression = 1.0 - eve_info - security_overhead
        return max(0.0, compression)


class ProtocolVisualizer:
    """
    Visualization tools for BB84 protocol analysis.

    Provides professional plotting functions for protocol metrics,
    security analysis, and comparative studies.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with matplotlib style.

        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)

    def plot_qber_vs_noise(
        self,
        noise_levels: np.ndarray,
        qber_mean: np.ndarray,
        qber_std: np.ndarray,
        title: str = "QBER vs Channel Noise",
        show_threshold: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot QBER against noise levels with confidence intervals.

        Args:
            noise_levels: Array of noise levels
            qber_mean: Mean QBER values
            qber_std: Standard deviation of QBER
            title: Plot title
            show_threshold: Whether to show 11% security threshold
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(noise_levels * 100, qber_mean * 100, 'o-',
                linewidth=2.5, markersize=8, label='Measured QBER',
                color=self.colors[0])

        ax.fill_between(noise_levels * 100,
                        (qber_mean - qber_std) * 100,
                        (qber_mean + qber_std) * 100,
                        alpha=0.3, color=self.colors[0],
                        label='Standard deviation')

        if show_threshold:
            ax.axhline(y=11, color='red', linestyle='--',
                      linewidth=2, label='Security threshold (11%)')

        ax.set_xlabel('Channel Noise Level (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('QBER (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_key_rate_comparison(
        self,
        qber_values: np.ndarray,
        theoretical_skr: np.ndarray,
        actual_skr: np.ndarray,
        title: str = "Secret Key Rate: Theoretical vs Actual",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare theoretical and actual secret key rates.

        Args:
            qber_values: Array of QBER values
            theoretical_skr: Theoretical secret key rates
            actual_skr: Actual measured secret key rates
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(qber_values * 100, theoretical_skr,
                'o-', linewidth=2.5, markersize=8,
                label='Theoretical (Shor-Preskill)',
                color=self.colors[1])

        ax.plot(qber_values * 100, actual_skr,
                's-', linewidth=2.5, markersize=8,
                label='Actual (measured)',
                color=self.colors[2])

        ax.set_xlabel('QBER (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Secret Key Rate (bits/qubit)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_protocol_metrics_dashboard(
        self,
        metrics_data: Dict[str, np.ndarray],
        parameter_name: str = "Noise Level",
        parameter_unit: str = "%",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive dashboard of protocol metrics.

        Args:
            metrics_data: Dictionary with metrics arrays
            parameter_name: Name of the varied parameter
            parameter_unit: Unit of the parameter
            save_path: Path to save figure

        Returns:
            Matplotlib figure with subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BB84 Protocol Performance Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)

        parameter = metrics_data.get('parameter', np.array([]))

        # Subplot 1: QBER
        if 'qber_mean' in metrics_data:
            ax = axes[0, 0]
            ax.plot(parameter, metrics_data['qber_mean'] * 100,
                   'o-', linewidth=2, markersize=6, color=self.colors[0])
            if 'qber_std' in metrics_data:
                ax.fill_between(parameter,
                              (metrics_data['qber_mean'] - metrics_data['qber_std']) * 100,
                              (metrics_data['qber_mean'] + metrics_data['qber_std']) * 100,
                              alpha=0.3, color=self.colors[0])
            ax.axhline(y=11, color='red', linestyle='--', linewidth=1.5)
            ax.set_xlabel(f'{parameter_name} ({parameter_unit})', fontweight='bold')
            ax.set_ylabel('QBER (%)', fontweight='bold')
            ax.set_title('Quantum Bit Error Rate', fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Subplot 2: Key Rate
        if 'key_rate_mean' in metrics_data:
            ax = axes[0, 1]
            ax.plot(parameter, metrics_data['key_rate_mean'] * 100,
                   's-', linewidth=2, markersize=6, color=self.colors[1])
            if 'key_rate_std' in metrics_data:
                ax.fill_between(parameter,
                              (metrics_data['key_rate_mean'] - metrics_data['key_rate_std']) * 100,
                              (metrics_data['key_rate_mean'] + metrics_data['key_rate_std']) * 100,
                              alpha=0.3, color=self.colors[1])
            ax.set_xlabel(f'{parameter_name} ({parameter_unit})', fontweight='bold')
            ax.set_ylabel('Key Rate (%)', fontweight='bold')
            ax.set_title('Final Key Generation Rate', fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Subplot 3: Key Length
        if 'key_length_mean' in metrics_data:
            ax = axes[1, 0]
            ax.plot(parameter, metrics_data['key_length_mean'],
                   '^-', linewidth=2, markersize=6, color=self.colors[2])
            if 'key_length_std' in metrics_data:
                ax.fill_between(parameter,
                              metrics_data['key_length_mean'] - metrics_data['key_length_std'],
                              metrics_data['key_length_mean'] + metrics_data['key_length_std'],
                              alpha=0.3, color=self.colors[2])
            ax.set_xlabel(f'{parameter_name} ({parameter_unit})', fontweight='bold')
            ax.set_ylabel('Key Length (bits)', fontweight='bold')
            ax.set_title('Final Key Length', fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Subplot 4: Security Status
        if 'security_fraction' in metrics_data:
            ax = axes[1, 1]
            ax.bar(parameter, metrics_data['security_fraction'] * 100,
                  width=(parameter[1] - parameter[0]) * 0.8 if len(parameter) > 1 else 1,
                  color=self.colors[3], alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'{parameter_name} ({parameter_unit})', fontweight='bold')
            ax.set_ylabel('Secure Runs (%)', fontweight='bold')
            ax.set_title('Security Success Rate', fontweight='bold')
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_eavesdropping_analysis(
        self,
        intercept_rates: np.ndarray,
        qber_values: np.ndarray,
        detection_rate: np.ndarray,
        title: str = "Eavesdropping Attack Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize impact of eavesdropping attacks.

        Args:
            intercept_rates: Array of intercept rates
            qber_values: Corresponding QBER values
            detection_rate: Fraction of runs where attack was detected
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Plot 1: QBER vs Intercept Rate
        ax1.plot(intercept_rates * 100, qber_values * 100,
                'o-', linewidth=2.5, markersize=8,
                color='darkred', label='Measured QBER')
        ax1.axhline(y=11, color='orange', linestyle='--',
                   linewidth=2, label='Detection threshold (11%)')

        # Theoretical line
        theoretical_qber = intercept_rates * 25  # 25% QBER at 100% intercept
        ax1.plot(intercept_rates * 100, theoretical_qber,
                '--', linewidth=2, color='gray',
                label='Theoretical (intercept_rate × 25%)')

        ax1.set_xlabel('Intercept Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('QBER (%)', fontsize=12, fontweight='bold')
        ax1.set_title('QBER vs Eavesdropping Rate', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Detection Rate
        colors = ['green' if d > 0.5 else 'orange' for d in detection_rate]
        ax2.bar(intercept_rates * 100, detection_rate * 100,
               width=4, color=colors, alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Intercept Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Attack Detected (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Eavesdropping Detection Success', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_protocol_comparison(
        self,
        protocols: Dict[str, Dict[str, float]],
        metrics: List[str] = ['qber', 'key_rate', 'key_length'],
        title: str = "Protocol Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare different protocol configurations or types.

        Args:
            protocols: Dictionary mapping protocol names to their metrics
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight='bold')

        protocol_names = list(protocols.keys())
        x_pos = np.arange(len(protocol_names))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [protocols[p].get(metric, 0) for p in protocol_names]

            bars = ax.bar(x_pos, values, color=self.colors[:len(protocol_names)],
                         alpha=0.7, edgecolor='black', linewidth=1.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(protocol_names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class NoiseModelAnalyzer:
    """
    Analyzer for different quantum noise models and their impact.

    Provides tools to analyze and compare various noise models:
    - Depolarizing noise
    - Amplitude damping
    - Phase damping
    - Custom noise combinations
    """

    @staticmethod
    def analyze_depolarizing_noise(p: float, n_qubits: int = 1) -> Dict:
        """
        Analyze impact of depolarizing noise.

        Args:
            p: Depolarizing probability
            n_qubits: Number of qubits

        Returns:
            Dictionary with noise analysis
        """
        # Depolarizing channel: ρ → (1-p)ρ + p*I/2^n
        fidelity = 1 - p * (1 - 1 / (2**n_qubits))

        # Expected error rate
        expected_qber = p / 2

        return {
            'noise_type': 'depolarizing',
            'probability': p,
            'fidelity': fidelity,
            'expected_qber': expected_qber,
            'description': f'Depolarizing noise with p={p:.4f}'
        }

    @staticmethod
    def analyze_amplitude_damping(gamma: float) -> Dict:
        """
        Analyze impact of amplitude damping (energy dissipation).

        Args:
            gamma: Damping parameter

        Returns:
            Dictionary with noise analysis
        """
        # Amplitude damping: models energy loss
        # |1⟩ can decay to |0⟩ with probability gamma

        # For |+⟩ state, expected QBER
        expected_qber = gamma / 4

        return {
            'noise_type': 'amplitude_damping',
            'gamma': gamma,
            'expected_qber': expected_qber,
            'description': f'Amplitude damping with γ={gamma:.4f}'
        }

    @staticmethod
    def compare_noise_models(
        noise_params: List[Tuple[str, float]],
        n_simulations: int = 100
    ) -> Dict:
        """
        Compare different noise models.

        Args:
            noise_params: List of (noise_type, parameter) tuples
            n_simulations: Number of simulations per model

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for noise_type, param in noise_params:
            if noise_type == 'depolarizing':
                results[f'depol_p={param:.3f}'] = (
                    NoiseModelAnalyzer.analyze_depolarizing_noise(param)
                )
            elif noise_type == 'amplitude_damping':
                results[f'damping_γ={param:.3f}'] = (
                    NoiseModelAnalyzer.analyze_amplitude_damping(param)
                )

        return results
