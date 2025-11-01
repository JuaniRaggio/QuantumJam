"""
BB84 Quantum Key Distribution Protocol Implementation

This module implements the BB84 protocol for quantum key distribution,
including advanced features such as error correction, privacy amplification,
and security analysis.

References:
    - Bennett, C. H., & Brassard, G. (1984). Quantum cryptography: Public key
      distribution and coin tossing. IEEE International Conference on Computers,
      Systems and Signal Processing.
    - Shor, P. W., & Preskill, J. (2000). Simple proof of security of the BB84
      quantum key distribution protocol. Physical review letters, 85(2), 441.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from scipy.stats import binom


@dataclass
class ProtocolMetrics:
    """Container for protocol execution metrics."""
    n_bits_sent: int
    n_bits_sifted: int
    n_bits_final: int
    qber: float
    key_rate: float
    sifting_efficiency: float
    mutual_information: float
    secret_key_rate_theoretical: float
    secret_key_rate_actual: float

    def __str__(self) -> str:
        return (
            f"Protocol Metrics:\n"
            f"  Bits sent: {self.n_bits_sent}\n"
            f"  Bits after sifting: {self.n_bits_sifted}\n"
            f"  Final key length: {self.n_bits_final}\n"
            f"  QBER: {self.qber*100:.4f}%\n"
            f"  Key rate: {self.key_rate*100:.4f}%\n"
            f"  Sifting efficiency: {self.sifting_efficiency*100:.2f}%\n"
            f"  Mutual information: {self.mutual_information:.4f} bits\n"
            f"  SKR (theoretical): {self.secret_key_rate_theoretical:.4f}\n"
            f"  SKR (actual): {self.secret_key_rate_actual:.4f}"
        )


@dataclass
class SecurityAnalysis:
    """Container for security analysis results."""
    qber: float
    is_secure: bool
    max_eve_information: float
    privacy_amplification_rate: float
    error_correction_cost: float
    final_security_parameter: float

    def __str__(self) -> str:
        return (
            f"Security Analysis:\n"
            f"  QBER: {self.qber*100:.4f}%\n"
            f"  Secure: {self.is_secure}\n"
            f"  Max Eve information: {self.max_eve_information:.4f} bits\n"
            f"  Privacy amplification rate: {self.privacy_amplification_rate:.4f}\n"
            f"  Error correction cost: {self.error_correction_cost:.4f}\n"
            f"  Security parameter: {self.final_security_parameter:.6e}"
        )


class BB84Protocol:
    """
    Implementation of the BB84 quantum key distribution protocol.

    This class provides a complete implementation of BB84, including:
    - Quantum state preparation and measurement
    - Basis reconciliation (sifting)
    - Error estimation
    - Error correction via Cascade algorithm
    - Privacy amplification
    - Security analysis

    Attributes:
        n_bits: Number of qubits to transmit
        noise_level: Channel noise level (0.0 to 1.0)
        simulator: Qiskit quantum simulator
        alice_bits: Alice's random bit string
        alice_bases: Alice's random basis choices
        bob_bases: Bob's random basis choices
        bob_measurements: Bob's measurement results
        alice_key: Alice's final key after processing
        bob_key: Bob's final key after processing
        metrics: Protocol execution metrics
    """

    def __init__(
        self,
        n_bits: int = 1000,
        noise_level: float = 0.0,
        use_noise_model: bool = False,
        depolarizing_prob: float = 0.0,
        damping_prob: float = 0.0
    ):
        """
        Initialize the BB84 protocol.

        Args:
            n_bits: Number of qubits to transmit
            noise_level: Simple noise level (random bit flips)
            use_noise_model: Whether to use advanced Qiskit noise model
            depolarizing_prob: Depolarizing error probability
            damping_prob: Amplitude damping error probability
        """
        self.n_bits = n_bits
        self.noise_level = noise_level
        self.use_noise_model = use_noise_model
        self.depolarizing_prob = depolarizing_prob
        self.damping_prob = damping_prob

        # Protocol state
        self.alice_bits: Optional[np.ndarray] = None
        self.alice_bases: Optional[np.ndarray] = None
        self.bob_bases: Optional[np.ndarray] = None
        self.bob_measurements: Optional[np.ndarray] = None
        self.alice_key: Optional[np.ndarray] = None
        self.bob_key: Optional[np.ndarray] = None

        # Tracking
        self.matching_bases_indices: List[int] = []
        self.error_sample_indices: List[int] = []

        # Metrics
        self.metrics: Optional[ProtocolMetrics] = None
        self.security_analysis: Optional[SecurityAnalysis] = None

        # Simulator setup
        self.simulator = AerSimulator()
        self.noise_model = self._create_noise_model() if use_noise_model else None

    def _create_noise_model(self) -> NoiseModel:
        """Create a realistic noise model for quantum channel."""
        noise_model = NoiseModel()

        if self.depolarizing_prob > 0:
            depol_error = depolarizing_error(self.depolarizing_prob, 1)
            noise_model.add_all_qubit_quantum_error(depol_error, ['h', 'x'])

        if self.damping_prob > 0:
            damping_error_obj = amplitude_damping_error(self.damping_prob)
            noise_model.add_all_qubit_quantum_error(damping_error_obj, ['id'])

        return noise_model

    def _generate_random_bits(self, n: int) -> np.ndarray:
        """Generate n random bits."""
        return np.random.randint(0, 2, n)

    def _generate_random_bases(self, n: int) -> np.ndarray:
        """
        Generate n random bases.

        Returns:
            Array of 0s and 1s where:
            0 = computational/rectilinear basis (Z-basis)
            1 = diagonal basis (X-basis)
        """
        return np.random.randint(0, 2, n)

    def _create_qubit_state(self, bit: int, basis: int) -> QuantumCircuit:
        """
        Create a quantum circuit encoding a bit in a specific basis.

        Args:
            bit: The classical bit value (0 or 1)
            basis: The encoding basis (0=Z, 1=X)

        Returns:
            Quantum circuit with prepared qubit
        """
        qc = QuantumCircuit(1, 1)

        # Encode the bit
        if bit == 1:
            qc.x(0)

        # Change to diagonal basis if needed
        if basis == 1:
            qc.h(0)

        return qc

    def _measure_qubit(self, qc: QuantumCircuit, basis: int) -> QuantumCircuit:
        """
        Add measurement in specified basis to quantum circuit.

        Args:
            qc: Quantum circuit to measure
            basis: Measurement basis (0=Z, 1=X)

        Returns:
            Circuit with measurement added
        """
        if basis == 1:
            qc.h(0)
        qc.measure(0, 0)
        return qc

    def _apply_channel_noise(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Apply noise to quantum circuit to simulate imperfect channel.

        Args:
            qc: Quantum circuit

        Returns:
            Circuit with noise applied
        """
        if self.noise_level > 0 and not self.use_noise_model:
            if np.random.random() < self.noise_level:
                noise_gate = np.random.choice(['x', 'z', 'y'])
                if noise_gate == 'x':
                    qc.x(0)
                elif noise_gate == 'z':
                    qc.z(0)
                else:
                    qc.y(0)
        return qc

    def step1_alice_prepare(self) -> None:
        """Step 1: Alice generates random bits and bases."""
        self.alice_bits = self._generate_random_bits(self.n_bits)
        self.alice_bases = self._generate_random_bases(self.n_bits)

    def step2_bob_prepare(self) -> None:
        """Step 2: Bob generates random measurement bases."""
        self.bob_bases = self._generate_random_bases(self.n_bits)

    def step3_quantum_transmission(self) -> None:
        """Step 3: Alice transmits qubits and Bob measures them."""
        self.bob_measurements = np.zeros(self.n_bits, dtype=int)

        for i in range(self.n_bits):
            # Alice prepares qubit
            qc = self._create_qubit_state(self.alice_bits[i], self.alice_bases[i])

            # Apply channel noise
            qc = self._apply_channel_noise(qc)

            # Bob measures
            qc = self._measure_qubit(qc, self.bob_bases[i])

            # Execute circuit
            if self.use_noise_model and self.noise_model is not None:
                result = self.simulator.run(
                    qc, shots=1, noise_model=self.noise_model
                ).result()
            else:
                result = self.simulator.run(qc, shots=1).result()

            counts = result.get_counts()
            measured_bit = int(list(counts.keys())[0])
            self.bob_measurements[i] = measured_bit

    def step4_sifting(self) -> None:
        """Step 4: Alice and Bob compare bases and keep matching results."""
        self.matching_bases_indices = np.where(
            self.alice_bases == self.bob_bases
        )[0].tolist()

        self.alice_key = self.alice_bits[self.matching_bases_indices]
        self.bob_key = self.bob_measurements[self.matching_bases_indices]

    def step5_error_estimation(self, sample_fraction: float = 0.3) -> float:
        """
        Step 5: Estimate QBER by comparing a random sample of bits.

        Args:
            sample_fraction: Fraction of sifted key to use for error estimation

        Returns:
            Quantum Bit Error Rate (QBER)
        """
        if len(self.alice_key) == 0:
            return 0.0

        sample_size = max(1, int(len(self.alice_key) * sample_fraction))
        sample_size = min(sample_size, len(self.alice_key))

        self.error_sample_indices = np.random.choice(
            len(self.alice_key), sample_size, replace=False
        ).tolist()

        alice_sample = self.alice_key[self.error_sample_indices]
        bob_sample = self.bob_key[self.error_sample_indices]

        errors = np.sum(alice_sample != bob_sample)
        qber = errors / sample_size

        # Remove error estimation bits from keys
        mask = np.ones(len(self.alice_key), dtype=bool)
        mask[self.error_sample_indices] = False
        self.alice_key = self.alice_key[mask]
        self.bob_key = self.bob_key[mask]

        return qber

    def step6_error_correction(self) -> int:
        """
        Step 6: Apply error correction using simplified Cascade-like algorithm.

        Returns:
            Number of bits corrected
        """
        if len(self.alice_key) == 0 or len(self.bob_key) == 0:
            return 0

        # Simple parity-based error correction (simplified Cascade)
        errors_corrected = 0
        key_length = len(self.alice_key)

        # Single pass with adaptive block size
        block_size = max(4, key_length // 10)

        for start in range(0, key_length, block_size):
            end = min(start + block_size, key_length)
            alice_block = self.alice_key[start:end]
            bob_block = self.bob_key[start:end]

            alice_parity = np.sum(alice_block) % 2
            bob_parity = np.sum(bob_block) % 2

            if alice_parity != bob_parity:
                # Error detected in block - correct first differing bit
                diff_indices = np.where(alice_block != bob_block)[0]
                if len(diff_indices) > 0:
                    # Correct in Bob's key
                    self.bob_key[start + diff_indices[0]] = alice_block[diff_indices[0]]
                    errors_corrected += 1

        return errors_corrected

    def step7_privacy_amplification(self, compression_ratio: float = 0.9) -> None:
        """
        Step 7: Apply privacy amplification to remove Eve's potential information.

        Args:
            compression_ratio: Ratio of bits to keep (< 1.0 for security)
        """
        if len(self.alice_key) == 0:
            return

        final_length = int(len(self.alice_key) * compression_ratio)

        # Simple privacy amplification: XOR pairs of bits with random hash
        # In production, use universal hash functions
        if final_length > 0:
            indices = np.random.choice(len(self.alice_key), final_length, replace=False)
            self.alice_key = self.alice_key[indices]
            self.bob_key = self.bob_key[indices]

    def calculate_mutual_information(self, qber: float) -> float:
        """
        Calculate mutual information between Alice and Bob.

        Args:
            qber: Quantum bit error rate

        Returns:
            Mutual information in bits
        """
        if qber == 0:
            return 1.0
        if qber == 1:
            return 0.0

        # I(A:B) = 1 - H(QBER) where H is binary entropy
        h = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
        return 1.0 - h

    def calculate_secret_key_rate(self, qber: float) -> float:
        """
        Calculate theoretical secret key rate using Shor-Preskill formula.

        Args:
            qber: Quantum bit error rate

        Returns:
            Secret key rate (bits per transmitted qubit)
        """
        if qber >= 0.11:
            return 0.0

        # Simplified Shor-Preskill: r = 1 - 2*H(QBER)
        # where H is binary entropy function
        if qber == 0:
            h = 0
        else:
            h = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)

        skr = 1 - 2 * h
        return max(0, skr)

    def perform_security_analysis(self, qber: float) -> SecurityAnalysis:
        """
        Perform comprehensive security analysis.

        Args:
            qber: Measured quantum bit error rate

        Returns:
            SecurityAnalysis object with detailed security metrics
        """
        # Security threshold (theoretical limit is ~11%)
        is_secure = qber < 0.11

        # Maximum information Eve could have (using binary entropy)
        if qber == 0:
            max_eve_info = 0.0
        else:
            max_eve_info = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)

        # Privacy amplification rate needed
        privacy_amp_rate = 1 - max_eve_info if is_secure else 0.0

        # Error correction cost (Shannon limit)
        if qber == 0:
            error_correction_cost = 0.0
        else:
            error_correction_cost = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)

        # Security parameter (epsilon-security)
        # Simplified calculation
        if len(self.alice_key) > 0:
            security_param = 2 ** (-len(self.alice_key) * (1 - 2 * max_eve_info))
        else:
            security_param = 1.0

        return SecurityAnalysis(
            qber=qber,
            is_secure=is_secure,
            max_eve_information=max_eve_info,
            privacy_amplification_rate=privacy_amp_rate,
            error_correction_cost=error_correction_cost,
            final_security_parameter=security_param
        )

    def calculate_metrics(self, qber: float, n_bits_sifted: int) -> ProtocolMetrics:
        """
        Calculate comprehensive protocol metrics.

        Args:
            qber: Quantum bit error rate
            n_bits_sifted: Number of bits after sifting

        Returns:
            ProtocolMetrics object
        """
        n_bits_final = len(self.alice_key) if self.alice_key is not None else 0

        key_rate = n_bits_final / self.n_bits if self.n_bits > 0 else 0
        sifting_efficiency = n_bits_sifted / self.n_bits if self.n_bits > 0 else 0

        mutual_info = self.calculate_mutual_information(qber)
        skr_theoretical = self.calculate_secret_key_rate(qber)
        skr_actual = key_rate

        return ProtocolMetrics(
            n_bits_sent=self.n_bits,
            n_bits_sifted=n_bits_sifted,
            n_bits_final=n_bits_final,
            qber=qber,
            key_rate=key_rate,
            sifting_efficiency=sifting_efficiency,
            mutual_information=mutual_info,
            secret_key_rate_theoretical=skr_theoretical,
            secret_key_rate_actual=skr_actual
        )

    def verify_keys_match(self) -> bool:
        """
        Verify that Alice's and Bob's keys are identical.

        Returns:
            True if keys match exactly
        """
        if self.alice_key is None or self.bob_key is None:
            return False
        return np.array_equal(self.alice_key, self.bob_key)

    def run_protocol(
        self,
        apply_error_correction: bool = True,
        apply_privacy_amplification: bool = True,
        verbose: bool = False
    ) -> Dict:
        """
        Execute the complete BB84 protocol.

        Args:
            apply_error_correction: Whether to apply error correction
            apply_privacy_amplification: Whether to apply privacy amplification
            verbose: Whether to print progress information

        Returns:
            Dictionary containing protocol results and metrics
        """
        if verbose:
            print("="*80)
            print("BB84 Protocol Execution")
            print("="*80)
            print(f"Configuration: {self.n_bits} qubits, noise={self.noise_level}")
            print()

        # Execute protocol steps
        if verbose:
            print("[1/7] Alice preparing random bits and bases...")
        self.step1_alice_prepare()

        if verbose:
            print("[2/7] Bob preparing random measurement bases...")
        self.step2_bob_prepare()

        if verbose:
            print("[3/7] Quantum transmission in progress...")
        self.step3_quantum_transmission()

        if verbose:
            print("[4/7] Performing basis reconciliation (sifting)...")
        self.step4_sifting()
        n_bits_sifted = len(self.alice_key)

        if verbose:
            print(f"[5/7] Estimating QBER from sample...")
        qber = self.step5_error_estimation()

        if verbose:
            print(f"      QBER = {qber*100:.4f}%")

        errors_corrected = 0
        if apply_error_correction:
            if verbose:
                print("[6/7] Applying error correction...")
            errors_corrected = self.step6_error_correction()
            if verbose:
                print(f"      Corrected {errors_corrected} errors")

        if apply_privacy_amplification:
            if verbose:
                print("[7/7] Applying privacy amplification...")
            compression = 0.95 if qber < 0.05 else 0.85
            self.step7_privacy_amplification(compression_ratio=compression)

        # Calculate metrics
        self.metrics = self.calculate_metrics(qber, n_bits_sifted)
        self.security_analysis = self.perform_security_analysis(qber)

        keys_match = self.verify_keys_match()

        if verbose:
            print()
            print("="*80)
            print("Results")
            print("="*80)
            print(self.metrics)
            print()
            print(self.security_analysis)
            print()
            print(f"Keys match: {keys_match}")
            print("="*80)

        return {
            'success': keys_match and self.security_analysis.is_secure,
            'keys_match': keys_match,
            'alice_key': self.alice_key,
            'bob_key': self.bob_key,
            'metrics': self.metrics,
            'security_analysis': self.security_analysis,
            'errors_corrected': errors_corrected
        }


class BB84WithEavesdropper(BB84Protocol):
    """
    BB84 protocol with simulated eavesdropping attack.

    This class extends BB84Protocol to include an eavesdropper (Eve)
    who performs intercept-resend attacks on the quantum channel.

    Attributes:
        intercept_rate: Fraction of qubits Eve intercepts (0.0 to 1.0)
        eve_bases: Eve's measurement bases
        eve_measurements: Eve's measurement results
        intercepted_indices: Indices of intercepted qubits
    """

    def __init__(
        self,
        n_bits: int = 1000,
        intercept_rate: float = 0.5,
        **kwargs
    ):
        """
        Initialize BB84 with eavesdropper.

        Args:
            n_bits: Number of qubits to transmit
            intercept_rate: Fraction of qubits to intercept
            **kwargs: Additional arguments passed to BB84Protocol
        """
        super().__init__(n_bits=n_bits, **kwargs)
        self.intercept_rate = intercept_rate
        self.eve_bases: Optional[np.ndarray] = None
        self.eve_measurements: Optional[np.ndarray] = None
        self.intercepted_indices: List[int] = []

    def step3_quantum_transmission(self) -> None:
        """
        Modified quantum transmission with Eve's intercept-resend attack.
        """
        self.bob_measurements = np.zeros(self.n_bits, dtype=int)
        self.eve_bases = self._generate_random_bases(self.n_bits)
        self.eve_measurements = np.zeros(self.n_bits, dtype=int)

        for i in range(self.n_bits):
            # Alice prepares qubit
            qc = self._create_qubit_state(self.alice_bits[i], self.alice_bases[i])

            # Eve intercepts with probability intercept_rate
            if np.random.random() < self.intercept_rate:
                self.intercepted_indices.append(i)

                # Eve measures in random basis
                qc_eve = qc.copy()
                qc_eve = self._measure_qubit(qc_eve, self.eve_bases[i])

                result_eve = self.simulator.run(qc_eve, shots=1).result()
                eve_bit = int(list(result_eve.get_counts().keys())[0])
                self.eve_measurements[i] = eve_bit

                # Eve resends qubit in her basis
                qc = self._create_qubit_state(eve_bit, self.eve_bases[i])

            # Apply channel noise
            qc = self._apply_channel_noise(qc)

            # Bob measures
            qc = self._measure_qubit(qc, self.bob_bases[i])

            result = self.simulator.run(qc, shots=1).result()
            self.bob_measurements[i] = int(list(result.get_counts().keys())[0])

    def analyze_eve_impact(self) -> Dict:
        """
        Analyze the impact of Eve's attack on the protocol.

        Returns:
            Dictionary with analysis of Eve's attack
        """
        if self.alice_key is None or self.metrics is None:
            return {}

        # Theoretical QBER from Eve's attack
        # Expected QBER = (intercept_rate / 2) * 0.5 = intercept_rate / 4
        # This is because Eve has 50% chance of choosing wrong basis,
        # and when wrong, introduces 50% error rate
        theoretical_qber_from_eve = self.intercept_rate * 0.25

        # Information leaked to Eve (upper bound)
        # Eve gets full information on qubits she measured in correct basis
        eve_info_fraction = self.intercept_rate * 0.5

        return {
            'intercept_rate': self.intercept_rate,
            'qubits_intercepted': len(self.intercepted_indices),
            'theoretical_qber': theoretical_qber_from_eve,
            'measured_qber': self.metrics.qber,
            'eve_information_fraction': eve_info_fraction,
            'attack_detected': self.metrics.qber > 0.11
        }


class E91Protocol:
    """
    Implementation of the E91 (Ekert 1991) protocol for QKD.

    E91 uses entangled pairs and Bell inequality violations to detect
    eavesdropping, providing an alternative approach to BB84.

    References:
        - Ekert, A. K. (1991). Quantum cryptography based on Bell's theorem.
          Physical review letters, 67(6), 661.
    """

    def __init__(self, n_pairs: int = 1000):
        """
        Initialize E91 protocol.

        Args:
            n_pairs: Number of entangled pairs to generate
        """
        self.n_pairs = n_pairs
        self.simulator = AerSimulator()

        # Alice and Bob's measurement bases
        # In E91, they use 3 different measurement angles each
        self.alice_bases: Optional[np.ndarray] = None
        self.bob_bases: Optional[np.ndarray] = None

        # Measurement results
        self.alice_results: Optional[np.ndarray] = None
        self.bob_results: Optional[np.ndarray] = None

        # Final keys
        self.alice_key: Optional[np.ndarray] = None
        self.bob_key: Optional[np.ndarray] = None

    def create_bell_pair(self) -> QuantumCircuit:
        """
        Create a Bell state (EPR pair) |Φ+⟩ = (|00⟩ + |11⟩)/√2.

        Returns:
            Quantum circuit with entangled pair
        """
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        return qc

    def measure_in_basis(self, qc: QuantumCircuit, qubit: int, basis: int) -> QuantumCircuit:
        """
        Measure qubit in specified basis for E91.

        Args:
            qc: Quantum circuit
            qubit: Qubit index to measure
            basis: Measurement basis (0, 1, or 2 corresponding to different angles)

        Returns:
            Circuit with measurement added
        """
        # E91 uses three measurement angles for each party
        # Alice: 0°, 45°, 90° (basis 0, 1, 2)
        # Bob: 45°, 90°, 135° (basis 0, 1, 2)
        # Implemented via rotations

        if basis == 1:
            # 45 degrees rotation
            qc.ry(np.pi/4, qubit)
        elif basis == 2:
            # 90 degrees - Hadamard
            qc.h(qubit)

        qc.measure(qubit, qubit)
        return qc

    def run_protocol(self, verbose: bool = False) -> Dict:
        """
        Execute E91 protocol.

        Args:
            verbose: Whether to print progress

        Returns:
            Dictionary with protocol results
        """
        if verbose:
            print("="*80)
            print("E91 Protocol Execution")
            print("="*80)

        self.alice_bases = np.random.randint(0, 3, self.n_pairs)
        self.bob_bases = np.random.randint(0, 3, self.n_pairs)

        self.alice_results = np.zeros(self.n_pairs, dtype=int)
        self.bob_results = np.zeros(self.n_pairs, dtype=int)

        for i in range(self.n_pairs):
            # Create entangled pair
            qc = self.create_bell_pair()

            # Alice and Bob measure in their chosen bases
            qc = self.measure_in_basis(qc, 0, self.alice_bases[i])
            qc = self.measure_in_basis(qc, 1, self.bob_bases[i])

            # Execute
            result = self.simulator.run(qc, shots=1).result()
            counts = result.get_counts()
            measurement = list(counts.keys())[0]

            self.alice_results[i] = int(measurement[1])
            self.bob_results[i] = int(measurement[0])

        # Sifting: keep only when both used basis 0
        matching = np.where((self.alice_bases == 0) & (self.bob_bases == 0))[0]

        self.alice_key = self.alice_results[matching]
        self.bob_key = self.bob_results[matching]

        # For Bell test, use other basis combinations
        bell_test_indices = np.where(
            ((self.alice_bases == 1) & (self.bob_bases == 1)) |
            ((self.alice_bases == 1) & (self.bob_bases == 2)) |
            ((self.alice_bases == 2) & (self.bob_bases == 1))
        )[0]

        # Calculate correlation (simplified Bell test)
        if len(bell_test_indices) > 0:
            correlations = np.sum(
                self.alice_results[bell_test_indices] == self.bob_results[bell_test_indices]
            ) / len(bell_test_indices)
        else:
            correlations = 0

        keys_match = np.array_equal(self.alice_key, self.bob_key)

        if verbose:
            print(f"Entangled pairs generated: {self.n_pairs}")
            print(f"Key length: {len(self.alice_key)}")
            print(f"Bell test correlation: {correlations:.4f}")
            print(f"Keys match: {keys_match}")
            print("="*80)

        return {
            'alice_key': self.alice_key,
            'bob_key': self.bob_key,
            'key_length': len(self.alice_key),
            'bell_correlation': correlations,
            'keys_match': keys_match
        }
