# BB84 Quantum Key Distribution: Professional Implementation

## Quantum Jam 2025 - Technical Implementation

A comprehensive, production-grade implementation of the BB84 quantum key distribution protocol with advanced security analysis, error correction, and privacy amplification.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)
- [Testing](#testing)
- [Performance](#performance)
- [Security Analysis](#security-analysis)
- [References](#references)
- [License](#license)

---

## Overview

This project implements the BB84 quantum key distribution protocol, providing information-theoretically secure key exchange between two parties (Alice and Bob) using quantum communication. The implementation includes:

- Complete BB84 protocol with all processing steps
- E91 entanglement-based protocol for comparison
- Advanced noise modeling (depolarizing, amplitude damping)
- Eavesdropping attack simulation and detection
- Comprehensive security analysis
- Statistical evaluation tools
- Professional visualization suite

### What is BB84?

BB84, proposed by Bennett and Brassard in 1984, is the first and most widely studied quantum key distribution protocol. It leverages fundamental quantum mechanical principles to detect eavesdropping and establish secure shared keys.

**Security Foundation:**
- No-cloning theorem prevents copying of unknown quantum states
- Measurement disturbance makes eavesdropping detectable
- Information-theoretic security (not based on computational complexity)

---

## Features

### Core Protocol Implementation

- **Complete BB84 Protocol:**
  - Random bit and basis generation
  - Quantum state preparation (Z and X bases)
  - Quantum channel simulation with noise
  - Basis reconciliation (sifting)
  - Error estimation via sampling
  - Simplified Cascade error correction
  - Privacy amplification

- **E91 Protocol:**
  - Entangled pair generation
  - Bell state measurements
  - Correlation analysis
  - Bell inequality tests

### Security & Analysis

- **Security Metrics:**
  - QBER (Quantum Bit Error Rate) calculation
  - Mutual information I(A:B)
  - Secret key rate (Shor-Preskill bound)
  - Eve information estimation
  - Security parameter (ε-security)

- **Attack Simulation:**
  - Intercept-resend attacks
  - Detection probability analysis
  - Impact on QBER

### Noise Modeling

- Simple bit-flip noise
- Depolarizing channel (Pauli noise)
- Amplitude damping (energy dissipation)
- Custom noise model support

### Visualization & Statistics

- Protocol performance dashboards
- QBER vs noise analysis
- Secret key rate comparisons
- Eavesdropping impact plots
- Statistical analysis with confidence intervals

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Jupyter notebook support

### Standard Installation

```bash
# Navigate to project directory
cd quantum

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

For use in Google Colab (recommended for the hackathon):

```python
# Run in a Colab cell
!pip install qiskit qiskit-aer qiskit-ibm-runtime numpy scipy matplotlib seaborn pandas tqdm -q
```

Then upload the project files:
- `bb84.py`
- `analysis.py`
- `BB84_Professional.ipynb`

### Verification

```bash
# Run unit tests to verify installation
python test_bb84.py
```

---

## Quick Start

### Basic BB84 Execution

```python
from bb84 import BB84Protocol

# Create protocol instance
protocol = BB84Protocol(n_bits=500, noise_level=0.05)

# Run complete protocol
result = protocol.run_protocol(
    apply_error_correction=True,
    apply_privacy_amplification=True,
    verbose=True
)

# Access results
print(f"QBER: {result['metrics'].qber*100:.2f}%")
print(f"Key length: {result['metrics'].n_bits_final} bits")
print(f"Secure: {result['security_analysis'].is_secure}")
```

### Eavesdropping Simulation

```python
from bb84 import BB84WithEavesdropper

# Simulate Eve intercepting 50% of qubits
protocol_eve = BB84WithEavesdropper(
    n_bits=500,
    intercept_rate=0.5
)

result = protocol_eve.run_protocol(verbose=True)

# Analyze Eve's impact
analysis = protocol_eve.analyze_eve_impact()
print(f"Attack detected: {analysis['attack_detected']}")
```

### E91 Protocol

```python
from bb84 import E91Protocol

# Run E91 with entangled pairs
e91 = E91Protocol(n_pairs=500)
result = e91.run_protocol(verbose=True)

print(f"Bell correlation: {result['bell_correlation']:.4f}")
print(f"Key length: {result['key_length']} bits")
```

### Visualization

```python
from analysis import ProtocolVisualizer
import numpy as np

visualizer = ProtocolVisualizer()

# Example: Plot QBER vs noise
noise_levels = np.linspace(0, 0.15, 16)
# ... run simulations and collect data ...

fig = visualizer.plot_qber_vs_noise(
    noise_levels=noise_levels,
    qber_mean=qber_data,
    qber_std=qber_std_data
)
```

---

## Project Structure

```
quantum/
├── bb84.py                     # Core protocol implementations
│   ├── BB84Protocol            # Main BB84 class
│   ├── BB84WithEavesdropper    # Eavesdropping simulation
│   ├── E91Protocol             # Entanglement-based QKD
│   ├── ProtocolMetrics         # Metrics dataclass
│   └── SecurityAnalysis        # Security analysis dataclass
│
├── analysis.py                 # Analysis and visualization tools
│   ├── ProtocolAnalyzer        # Information-theoretic analysis
│   ├── ProtocolVisualizer      # Plotting functions
│   ├── NoiseModelAnalyzer      # Noise analysis
│   └── StatisticalResults      # Statistics dataclass
│
├── test_bb84.py               # Comprehensive unit tests
│   ├── TestBB84Protocol
│   ├── TestBB84WithEavesdropper
│   ├── TestE91Protocol
│   ├── TestProtocolAnalyzer
│   └── TestEdgeCases
│
├── BB84_Professional.ipynb    # Main technical notebook
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Usage Examples

### Example 1: Statistical Analysis Across Noise Levels

```python
import numpy as np
from bb84 import BB84Protocol

noise_levels = np.linspace(0, 0.15, 11)
results = []

for noise in noise_levels:
    # Run multiple trials
    qber_values = []
    for _ in range(10):
        protocol = BB84Protocol(n_bits=300, noise_level=noise)
        result = protocol.run_protocol(verbose=False)
        qber_values.append(result['metrics'].qber)

    results.append({
        'noise': noise,
        'qber_mean': np.mean(qber_values),
        'qber_std': np.std(qber_values)
    })

# Analyze results
for r in results:
    print(f"Noise {r['noise']*100:.1f}%: "
          f"QBER = {r['qber_mean']*100:.3f}% ± {r['qber_std']*100:.3f}%")
```

### Example 2: Security Threshold Analysis

```python
from bb84 import BB84Protocol
from analysis import ProtocolAnalyzer

qber_range = np.linspace(0, 0.2, 100)
secret_key_rates = []

for qber in qber_range:
    skr = ProtocolAnalyzer.secret_key_rate_shor_preskill(qber)
    secret_key_rates.append(skr)

# Find security threshold
threshold_idx = np.where(np.array(secret_key_rates) <= 0)[0][0]
threshold_qber = qber_range[threshold_idx]
print(f"Security threshold: {threshold_qber*100:.2f}%")
```

### Example 3: Noise Model Comparison

```python
from bb84 import BB84Protocol

# Test different noise models
configs = [
    {'name': 'Bit-flip', 'noise_level': 0.05},
    {'name': 'Depolarizing', 'use_noise_model': True, 'depolarizing_prob': 0.03},
    {'name': 'Damping', 'use_noise_model': True, 'damping_prob': 0.03}
]

for config in configs:
    name = config.pop('name')
    protocol = BB84Protocol(n_bits=400, **config)
    result = protocol.run_protocol(verbose=False)

    print(f"{name}: QBER = {result['metrics'].qber*100:.3f}%, "
          f"Key rate = {result['metrics'].key_rate*100:.2f}%")
```

---

## Technical Details

### Protocol Steps

**1. Preparation (Alice)**
```
For each qubit i:
  - Generate random bit a_i ∈ {0, 1}
  - Choose random basis b_i^A ∈ {Z, X}
  - Prepare state:
      |ψ_i⟩ = |a_i⟩           if b_i^A = Z
             = |+⟩/|-⟩        if b_i^A = X and a_i = 0/1
```

**2. Transmission**
```
Alice sends qubits through quantum channel to Bob
Channel may introduce noise/eavesdropping
```

**3. Measurement (Bob)**
```
For each qubit i:
  - Choose random basis b_i^B ∈ {Z, X}
  - Measure in chosen basis
  - Record result m_i
```

**4. Sifting**
```
Alice and Bob publicly compare bases
Keep only bits where b_i^A = b_i^B
Expected efficiency: ~50%
```

**5. Error Estimation**
```
Compare random sample of n bits
QBER = (number of errors) / n
If QBER > 11%, abort (eavesdropping detected)
```

**6. Error Correction**
```
Use Cascade algorithm to correct remaining errors
Cost: ~H(QBER) bits revealed publicly
```

**7. Privacy Amplification**
```
Apply universal hash function
Compress key to remove Eve's potential information
Final key length ≈ n × [1 - 2H(QBER)]
```

### Information-Theoretic Bounds

**Binary Entropy:**
```
H(p) = -p log₂(p) - (1-p) log₂(1-p)
```

**Mutual Information:**
```
I(A:B) = 1 - H(QBER)
```

**Secret Key Rate (Shor-Preskill):**
```
r ≥ 1 - 2H(QBER)
```

**Security Condition:**
```
I(A:B) > I(A:E)
⟹ QBER < 11% (approximately)
```

### Noise Models

**Depolarizing Channel:**
```
ρ → (1-p)ρ + p·I/2
Expected QBER ≈ p/2
```

**Amplitude Damping:**
```
|1⟩ → √(1-γ)|1⟩ + √γ|0⟩
|0⟩ → |0⟩
Expected QBER ≈ γ/4
```

---

## Testing

### Running Unit Tests

```bash
# Run all tests
python test_bb84.py

# Run with verbose output
python test_bb84.py -v

# Run specific test class
python -m unittest test_bb84.TestBB84Protocol
```

### Test Coverage

The test suite includes:
- Protocol initialization and execution
- Bit/basis generation randomness
- Sifting efficiency (~50%)
- Error correction functionality
- Privacy amplification
- QBER calculation accuracy
- Security threshold detection
- Eavesdropping impact
- Edge cases and boundary conditions

---

## Performance

### Computational Complexity

- **Quantum circuit simulation:** O(n) where n = number of qubits
- **Sifting:** O(n)
- **Error estimation:** O(n × sample_size)
- **Error correction:** O(n log n) for Cascade
- **Privacy amplification:** O(n)

**Overall:** O(n log n) for complete protocol

### Benchmarks

Typical execution times on modern CPU (n_bits=1000):

| Operation | Time |
|-----------|------|
| State preparation | ~50ms |
| Transmission & measurement | ~100ms |
| Sifting | ~1ms |
| Error estimation | ~5ms |
| Error correction | ~10ms |
| Privacy amplification | ~2ms |
| **Total** | **~170ms** |

---

## Security Analysis

### Security Guarantees

**Information-Theoretic Security:**
- Does not depend on computational assumptions
- Secure against adversaries with unlimited computational power
- Based on fundamental laws of quantum mechanics

**Key Security Properties:**
1. **Eavesdropping Detection:** Any measurement by Eve disturbs quantum states, introducing detectable errors
2. **No-Cloning:** Eve cannot copy quantum states without disturbing them
3. **Privacy Amplification:** Removes any residual information Eve might have obtained

### Attack Resistance

**Intercept-Resend Attack:**
- Eve intercepts qubits, measures, and resends
- Introduces QBER ≈ (intercept_rate × 25%)
- Reliably detected when intercept_rate > 30%

---

## References

### Foundational Papers

1. **Bennett, C. H., & Brassard, G. (1984).** Quantum cryptography: Public key distribution and coin tossing. *Proceedings of IEEE International Conference on Computers, Systems and Signal Processing*.

2. **Shor, P. W., & Preskill, J. (2000).** Simple proof of security of the BB84 quantum key distribution protocol. *Physical Review Letters, 85(2)*, 441.

3. **Ekert, A. K. (1991).** Quantum cryptography based on Bell's theorem. *Physical Review Letters, 67(6)*, 661.

### Security Analysis

4. **Renner, R. (2008).** Security of quantum key distribution. *International Journal of Quantum Information, 6(01)*, 1-127.

5. **Scarani, V., et al. (2009).** The security of practical quantum key distribution. *Reviews of Modern Physics, 81(3)*, 1301.

### Reviews

6. **Gisin, N., et al. (2002).** Quantum cryptography. *Reviews of Modern Physics, 74(1)*, 145.

7. **Pirandola, S., et al. (2020).** Advances in quantum cryptography. *Advances in Optics and Photonics, 12(4)*, 1012-1236.

### Textbooks

8. **Nielsen, M. A., & Chuang, I. L. (2010).** Quantum Computation and Quantum Information. *Cambridge University Press*.

---

## Contributing

This project was developed for the Quantum Jam 2025 hackathon. Contributions, suggestions, and improvements are welcome.

---

## License

This project is provided for educational and research purposes as part of the Quantum Jam 2025 hackathon.

---

## Acknowledgments

- **IBM Quantum** for providing Qiskit framework
- **Bennett & Brassard** for inventing BB84
- **Quantum Jam 2025 organizers** for the opportunity

---

**Last Updated:** 2025
**Version:** 1.0.0
**Status:** Production-ready for hackathon submission
# QuantumJam
