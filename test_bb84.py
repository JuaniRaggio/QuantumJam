"""
Unit tests for BB84 protocol implementation.

Tests cover:
- Protocol correctness
- Error rates
- Security properties
- Edge cases
- Statistical properties
"""

import unittest
import numpy as np
from bb84 import BB84Protocol, BB84WithEavesdropper, E91Protocol
from analysis import ProtocolAnalyzer, StatisticalResults


class TestBB84Protocol(unittest.TestCase):
    """Test cases for BB84Protocol class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducibility
        self.protocol = BB84Protocol(n_bits=100, noise_level=0.0)

    def test_initialization(self):
        """Test protocol initialization."""
        self.assertEqual(self.protocol.n_bits, 100)
        self.assertEqual(self.protocol.noise_level, 0.0)
        self.assertIsNone(self.protocol.alice_bits)
        self.assertIsNone(self.protocol.bob_measurements)

    def test_bit_generation(self):
        """Test random bit generation."""
        bits = self.protocol._generate_random_bits(100)
        self.assertEqual(len(bits), 100)
        self.assertTrue(np.all((bits == 0) | (bits == 1)))

        # Test randomness (should have roughly 50% 0s and 50% 1s)
        proportion_ones = np.mean(bits)
        self.assertGreater(proportion_ones, 0.3)
        self.assertLess(proportion_ones, 0.7)

    def test_basis_generation(self):
        """Test random basis generation."""
        bases = self.protocol._generate_random_bases(100)
        self.assertEqual(len(bases), 100)
        self.assertTrue(np.all((bases == 0) | (bases == 1)))

    def test_ideal_channel_zero_qber(self):
        """Test that ideal channel produces zero QBER."""
        protocol = BB84Protocol(n_bits=200, noise_level=0.0)
        result = protocol.run_protocol(verbose=False)

        self.assertTrue(result['keys_match'])
        self.assertLess(result['metrics'].qber, 0.05)  # Should be very low
        self.assertTrue(result['security_analysis'].is_secure)

    def test_noisy_channel_positive_qber(self):
        """Test that noisy channel produces non-zero QBER."""
        protocol = BB84Protocol(n_bits=200, noise_level=0.05)
        result = protocol.run_protocol(verbose=False)

        self.assertGreater(result['metrics'].qber, 0.0)
        # QBER should be roughly proportional to noise level
        self.assertLess(result['metrics'].qber, 0.15)

    def test_sifting_efficiency(self):
        """Test that sifting produces ~50% efficiency."""
        protocol = BB84Protocol(n_bits=1000, noise_level=0.0)
        protocol.step1_alice_prepare()
        protocol.step2_bob_prepare()
        protocol.step4_sifting()

        efficiency = len(protocol.matching_bases_indices) / protocol.n_bits
        # Should be around 0.5 with some statistical variance
        self.assertGreater(efficiency, 0.4)
        self.assertLess(efficiency, 0.6)

    def test_error_correction_reduces_errors(self):
        """Test that error correction improves key quality."""
        protocol = BB84Protocol(n_bits=200, noise_level=0.03)

        # Run without error correction
        result_no_ec = protocol.run_protocol(
            apply_error_correction=False,
            apply_privacy_amplification=False,
            verbose=False
        )

        # Run with error correction
        protocol2 = BB84Protocol(n_bits=200, noise_level=0.03)
        result_with_ec = protocol2.run_protocol(
            apply_error_correction=True,
            apply_privacy_amplification=False,
            verbose=False
        )

        # Keys should match better with error correction
        if not result_no_ec['keys_match']:
            self.assertTrue(result_with_ec['errors_corrected'] > 0)

    def test_high_noise_detection(self):
        """Test that high noise is detected as insecure."""
        protocol = BB84Protocol(n_bits=200, noise_level=0.15)
        result = protocol.run_protocol(verbose=False)

        self.assertFalse(result['security_analysis'].is_secure)
        self.assertGreater(result['metrics'].qber, 0.11)

    def test_privacy_amplification_reduces_key_length(self):
        """Test that privacy amplification reduces key length."""
        protocol = BB84Protocol(n_bits=500, noise_level=0.02)

        protocol.run_protocol(
            apply_error_correction=True,
            apply_privacy_amplification=False,
            verbose=False
        )
        length_before = len(protocol.alice_key)

        protocol2 = BB84Protocol(n_bits=500, noise_level=0.02)
        protocol2.run_protocol(
            apply_error_correction=True,
            apply_privacy_amplification=True,
            verbose=False
        )
        length_after = len(protocol2.alice_key)

        self.assertLessEqual(length_after, length_before)

    def test_keys_match_after_protocol(self):
        """Test that Alice and Bob's keys match in ideal conditions."""
        protocol = BB84Protocol(n_bits=300, noise_level=0.0)
        result = protocol.run_protocol(
            apply_error_correction=True,
            verbose=False
        )

        self.assertTrue(result['keys_match'])
        self.assertTrue(np.array_equal(result['alice_key'], result['bob_key']))


class TestBB84WithEavesdropper(unittest.TestCase):
    """Test cases for eavesdropping scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_no_intercept_behaves_like_bb84(self):
        """Test that 0% intercept is equivalent to normal BB84."""
        protocol = BB84WithEavesdropper(n_bits=200, intercept_rate=0.0)
        result = protocol.run_protocol(verbose=False)

        self.assertTrue(result['keys_match'])
        self.assertLess(result['metrics'].qber, 0.05)

    def test_intercept_increases_qber(self):
        """Test that interception increases QBER."""
        protocol_no_eve = BB84Protocol(n_bits=300, noise_level=0.0)
        result_no_eve = protocol_no_eve.run_protocol(verbose=False)

        protocol_eve = BB84WithEavesdropper(n_bits=300, intercept_rate=0.5)
        result_eve = protocol_eve.run_protocol(verbose=False)

        self.assertGreater(
            result_eve['metrics'].qber,
            result_no_eve['metrics'].qber
        )

    def test_full_intercept_high_qber(self):
        """Test that 100% interception produces high QBER."""
        protocol = BB84WithEavesdropper(n_bits=300, intercept_rate=1.0)
        result = protocol.run_protocol(verbose=False)

        # Theoretical QBER = 25% for 100% intercept
        self.assertGreater(result['metrics'].qber, 0.15)
        self.assertFalse(result['security_analysis'].is_secure)

    def test_eve_impact_analysis(self):
        """Test Eve impact analysis functionality."""
        protocol = BB84WithEavesdropper(n_bits=200, intercept_rate=0.5)
        protocol.run_protocol(verbose=False)

        analysis = protocol.analyze_eve_impact()

        self.assertEqual(analysis['intercept_rate'], 0.5)
        self.assertIn('theoretical_qber', analysis)
        self.assertIn('attack_detected', analysis)

    def test_detection_threshold(self):
        """Test that sufficient interception triggers detection."""
        protocol = BB84WithEavesdropper(n_bits=400, intercept_rate=0.6)
        result = protocol.run_protocol(verbose=False)

        # Should be detected (QBER > 11%)
        self.assertFalse(result['security_analysis'].is_secure)


class TestE91Protocol(unittest.TestCase):
    """Test cases for E91 protocol."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_initialization(self):
        """Test E91 initialization."""
        protocol = E91Protocol(n_pairs=100)
        self.assertEqual(protocol.n_pairs, 100)

    def test_protocol_execution(self):
        """Test basic E91 execution."""
        protocol = E91Protocol(n_pairs=200)
        result = protocol.run_protocol(verbose=False)

        self.assertIsNotNone(result['alice_key'])
        self.assertIsNotNone(result['bob_key'])
        self.assertGreater(result['key_length'], 0)

    def test_bell_correlation(self):
        """Test that Bell correlation is computed."""
        protocol = E91Protocol(n_pairs=300)
        result = protocol.run_protocol(verbose=False)

        self.assertIn('bell_correlation', result)
        # Correlation should be between 0 and 1
        self.assertGreaterEqual(result['bell_correlation'], 0)
        self.assertLessEqual(result['bell_correlation'], 1)


class TestProtocolAnalyzer(unittest.TestCase):
    """Test cases for protocol analysis functions."""

    def test_binary_entropy_bounds(self):
        """Test binary entropy function bounds."""
        # H(0) = 0
        self.assertEqual(ProtocolAnalyzer.binary_entropy(0.0), 0.0)

        # H(1) = 0
        self.assertEqual(ProtocolAnalyzer.binary_entropy(1.0), 0.0)

        # H(0.5) = 1 (maximum)
        self.assertAlmostEqual(
            ProtocolAnalyzer.binary_entropy(0.5),
            1.0,
            places=10
        )

    def test_mutual_information(self):
        """Test mutual information calculation."""
        # Perfect channel (QBER=0)
        self.assertEqual(ProtocolAnalyzer.mutual_information(0.0), 1.0)

        # Completely noisy channel (QBER=0.5)
        self.assertAlmostEqual(
            ProtocolAnalyzer.mutual_information(0.5),
            0.0,
            places=10
        )

    def test_secret_key_rate_threshold(self):
        """Test secret key rate at security threshold."""
        # Below threshold should give positive rate
        skr_safe = ProtocolAnalyzer.secret_key_rate_shor_preskill(0.10)
        self.assertGreater(skr_safe, 0.0)

        # Above threshold should give zero rate
        skr_unsafe = ProtocolAnalyzer.secret_key_rate_shor_preskill(0.12)
        self.assertEqual(skr_unsafe, 0.0)

    def test_channel_capacity(self):
        """Test channel capacity calculation."""
        # Perfect channel
        self.assertEqual(ProtocolAnalyzer.channel_capacity(0.0), 1.0)

        # QBER=0.5 gives zero capacity
        self.assertAlmostEqual(
            ProtocolAnalyzer.channel_capacity(0.5),
            0.0,
            places=10
        )

    def test_statistics_calculation(self):
        """Test statistical analysis."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = ProtocolAnalyzer.calculate_statistics(data)

        self.assertEqual(stats.mean, 3.0)
        self.assertEqual(stats.median, 3.0)
        self.assertEqual(stats.min_val, 1.0)
        self.assertEqual(stats.max_val, 5.0)
        self.assertGreater(stats.std, 0)

    def test_eve_information_estimate(self):
        """Test Eve information estimation."""
        # Zero QBER means no information leaked
        self.assertEqual(ProtocolAnalyzer.estimate_eve_information(0.0), 0.0)

        # Some QBER means potential information leak
        eve_info = ProtocolAnalyzer.estimate_eve_information(0.05)
        self.assertGreater(eve_info, 0.0)
        self.assertLess(eve_info, 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_very_small_n_bits(self):
        """Test protocol with very few qubits."""
        protocol = BB84Protocol(n_bits=10, noise_level=0.0)
        result = protocol.run_protocol(verbose=False)

        # Should still execute without errors
        self.assertIsNotNone(result['alice_key'])
        self.assertIsNotNone(result['bob_key'])

    def test_zero_noise(self):
        """Test protocol with exactly zero noise."""
        protocol = BB84Protocol(n_bits=100, noise_level=0.0)
        result = protocol.run_protocol(verbose=False)

        self.assertTrue(result['keys_match'])
        self.assertEqual(result['metrics'].noise_level, 0.0)

    def test_maximum_realistic_noise(self):
        """Test protocol at maximum realistic noise level."""
        protocol = BB84Protocol(n_bits=200, noise_level=0.11)
        result = protocol.run_protocol(verbose=False)

        # Should be at or near security threshold
        self.assertGreaterEqual(result['metrics'].qber, 0.09)


def run_tests():
    """Run all unit tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
