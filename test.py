import numpy as np
from scipy import stats
from scipy.stats import norm

# MT (https://numpy.org/doc/2.2/reference/random/generator.html#numpy.random.Generator)
np.random.seed(42)

def test_normal_distribution():
  samples = np.random.normal(loc=0, scale=1, size=1000000)
  ks_stat, ks_pvalue = stats.kstest(samples, 'norm')
  print(f"KS test: stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

def dft_test(bits, significance_level=0.05):
  # convert bits to -1 and 1 sequence
  x = np.array(bits) * 2 - 1
  n = x.size
  S = np.fft.fft(x)
  M = np.abs(S)[:n >> 1]  # only first n/2 frequencies
  T = np.sqrt(np.log(1 / significance_level) * n)

  # count peaks below threshold
  N0 = 0.001 * n / 2.0
  N1 = np.count_nonzero(M < T)

  # compute test statistic d and p-value
  d = (N1 - N0) / np.sqrt(n * (1 - significance_level) * significance_level / 4.0)
  p_value = 2 * norm.sf(abs(d))
  return p_value, (p_value >= significance_level)

def second_level_uniformity_test(p_values, significance_level=1e-4):
  d_stat, p = stats.kstest(p_values, 'uniform')
  return p, (p >= significance_level)

def get_random_bits(n_bits):
  n_bytes = (n_bits + 7) >> 3
  byte_array = np.frombuffer(np.random.bytes(n_bytes), dtype=np.uint8)
  bit_array = np.unpackbits(byte_array)
  return bit_array

def test_fourier(n_bits=100000):
  # 1st level: get p-values
  results = [dft_test(get_random_bits(n_bits)) for _ in range(1000)]
  p_values = [r[0] for r in results]
  passes = [r[1] for r in results]
  num_passed = np.count_nonzero(passes)

  # 2nd level: check uniformity of p-values
  p, passed = second_level_uniformity_test(p_values)
  print(f"2nd-level uniformity test: p={p} -> {'PASS' if passed else 'FAIL'}; first-level tests passed: {num_passed}")

if __name__ == "__main__":
  print("Testing normal distribution:")
  test_normal_distribution()
  test_fourier()
