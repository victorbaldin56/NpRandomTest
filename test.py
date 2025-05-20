import numpy as np
from scipy import stats
from scipy.special import erfc

# MT (https://numpy.org/doc/2.2/reference/random/generator.html#numpy.random.Generator)
np.random.seed(42)

def test_normal_distribution():
  samples = np.random.normal(loc=0, scale=1, size=1000000)
  ks_stat, ks_pvalue = stats.kstest(samples, 'norm')
  print(f"KS test: stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

def dft_test(bits):
  # convert bits to -1 and 1 sequence
  x = np.array(bits) * 2 - 1
  n = x.size
  S = np.fft.fft(x)
  M = np.abs(S)[:n // 2]  # only first n/2 frequencies
  T = np.sqrt(np.log(1 / 0.05) * n)

  # count peaks below threshold
  N0 = 0.95 * n / 2.0
  N1 = np.count_nonzero(M < T)

  # compute test statistic d and p-value
  d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
  p_value = erfc(abs(d) / np.sqrt(2.0))
  return p_value, (p_value >= 0.05)

def second_level_uniformity_test(p_values, significance_level=0.01):
  num_bins = 10
  expected = len(p_values) / num_bins
  hist, _ = np.histogram(p_values, bins=num_bins, range=(0, 1))
  chi2_stat, p = stats.chisquare(hist, f_exp=[expected] * num_bins)
  return p, (p >= significance_level)

def test_fourier(n_bits=1000000):
  # 1st level: get p-values
  p_values = [dft_test(np.random.rand(n_bits) > 0.5)[0] for _ in range(100)]

  # 2nd level: check uniformity of p-values
  p, passed = second_level_uniformity_test(p_values)
  print(f"2nd-level uniformity test: p={p:.4f} -> {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
  print("Testing normal distribution:")
  test_normal_distribution()
  test_fourier()
