import numpy as np
from scipy import stats
from scipy.special import erfc

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

def test_fourier(n_bits=1000000):
  # generate bits (uniform [0,1))
  raw = np.random.rand(n_bits)
  bits = (raw > 0.5).astype(int)

  p_val, passed = dft_test(bits)
  print(f"DFT test: p-value = {p_val:.4f} -> {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
  print("Testing normal distribution:")
  test_normal_distribution()
  test_fourier()
