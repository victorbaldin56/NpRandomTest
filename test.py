import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

def test_normal_distribution():
  samples = np.random.normal(loc=0, scale=1, size=1000)
  ks_stat, ks_pvalue = stats.kstest(samples, 'norm')
  print(f"KS test: stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
  ad_result = stats.anderson(samples, dist='norm')
  print(f"AD test: stat={ad_result.statistic:.4f}, critical values={ad_result.critical_values}")

if __name__ == "__main__":
  print("Testing normal distribution:")
  test_normal_distribution()
