import numpy as np

from mks_test import mkstest
from scipy.stats import ttest_ind

##### testing mkstest #####
# Generate two samples from a 5D Normal distribution
n = 100
d = 5
mu = np.zeros(d)
sigma = np.eye(d)
X = np.random.multivariate_normal(mu, sigma, n)
Y = np.random.multivariate_normal(mu, sigma, n)
mkstest(X, Y, alpha=0.05, verbose=True)

##### test for real data #####
# NOTE mkstest takes a long time to run, and the library itself is a bit buggy

# print("VAE mkstest:")
# mkstest(gen_vae, data, alpha=0.05, verbose=True)
# print("IWAE mkstest:")
# mkstest(gen_iwae, data, alpha=0.05, verbose=True)
# print("KDE-MCMC mkstest:")
# mkstest(gen_mcmc, data, alpha=0.05, verbose=True)

# NOTE shannon diversity can't compute negative values so it's not included in the analysis
# H_orig = shannon(data)
# H_vae = shannon(gen_vae)
# H_iwae = shannon(gen_iwae)
# H_mcmc = shannon(gen_mcmc)

# t_vae, p_vae = ttest_ind(H_orig, H_vae)
# t_iwae, p_iwae = ttest_ind(H_orig, H_iwae)
# t_mcmc, p_mcmc = ttest_ind(H_orig, H_mcmc)

# print("Shannon diversity t-test results:")
# print(f" VAE:  t={t_vae:.3f}, p={p_vae:.3e}")
# print(f" IWAE: t={t_iwae:.3f}, p={p_iwae:.3e}")
# print(f" MCMC: t={t_mcmc:.3f}, p={p_mcmc:.3e}")

# means (axis=0, less samples)
# mean_orig = np.mean(data, axis=0)
# mean_vae = np.mean(gen_vae, axis=0)
# mean_iwae = np.mean(gen_iwae, axis=0)
# mean_mcmc = np.mean(gen_mcmc, axis=0)

# t-tests on means
# t_mean_vae, p_mean_vae = ttest_ind(mean_orig, mean_vae)
# t_mean_iwae, p_mean_iwae = ttest_ind(mean_orig, mean_iwae)
# t_mean_mcmc, p_mean_mcmc = ttest_ind(mean_orig, mean_mcmc)

# print("Mean t-test results:")
# print(f" VAE:  t={t_mean_vae:.3f}, p={p_mean_vae:.3e}")
# print(f" IWAE: t={t_mean_iwae:.3f}, p={p_mean_iwae:.3e}")
# print(f" MCMC: t={t_mean_mcmc:.3f}, p={p_mean_mcmc:.3e}")

# medians
# med_orig = np.median(data, axis=0)
# med_vae = np.median(gen_vae, axis=0)
# med_iwae = np.median(gen_iwae, axis=0)
# med_mcmc = np.median(gen_mcmc, axis=0)

# from scipy.stats import wilcoxon

# paired Wilcoxon signed-rank tests on medians
# w_med_vae, p_w_med_vae = wilcoxon(med_orig, med_vae)
# w_med_iwae, p_w_med_iwae = wilcoxon(med_orig, med_iwae)
# if you had paired KDE-MCMC medians too:
# w_med_mcmc, p_w_med_mcmc = wilcoxon(med_orig, med_mcmc)

# print("Wilcoxon signed-rank test on medians:")
# print(f" VAE:  W={w_med_vae:.3f}, p={p_w_med_vae:.3e}")
# print(f" IWAE: W={w_med_iwae:.3f}, p={p_w_med_iwae:.3e}")
# print(f" MCMC: W={w_med_mcmc:.3f}, p={p_w_med_mcmc:.3e}")

# from scipy.stats import mannwhitneyu

# u_med_vae, p_u_med_vae = mannwhitneyu(med_orig, med_vae)
# u_med_iwae, p_u_med_iwae = mannwhitneyu(med_orig, med_iwae)

# print("Mann–Whitney U test on medians:")
# print(f" VAE:  W={u_med_vae:.3f}, p={p_u_med_vae:.3e}")
# print(f" IWAE: W={u_med_iwae:.3f}, p={p_u_med_iwae:.3e}")
# print(f" MCMC: W={w_med_mcmc:.3f}, p={p_w_med_mcmc:.3e}")
