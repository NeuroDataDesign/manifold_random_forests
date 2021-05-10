
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load true values
#test_df = pd.read_csv("Sparse_parity_test.csv", header=None).to_numpy()
test_df = np.load("data/sparse_parity_test.npy")
y = test_df[:, -1]

ofpreds1k = np.load("output/of_sparse_parity_preds_1000.npy")
ofpreds5k = np.load("output/of_sparse_parity_preds_5000.npy")
ofpreds10k = np.load("output/of_sparse_parity_preds_10000.npy")

reps = ofpreds1k.shape[0]

rfpreds1k = np.load("output/rf_sparse_parity_preds_1000.npy")
rfpreds5k = np.load("output/rf_sparse_parity_preds_5000.npy")
rfpreds10k = np.load("output/rf_sparse_parity_preds_10000.npy")

rerfpreds1k = np.load("output/rerf_sparse_parity_preds_1000.npy")
rerfpreds5k = np.load("output/rerf_sparse_parity_preds_5000.npy")
rerfpreds10k = np.load("output/rerf_sparse_parity_preds_10000.npy")

oferr1k = 1 - np.sum(y == ofpreds1k, axis=1) / 10000
oferr5k = 1 - np.sum(y == ofpreds5k, axis=1) / 10000
oferr10k = 1 - np.sum(y == ofpreds10k, axis=1) / 10000

rferr1k = 1 - np.sum(y == rfpreds1k, axis=1) / 10000
rferr5k = 1 - np.sum(y == rfpreds5k, axis=1) / 10000
rferr10k = 1 - np.sum(y == rfpreds10k, axis=1) / 10000

rerferr1k = 1 - np.sum(y == rerfpreds1k, axis=1) / 10000
rerferr5k = 1 - np.sum(y == rerfpreds5k, axis=1) / 10000
rerferr10k = 1 - np.sum(y == rerfpreds10k, axis=1) / 10000

ofmeans = [np.mean(oferr1k), np.mean(oferr5k), np.mean(oferr10k)]
ofsterr = [np.std(oferr1k), np.std(oferr5k), np.std(oferr10k)] / np.sqrt(reps)

rfmeans = [np.mean(rferr1k), np.mean(rferr5k), np.mean(rferr10k)]
rfsterr = [np.std(rferr1k), np.std(rferr5k), np.std(rferr10k)] / np.sqrt(reps)

rerfmeans = [np.mean(rerferr1k), np.mean(rerferr5k), np.mean(rerferr10k)]
rerfsterr = [np.std(rerferr1k), np.std(rerferr5k), np.std(rerferr10k)] / np.sqrt(reps)


n = [1000, 5000, 10000]
logn = np.log(n)
plt.figure(1)
plt.errorbar(logn, ofmeans, yerr=ofsterr)
plt.errorbar(logn, rfmeans, yerr=rfsterr)
plt.errorbar(logn, rerfmeans, yerr=rerfsterr)
plt.ylim(ymax=0.5, ymin=0)
plt.xticks(logn, n)
plt.title("Sparse Parity")
plt.xlabel("Number of training samples")
plt.ylabel("Error")
plt.legend(["PySPORF", "RF", "RerF"])

plt.savefig("figures/sparse_parity_experiment")
