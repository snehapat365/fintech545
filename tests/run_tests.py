import pandas as pd
import numpy as np
from scipy.stats import norm, t, kurtosis
from sklearn.decomposition import PCA
import utils

# Test 1 - missing covariance calculations
# Generate some random numbers with missing values.

x = pd.read_csv("data/input/test1.csv")
# 1.1 Skip Missing rows - Covariance
cout = x.dropna(axis=0).cov()
cout.to_csv("data/output/testout_1.1.csv", index=False)
# 1.2 Skip Missing rows - Correlation
cout = x.dropna(axis=0).corr()
cout.to_csv("data/output/testout_1.2.csv", index=False)
# 1.3 Pairwise - Covariance
cout = x.cov()
cout.to_csv("data/output/testout_1.3.csv", index=False)
# 1.4 Pairwise - Correlation
cout = x.corr()
cout.to_csv("data/output/testout_1.4.csv", index=False)

# Test 2 - EW Covariance
# np.random.seed(3)

x = pd.read_csv("data/input/test2.csv")
x_headers = x.columns.tolist()
# 2.1 EW Covariance λ=0.97
ew_lambda = 0.97
ew_cov = utils.exponential_covariance(x, ew_lambda)
pd.DataFrame(ew_cov, columns = x_headers).to_csv("data/output/testout_2.1.csv", index=False)

# 2.2 EW Correlation λ=0.94
ew_lambda = 0.94
ew_cov = utils.exponential_covariance(x, ew_lambda)
sd = 1 / np.sqrt(np.diag(ew_cov))
# normalize the covariance matrix by the standard deviation to get the correlation matrix
ew_cor = ew_cov * sd[:, np.newaxis] * sd[np.newaxis, :]
pd.DataFrame(ew_cor, columns = x_headers).to_csv("data/output/testout_2.2.csv", index=False)


# 2.3 EW Cov w/ EW Var(λ=0.94) EW Correlation(λ=0.97)
# Calculate the exponentially weighted covariance matrix for variance
ew_var_cov =  utils.exponential_covariance(x, 0.94)
# Calculate the exponentially weighted covariance matrix for correlation
ew_cor_cov =  utils.exponential_covariance(x, 0.97)
# Calculate the standard deviations
sd1 = np.sqrt(np.diag(ew_cor_cov))
sd = 1 / np.sqrt(np.diag(ew_var_cov))
# Combine the two covariance matrices
ew_cov = np.diag(sd1) @ np.diag(sd) @ ew_var_cov @ np.diag(sd) @ np.diag(sd1)
print(type(ew_cov))
pd.DataFrame(ew_cov, columns= x_headers).to_csv("data/output/testout_2.3.csv", index=False)

# Test 3 - non-psd matrices

# 3.1 near_psd covariance
cin = pd.read_csv("data/output/testout_1.3.csv")
cin_headers = cin.columns.tolist()
cout = utils.near_psd(cin)
pd.DataFrame(cout, columns=cin_headers).to_csv("data/output/testout_3.1.csv", index=False)

# 3.2 near_psd Correlation
cin = pd.read_csv("data/output/testout_1.4.csv")
cin_headers = cin.columns.tolist()
cin = np.array(cin)
cout = utils.near_psd(cin)
print(type(cout))
pd.DataFrame(cout, columns=cin_headers).to_csv("data/output/testout_3.2.csv", index=False)

# 3.3 Higham covariance
cin = pd.read_csv("data/output/testout_1.3.csv")
cin_headers = cin.columns.tolist()
cin = np.array(cin)
cout = utils.higham_nearestPSD(cin)
pd.DataFrame(cout, columns= cin_headers).to_csv("data/output/testout_3.3.csv", index=False)

# 3.4 Higham Correlation
cin = pd.read_csv("data/output/testout_1.4.csv")
cin_headers = cin.columns.tolist()
cin = np.array(cin)
cout = utils.higham_nearestPSD(cin)
pd.DataFrame(cout, columns= cin_headers).to_csv("data/output/testout_3.4.csv", index=False)

# 4 cholesky factorization
cin = pd.read_csv("data/output/testout_3.1.csv")
cin_headers = cin.columns.tolist()
cin = np.array(cin)
n, m = cin.shape
cout = np.zeros((n, m))
utils.chol_psd(cout, cin)
pd.DataFrame(cout, columns= cin_headers).to_csv("data/output/testout_4.1.csv", index=False)

# 5 Normal Simulation

# # 5.1 PD Input
# cin = pd.read_csv("data/input/test5_1.csv")
# cout = np.cov(utils.simulate_normal(100000, cin))
# pd.DataFrame(cout).to_csv("data/output/testout_5.1.csv", index=False)
# 
# # 5.2 PSD Input
# cin = pd.read_csv("data/input/test5_2.csv")
# cout = np.cov(utils.simulate_normal(100000, cin))
# pd.DataFrame(cout).to_csv("data/output/testout_5.2.csv", index=False)
# 
# # 5.3 nonPSD Input, near_psd fix
# cin = pd.read_csv("data/input/test5_3.csv")
# cout = np.cov(utils.simulate_normal(100000, cin, fixMethod=utils.near_psd))
# pd.DataFrame(cout).to_csv("data/output/testout_5.3.csv", index=False)
# # 
# # 5.4 nonPSD Input Higham Fix
# cin = pd.read_csv("data/input/test5_3.csv")
# cout = np.cov(utils.simulate_normal(100000, cin, fixMethod=utils.higham_nearestPSD))
# pd.DataFrame(cout).to_csv("data/output/testout_5.4.csv", index=False)

# 5.5 PSD Input - PCA Simulation
# print("5.5")
# cin = pd.read_csv("data/test5_2.csv")
# # cout = np.cov(utils.simulate_pca(cin, 100000, pctExp=0.99))
# cout = np.cov(utils.simulate_pca(cin, 1000, pctExp=0.99))
# pd.DataFrame(cout).to_csv("data/testout_5.5.csv", index=False)
# 
# Test 6
# print("6.1")
# # 6.1 Arithmetic returns
# prices = pd.read_csv("data/input/test6.csv")
# rout = utils.return_calculate(prices)
# rout.to_csv("data/test6_1.csv", index=False)
# print("6.2")
# # 6.2 Log returns
# prices = pd.read_csv("data/test6.csv")
# rout = utils.return_calculate(prices, method="LOG")
# rout.to_csv("data/test6_2.csv", index=False)

# Test 7
# 
# d = norm(loc=0.05, scale=0.05)
# x = d.rvs(100)
# pd.DataFrame(x).to_csv("data/test7_1.csv", index=False)
# 
# d = t(df=10, loc=0.05, scale=0.05)
# x = d.rvs(100)
# kurtosis(x)
# pd.DataFrame(x).to_csv("data/test7_2.csv", index=False)
# 
# corr = np.full((3, 3), 0.5) + np.eye(3) * 0.5
# sd = [0.02, 0.03, 0.04]
# covar = np.diag(sd) @ corr @ np.diag(sd)
# x = np.random.multivariate_normal([0, 0, 0], covar, 100).T
# e = (np.random.standard_t(10, size=100) * 0.05) + 0.05
# B = [1, 2, 3]
# y = x * B + e
# cout = pd.DataFrame(x.T, columns=["x1", "x2", "x3"])
# cout["y"] = y
# cout.to_csv("data/test7_3.csv", index=False)
# 
# 7.1 Fit Normal Distribution
print("7.1")
cin = pd.read_csv("data/input/test7_1.csv")
mu, sigma = norm.fit(cin.values.flatten())
pd.DataFrame({"mu": [mu], "sigma": [sigma]}).to_csv("data/output/testout7_1.csv", index=False)
#
# 7.2 Fit TDist
print("7.2")
cin = pd.read_csv("data/input/test7_2.csv")
params = t.fit(cin.values.flatten())
mu, sigma, nu = params[1], params[2], params[0]
pd.DataFrame({"mu": [mu], "sigma": [sigma], "nu": [nu]}).to_csv("data/output/testout7_2.csv", index=False)

# 7.3 Fit T Regression
# print("7.3")
# cin = pd.read_csv("data/input/test7_3.csv")
# x = cin[["x1", "x2", "x3"]].values
# y = cin["y"].values
# params = np.linalg.lstsq(x, y, rcond=None)[0]
# print(params)
# mu, sigma, nu = norm.fit(y - x @ params), np.std(y - x @ params), t.fit(y - x @ params)[0]
# pd.DataFrame({"mu": [mu], "sigma": [sigma], "nu": [nu], "Alpha": [params[0]], "B1": [params[1]], "B2": [params[2]], "B3": [params[2]]}).to_csv("data/output/testout7_3.csv", index=False)

# Test 8

# Test 8.1 VaR Normal
print("8.1")
cin = pd.read_csv("data/input/test7_1.csv")
mu, sigma = norm.fit(cin.values.flatten())
VaR_abs = norm.ppf(0.05, loc=mu, scale=sigma)
VaR_diff = mu - norm.ppf(0.05, scale=sigma)
pd.DataFrame({"VaR Absolute": [VaR_abs], "VaR Diff from Mean": [VaR_diff]}).to_csv("data/output/testout8_1.csv", index=False)

# Test 8.2 VaR TDist
print("8.2")
cin = pd.read_csv("data/input/test7_2.csv")
params = t.fit(cin.values.flatten())
mu, sigma, nu = params[1], params[2], params[0]
VaR_abs = t.ppf(0.05, df=nu, loc=mu, scale=sigma)
VaR_diff = mu - t.ppf(0.05, df=nu, scale=sigma)
pd.DataFrame({"VaR Absolute": [VaR_abs], "VaR Diff from Mean": [VaR_diff]}).to_csv("data/output/testout8_2.csv", index=False)

# Test 8.3 VaR Simulation
print("8.3")
cin = pd.read_csv("data/input/test7_2.csv")
params = t.fit(cin.values.flatten())
sim = t.rvs(df=params[0], loc=params[1], scale=params[2], size=10000)
VaR_abs = np.percentile(sim, 5)
VaR_diff = np.percentile(sim - np.mean(sim), 5)
pd.DataFrame({"VaR Absolute": [VaR_abs], "VaR Diff from Mean": [VaR_diff]}).to_csv("data/output/testout8_3.csv", index=False)

# Test 8.4 ES Normal
print("8.4")
cin = pd.read_csv("data/input/test7_1.csv")
mu, sigma = norm.fit(cin.values.flatten())
ES_abs = norm.expect(lambda x: x, loc=mu, scale=sigma, lb=norm.ppf(0.05, loc=mu, scale=sigma))
ES_diff = norm.expect(lambda x: x, loc=mu, scale=sigma, lb=mu - norm.ppf(0.05, scale=sigma))
pd.DataFrame({"ES Absolute": [ES_abs], "ES Diff from Mean": [ES_diff]}).to_csv("data/output/testout8_4.csv", index=False)

# Test 8.5 ES TDist
# print("8.5")
# cin = pd.read_csv("data/input/test7_2.csv")
# params = t.fit(cin.values.flatten())
# mu, sigma, nu = params[1], params[2], params[0]
# ES_abs = t.expect(lambda x: x, df=nu, loc=mu, scale=sigma, lb=t.ppf(0.05, df=nu, loc=mu, scale=sigma))
# ES_diff = t.expect(lambda x: x, df=nu, loc=mu, scale=sigma, lb=mu - t.ppf(0.05, df=nu, scale=sigma))
# pd.DataFrame({"ES Absolute": [ES_abs], "ES Diff from Mean": [ES_diff]}).to_csv("data/output/testout8_5.csv", index=False)

# Test 8.6 ES Simulation
print("8.6")
cin = pd.read_csv("data/input/test7_2.csv")
params = t.fit(cin.values.flatten())
sim = t.rvs(df=params[0], loc=params[1], scale=params[2], size=10000)
ES_abs = np.mean(sim[sim <= np.percentile(sim, 5)])
ES_diff = np.mean(sim[sim <= np.percentile(sim - np.mean(sim), 5)])
pd.DataFrame({"ES Absolute": [ES_abs], "ES Diff from Mean": [ES_diff]}).to_csv("data/output/testout8_6.csv", index=False)

# # Test 9
# A = np.random.normal(0, 0.03, 200)
# B = 0.1 * A + np.random.standard_t(10, size=200) * 0.02
# pd.DataFrame({"A": A, "B": B}).to_csv("data/test9_1_returns.csv", index=False)
# 
# # 9.1
# print("9.1")
# cin = pd.read_csv("data/test9_1_returns.csv")
# portfolio = {"A": 20.0, "B": 30.0}
# models = {
#     "A": {"mu": cin["A"].mean(), "sigma": cin["A"].std()},
#     "B": {"mu": cin["B"].mean(), "sigma": cin["B"].std(), "nu": t.fit(cin["B"])[0]}
# }
# nSim = 100000
# nSim = 10000
# 
# U = np.column_stack([norm.ppf(np.linspace(0, 1, nSim), loc=mu, scale=sigma) for mu, sigma in [(models["A"]["mu"], models["A"]["sigma"]), (models["B"]["mu"], models["B"]["sigma"])]])
# spcor = np.corrcoef(U, rowvar=False)
# pca = PCA(n_components=2)
# pca.fit(spcor)
# uSim = pca.inverse_transform(np.random.normal(size=(nSim, 2)))
# uSim = norm.cdf(uSim)
# 
# simRet = pd.DataFrame({"A": norm.ppf(uSim[:, 0], loc=models["A"]["mu"], scale=models["A"]["sigma"]), "B": t.ppf(uSim[:, 1], df=models["B"]["nu"], loc=models["B"]["mu"], scale=models["B"]["sigma"])})
# 
# portfolio = pd.DataFrame({"Stock": ["A", "B"], "currentValue": [2000.0, 3000.0]})
# iteration = np.arange(1, nSim + 1)
# values = pd.DataFrame(np.array(np.meshgrid(portfolio["Stock"], portfolio["currentValue"], iteration)).T.reshape(-1, 3), columns=["Stock", "currentValue", "iteration"])
# 
# pnl = (values["currentValue"].astype(float) * (1 + simRet.loc[values["iteration"], values["Stock"]].values) - values["currentValue"].astype(float)).values
# values["pnl"] = pnl
# values["simulatedValue"] = values["currentValue"].astype(float) + pnl
# 
# risk = values.groupby("Stock").agg({"simulatedValue": lambda x: np.percentile(x, 95), "pnl": lambda x: np.percentile(x, 95), "simulatedValue": lambda x: np.percentile(x, 95) / portfolio.set_index("Stock")["currentValue"]}).reset_index()
# risk.columns = ["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]
# 
# risk.to_csv("data/testout9_1.csv", index=False)
