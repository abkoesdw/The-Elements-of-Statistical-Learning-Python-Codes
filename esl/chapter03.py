import numpy as np
import sys
import scipy.stats
import pandas as pd
from math import pi, gamma
from itertools import combinations
from tqdm import tqdm


class Gaussian:
    def __init__(self, *args, **kwargs):
        self.num_data = kwargs.get("num_data", 10001)

        # this is to get the all possible values so we can compute the cdf
        np.random.seed(0)
        data = np.random.randn(self.num_data)
        data_min = data.min()
        data_max = data.max()
        self.x_data = np.linspace(data_min, data_max, self.num_data)

    def pdf(self, **kwargs):
        self.p_x = np.exp(-(self.x_data ** 2) / 2) / np.sqrt(2 * pi)
        return self.p_x

    def cdf(self, x, **kwargs):
        p_x = self.pdf(**kwargs)
        cdf_temp = []
        for i in range(len(p_x)):
            temp = np.sum(p_x[: i + 1])
            cdf_temp.append(temp)

        # this is to generate cdf only for some specified range
        idx1 = np.argmin(np.abs(self.x_data - x[0]))
        idx2 = np.argmin(np.abs(self.x_data - x[-1]))

        if x[0] < x[-1]:
            self.x = self.x_data[idx1 : idx2 + 1]
            return cdf_temp[idx1 : idx2 + 1] / cdf_temp[-1]

        else:
            return cdf_temp[idx2 : idx1 + 1][::-1] / cdf_temp[-1]


class StudentT(Gaussian):
    def __init__(self, *args, **kwargs):
        super(StudentT, self).__init__(*args, **kwargs)

    def pdf(self, **kwargs):
        dof = kwargs.get("dof", "None")

        # we use gamma function from math package
        self.p_x = (gamma((dof + 1) / 2) / (np.sqrt(pi * dof) * gamma(dof / 2))) * (
            1 + self.x_data ** 2 / dof
        ) ** -((dof + 1) / 2)
        return self.p_x


class LeastSquaresRegression:
    def __init__(self, *args, **kwargs):
        self.intercept = kwargs.get("intercept", True)

    def fit(self, x, y):
        self.y = y

        x = self._check_intercept(x)

        # check if dot(x.T, x) is invertible
        if self._check_condition(x) == 1:
            xtx = x.T @ x
            xtx_inv = np.linalg.inv(xtx)
            self.beta_hat = xtx_inv @ x.T @ self.y

    def _check_intercept(self, x):
        if self.intercept:
            if not np.isnan(x).any():
                # append intercept in the input x, so x = [1, x1, x2, ...]
                x = np.concatenate((np.ones_like(x[:, 0].reshape(-1, 1)), x), axis=1)
            else:
                # if x is nan, it means the input is just the intercept
                x = np.ones_like(x.reshape(-1, 1))
        return x

    def _check_condition(self, x):
        xtx = x.T @ x

        if np.linalg.cond(xtx) < 1 / sys.float_info.epsilon:
            return 1

        else:
            print("dot(x.T, x) is ill-conditioned")
            sys.exit()

    def calculate_z(self, x, y):
        # use (1 / (x.shape[0] - x.shape[1])) instead of (1 / (x.shape[0] - x.shape[1] - 1))
        # since it already includes the intercept
        x = self._check_intercept(x)

        if self._check_condition(x) == 1:
            xtx = x.T @ x
            xtx_inv = np.linalg.inv(xtx)

        self.variance_hat = (1 / (x.shape[0] - x.shape[1])) * np.sum(
            (y - x @ self.beta_hat) ** 2
        )
        self.beta_hat_var = np.sqrt(np.diag(xtx_inv) * self.variance_hat)
        self.z = self.beta_hat / self.beta_hat_var

    def predict(self, x):
        x = self._check_intercept(x)
        return x @ self.beta_hat

    def calculate_f(self, x, y, x0):
        rss1 = self.RSS(x, y)
        x = self._check_intercept(x)

        x0 = self._check_intercept(x0)

        if self._check_condition(x) == 1:
            xtx0 = x0.T @ x0
            xtx0_inv = np.linalg.inv(xtx0)
            beta_hat_0 = xtx0_inv @ x0.T @ self.y

        rss0 = self.RSS(x0[:, 1:], y, beta_hat=beta_hat_0)

        self.F = ((rss0 - rss1) / (x.shape[1] - x0.shape[1])) / (
            rss1 / (len(y) - x.shape[1])
        )
        self.dfn = x.shape[1] - x0.shape[1]
        self.dfd = x.shape[0] - x.shape[1]
        self.p_value = 1 - scipy.stats.f.cdf(self.F, dfn=self.dfn, dfd=self.dfd)

    def RSS(self, x, y, beta_hat=None):
        x = self._check_intercept(x)

        if beta_hat is None:
            beta_hat = self.beta_hat

        y_hat = x @ beta_hat
        return (y - y_hat).T @ (y - y_hat)


class LSOrthogonalization(LeastSquaresRegression):
    def __init__(self, *args, **kwargs):

        super(LSOrthogonalization, self).__init__(*args, **kwargs)

    def fit(self, x, y):
        x = self._check_intercept(x)

        self.compute_gamma(x)

        self.compute_beta_hat_z(y)

        self.beta_hat_x = np.linalg.inv(self.gamma) @ self.beta_hat_z

    def compute_beta_hat_z(self, y):
        self.beta_hat_z = np.zeros(self.z.shape[1])

        for i in range(self.z.shape[1]):
            self.beta_hat_z[i] = (self.z[:, i] @ y) / (self.z[:, i] @ self.z[:, i])

    def compute_gamma(self, x):
        self.z = np.zeros_like(x)
        self.z[:, 0] = 1.0
        self.gamma = np.zeros((x.shape[1], x.shape[1]))

        for i in range(x.shape[1]):
            sum_gamma_z = 0

            for j in range(i):
                self.gamma[j, i] = (self.z[:, j] @ x[:, i]) / (
                    self.z[:, j] @ self.z[:, j]
                )
                sum_gamma_z = sum_gamma_z + (self.gamma[j, i] * self.z[:, j])

            self.z[:, i] = x[:, i] - sum_gamma_z
            self.gamma[i, i] = (self.z[:, i] @ x[:, i]) / (self.z[:, i] @ self.z[:, i])

    def predict(self, x, mode="x"):
        x = self._check_intercept(x)

        if mode == "x":
            return x @ self.beta_hat_x

        elif mode == "z":
            return x @ np.linalg.inv(self.gamma) @ self.beta_hat_z


class LSQR(LSOrthogonalization):
    def __init__(self, *args, **kwargs):
        super(LSQR, self).__init__(*args, **kwargs)

    def fit(self, x, y):
        x = self._check_intercept(x)
        self.compute_gamma(x)
        self.compute_QR()
        self.beta_hat = np.linalg.inv(self.R) @ self.Q.T @ y

    def compute_QR(self):
        D = np.diag(np.linalg.norm(self.z, axis=0))

        self.Q = self.z @ np.linalg.inv(D)
        self.R = D @ self.gamma

    def predict(self, x):
        x = self._check_intercept(x)
        return x @ self.beta_hat


class SubsetSelection:
    def __init__(self, *args, **kwargs):
        self.subsets = kwargs.get("subsets", None)
        self.beta_path = []

    def fit(self, feature_index):
        if feature_index:
            x_temp = self.x[:, feature_index].reshape(-1, len(feature_index))

        else:
            x_temp = np.zeros_like(self.x[:, 0]).reshape(-1, 1)
            x_temp[:, 0] = np.nan

        lsqr = LeastSquaresRegression()
        lsqr.fit(x_temp, self.y)

        self.beta_path.append(lsqr.beta_hat)
        return lsqr.RSS(x_temp, self.y)

    def process_subset_best(self):
        result = dict()
        result["num_feat"] = []
        result["feat_subset"] = []
        result["RSS"] = []

        for subset_size in range(len(self.subsets) + 1):
            subset = combinations(self.subsets, subset_size)
            subset = [list(i) for i in subset]

            if any(subset):
                for feature_index in tqdm(
                    subset,
                    total=len(subset),
                    desc="processing Best subset: {}".format(subset_size),
                ):
                    RSS_temp = self.fit(feature_index)

                    result["RSS"].append(RSS_temp)
                    result["feat_subset"].append(feature_index)
                    result["num_feat"].append(subset_size)
            else:

                result["RSS"].append(self.fit(None))
                result["feat_subset"].append("intercept")
                result["num_feat"].append(0)

        result = pd.DataFrame(result)
        result_best = result[
            result.groupby("num_feat")["RSS"].transform(min) == result["RSS"]
        ]

        return result, result_best.reset_index(drop=True)

    def process_subset_forward(self):
        result = dict()
        result["num_feat"] = []
        result["feat_subset"] = []
        result["RSS"] = []

        subsets_remaining = list(self.subsets)

        subsets_keep = []
        for subset_size in range(len(self.subsets) + 1):
            subset = combinations(subsets_remaining, 1)
            subset = [list(i) for i in subset]
            best_RSS = np.inf

            if subset_size != 0:
                for feature_index in tqdm(
                    subset,
                    total=len(subset),
                    desc="processing Forward subset: {}".format(subset_size),
                ):
                    idx_temp = subsets_keep + feature_index
                    RSS_temp = self.fit(idx_temp)

                    if RSS_temp < best_RSS:
                        best_RSS = RSS_temp
                        best_feature = feature_index[0]

                subsets_keep.append(best_feature)
                subsets_remaining.remove(best_feature)
                result["RSS"].append(best_RSS)

                subsets_keep_copy = np.sort(subsets_keep.copy())
                result["feat_subset"].append(subsets_keep_copy)
                result["num_feat"].append(subset_size)

            else:
                result["RSS"].append(self.fit(None))
                result["feat_subset"].append("intercept")
                result["num_feat"].append(0)

        result = pd.DataFrame(result)
        return result.reset_index(drop=True)

    def process_subset_backward(self):
        result = dict()
        result["num_feat"] = []
        result["feat_subset"] = []
        result["RSS"] = []

        best_feature = list(self.subsets)

        for subset_size in range(len(self.subsets) + 1):
            subset = combinations(best_feature, len(list(self.subsets)) - subset_size)
            subset = [list(i) for i in subset]
            best_RSS = np.inf

            if len(list(self.subsets)) - subset_size != 0:
                for feature_index in tqdm(
                    subset,
                    total=len(subset),
                    desc="processing Backward subset: {}".format(subset_size),
                ):
                    RSS_temp = self.fit(list(feature_index))

                    if RSS_temp < best_RSS:
                        best_RSS = RSS_temp
                        best_feature = feature_index

                result["RSS"].append(best_RSS)
                result["feat_subset"].append(best_feature)
                result["num_feat"].append(len(best_feature))

            else:
                result["RSS"].append(self.fit(None))
                result["feat_subset"].append("intercept")
                result["num_feat"].append(0)

        result = pd.DataFrame(result).sort_values(by=["num_feat"], ascending=False)
        return result.reset_index(drop=True)

    def run(self, x, y, mode="best"):
        if self.subsets is not None:
            self.x = x
            self.y = y

            if mode == "best":
                return self.process_subset_best()

            elif mode == "forward":
                return self.process_subset_forward()

            elif mode == "backward":
                return self.process_subset_backward()

            else:
                print("Incorrect mode selection, available: best, forward, backward")

        else:
            print("Subsets need to be defined!")
            sys.exit()

    def get_statistics(self, result):
        result = self.get_mallow_cp(result)
        result = self.get_aic(result)
        result = self.get_bic(result)
        return result

    def get_mallow_cp(self, result):
        m, p = self.x.shape
        sigma_hat_squared = (1 / (m - p)) * min(result["RSS"])
        result["Mallow's Cp"] = (1 / m) * (
            result["RSS"] + 2 * result["num_feat"] * sigma_hat_squared
        )
        return result

    def get_aic(self, result):
        m, p = self.x.shape
        sigma_hat_squared = (1 / (m - p - 1)) * min(result["RSS"])
        result["AIC"] = (1 / (m * sigma_hat_squared)) * (
            result["RSS"] + 2 * result["num_feat"] * sigma_hat_squared
        )
        return result

    def get_bic(self, result):
        m, p = self.x.shape
        sigma_hat_squared = (1 / (m - p - 1)) * min(result["RSS"])
        result["BIC"] = (1 / (m * sigma_hat_squared)) * (
            result["RSS"] + np.log(m) * result["num_feat"] * sigma_hat_squared
        )
        return result


def kfold(x, y, nfold=10, seed=2):
    num_data = len(y)
    index = np.arange(num_data)
    np.random.seed(seed)
    np.random.shuffle(index)

    num_data = len(y)
    avg = len(index) / float(nfold)
    last = 0.0

    x_dict = dict()
    y_dict = dict()

    i = 0
    while last < num_data:
        index_val = index[int(last) : int(last + avg)]
        index_train = np.array([j for j in range(num_data) if j not in index_val])

        y_dict[i] = {"train": y[index_train], "val": y[index_val]}
        x_dict[i] = {"train": x[index_train, :], "val": x[index_val, :]}

        last += avg
        i += 1

    return x_dict, y_dict


class RidgeRegression:
    def fit(self, x, y, lambda_):
        # calculate beta_hat using SVD
        self.u, self.d, self.vT = np.linalg.svd(x, full_matrices=False)

        # divide the equation into 3 parts because it's too long otherwise
        self.diag_mat = np.diag(self.d)
        self.inv_diag_lambda = np.linalg.inv(
            self.diag_mat ** 2 + lambda_ * np.eye(x.shape[1])
        )
        self.ut_y = self.u.T @ y
        self.beta_hat = self.vT.T @ self.inv_diag_lambda @ self.diag_mat @ self.ut_y

    def predict(self, x):
        return x @ self.beta_hat

    def RSS(self, x, y):
        y_hat = x @ self.beta_hat
        return (y - y_hat).T @ (y - y_hat)

    def MRSS(self, x, y):
        y_hat = x @ self.beta_hat
        return ((y - y_hat).T @ (y - y_hat)) / len(y)


class Lasso:
    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)

        r = y.copy()
        corr = X.T @ r
        best_feat = np.argmax(np.abs(corr))
        A = [best_feat]
        beta_path = np.zeros((p, 1))

        while True:
            d = np.zeros(p)
            d[A] = np.linalg.inv(X[:, A].T @ X[:, A]) @ X[:, A].T @ r
            alpha = 1.0
            for j in range(p):
                if j not in A and beta[j] == 0:
                    alpha_temp = dict()
                    alpha_temp["+"] = (corr[j] - corr[best_feat]) / (
                        X[:, j].T @ X[:, A] @ d[A] - X[:, best_feat].T @ X[:, A] @ d[A]
                    )
                    alpha_temp["-"] = (corr[j] + corr[best_feat]) / (
                        X[:, j].T @ X[:, A] @ d[A] + X[:, best_feat].T @ X[:, A] @ d[A]
                    )

                    for key in alpha_temp.keys():
                        if alpha_temp[key] <= 0.0:
                            alpha_temp[key] = np.inf

                        if alpha_temp[key] <= alpha:
                            alpha = alpha_temp[key]
                            best_feat_temp = j

                elif beta[j] != 0 and j in A:
                    if d[j] != 0:
                        alpha_temp = -beta[j] / d[j]
                        if alpha_temp <= alpha and alpha_temp >= 0.0:
                            alpha = alpha_temp
                            best_feat_temp = -j

            beta = beta + alpha * d
            beta_path = np.concatenate((beta_path, beta.reshape(-1, 1)), axis=1)

            if alpha == 1.0:
                break

            if best_feat_temp < 0:
                A = [k for k in A if k != np.abs(best_feat_temp)]
                best_feat = A[0]

            else:
                A.append(best_feat_temp)
                best_feat = best_feat_temp

            r = r - alpha * (X @ d)
            corr = X.T @ r

        return A, beta_path


class PrincipalComponentRegression:
    def __init__(self, **kwargs):
        self.num_components = kwargs.get("num_components", 1)

    def fit(self, x, y):
        xTx = x.T @ x
        v, d2, vt = np.linalg.svd(xTx)

        beta = np.zeros((x.shape[1], x.shape[1]))
        for m in range(self.num_components):
            z_m = x @ v[:, m]
            theta_m = np.inner(z_m, y) / d2[m]
            beta[:, m] = theta_m * v[:, m]

        self.beta_hat = np.sum(beta, axis=1)

    def predict(self, x):
        return x @ self.beta_hat

    def error(self, x, y):
        y_hat = self.predict(x)
        error = y_hat - y

        return error.T @ error, (error.T @ error) / len(error)


class PartialLeastSquares:
    def __init__(self, **kwargs):
        self.num_direction = kwargs.get("num_direction", 1)

    def fit(self, x, y):
        x_prev = x.copy()
        y_hat_prev = np.ones_like(y) * np.mean(y)
        z = np.zeros_like(x)
        phi = np.zeros((x.shape[1], x.shape[1]))
        theta = np.zeros(x.shape[1])
        for m in range(1, self.num_direction + 1):
            for j in range(x.shape[1]):
                phi[m - 1, j] = x_prev[:, j].T @ y
                z[:, m - 1] += phi[m - 1, j] * x_prev[:, j]

            theta[m - 1] = (z[:, m - 1].T @ y) / (z[:, m - 1].T @ z[:, m - 1])

            y_hat_prev = y_hat_prev + theta[m - 1] * z[:, m - 1]

            for j in range(x.shape[1]):
                x_prev[:, j] = (
                    x_prev[:, j]
                    - ((z[:, m - 1].T @ x_prev[:, j]) / (z[:, m - 1].T @ z[:, m - 1]))
                    * z[:, m - 1]
                )

        self.beta_hat = np.linalg.inv(x.T @ x) @ x.T @ y_hat_prev

    def predict(self, x):
        return x @ self.beta_hat

    def error(self, x, y):
        y_hat = self.predict(x)
        error = y_hat - y

        return error.T @ error, (error.T @ error) / len(error)
