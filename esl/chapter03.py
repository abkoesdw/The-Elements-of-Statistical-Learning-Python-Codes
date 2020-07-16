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


class BestSelection:
    def __init__(self, *args, **kwargs):
        self.subsets = kwargs.get("subsets", None)

    def _fit(self, feature_index):
        if feature_index is not None:
            x_temp = self.x[:, feature_index].reshape(-1, len(feature_index))

        else:
            x_temp = np.zeros_like(self.x[:, 0]).reshape(-1, 1)
            x_temp[:, 0] = np.nan

        lsqr = LeastSquaresRegression()
        lsqr.fit(x_temp, self.y)

        return lsqr.RSS(x_temp, self.y)

    def _process_subset(self):
        result = dict()
        result["num_feat"] = []
        result["feat_subset"] = []
        result["RSS"] = []
        for subset_size in range(len(self.subsets) + 1):
            subset = list(combinations(self.subsets, subset_size))

            if any(subset):
                for feature_index in tqdm(
                    subset,
                    total=len(subset),
                    desc="processing subset: {}".format(subset_size),
                ):
                    result["RSS"].append(self._fit(feature_index))
                    result["feat_subset"].append(feature_index)
                    result["num_feat"].append(subset_size)
            else:

                result["RSS"].append(self._fit(None))
                result["feat_subset"].append("intercept")
                result["num_feat"].append(0)

        return pd.DataFrame(result)

    def run(self, x, y):
        if self.subsets is not None:
            self.x = x
            self.y = y
            return self._process_subset()

        else:
            print("Subsets need to be defined!")
            sys.exit()
