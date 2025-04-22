import numpy as np
from scipy.stats import multivariate_normal

class DPMM:
    def __init__(self, alpha, mu_0=None, sigma_0=None, sigma_x=1.0, covariance_type='same', n_iter=10):
        self.alpha = alpha
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.sigma_x = sigma_x
        self.n_iter = n_iter
        self.cluster_params = []
        self.covariance_type = covariance_type
        self.z = None

    def likelihood(self, x, mu, sigma):
        # it's diagonal covariance matrix
        sigma=np.diag(sigma)
        return multivariate_normal(mu, sigma).pdf(x)

    def gibbs_sampling(self, data):
        if self.mu_0 is None:
            self.mu_0 = np.mean(data, axis=0)
        if self.sigma_0 is None:
            # sigma_0 is diagonal covariance matrix
            self.sigma_0 = np.var(data, axis=0, ddof=1)

        n, d = data.shape
        self.z = np.zeros(n, dtype=int)
        self.cluster_params = [(self.mu_0, self.sigma_0)]

        for it in range(self.n_iter):
            for i, x_i in enumerate(data):
                cluster_weights = []
                current_cluster = self.z[i]

                for k, (mu_k, sigma_k) in enumerate(self.cluster_params):
                    if k == current_cluster:
                        n_k = np.sum(self.z == k) - 1
                    else:
                        n_k = np.sum(self.z == k)

                    if n_k > 0:
                        combined_sigma = self.sigma_x * np.ones(d) #sigma_k + self.sigma_x #* np.eye(d)
                        weight = n_k / (n - 1 + self.alpha) * self.likelihood(x_i, mu_k, combined_sigma)
                    else:
                        weight = 0
                    cluster_weights.append(weight)

                new_cluster_weight = self.alpha / (n - 1 + self.alpha) * self.likelihood(x_i, self.mu_0, self.sigma_x * np.ones(d))
                cluster_weights.append(new_cluster_weight)

                probabilities = np.array(cluster_weights) / sum(cluster_weights)
                new_cluster = np.random.choice(range(len(cluster_weights)), p=probabilities)
                self.z[i] = new_cluster

                if new_cluster < len(self.cluster_params):
                    assigned_data = data[self.z == new_cluster]
                    updated_mu = np.mean(assigned_data, axis=0)
                    if self.covariance_type == 'diag':
                        updated_sigma = np.var(assigned_data, axis=0,ddof=1) + self.sigma_x #* np.eye(d)
                    elif self.covariance_type == 'same':
                        updated_sigma = self.sigma_x
                    self.cluster_params[new_cluster] = (updated_mu, updated_sigma)
                else:
                    self.cluster_params.append((np.random.multivariate_normal(self.mu_0, np.diag(self.sigma_0)), self.sigma_x)) #* np.eye(d)))

    def fit(self, data):
        self.gibbs_sampling(data)

    def predict(self, data):
        n, d = data.shape
        predictions = np.zeros(n, dtype=int)

        for i, x_i in enumerate(data):
            cluster_likelihoods = []
            for k, (mu_k, sigma_k) in enumerate(self.cluster_params):
                combined_sigma = sigma_k + self.sigma_x * np.eye(d)
                likelihood = self.likelihood(x_i, mu_k, combined_sigma)
                cluster_likelihoods.append(likelihood)

            predictions[i] = np.argmax(cluster_likelihoods)

        return predictions
