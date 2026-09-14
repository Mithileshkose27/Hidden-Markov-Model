import numpy as np

def log_sum_exp(log_vars):
    """Numerically stable Log-Sum-Exp computation."""
    max_val = np.max(log_vars)
    if np.isneginf(max_val):
        return -np.inf
    return max_val + np.log(np.sum(np.exp(log_vars - max_val)))

def log_sum_exp_axis(log_matrix, axis=0):
    """Log-Sum-Exp along specified axis."""
    max_val = np.max(log_matrix, axis=axis, keepdims=True)
    max_val_squeezed = np.squeeze(max_val, axis=axis)
    
    # Handle infinite values cleanly
    safe_max = np.where(np.isneginf(max_val_squeezed), 0.0, max_val_squeezed)
    exp_diff = np.exp(log_matrix - max_val)
    sum_exp = np.sum(exp_diff, axis=axis)
    
    result = np.where(sum_exp > 0, safe_max + np.log(sum_exp), -np.inf)
    return result


class GaussianHMM:
    """
    Continuous Gaussian Hidden Markov Model with Left-to-Right (Bakis) Topology.
    Supports Log-domain Forward, Backward, Viterbi dynamic programming, and Baum-Welch EM.
    """

    def __init__(self, n_states=3, n_features=13, topology="left_right", min_covar=1e-3):
        self.n_states = n_states
        self.n_features = n_features
        self.topology = topology
        self.min_covar = min_covar
        
        # 1. Initial State Probabilities pi (Left-to-Right starts at state 0)
        self.pi = np.zeros(n_states)
        self.pi[0] = 1.0
        self.log_pi = np.full(n_states, -np.inf)
        self.log_pi[0] = 0.0

        # 2. Transition Probability Matrix A
        self.A = np.zeros((n_states, n_states))
        self.init_topology()
        self.log_A = np.where(self.A > 0, np.log(self.A), -np.inf)

        # 3. Gaussian Emission parameters: Means and Diagonal Covariances
        self.means = np.zeros((n_states, n_features))
        self.covars = np.ones((n_states, n_features)) # diagonal covariances

    def init_topology(self):
        """Initializes transition matrix according to Bakis left-to-right topology."""
        self.A = np.zeros((self.n_states, self.n_states))
        if self.topology == "left_right":
            for i in range(self.n_states):
                if i == self.n_states - 1:
                    self.A[i, i] = 1.0
                elif i == self.n_states - 2:
                    self.A[i, i] = 0.6
                    self.A[i, i + 1] = 0.4
                else:
                    self.A[i, i] = 0.5
                    self.A[i, i + 1] = 0.35
                    self.A[i, i + 2] = 0.15
        else:
            # Ergodic (fully connected)
            self.A = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
            
        # Normalize rows
        self.A /= self.A.sum(axis=1, keepdims=True)

    def compute_log_b(self, X):
        """
        Computes log-emission probability log b_j(x_t) for each frame t and state j.
        X shape: (T, D)
        Returns shape: (T, N)
        """
        T, D = X.shape
        log_B = np.zeros((T, self.n_states))
        
        for j in range(self.n_states):
            mean = self.means[j]
            var = np.maximum(self.covars[j], self.min_covar)
            
            # Log multivariate diagonal Gaussian density formula
            diff = X - mean # (T, D)
            log_det = np.sum(np.log(2 * np.pi * var))
            mah_dist = np.sum((diff ** 2) / var, axis=1) # (T,)
            
            log_B[:, j] = -0.5 * (log_det + mah_dist)
            
        return log_B

    def forward(self, X):
        """
        Computes log-alpha matrix using the Forward algorithm.
        Returns: log_alpha (T, N), log_likelihood scalar
        """
        T = X.shape[0]
        log_B = self.compute_log_b(X)
        log_alpha = np.full((T, self.n_states), -np.inf)

        # Initialization at t = 0
        log_alpha[0, :] = self.log_pi + log_B[0, :]

        # Induction for t = 1 ... T-1
        for t in range(1, T):
            for j in range(self.n_states):
                prev_terms = log_alpha[t - 1, :] + self.log_A[:, j]
                log_alpha[t, j] = log_sum_exp(prev_terms) + log_B[t, j]

        log_likelihood = log_sum_exp(log_alpha[T - 1, :])
        return log_alpha, log_likelihood

    def backward(self, X):
        """
        Computes log-beta matrix using the Backward algorithm.
        Returns: log_beta (T, N)
        """
        T = X.shape[0]
        log_B = self.compute_log_b(X)
        log_beta = np.full((T, self.n_states), -np.inf)

        # Initialization at t = T-1
        log_beta[T - 1, :] = 0.0 # log(1.0) = 0.0

        # Induction for t = T-2 ... 0
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                next_terms = self.log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :]
                log_beta[t, i] = log_sum_exp(next_terms)

        return log_beta

    def viterbi(self, X):
        """
        Viterbi Decoding Algorithm. Finds optimal state path Q* and log-likelihood score.
        Returns: best_path (list of state indices), best_log_prob (float), delta matrix (T, N)
        """
        T = X.shape[0]
        log_B = self.compute_log_b(X)
        
        delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization at t = 0
        delta[0, :] = self.log_pi + log_B[0, :]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                scores = delta[t - 1, :] + self.log_A[:, j]
                best_prev_state = np.argmax(scores)
                delta[t, j] = scores[best_prev_state] + log_B[t, j]
                psi[t, j] = best_prev_state

        # Termination
        best_last_state = np.argmax(delta[T - 1, :])
        best_log_prob = delta[T - 1, best_last_state]

        # Path Backtracking
        best_path = [0] * T
        best_path[T - 1] = best_last_state
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        return best_path, best_log_prob, delta

    def fit(self, sequences, n_iter=15, tol=1e-4):
        """
        Baum-Welch (EM) Training across multiple feature observation sequences.
        Fits transition matrix log_A, means, and diagonal covars.
        """
        if len(sequences) == 0:
            return

        # 1. Initialize Means & Covariances using uniform temporal segmentation
        all_frames = np.vstack(sequences)
        n_total_frames = len(all_frames)
        
        # Segment sequences uniformly across N states to initialize Gaussian means
        state_frames = {j: [] for j in range(self.n_states)}
        for seq in sequences:
            T = len(seq)
            for t in range(T):
                assigned_state = min(int(self.n_states * t / T), self.n_states - 1)
                state_frames[assigned_state].append(seq[t])

        for j in range(self.n_states):
            if len(state_frames[j]) > 0:
                sf = np.array(state_frames[j])
                self.means[j] = np.mean(sf, axis=0)
                self.covars[j] = np.var(sf, axis=0) + self.min_covar
            else:
                self.means[j] = np.mean(all_frames, axis=0) + np.random.randn(self.n_features) * 0.1
                self.covars[j] = np.var(all_frames, axis=0) + self.min_covar

        prev_total_log_likelihood = -np.inf

        # 2. EM Loop
        for iteration in range(n_iter):
            total_log_likelihood = 0.0
            
            # Accumulators for parameter updates
            log_A_acc = np.full((self.n_states, self.n_states), -np.inf)
            gamma_acc = np.zeros((self.n_states, self.n_features)) # weighted mean accumulator
            gamma_sum_acc = np.zeros(self.n_states)
            covar_acc = np.zeros((self.n_states, self.n_features))

            for X in sequences:
                T = len(X)
                log_B = self.compute_log_b(X)
                log_alpha, log_p = self.forward(X)
                log_beta = self.backward(X)

                total_log_likelihood += log_p

                # Compute log gamma_t(i) = log_alpha_t(i) + log_beta_t(i) - log_p
                log_gamma = log_alpha + log_beta - log_p # (T, N)
                gamma = np.exp(log_gamma) # (T, N)

                # Compute log xi_t(i, j)
                log_xi = np.full((T - 1, self.n_states, self.n_states), -np.inf)
                for t in range(T - 1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            if self.A[i, j] > 0:
                                log_xi[t, i, j] = (log_alpha[t, i] + self.log_A[i, j] + 
                                                   log_B[t + 1, j] + log_beta[t + 1, j] - log_p)

                # Accumulate transition probabilities
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        if self.A[i, j] > 0:
                            sum_xi_ij = log_sum_exp(log_xi[:, i, j])
                            log_A_acc[i, j] = log_sum_exp(np.array([log_A_acc[i, j], sum_xi_ij]))

                # Accumulate emission parameters
                for j in range(self.n_states):
                    g_sum = np.sum(gamma[:, j])
                    gamma_sum_acc[j] += g_sum
                    gamma_acc[j] += np.sum(gamma[:, j, np.newaxis] * X, axis=0)

            # M-Step: Parameter Updates
            # Update A
            for i in range(self.n_states):
                row_log_sum = log_sum_exp(log_A_acc[i, :])
                if not np.isneginf(row_log_sum):
                    self.log_A[i, :] = log_A_acc[i, :] - row_log_sum
                    self.A[i, :] = np.exp(self.log_A[i, :])

            # Update Means and Covariances
            for j in range(self.n_states):
                if gamma_sum_acc[j] > 1e-6:
                    self.means[j] = gamma_acc[j] / gamma_sum_acc[j]

            # Re-accumulate covariances with updated means
            for X in sequences:
                log_alpha, log_p = self.forward(X)
                log_beta = self.backward(X)
                log_gamma = log_alpha + log_beta - log_p
                gamma = np.exp(log_gamma)
                for j in range(self.n_states):
                    diff = X - self.means[j]
                    covar_acc[j] += np.sum(gamma[:, j, np.newaxis] * (diff ** 2), axis=0)

            for j in range(self.n_states):
                if gamma_sum_acc[j] > 1e-6:
                    self.covars[j] = np.maximum(covar_acc[j] / gamma_sum_acc[j], self.min_covar)

            # Convergence Check
            diff_ll = total_log_likelihood - prev_total_log_likelihood
            if abs(diff_ll) < tol and iteration > 2:
                break
            prev_total_log_likelihood = total_log_likelihood

    def score(self, X):
        """Returns normalized log-likelihood score log P(O|lambda) / T."""
        if len(X) == 0:
            return -np.inf
        _, log_p = self.forward(X)
        return log_p / len(X)
