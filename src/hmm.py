"""Diagonal-covariance Gaussian HMM in NumPy.

Baum-Welch (EM) learns the start vector, the transition matrix, and a
diagonal Gaussian emission for each hidden state. Viterbi decodes the
single most likely state path. Forward and backward messages stay in
log space so a long recording does not underflow.
"""

from __future__ import annotations

import numpy as np


def logsumexp(values, axis):
    """log(sum(exp(values))) along axis, stable when values are large and negative."""
    peak = np.max(values, axis=axis, keepdims=True)
    total = peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))
    return np.squeeze(total, axis=axis)


class GaussianHMM:
    """HMM whose observations are continuous feature vectors.

    Emissions are diagonal Gaussians. A full covariance is unnecessary
    here: after z-scoring, the features are only weakly dependent, and a
    diagonal model has far fewer parameters than a full one on a few
    hundred windows.
    """

    def __init__(self, n_states, n_features, var_floor=1e-3):
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.var_floor = float(var_floor)
        self.pi = np.full(self.n_states, 1.0 / self.n_states)
        self.A = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        self.mu = np.zeros((self.n_states, self.n_features))
        self.var = np.ones((self.n_states, self.n_features))
        self.history_ = []

    def initialize_from_labels(self, sequences, label_seqs, prior=1.0):
        """Seed every parameter from labeled windows.

        `prior` is a Dirichlet count added to every transition so a jump
        that never appears in training stays possible. States that already
        match the activity names stay aligned when EM starts from here.
        """
        n = self.n_states
        pi = np.zeros(n)
        trans = np.full((n, n), float(prior))
        buckets = [[] for _ in range(n)]
        for obs, labels in zip(sequences, label_seqs):
            labels = np.asarray(labels, dtype=int)
            if len(labels) == 0:
                continue
            pi[labels[0]] += 1.0
            if len(labels) > 1:
                for left, right in zip(labels[:-1], labels[1:]):
                    trans[left, right] += 1.0
            for state in range(n):
                taken = obs[labels == state]
                if len(taken):
                    buckets[state].append(taken)
        if pi.sum() == 0:
            raise ValueError("no labeled frames to initialize from")
        self.pi = pi / pi.sum()
        self.A = trans / trans.sum(axis=1, keepdims=True)
        means = []
        variances = []
        for state in range(n):
            if not buckets[state]:
                means.append(np.zeros(self.n_features))
                variances.append(np.ones(self.n_features))
                continue
            block = np.concatenate(buckets[state], axis=0)
            means.append(block.mean(axis=0))
            variances.append(np.maximum(block.var(axis=0), self.var_floor))
        self.mu = np.stack(means)
        self.var = np.stack(variances)

    def _log_emit(self, obs):
        """Log density of each frame under each state's diagonal Gaussian."""
        variance = np.maximum(self.var, self.var_floor)
        delta = obs[:, None, :] - self.mu[None, :, :]
        quad = np.sum(delta * delta / variance[None, :, :], axis=2)
        log_det = np.sum(np.log(variance), axis=1)
        width = obs.shape[1]
        return -0.5 * (width * np.log(2.0 * np.pi) + log_det[None, :] + quad)

    def _forward(self, log_b):
        log_alpha = np.empty_like(log_b)
        log_alpha[0] = np.log(self.pi) + log_b[0]
        log_A = np.log(self.A)
        for t in range(1, len(log_b)):
            arrived = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = log_b[t] + logsumexp(arrived, axis=0)
        return log_alpha

    def _backward(self, log_b):
        log_beta = np.empty_like(log_b)
        log_beta[-1] = 0.0
        log_A = np.log(self.A)
        for t in range(len(log_b) - 2, -1, -1):
            leave = log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :]
            log_beta[t] = logsumexp(leave, axis=1)
        return log_beta

    def loglikelihood(self, obs):
        log_b = self._log_emit(obs)
        log_alpha = self._forward(log_b)
        return float(logsumexp(log_alpha[-1], axis=0))

    def expectations(self, obs):
        """One E-step: log-likelihood, state posteriors, transition posteriors."""
        log_b = self._log_emit(obs)
        log_alpha = self._forward(log_b)
        log_beta = self._backward(log_b)
        loglik = logsumexp(log_alpha[-1], axis=0)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)
        if len(obs) < 2:
            xi = np.zeros((0, self.n_states, self.n_states))
            return float(loglik), gamma, xi
        log_xi = (
            log_alpha[:-1, :, None]
            + np.log(self.A)[None, :, :]
            + log_b[1:][:, None, :]
            + log_beta[1:][:, None, :]
        )
        flat = log_xi.reshape(len(obs) - 1, -1)
        log_xi = log_xi - logsumexp(flat, axis=1)[:, None, None]
        return float(loglik), gamma, np.exp(log_xi)

    def fit(self, sequences, max_iter=40, tol=1e-6):
        """Baum-Welch. Stops when the relative change in log-likelihood is below `tol`.

        The likelihood stored for an iteration is the one under the parameters
        at the start of that iteration. EM updates after the check, so the
        last value is the converged likelihood and is not followed by a step
        that the stop test did not accept.
        """
        if not sequences:
            raise ValueError("fit needs at least one sequence")
        width = sequences[0].shape[1]
        history = []
        for iteration in range(max_iter):
            pi_acc = np.zeros(self.n_states)
            a_num = np.zeros((self.n_states, self.n_states))
            a_den = np.zeros(self.n_states)
            weight = np.zeros(self.n_states)
            moment = np.zeros((self.n_states, width))
            second = np.zeros((self.n_states, width))
            total = 0.0
            for obs in sequences:
                loglik, gamma, xi = self.expectations(obs)
                total += loglik
                pi_acc += gamma[0]
                weight += gamma.sum(axis=0)
                moment += gamma.T @ obs
                second += gamma.T @ (obs * obs)
                if len(xi):
                    a_num += xi.sum(axis=0)
                    a_den += gamma[:-1].sum(axis=0)
            history.append(total)
            if iteration > 0:
                rel = abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-8)
                if rel < tol:
                    break
            pi = np.maximum(pi_acc, 1e-12)
            self.pi = pi / pi.sum()
            transitions = a_num / np.maximum(a_den[:, None], 1e-12)
            transitions = np.maximum(transitions, 1e-8)
            self.A = transitions / transitions.sum(axis=1, keepdims=True)
            safe = np.maximum(weight[:, None], 1e-12)
            mu = moment / safe
            var = second / safe - mu * mu
            self.mu = mu
            self.var = np.maximum(var, self.var_floor)
        self.history_ = history
        return history

    def viterbi(self, obs):
        """Most likely state index for every frame."""
        log_b = self._log_emit(obs)
        log_delta = np.empty_like(log_b)
        back = np.zeros(log_b.shape, dtype=int)
        log_delta[0] = np.log(self.pi) + log_b[0]
        log_A = np.log(self.A)
        for t in range(1, len(obs)):
            scores = log_delta[t - 1][:, None] + log_A
            back[t] = np.argmax(scores, axis=0)
            log_delta[t] = log_b[t] + np.max(scores, axis=0)
        path = np.empty(len(obs), dtype=int)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(len(obs) - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        return path

    def align_to_labels(self, sequences, label_seqs):
        """Rename states so each index matches the majority training label.

        EM does not know the names of the activities. Matching is fit on
        training paths only; the test recordings are not used.
        """
        counts = np.zeros((self.n_states, self.n_states))
        for obs, labels in zip(sequences, label_seqs):
            path = self.viterbi(obs)
            for truth, guess in zip(labels, path):
                counts[int(truth), int(guess)] += 1
        pairs = [
            (counts[truth, guess], truth, guess)
            for truth in range(self.n_states)
            for guess in range(self.n_states)
        ]
        pairs.sort(reverse=True)
        pred_to_true = -np.ones(self.n_states, dtype=int)
        used_truth = set()
        for _, truth, guess in pairs:
            if pred_to_true[guess] >= 0 or truth in used_truth:
                continue
            pred_to_true[guess] = truth
            used_truth.add(truth)
        order = np.array([int(np.where(pred_to_true == i)[0][0]) for i in range(self.n_states)])
        self.pi = self.pi[order]
        self.A = self.A[np.ix_(order, order)]
        self.mu = self.mu[order]
        self.var = self.var[order]
        return order


def _sample(rng, pi, transitions, mu, var, length):
    state = int(rng.choice(len(pi), p=pi))
    obs = np.empty((length, mu.shape[1]))
    labels = np.empty(length, dtype=int)
    for t in range(length):
        labels[t] = state
        obs[t] = rng.normal(mu[state], np.sqrt(var[state]))
        state = int(rng.choice(len(pi), p=transitions[state]))
    return obs, labels


def self_test(seed=0):
    """Check Viterbi, posterior normalization, and that EM raises the likelihood."""
    rng = np.random.default_rng(seed)
    pi = np.array([0.65, 0.35])
    transitions = np.array([[0.92, 0.08], [0.12, 0.88]])
    mu = np.array([[0.0], [3.5]])
    var = np.array([[0.35], [0.35]])
    sequences = []
    labels = []
    for _ in range(12):
        obs, path = _sample(rng, pi, transitions, mu, var, 60)
        sequences.append(obs)
        labels.append(path)

    probe = GaussianHMM(2, 1)
    probe.pi = pi.copy()
    probe.A = transitions.copy()
    probe.mu = mu.copy()
    probe.var = var.copy()
    decoded = probe.viterbi(sequences[0])
    accuracy = float(np.mean(decoded == labels[0]))
    if accuracy < 0.85:
        raise AssertionError(f"Viterbi accuracy {accuracy:.3f} on a separated model")

    loglik, gamma, xi = probe.expectations(sequences[0])
    if not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("state posteriors do not sum to 1")
    if not np.allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-6):
        raise AssertionError("transition posteriors do not sum to 1")
    # The same marginal must fall out of forward and of forward-backward.
    log_b = probe._log_emit(sequences[0])
    log_alpha = probe._forward(log_b)
    log_beta = probe._backward(log_b)
    by_time = logsumexp(log_alpha + log_beta, axis=1)
    if not np.allclose(by_time, loglik, atol=1e-5):
        raise AssertionError("forward likelihood and forward-backward marginal disagree")

    learner = GaussianHMM(2, 1)
    learner.initialize_from_labels(sequences, labels)
    history = learner.fit(sequences, max_iter=20, tol=1e-9)
    steps = np.diff(history)
    if np.any(steps < -1e-3):
        raise AssertionError(f"log-likelihood fell during EM: {history}")
    return {
        "viterbi_accuracy": accuracy,
        "loglik_start": history[0],
        "loglik_end": history[-1],
        "iterations": len(history),
    }


if __name__ == "__main__":
    report = self_test()
    print(
        "self-test ok  "
        f"viterbi={report['viterbi_accuracy']:.3f}  "
        f"ll {report['loglik_start']:.2f} -> {report['loglik_end']:.2f}  "
        f"iters={report['iterations']}"
    )
