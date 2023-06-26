import math
import numpy as np
from scipy.special import gammaln  # log(gamma(x))
from scipy.special import psi  # digamma(x)
from scipy.special import polygamma
from scipy.special import logsumexp


class LDA:
    """
    Vanilla LDA optimized with variational EM, treating topics as parameters, with scalar smoothing parameter
    """

    def __init__(self, K, V, alpha=0.1):
        """
        Create an LDA model
        :param K: The number of topics (int)
        :param V: The size of the vocabulary
        :param alpha: Initial hyperparameter for document-topic distributions
        """
        self.K = K  # scalar number of topics
        self.V = V  # scalar size of vocabulary
        self.D = None  # scalar number of documents
        self.gammas = None  # D x K matrix of gammas
        self.log_betas = None  # V x K matrix of log(\beta)
        self.alpha = alpha  # scalar initial hyperparameter for p(\theta)
        self.bound = 0  # the variational bound
        self.token_bound = 0

    def fit(self, X, tolerance=1e-4, max_epochs=10, initial_smoothing=1.0, n_initial_docs=1,
            max_inner_iterations=20, inner_tol=1e-6, vocab=None, display_topics=False):
        """
        fit a model to data
        :param X: a list of documents (each a list of (word_index, count) tuples)
        :param tolerance: stopping criteria (relative change in bound)
        :param max_epochs: the maximum number of epochs
        :param initial_smoothing: smoothing to use when initializing topics
        :param n_initial_docs: the number of documents to use to randomly initialize each topic
        :param max_inner_iterations: maximum number of iterations for inner optimization loop (E-step)
        :param inner_tol: the tolerance for the inner optimization loop (E-step)
        :param vocab: a list of words in the vocabulary (used for displaying topics)
        :param display_topics: if True, print topics after each epoch
        :return: None
        """
        # set up initial values for monitoring convergence
        prev_bound = -np.inf
        delta = np.inf
        self.D = len(X)

        # initialize model parameters based on data
        self.init_parameters(X, initial_smoothing, n_initial_docs=n_initial_docs)

        # repeat optimization until convergence
        print("Iter bound\tLL/token\tDelta")
        for i in range(max_epochs):
            # update parameters (this also computes the bound)
            self.update_parameters(X, max_inner_iterations=max_inner_iterations, inner_tol=inner_tol)

            # compute the relative change in the bound
            if i > 0:
                delta = (prev_bound - self.bound) / float(prev_bound)

            # print progress
            print('%d\t%0.3f\t%0.3f\t%0.5f' % (i, self.bound, self.token_bound / self.D, delta))

            # store the new value of the bound
            prev_bound = self.bound

            # check for convergence
            if (0 < delta < tolerance) or (i + 1) >= max_epochs:
                break

            if vocab is not None and display_topics:
                self.print_topics(vocab)

    def init_parameters(self, X, initital_smoothing, n_initial_docs=1):
        """
        Initialize parameters using recommended values from the original LDA paper
        :param X: the data (as above)
        :param initial_smoothing: the amount of smoothing to use in initializing the topics
        :param n_initial_docs: the number of documents to use to initialize each topic
        :return: None
        """

        phi_total = np.zeros([self.V, self.K]) # V x K matrix of expected counts of each word type in each topic
        random_docs = list(np.random.choice(np.arange(self.D), size=n_initial_docs * self.K, replace=False))
        # initialize each topic with word counts from a subset of documents
        # 使用子集文档的词频计数初始化每个主题
        for k in range(self.K):
            docs = random_docs[k * n_initial_docs: (k + 1) * n_initial_docs]
            for d in docs:
                for w, c in X[d]:
                    phi_total[w, k] += c
        # smooth the counts - you should know why!
        # 平滑处理旨在避免计数为0的情况，提高模型的鲁棒性
        phi_total += initital_smoothing
        # compute the corresponding topics
        self.log_betas = self.compute_log_betas_mle(phi_total)
        self.gammas = np.zeros([self.D, self.K])

    def compute_log_betas_mle(self, phi_total):
        """
        M-step for topics: compute the values of log betas to maximize the bound
        :param phi_total: np.array (V,K): Expected number of each type of token assigned to each class k
        :return: np.array (V, K): log(beta)
        """
        # sum counts over vocbaulary
        topic_word_totals = np.sum(phi_total, axis=0)
        # compute new optimal values for log betas
        # log_betas表示每个词汇在每个主题下的概率对数值
        log_betas = np.log(phi_total) - np.log(topic_word_totals) 
        # avoid negative infinities
        log_betas[phi_total == 0] = -100
        return log_betas

    def update_parameters(self, X, max_inner_iterations=20, inner_tol=1e-6):
        """
        Do one epoch of updates for all parameters
        :param X: the data (D x V np.array)
        :param max_inner_iterations: the maximum number of iterations for optimizing document parameters
        :param inner_tol: the tolerance for optimizing  document parameters
        :return: None
        """
        self.bound = 0
        self.token_bound = 0
        phi_total = np.zeros_like(self.log_betas)

        # make one update for each document
        for d in range(self.D):
            if d % 1000 == 0 and d > 0:
                print(d)
            counts_d = X[d]

            # optimize the phi and gamma parameter for this document
            bound, phi_d, gammas = self.update_parameters_for_one_item(counts_d, max_iter_d=max_inner_iterations,
                                                                       tol_d=inner_tol)
            self.gammas[d, :] = gammas

            # only need to store the running sum of phi over the documents
            N_d = 0
            for n, (w, c) in enumerate(counts_d):
                phi_total[w, :] += c * phi_d[n, :]
                N_d += c

            # add the contribution of this document to the bound
            self.bound += bound
            self.token_bound += bound / float(N_d)

        # finally update the topic-word distributions and hyperparameters
        self.log_betas = self.compute_log_betas_mle(phi_total)
        self.update_alpha()

    def update_parameters_for_one_item(self, count_tuples, max_iter_d=20, tol_d=1e-6):
        """
        Update gamma and compute updates for beta and the bound for one document
        :param counts: the word counts for the corresponding document (length-V np.array)
        :param max_iter_d: the maximum number of epochs for this inner optimization
        :param tol_d: the tolerance required for convergence of the inner optimization problem
        :return: (contribution to the bound, phi values for this doc, gammas for this doc)
        """

        # unzip counts into lists of word indices and counts of those words
        word_indices, counts = zip(*count_tuples)
        # convert the lists into vectors of the required shapes
        word_indices = np.reshape(np.array(word_indices, dtype=np.int32), (len(word_indices),))
        counts = np.reshape(np.array(counts, dtype=np.int32), (len(word_indices),))
        count_vector_2d = np.reshape(np.array(counts), (len(word_indices), 1))

        # count the total number of words
        N_d = int(count_vector_2d.sum())
        # and the number of distinct word types
        n_word_types = len(word_indices)

        # initialize gamma values to alpha + 1/K
        gammas = self.alpha + N_d * np.ones(self.K) / float(self.K)
        # initialize phis to 1/K; only need to consider each word type
        phi_d = np.ones([n_word_types, self.K]) / float(self.K)

        # do the optimization step
        bound = update_params_for_one_item(self.K, n_word_types, word_indices, counts, gammas, phi_d, self.log_betas,
                                           self.alpha, max_iter_d, tol_d)

        return bound, phi_d, gammas

    def update_alpha(self, newton_thresh=1e-5, max_iter=1000):
        """
        Update hyperparameters of p(\theta) using Netwon's method [ported from lda-c]
        :param newton_thresh: tolerance for Newton optimization
        :param max_iter: maximum number of iterations
        :return: None
        """

        init_alpha = 100
        log_alpha = np.log(init_alpha)

        psi_gammas = psi(self.gammas)
        psi_sum_gammas = psi(np.sum(self.gammas, axis=1))
        E_ln_thetas = psi_gammas - np.reshape(psi_sum_gammas, (self.D, 1))
        sum_E_ln_theta = np.sum(E_ln_thetas)  # called ss (sufficient statistics) in lda-c

        # repeat until convergence
        print("alpha\tL(alpha)\tdL(alpha)")
        for i in range(max_iter):
            alpha = np.exp(log_alpha)
            if np.isnan(alpha):
                init_alpha *= 10
                print("warning : alpha is nan; new init = %0.5f" % init_alpha)
                alpha = init_alpha
                log_alpha = np.log(alpha)

            L_alpha = self.compute_L_alpha(alpha, sum_E_ln_theta)
            dL_alpha = self.compute_dL_alpha(alpha, sum_E_ln_theta)
            d2L_alpha = self.compute_d2L_alpha(alpha)
            log_alpha = log_alpha - dL_alpha / (d2L_alpha * alpha + dL_alpha)

            print("alpha maximization: %5.5f\t%5.5f\t%5.5f" % (np.exp(log_alpha), L_alpha, dL_alpha))
            if np.abs(dL_alpha) <= newton_thresh:
                break

        self.alpha = np.exp(log_alpha)

    def compute_L_alpha(self, alpha, sum_E_ln_theta):
        return self.D * (gammaln(self.K * alpha) - self.K * gammaln(alpha)) + (alpha - 1) * sum_E_ln_theta

    def compute_dL_alpha(self, alpha, sum_E_ln_theta):
        return self.D * (self.K * psi(self.K * alpha) - self.K * psi(alpha)) + sum_E_ln_theta

    def compute_d2L_alpha(self, alpha):
        return self.D * (self.K ** 2 * polygamma(1, self.K * alpha) - self.K * polygamma(1, alpha))

    def print_topics(self, vocab, n_words=8):
        """
        Display the top words in each topic
        :param vocab: a list of words in the vocabulary
        """
        for k in range(self.K):
            order = list(np.argsort(self.log_betas[:, k]).tolist())
            order.reverse()
            print("%d %s" % (k, ' '.join([vocab[i] for i in order[:n_words]])))


def update_params_for_one_item(K, n_word_types, word_indices, counts, gammas, phi_d, log_betas, alpha, max_iter_d,
                               tol_d):
    """
    Optimize the per-document variational parameters for one document, namely gamma and phi.
    :param K: the number of topics
    :param n_word_types: the number of word types in this document (length of word_indices)
    :param word_indices: a typed memory view of the indices of the words in the document
    :param counts: the corresponding counts of each word in the document
    :param gammas: the variational parameters of this document to be updated (length-K)
    :param phi_d: n_word_types x K memory view of expected distribution of topics for each word
    :param log_betas: V x K memoryview of the current value of the log of the topic distributions
    :param alpha: current value of the hyperparmeter alpha
    :param max_iter_d: the maximum number of iterations for this inner optimization loop
    :param tol_d: the tolerance required for convergence of this inner optimization loop
    """
    i = 0
    prev_bound = -1000000.0
    bound = 0.0
    delta = tol_d
    new_phi_dn = np.empty(K)

    psi_gammas = psi(gammas)

    # repeat until convergence
    while i < max_iter_d and delta >= tol_d:
        # process all the word index: count pairs in this documents
        n = 0
        while n < n_word_types:
            w = word_indices[n]
            c = counts[n]
            
            # TODO: update variational parameters inplace: gamma and phi
            log_phi_dn = log_betas[w, :] + psi_gammas
            max_log_phi_dn = np.max(log_phi_dn)
            exp_phi_dn = np.exp(log_phi_dn - max_log_phi_dn)
            phi_dn = exp_phi_dn / np.sum(exp_phi_dn)
            new_phi_dn[:] = phi_dn
            
            phi_d[n, :] = new_phi_dn
            gammas += c * new_phi_dn

            n += 1

        # compute the part of the variational bound corresponding to this document
        bound = compute_bound_for_one_item(K, n_word_types, word_indices, counts, alpha, gammas, psi_gammas, phi_d, log_betas)

        # compute the relative change in the bound
        delta = (prev_bound - bound) / prev_bound

        # save the new value of the bound
        prev_bound = bound
        i += 1

    return bound


def compute_bound_for_one_item(K, n_word_types, word_indices, count_vector, alpha, gammas, psi_gammas, phi_d,
                               log_betas):
    """
    Compute the parts of the variational bound corresponding to the particular document
    :param K: the number of topics
    :param n_word_types: the number of word types in this document (length of count_vector)
    :param word_indices: a vector of vocabulary indices of the word types in this document
    :param count_vector: the corresponding vector of counts of each word type
    :param alpha: the current value of the hyperparameter alpha
    :param gammas: the current value of gammas for this document
    :param psi_gammas: pre-computed values of psi(gammas)
    :param phi_d: the expected distribution of topics for each word type in this document
    :param log_betas: the current value of the log of the topic distributions
    """

    bound = 0.0

    # TODO: calculate ELBO, return it as bprintound                             
    for n in range(n_word_types):
        w = word_indices[n]
        c = count_vector[n]
        bound += c * (log_betas[w, :] @ phi_d[n, :])
    bound += np.sum((alpha - 1) * (psi_gammas - psi(np.sum(gammas))))
    bound += np.sum(phi_d.T * (log_betas[word_indices, :].T - np.log(phi_d.T)))
    return np.abs(bound)


if __name__ == "__main__":
    from utils import preprocess
    docs, _, vocab = preprocess("./dataset/dataset_cn_full.txt")
    lda = LDA(K=5, V=len(vocab))
    lda.fit(docs, vocab=vocab, display_topics=True)
