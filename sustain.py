"""
SUSTAIN: Supervised and Unsupervised STratified Adaptive Incremental Network
Implementation based on:
    Love, B.C., Medin, D.L., & Gureckis, T.M. (2004). SUSTAIN: A Network Model
    of Category Learning. Psychological Review, 111(2), 309-332.

SUSTAIN is a clustering model of human category learning. It starts with a
single cluster and recruits new clusters in response to surprising events
(prediction errors in supervised learning, or low similarity in unsupervised).

Key equations from the paper:
    Eq 1:  Receptive field response: lambda(delta) = exp(-lambda * delta)
    Eq 4:  Distance between stimulus and cluster on dim i
    Eq 5:  Cluster activation (normalized, attention-weighted)
    Eq 6:  Cluster competition (lateral inhibition / winner-take-all output)
    Eq 7:  Output unit activation (from winning cluster)
    Eq 8:  Response probability (Luce choice rule with decision consistency d)
    Eq 9:  Humble teacher target signal
    Eq 10: Supervised cluster recruitment (on prediction error)
    Eq 11: Unsupervised cluster recruitment (on low activation)
    Eq 12: Cluster position update (Kohonen rule)
    Eq 13: Receptive field tuning update
    Eq 14: Weight update (delta rule)
"""

import numpy as np
from typing import Optional


class SUSTAIN:
    """
    SUSTAIN category learning model.

    Parameters
    ----------
    r : float
        Attentional focus parameter (>=0). Higher r emphasises dimensions
        with tighter (more predictive) tunings.
    beta : float
        Lateral inhibition / cluster competition parameter (>=0). Higher beta
        means the winner is less inhibited by competitors.
    d : float
        Decision consistency parameter (>=0). Higher d → responses approach
        the maximum-output unit (more deterministic).
    eta : float
        Learning rate for position updates, tuning updates, and weight updates.
    tau : float
        Unsupervised recruitment threshold in (0, 1). A new cluster is
        recruited when the winning cluster's activation < tau.
    supervised : bool
        If True, use supervised learning (corrective feedback on a queried
        dimension). If False, use unsupervised learning (no external feedback).
    queried_dim : int or None
        Index of the stimulus dimension that is queried (unknown) on each
        trial. Typically the last dimension (the category label) for
        classification learning. In unsupervised mode this is ignored.
    """

    def __init__(
        self,
        r: float = 9.01245, #2.844642,
        beta: float = 1.252233, #2.386305,
        d: float = 16.924073, #12.0,
        eta: float = 0.092327, #0.09361126,
        tau: float = 0.5,
        supervised: bool = True,
        queried_dim: int = -1,
    ):
        self.r = r
        self.beta = beta
        self.d = d
        self.eta = eta
        self.tau = tau
        self.supervised = supervised
        self.queried_dim = queried_dim  # will be resolved after first stimulus

        # These are set after the first call to reset() or the first trial
        self.n_dims: int = 0
        self.dim_sizes: list[int] = []    # number of values per dimension
        self.dim_offsets: list[int] = []  # start index in flat input vector

        # Cluster state (lists grow as clusters are recruited)
        self.n_clusters: int = 0
        self.H_pos: list[np.ndarray] = []   # cluster positions (flat vectors)
        self.lambdas: np.ndarray = np.array([])  # tunings per dimension (shape: n_dims)
        self.weights: list[np.ndarray] = []  # weight matrices per cluster

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_queried_dim(self, n_dims: int) -> int:
        q = self.queried_dim
        if q < 0:
            q = n_dims + q
        return q

    def _setup(self, dim_sizes: list[int]):
        """Initialise internal structure for given stimulus dimensionality."""
        self.n_dims = len(dim_sizes)
        self.dim_sizes = list(dim_sizes)
        self.dim_offsets = []
        offset = 0
        for s in dim_sizes:
            self.dim_offsets.append(offset)
            offset += s
        self.flat_size = offset

        # Initial tuning: lambda_i = 1 for all dimensions (broadly tuned)
        self.lambdas = np.ones(self.n_dims, dtype=float)

        # No clusters yet
        self.n_clusters = 0
        self.H_pos = []
        self.weights = []

        self._queried_dim_idx = self._resolve_queried_dim(self.n_dims)

    def reset(
        self,
        dim_sizes: list[int],
        initial_lambdas: Optional[list[float]] = None,
    ):
        """
        Reset the model for a new learning episode.

        Parameters
        ----------
        dim_sizes : list of int
            Number of nominal values per dimension.
        initial_lambdas : list of float, optional
            Initial receptive-field tuning per dimension. Defaults to 1.0
            for every dimension (the paper's standard setting). Studies that
            give certain dimensions different a priori salience override this
            (e.g., lambda_label in Yamauchi et al. 2002,
            lambda_distinct in Medin et al. 1983).
        """
        self._setup(dim_sizes)
        if initial_lambdas is not None:
            if len(initial_lambdas) != self.n_dims:
                raise ValueError(
                    f"initial_lambdas has length {len(initial_lambdas)} "
                    f"but dim_sizes has {self.n_dims} dimensions."
                )
            self.lambdas = np.array(initial_lambdas, dtype=float)

    # ------------------------------------------------------------------
    # Encoding / decoding helpers
    # ------------------------------------------------------------------

    def _encode_stimulus(self, stimulus: list[int]) -> np.ndarray:
        """
        Convert a nominal stimulus (list of integer values, one per dimension)
        into a flat one-hot vector I^pos.

        Parameters
        ----------
        stimulus : list of int
            Each entry is the value (0-indexed) on that dimension.
            Use -1 to mark a dimension as queried/unknown.

        Returns
        -------
        np.ndarray of shape (flat_size,)
        """
        vec = np.zeros(self.flat_size, dtype=float)
        for i, val in enumerate(stimulus):
            if val >= 0:  # known dimension
                idx = self.dim_offsets[i] + val
                vec[idx] = 1.0
        return vec

    def _dim_slice(self, dim: int):
        """Return slice for dimension dim in the flat vector."""
        start = self.dim_offsets[dim]
        end = start + self.dim_sizes[dim]
        return slice(start, end)

    # ------------------------------------------------------------------
    # Core equations
    # ------------------------------------------------------------------

    def _distance(self, I_pos: np.ndarray, H_pos_j: np.ndarray, dim: int) -> float:
        """
        Eq. 4 (Love et al. 2004, p. 314): nominal-dimension distance.

            mu_ij = (1/2) * sum_{k=1}^{v_i} |I^pos_ik - H^pos_jk|

        v_i is the upper limit of the summation (the number of values on
        the dim), NOT a denominator. The (1/2) factor alone bounds mu in
        [0, 1] because one-hot input vectors yield a maximum absolute-
        difference sum of 2 regardless of v_i.
        """
        sl = self._dim_slice(dim)
        diff = I_pos[sl] - H_pos_j[sl]
        return 0.5 * float(np.sum(np.abs(diff)))

    def _receptive_field(self, lam: float, delta: float) -> float:
        """
        Exponential kernel used inside Eq. 5: exp(-lambda * delta).

        (This is NOT the full Eq. 1 receptive field alpha(mu) = lambda *
        exp(-lambda * mu) — that form integrates to 1 over mu, but Eq. 5
        uses just the exponential and normalises across dimensions via
        the (lambda)^r weights instead.)
        """
        return np.exp(-lam * delta)

    def _cluster_activation(
        self, I_pos: np.ndarray, H_pos_j: np.ndarray, known_dims: list[int]
    ) -> float:
        """
        Eq. 5: Activation of a cluster.
        Only known (non-queried) dimensions contribute.
        """
        numerator = 0.0
        denominator = 0.0
        for i in known_dims:
            lam_i = self.lambdas[i]
            delta_ij = self._distance(I_pos, H_pos_j, i)
            rf = self._receptive_field(lam_i, delta_ij)
            weight = lam_i ** self.r
            numerator += weight * rf
            denominator += weight
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def _cluster_output(self, activations: np.ndarray) -> np.ndarray:
        """
        Eq. 6 (Love et al. 2004, p. 315): lateral-inhibition output.

            H^out_j = (H^act_j)^beta / sum_i (H^act_i)^beta   for the winner
            H^out_j = 0                                       for all other clusters

        Power-normalisation: larger beta lets the winner dominate the
        denominator (less inhibited); smaller beta spreads mass across
        competitors (more inhibited). The returned array has the same
        shape / dtype as `activations`, so downstream callers
        (`_output_units`, `query_output`) are unaffected.
        """
        if len(activations) == 0:
            return np.array([])
        winner = int(np.argmax(activations))
        powered = np.power(activations, self.beta)
        denom = float(powered.sum())
        H_out = np.zeros(len(activations))
        if denom > 0:
            H_out[winner] = powered[winner] / denom
        return H_out

    def _output_units(self, H_out: np.ndarray, dim: int) -> np.ndarray:
        """
        Eq. 7: Activation of output units for the queried dimension.
        C^out_zk = sum_j (w_j,zk * H^out_j)
        """
        v_z = self.dim_sizes[dim]
        C_out = np.zeros(v_z)
        for j in range(self.n_clusters):
            if H_out[j] != 0.0:
                C_out += self.weights[j][dim] * H_out[j]
        return C_out

    def _response_prob(self, C_out: np.ndarray) -> np.ndarray:
        """
        Eq. 8: Response probabilities using Luce choice rule.
        Pr(k) = exp(d * C^out_k) / sum_k exp(d * C^out_k)
        """
        scaled = self.d * C_out
        scaled -= scaled.max()  # numerical stability
        exp_vals = np.exp(scaled)
        return exp_vals / exp_vals.sum()

    def _humble_teacher(self, C_out: np.ndarray, I_pos: np.ndarray, dim: int) -> np.ndarray:
        """
        Eq. 9: Humble teacher target signal.
        For correct values: t_zk = max(C^out_zk, 1)  [if I^pos_zk == 1]
        For incorrect values: t_zk = min(C^out_zk, 0)  [if I^pos_zk == 0]
        """
        sl = self._dim_slice(dim)
        I_dim = I_pos[sl]
        t = np.where(I_dim == 1,
                     np.maximum(C_out, 1.0),
                     np.minimum(C_out, 0.0))
        return t

    def _recruit_cluster(self, I_pos: np.ndarray):
        """
        Recruit a new cluster centred on the current stimulus.
        Weights are initialised to zero.
        """
        new_pos = I_pos.copy()
        self.H_pos.append(new_pos)
        # Weights: one array per dimension, shape (dim_size,)
        new_weights = [np.zeros(s, dtype=float) for s in self.dim_sizes]
        self.weights.append(new_weights)
        self.n_clusters += 1

    def _update_position(self, winner: int, I_pos: np.ndarray):
        """Eq. 12: Kohonen position update for the winning cluster."""
        self.H_pos[winner] += self.eta * (I_pos - self.H_pos[winner])

    def _update_tuning(self, winner: int, I_pos: np.ndarray, known_dims: list[int]):
        """
        Eq. 13: Receptive field tuning update.
        lambda_i += exp(-lambda_i * delta_ij) * (1 - lambda_i * delta_ij)
        (Only for known dimensions; only the winner updates lambdas.)
        """
        for i in known_dims:
            delta_ij = self._distance(I_pos, self.H_pos[winner], i)
            lam_i = self.lambdas[i]
            self.lambdas[i] += self.eta * np.exp(-lam_i * delta_ij) * (1 - lam_i * delta_ij)
            # lambda cannot go below 1 (per paper: initial value = 1, cannot decrease)
            self.lambdas[i] = max(self.lambdas[i], 1.0)

    def _update_weights(self, winner: int, H_out_winner: float,
                        C_out: np.ndarray, t: np.ndarray, dim: int):
        """Eq. 14: Delta rule weight update for the queried dimension."""
        delta_w = (t - C_out) * H_out_winner
        self.weights[winner][dim] += self.eta * delta_w

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def present_stimulus(
        self, stimulus: list[int], queried_dim: Optional[int] = None
    ) -> dict:
        """
        Present one stimulus to SUSTAIN and perform one learning step.

        Parameters
        ----------
        stimulus : list of int
            Nominal values for each dimension. Use -1 for the queried
            (unknown) dimension in supervised mode.
        queried_dim : int or None
            Override the default queried dimension for this trial.

        Returns
        -------
        dict with keys:
            'response'       : int  – chosen value on the queried dimension
            'prob'           : np.ndarray – response probabilities
            'correct'        : bool – whether the response matched target
            'n_clusters'     : int  – current number of clusters
            'winner'         : int  – index of winning cluster
            'recruited'      : bool – whether a new cluster was recruited
        """
        # Lazy initialisation on the first trial
        if self.n_dims == 0:
            dim_sizes = []
            for v in stimulus:
                # For unknown dims (-1) we still need the size; caller should
                # have called reset() first.  Raise a helpful error.
                if v < 0:
                    raise ValueError(
                        "Call reset(dim_sizes) before presenting stimuli, or "
                        "ensure the first stimulus has all known dimensions."
                    )
                dim_sizes.append(max(v + 1, 2))  # minimal inference
            self._setup(dim_sizes)

        q_dim = queried_dim if queried_dim is not None else self._queried_dim_idx

        # Identify known / unknown dimensions
        known_dims = [i for i, v in enumerate(stimulus) if v >= 0 and i != q_dim]
        # The target value (correct answer) is stored in stimulus[q_dim]
        # stimulus may have -1 for the queried dim (it is unknown to model)
        target_val = stimulus[q_dim]  # -1 means we don't know (unsupervised)

        # Build flat input vector (queried dimension excluded)
        I_pos = self._encode_stimulus(stimulus)
        # Also build full stimulus vector (including queried dim) for updates
        full_stimulus = list(stimulus)
        if target_val >= 0:
            full_I_pos = self._encode_stimulus(full_stimulus)
        else:
            full_I_pos = I_pos.copy()

        # --- Bootstrap: first cluster on first stimulus ---
        recruited = False
        if self.n_clusters == 0:
            self._recruit_cluster(full_I_pos)
            recruited = True

        # --- Step 1: Compute cluster activations (Eq. 5) ---
        activations = np.array([
            self._cluster_activation(I_pos, self.H_pos[j], known_dims)
            for j in range(self.n_clusters)
        ])

        # --- Step 2: Cluster competition / output (Eq. 6) ---
        H_out = self._cluster_output(activations)
        winner = int(np.argmax(H_out))

        # --- Unsupervised recruitment (Eq. 11) ---
        if not self.supervised:
            if activations[winner] < self.tau:
                self._recruit_cluster(full_I_pos)
                recruited = True
                activations = np.array([
                    self._cluster_activation(I_pos, self.H_pos[j], known_dims)
                    for j in range(self.n_clusters)
                ])
                H_out = self._cluster_output(activations)
                winner = int(np.argmax(H_out))

        # --- Step 3: Output units for queried dimension (Eq. 7) ---
        C_out = self._output_units(H_out, q_dim)

        # --- Step 4: Response probabilities and choice (Eq. 8) ---
        if np.all(C_out == 0):
            probs = np.ones(self.dim_sizes[q_dim]) / self.dim_sizes[q_dim]
        else:
            probs = self._response_prob(C_out)
        response = int(np.argmax(probs))
        correct = (target_val >= 0) and (response == target_val)

        # --- Step 5: Supervised cluster recruitment (Eq. 10) ---
        if self.supervised and target_val >= 0:
            # Recruit if the most activated output unit is NOT the correct one
            if not correct and not recruited:
                self._recruit_cluster(full_I_pos)
                recruited = True
                # Recalculate everything with the new cluster
                activations = np.array([
                    self._cluster_activation(I_pos, self.H_pos[j], known_dims)
                    for j in range(self.n_clusters)
                ])
                H_out = self._cluster_output(activations)
                winner = int(np.argmax(H_out))
                C_out = self._output_units(H_out, q_dim)
                if np.all(C_out == 0):
                    probs = np.ones(self.dim_sizes[q_dim]) / self.dim_sizes[q_dim]
                else:
                    probs = self._response_prob(C_out)
                response = int(np.argmax(probs))

        # --- Step 6: Learning updates (only when feedback is available) ---
        if target_val >= 0:
            # Humble teacher target (Eq. 9)
            t = self._humble_teacher(C_out, full_I_pos, q_dim)

            # Update weights (Eq. 14)
            self._update_weights(winner, H_out[winner], C_out, t, q_dim)

            # Update cluster position (Eq. 12)
            self._update_position(winner, full_I_pos)

            # Update tuning (Eq. 13) — known dims + queried dim (full stimulus)
            all_dims = list(range(self.n_dims))
            self._update_tuning(winner, full_I_pos, all_dims)

        return {
            'response': response,
            'prob': probs,
            'correct': correct,
            'n_clusters': self.n_clusters,
            'winner': winner,
            'recruited': recruited,
            'activations': activations,
        }

    def predict(self, stimulus: list[int], queried_dim: Optional[int] = None) -> np.ndarray:
        """
        Return response probabilities for a stimulus without updating the model.
        Unknown values in stimulus should be marked with -1.
        """
        q_dim = queried_dim if queried_dim is not None else self._queried_dim_idx
        known_dims = [i for i, v in enumerate(stimulus) if v >= 0 and i != q_dim]
        I_pos = self._encode_stimulus(stimulus)

        if self.n_clusters == 0:
            return np.ones(self.dim_sizes[q_dim]) / self.dim_sizes[q_dim]

        activations = np.array([
            self._cluster_activation(I_pos, self.H_pos[j], known_dims)
            for j in range(self.n_clusters)
        ])
        H_out = self._cluster_output(activations)
        C_out = self._output_units(H_out, q_dim)

        if np.all(C_out == 0):
            return np.ones(self.dim_sizes[q_dim]) / self.dim_sizes[q_dim]
        return self._response_prob(C_out)

    def query_output(self, stimulus: list[int], queried_dim: Optional[int] = None) -> np.ndarray:
        """
        Return the raw C^out activations (Eq. 7) for the queried dimension
        without applying the Luce choice rule. Needed for forced-choice tasks
        such as Billman & Knutson's (1996) test phase, where Eq. 8 is applied
        across two stimuli rather than across values of a single dimension.
        """
        q_dim = queried_dim if queried_dim is not None else self._queried_dim_idx
        known_dims = [i for i, v in enumerate(stimulus) if v >= 0 and i != q_dim]
        I_pos = self._encode_stimulus(stimulus)

        if self.n_clusters == 0:
            return np.zeros(self.dim_sizes[q_dim])

        activations = np.array([
            self._cluster_activation(I_pos, self.H_pos[j], known_dims)
            for j in range(self.n_clusters)
        ])
        H_out = self._cluster_output(activations)
        return self._output_units(H_out, q_dim)
