"""
Replications of the four data sets fit by SUSTAIN in
Love, Medin & Gureckis (2004), Psychological Review, 111(2), 309-332.

Each `run_*` function corresponds to one section of the paper:

  run_shepard_six_types        — Shepard, Hovland & Jenkins (1961);
                                 Nosofsky et al. (1994). Six classification
                                 problem types.
  run_medin_1983               — Medin, Dewey & Murphy (1983). First-name
                                 (identification) vs. last-name (categorisation).
  run_yamauchi_inference_class — Yamauchi & Markman (1998); Yamauchi et al.
                                 (2002). Inference vs. classification learning
                                 on linear / nonlinear category structures.
  run_billman_knutson_1996     — Billman & Knutson (1996) Experiments 2 & 3.
                                 Unsupervised correlation learning.

Best-fitting parameters are taken from Table 1 of the paper.
"""

from typing import Optional

import numpy as np

from src.sustain import SUSTAIN


def _trial_correct(probs: np.ndarray, target: int,
                   rng: np.random.Generator,
                   stochastic: bool) -> bool:
    """
    Decide whether a trial is "correct".

    If stochastic is True, sample a response from the Luce-choice
    distribution (Eq. 8); otherwise take the argmax (deterministic).

    The SUSTAIN paper does not specify this exactly: argmax under-counts
    blocks-to-criterion once weights become moderately confident,
    while sampling can over-count when d is small enough that response
    probabilities never saturate. Both options are exposed to 
    explore which fits the paper's reported numbers most closely.
    """
    if stochastic:
        response = int(rng.choice(len(probs), p=probs))
    else:
        response = int(np.argmax(probs))
    return response == target


# =============================================================================
# Shepard, Hovland & Jenkins (1961) — Six classification problem types
# =============================================================================

def run_shepard_six_types(
    n_simulations: int = 500,
    max_blocks: int = 32,
    criterion_blocks: int = 4,
    params: Optional[dict] = None,
    verbose: bool = True,
    plot: bool = True,
    seed: Optional[int] = None,
    stochastic_response: bool = False,
) -> dict:
    """
    Replicate the Shepard, Hovland & Jenkins (1961) six classification
    problem types as fit by SUSTAIN in Love et al. (2004).

    Stimuli have 3 binary perceptual dimensions + 1 binary category label
    (= 4 dimensions total). Dimension 3 (0-indexed) is the category label,
    queried on every trial.
    """
    if params is None:
        params = dict(r=9.01245, beta=1.252233, d=16.924073, eta=0.092327)

    rng = np.random.default_rng(seed)

    # Table 2 of Love et al. (2004): cols = Type I…VI
    type_assignments = {
        (0, 0, 0): [0, 0, 1, 1, 1, 1],
        (0, 0, 1): [0, 0, 1, 1, 1, 0],
        (0, 1, 0): [0, 1, 1, 1, 1, 0],
        (0, 1, 1): [0, 1, 0, 0, 0, 1],
        (1, 0, 0): [1, 1, 0, 1, 0, 0],
        (1, 0, 1): [1, 1, 1, 0, 0, 1],
        (1, 1, 0): [1, 0, 0, 0, 0, 1],
        (1, 1, 1): [1, 0, 0, 0, 1, 0],
    }

    results = {}
    for type_idx in range(6):
        type_id = type_idx + 1
        stimuli = []
        for (d1, d2, d3), labels in type_assignments.items():
            stimuli.append([d1, d2, d3, labels[type_idx]])

        blocks_list = []
        clusters_list = []
        block_error_rates = np.zeros((n_simulations, max_blocks))

        for sim_idx in range(n_simulations):
            model = SUSTAIN(
                r=params['r'], beta=params['beta'],
                d=params['d'], eta=params['eta'],
                supervised=True, queried_dim=3,
            )
            model.reset(dim_sizes=[2, 2, 2, 2])

            blocks_to_criterion = max_blocks
            consecutive_correct_blocks = 0
            criterion_reached = False

            for block in range(1, max_blocks + 1):
                block_stimuli = stimuli.copy()
                rng.shuffle(block_stimuli)

                block_correct = 0
                block_p_error_sum = 0.0
                for stim in block_stimuli:
                    target_val = stim[3]
                    result = model.present_stimulus(stim, queried_dim=3)
                    if _trial_correct(result['prob'], target_val, rng,
                                      stochastic_response):
                        block_correct += 1
                    block_p_error_sum += 1.0 - result['prob'][target_val]

                block_error_rates[sim_idx, block - 1] = (
                    block_p_error_sum / len(stimuli)
                )

                if block_correct == len(stimuli):
                    consecutive_correct_blocks += 1
                else:
                    consecutive_correct_blocks = 0

                if (not criterion_reached
                        and consecutive_correct_blocks >= criterion_blocks):
                    blocks_to_criterion = block - criterion_blocks + 1
                    criterion_reached = True

            blocks_list.append(blocks_to_criterion)
            clusters_list.append(model.n_clusters)

        results[type_id] = {
            'blocks': float(np.mean(blocks_list)),
            'blocks_std': float(np.std(blocks_list)),
            'n_clusters': float(np.mean(clusters_list)),
            'learning_curve': block_error_rates.mean(axis=0),
        }
        if verbose:
            print(f"  Type {type_id}: "
                  f"{results[type_id]['blocks']:.2f} blocks "
                  f"(±{results[type_id]['blocks_std']:.2f}), "
                  f"{results[type_id]['n_clusters']:.2f} clusters")

    if plot:
        import matplotlib.pyplot as plt
        type_labels = ['I', 'II', 'III', 'IV', 'V', 'VI']
        fig, ax = plt.subplots(figsize=(8, 6))
        for type_idx in range(6):
            ax.plot(
                range(1, max_blocks + 1),
                results[type_idx + 1]['learning_curve'],
                marker='o', markersize=4,
                label=f'Type {type_labels[type_idx]}',
            )
        ax.set_xlabel('Learning block')
        ax.set_ylabel('P(error)')
        ax.set_title('SUSTAIN — Shepard et al. (1961) six types')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results


# =============================================================================
# Medin, Dewey & Murphy (1983) — Identification vs categorisation
# =============================================================================

# Table 3 of Love et al. (2004) — paper uses values 1/2 and labels A–I, A/B.
# Converted to 0-indexed integers.
MEDIN_STIMULI = [
    # (d1, d2, d3, d4, first_name_label, last_name_label)
    (0, 0, 0, 1, 0, 0),   # 1 1 1 2  A A
    (0, 1, 0, 1, 1, 0),   # 1 2 1 2  B A
    (0, 1, 0, 0, 2, 0),   # 1 2 1 1  C A
    (0, 0, 1, 0, 3, 0),   # 1 1 2 1  D A
    (1, 0, 0, 0, 4, 0),   # 2 1 1 1  E A
    (0, 0, 1, 1, 5, 1),   # 1 1 2 2  F B
    (1, 0, 0, 1, 6, 1),   # 2 1 1 2  G B
    (1, 1, 1, 0, 7, 1),   # 2 2 2 1  H B
    (1, 1, 1, 1, 8, 1),   # 2 2 2 2  I B
]


def run_medin_1983(
    n_simulations: int = 200,
    max_blocks: int = 16,
    criterion_blocks: int = 2,
    params: Optional[dict] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    stochastic_response: bool = False,
) -> dict:
    """
    Replicate the first-name vs. last-name conditions of Medin, Dewey &
    Murphy (1983) as fit by SUSTAIN in Love et al. (2004).

    First-name condition (identification): each of the 9 stimuli has a
    unique label (9 categories total).
    Last-name condition (categorisation):  stimuli are partitioned into
    just 2 categories (A vs. B).

    Following the paper, an extra "distinctive" stimulus dimension carries
    a unique value for every item (idiosyncratic photographic detail), and
    that dimension is given an elevated initial tuning lambda_distinct.

    Expected (human / SUSTAIN, Love et al. Table 4):
        First name : 7.1 / 7.2 blocks
        Last name  : 9.7 / 9.7 blocks
    """
    if params is None:
        params = dict(
            r=4.349951, beta=5.925613, d=15.19877, eta=0.0807908,
            lambda_distinct=4.61733,
        )

    n_stim = len(MEDIN_STIMULI)
    n_perceptual = 4
    label_dim = 5

    rng = np.random.default_rng(seed)

    conditions = {
        'first_name': {'label_idx': 4, 'n_labels': 9},
        'last_name':  {'label_idx': 5, 'n_labels': 2},
    }

    results = {}
    for cond_name, cond in conditions.items():
        dim_sizes = [2, 2, 2, 2, n_stim, cond['n_labels']]
        initial_lambdas = [1.0] * n_perceptual + [params['lambda_distinct'], 1.0]

        blocks_list = []
        clusters_list = []
        for _ in range(n_simulations):
            model = SUSTAIN(
                r=params['r'], beta=params['beta'],
                d=params['d'], eta=params['eta'],
                supervised=True, queried_dim=label_dim,
            )
            model.reset(dim_sizes=dim_sizes, initial_lambdas=initial_lambdas)

            # Build stimuli for this condition
            stimuli = []
            for i, row in enumerate(MEDIN_STIMULI):
                d1, d2, d3, d4 = row[0], row[1], row[2], row[3]
                lab = row[cond['label_idx']]
                # Distinctive dim value = item index i (unique per stimulus)
                stimuli.append([d1, d2, d3, d4, i, lab])

            blocks_to_criterion = max_blocks
            consecutive_correct_blocks = 0

            for block in range(1, max_blocks + 1):
                block_stimuli = stimuli.copy()
                rng.shuffle(block_stimuli)

                block_correct = 0
                for stim in block_stimuli:
                    result = model.present_stimulus(stim, queried_dim=label_dim)
                    if _trial_correct(result['prob'], stim[label_dim], rng,
                                      stochastic_response):
                        block_correct += 1

                if block_correct == n_stim:
                    consecutive_correct_blocks += 1
                else:
                    consecutive_correct_blocks = 0

                if consecutive_correct_blocks >= criterion_blocks:
                    blocks_to_criterion = block - criterion_blocks + 1
                    break

            blocks_list.append(blocks_to_criterion)
            clusters_list.append(model.n_clusters)

        results[cond_name] = {
            'blocks': float(np.mean(blocks_list)),
            'blocks_std': float(np.std(blocks_list)),
            'n_clusters': float(np.mean(clusters_list)),
        }
        if verbose:
            print(f"  {cond_name:>10s}: "
                  f"{results[cond_name]['blocks']:.2f} blocks "
                  f"(±{results[cond_name]['blocks_std']:.2f}), "
                  f"{results[cond_name]['n_clusters']:.2f} clusters")

    return results


# =============================================================================
# Yamauchi & Markman (1998) + Yamauchi et al. (2002)
# Inference vs. classification learning
# =============================================================================

# Table 5 of Love et al. (2004) / Table 1 of Yamauchi & Markman (1998) —
# linear structure. Category prototypes are (1,1,1,1) and (0,0,0,0); each
# exemplar's "exception feature" is the dim where it differs from its
# prototype, and those dims are excluded from inference queries per the
# original procedure (Yamauchi & Markman 1998, p. 69).
YAMAUCHI_LINEAR = [
    # (d1, d2, d3, d4, category)
    (1, 1, 1, 0, 0), (1, 1, 0, 1, 0), (1, 0, 1, 1, 0), (0, 1, 1, 1, 0),
    (0, 0, 0, 1, 1), (0, 0, 1, 0, 1), (0, 1, 0, 0, 1), (1, 0, 0, 0, 1),
]
YAMAUCHI_LINEAR_PROTOTYPES = {0: (1, 1, 1, 1), 1: (0, 0, 0, 0)}

# Table 6 of Love et al. (2004): Yamauchi et al. (2002) — nonlinear structure.
# No prototype-based exception-feature exclusion is specified for the
# nonlinear case, so all four perceptual dims are queryable.
YAMAUCHI_NONLINEAR = [
    (1, 1, 1, 1, 0), (1, 1, 0, 0, 0), (0, 0, 1, 1, 0),
    (1, 1, 0, 1, 1), (0, 1, 1, 0, 1), (1, 0, 0, 0, 1),
]
YAMAUCHI_NONLINEAR_PROTOTYPES = None


def _queryable_perceptual_dims(stim, prototypes, n_perceptual: int) -> list[int]:
    """
    Return the perceptual dims that may be queried on an inference trial.
    For the linear structure (Yamauchi & Markman 1998), the dim that
    differs from the stim's category prototype is the "exception feature"
    and is excluded. For the nonlinear structure, all dims are queryable.
    """
    if prototypes is None:
        return list(range(n_perceptual))
    cat = stim[-1]
    proto = prototypes[cat]
    return [d for d in range(n_perceptual) if stim[d] == proto[d]]


def _yamauchi_accuracy_criterion(block_accs: list[float],
                                 threshold: float = 0.9,
                                 span: int = 3) -> bool:
    """True once the mean accuracy across the last `span` blocks reaches threshold."""
    if len(block_accs) < span:
        return False
    return float(np.mean(block_accs[-span:])) >= threshold


def run_yamauchi_inference_class(
    n_simulations: int = 200,
    max_blocks: int = 30,
    params: Optional[dict] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    stochastic_response: bool = False,
) -> dict:
    """
    Replicate Yamauchi & Markman (1998) and Yamauchi et al. (2002):
    inference vs. classification learning on linear and nonlinear
    category structures.

    Both tasks use the same 4-perceptual-dim + 1-category-label stimuli.
    - Classification: the category label is queried; perceptual dims given.
    - Inference: the category label is given; one perceptual dim is queried.

    Block construction (Yamauchi & Markman 1998, p. 69: "Each stimulus
    appeared once in each block"):
    - Classification block = each stimulus presented once (label queried).
    - Inference block      = each stimulus presented once with a single
      random non-exception perceptual dim queried. Exception features
      (the dim where the stim differs from its category prototype) are
      excluded from queries.

    Criterion: mean accuracy >= 90% across 3 consecutive blocks; max 30 blocks.

    Expected (humans / SUSTAIN, Love et al. Table 7):
        Linear    Inference 6.5 / 7.5,  Classification 12.3 / 11.2
        Nonlinear Inference 27.4 / 28.6, Classification 10.4 / 10.6
    """
    if params is None:
        params = dict(
            r=1.016423, beta=3.97491, d=6.514972, eta=0.1150532,
            lambda_label=5.150151,
        )

    n_perceptual = 4
    label_dim = 4
    dim_sizes = [2, 2, 2, 2, 2]
    initial_lambdas = [1.0] * n_perceptual + [params['lambda_label']]

    rng = np.random.default_rng(seed)

    structures = {
        'linear':    (YAMAUCHI_LINEAR, YAMAUCHI_LINEAR_PROTOTYPES),
        'nonlinear': (YAMAUCHI_NONLINEAR, YAMAUCHI_NONLINEAR_PROTOTYPES),
    }

    results = {}
    for struct_name, (stimuli_table, prototypes) in structures.items():
        for task in ('inference', 'classification'):
            blocks_list = []
            for _ in range(n_simulations):
                model = SUSTAIN(
                    r=params['r'], beta=params['beta'],
                    d=params['d'], eta=params['eta'],
                    supervised=True, queried_dim=label_dim,
                )
                model.reset(dim_sizes=dim_sizes, initial_lambdas=initial_lambdas)

                block_accs: list[float] = []
                blocks_to_criterion = max_blocks
                for block in range(1, max_blocks + 1):
                    # Build this block's trial list: every stim appears
                    # exactly once. For inference the queried dim is a
                    # fresh random non-exception perceptual dim per stim
                    # per block.
                    block_stim_order = list(stimuli_table)
                    rng.shuffle(block_stim_order)
                    if task == 'classification':
                        trials = [(list(stim), label_dim)
                                  for stim in block_stim_order]
                    else:  # inference
                        trials = []
                        for stim in block_stim_order:
                            queryable = _queryable_perceptual_dims(
                                stim, prototypes, n_perceptual)
                            q = int(rng.choice(queryable))
                            trials.append((list(stim), q))

                    n_correct = 0
                    for stim, q in trials:
                        result = model.present_stimulus(stim, queried_dim=q)
                        if _trial_correct(result['prob'], stim[q], rng,
                                          stochastic_response):
                            n_correct += 1
                    block_accs.append(n_correct / len(trials))

                    if _yamauchi_accuracy_criterion(block_accs):
                        # Block at which the 3-block span first hits criterion
                        blocks_to_criterion = block - 2
                        break

                blocks_list.append(blocks_to_criterion)

            key = f"{struct_name}_{task}"
            results[key] = {
                'blocks': float(np.mean(blocks_list)),
                'blocks_std': float(np.std(blocks_list)),
            }
            if verbose:
                print(f"  {key:>25s}: "
                      f"{results[key]['blocks']:.2f} blocks "
                      f"(±{results[key]['blocks_std']:.2f})")

    return results


# =============================================================================
# Billman & Knutson (1996) — Unsupervised correlation learning
# =============================================================================

def _billman_make_item(
    template: list,                  # length 7, values in {0,1,2} or None for "free"
    rng: np.random.Generator,
) -> list[int]:
    """Fill in the 'free' slots (None) of a 7-dim template with random ternary values."""
    return [
        int(rng.integers(0, 3)) if v is None else int(v)
        for v in template
    ]


def _billman_templates(
    condition: str, experiment: int,
) -> tuple[list[list], list[list[int]]]:
    """
    Return the templates and the correlation groups for a Billman &
    Knutson condition.

    Each "correlation group" is a list of dim indices that all share a
    single covarying value within a template (the participant could
    learn that group as a single rule, and any 2-of-N pair from the
    group is a valid target rule for the missing-parts test).

    Templates encode the correlations of Table 8 of Love et al. (2004):
        Exp 2  noninter.    : (v,v,*,*,*,*,*) for v in 0..2          (3 templates)
                              1 correlation group: [d0, d1]
        Exp 2  intercorr.   : (v,v,v,v,*,*,*) for v in 0..2          (3 templates)
                              1 correlation group: [d0, d1, d2, d3]
        Exp 3  noninter.    : (v,v,w,w,*,*,*)                        (9 templates)
                              2 correlation groups: [d0,d1] and [d2,d3]
        Exp 3  intercorr.   : (v,v,v,v,*,*,*) — same as Exp 2 intercorr.
    """
    if experiment == 2:
        if condition == 'nonintercorrelated':
            return ([[v, v, None, None, None, None, None] for v in range(3)],
                    [[0, 1]])
        elif condition == 'intercorrelated':
            return ([[v, v, v, v, None, None, None] for v in range(3)],
                    [[0, 1, 2, 3]])
    elif experiment == 3:
        if condition == 'nonintercorrelated':
            templates = [
                [v, v, w, w, None, None, None]
                for v in range(3) for w in range(3)
            ]
            return (templates, [[0, 1], [2, 3]])
        elif condition == 'intercorrelated':
            return ([[v, v, v, v, None, None, None] for v in range(3)],
                    [[0, 1, 2, 3]])
    raise ValueError(f"Unknown Billman condition/experiment: {condition}, {experiment}")


def _billman_make_pair(
    template: list,
    correlation_groups: list[list[int]],
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """
    Build a forced-choice pair from one template, following the
    "missing-parts method" of Billman & Knutson (1996, p. 463 and
    Table 2).

    A "target rule" is a pair of dims drawn from the SAME correlation
    group (i.e. two dims that actually covary in the training data).
    The distractor mispairs those two dims' values, violating that one
    correlation. Every other fixed-value dim — both the remaining dims
    in the target rule's correlation group AND any dims belonging to
    other correlation groups — is blanked, so the model cannot use a
    different correlation to make the judgment. Free (noise) dims stay
    visible.

    This matters: it makes the visible structure for both items consist
    of (2 correlated target-rule dims) + (free noise dims). For the
    distractor, two clusters now become equally-good matches (one
    matching the correct value on dim A, another on dim B), so cluster
    competition through Eq. 6 stops saturating to a single winner and
    `C_out` actually differs between the items.
    """
    correct = _billman_make_item(template, rng)
    fixed_positions = [i for i, v in enumerate(template) if v is not None]

    # Pick a real correlation group as the target rule, then two dims
    # within it.
    groups_with_2plus = [g for g in correlation_groups if len(g) >= 2]
    if groups_with_2plus:
        group = groups_with_2plus[int(rng.integers(0, len(groups_with_2plus)))]
        target = list(rng.choice(group, size=2, replace=False))
        target_a, target_b = int(target[0]), int(target[1])

        # Distractor mispairs the two target-rule values.
        new_val = (correct[target_b] + int(rng.integers(1, 3))) % 3
        distractor = correct.copy()
        distractor[target_b] = new_val

        # Blank every fixed dim that is NOT in the target rule — both the
        # other members of the target's correlation group and any dims
        # from other correlation groups.
        for i in fixed_positions:
            if i != target_a and i != target_b:
                correct[i] = -1
                distractor[i] = -1
    else:
        # Degenerate fallback: no usable correlation group.
        violated = fixed_positions[0]
        new_val = (correct[violated] + int(rng.integers(1, 3))) % 3
        distractor = correct.copy()
        distractor[violated] = new_val
    return correct, distractor


def run_billman_knutson_1996(
    n_simulations: int = 200,
    n_items_per_template: int = 8,
    n_study_blocks: int = 4,
    n_test_pairs: int = 45,
    params: Optional[dict] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """
    Replicate Billman & Knutson (1996) Experiments 2 & 3 as fit by SUSTAIN
    in Love et al. (2004).

    Stimuli: 7 ternary perceptual dimensions; one trivial (unitary)
    category-label dimension is appended so that SUSTAIN's machinery for
    output-unit activations can be reused at test.

    Procedure:
      Study phase (unsupervised):
        For each block, present a freshly sampled set of items per
        template — each item respects the correlation structure of its
        condition. Cluster recruitment is governed by Eq. 11 (activation
        below tau).
      Test phase (forced choice):
        45 pairs of (correct, distractor). The correct item preserves the
        condition's correlation structure; the distractor violates it.
        For each pair the model's preference probability is given by
        Eq. 8 applied across the C^out of the (single) category unit:
            Pr(correct) = exp(d * C_correct) / (exp(d * C_correct)
                                                + exp(d * C_distractor))

    Expected (humans / SUSTAIN, Love et al. Table 9):
        Exp 2 noninter.    0.62 / 0.66    intercorr.    0.73 / 0.78
        Exp 3 noninter.    0.66 / 0.60    intercorr.    0.77 / 0.78
    """
    if params is None:
        params = dict(
            r=9.998779, beta=6.396300, d=1.977312, eta=0.096564, tau=0.5,
        )

    n_perceptual = 7
    label_dim = 7
    # Unitary category dim (size 1) so SUSTAIN's plumbing has somewhere
    # to put the output unit; every study item has category value 0.
    dim_sizes = [3] * n_perceptual + [1]
    rng_global = np.random.default_rng(seed)

    experiments = [
        (2, 'nonintercorrelated'),
        (2, 'intercorrelated'),
        (3, 'nonintercorrelated'),
        (3, 'intercorrelated'),
    ]

    results = {}
    for experiment, condition in experiments:
        templates, correlation_groups = _billman_templates(condition, experiment)

        accuracies = []
        for sim_idx in range(n_simulations):
            sim_seed = int(rng_global.integers(0, 2**31 - 1))
            rng = np.random.default_rng(sim_seed)

            model = SUSTAIN(
                r=params['r'], beta=params['beta'],
                d=params['d'], eta=params['eta'], tau=params['tau'],
                supervised=False, queried_dim=label_dim,
            )
            model.reset(dim_sizes=dim_sizes)

            # ----- Study phase -----
            for _ in range(n_study_blocks):
                study_items = []
                for tmpl in templates:
                    for _ in range(n_items_per_template):
                        study_items.append(
                            _billman_make_item(tmpl, rng) + [0]
                        )
                rng.shuffle(study_items)
                for item in study_items:
                    model.present_stimulus(item, queried_dim=label_dim)

            # ----- Test phase: forced choice -----
            p_correct_sum = 0.0
            for _ in range(n_test_pairs):
                tmpl = templates[int(rng.integers(0, len(templates)))]
                correct, distractor = _billman_make_pair(
                    tmpl, correlation_groups, rng,
                )
                # Append unitary category dim as queried (value irrelevant
                # since the dim has only one possible value)
                correct_full = correct + [0]
                distractor_full = distractor + [0]

                c_correct = float(
                    model.query_output(correct_full, queried_dim=label_dim)[0]
                )
                c_distractor = float(
                    model.query_output(distractor_full, queried_dim=label_dim)[0]
                )
                # Eq. 8 across the two test items
                vals = params['d'] * np.array([c_correct, c_distractor])
                vals -= vals.max()
                exp_vals = np.exp(vals)
                p_correct_sum += float(exp_vals[0] / exp_vals.sum())
            accuracies.append(p_correct_sum / n_test_pairs)

        key = f"exp{experiment}_{condition}"
        results[key] = {
            'accuracy': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
        }
        if verbose:
            print(f"  {key:>30s}: "
                  f"{results[key]['accuracy']:.3f} "
                  f"(±{results[key]['accuracy_std']:.3f})")

    return results


# =============================================================================
# Driver
# =============================================================================

def demo_simple_classification():
    """Single-dim category learning sanity check (Type-I-like rule)."""
    print("\n--- Simple unidimensional classification demo ---")
    model = SUSTAIN(
        r=2.844642, beta=2.386305, d=12.0, eta=0.09361126,
        supervised=True, queried_dim=1,
    )
    model.reset(dim_sizes=[2, 2])
    stimuli = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [1, 1]]
    for i, stim in enumerate(stimuli):
        res = model.present_stimulus(stim, queried_dim=1)
        print(f"  Trial {i+1}: stim={stim}, response={res['response']}, "
              f"correct={res['correct']}, clusters={res['n_clusters']}")


if __name__ == "__main__":
    np.random.seed(42)

    demo_simple_classification()

    print("\n--- Shepard, Hovland & Jenkins (1961) six types ---")
    print("Expected order of difficulty: Type I < II < III ≈ IV ≈ V < VI\n")
    shepard = run_shepard_six_types(
        n_simulations=100, plot=False, seed=42,
        stochastic_response=True,
    )

    print("\n--- Medin, Dewey & Murphy (1983) ---")
    print("Expected (humans/SUSTAIN, Table 4): "
          "first_name 7.1/7.2, last_name 9.7/9.7\n")
    medin = run_medin_1983(
        n_simulations=100, seed=42, stochastic_response=True,
    )

    print("\n--- Yamauchi & Markman (1998) + Yamauchi et al. (2002) ---")
    print("Expected (humans/SUSTAIN, Table 7):")
    print("  linear    inference     6.5 /  7.5")
    print("  linear    classification 12.3 / 11.2")
    print("  nonlinear inference     27.4 / 28.6")
    print("  nonlinear classification 10.4 / 10.6\n")
    yamauchi = run_yamauchi_inference_class(
        n_simulations=50, seed=42, stochastic_response=True,
    )

    print("\n--- Billman & Knutson (1996) Experiments 2 & 3 ---")
    print("Expected (humans/SUSTAIN, Table 9):")
    print("  Exp 2 noninter.     0.62 / 0.66    intercorr.    0.73 / 0.78")
    print("  Exp 3 noninter.     0.66 / 0.60    intercorr.    0.77 / 0.78\n")
    billman = run_billman_knutson_1996(n_simulations=50, seed=42)
