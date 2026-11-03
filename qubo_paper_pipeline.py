"""
Reproduction pipeline for
  Verma & Lewis, "Penalty and partitioning techniques to improve performance of QUBO solvers",
  Discrete Optimization 44 (2022) 100594.

NOTATION (same as the paper)
  Q        n x n UPPER-TRIANGULAR matrix:  f(x) = x'Qx = sum_i Q[i,i] x_i + sum_{i<j} Q[i,j] x_i x_j
  A, b     constraint  A x = b  (or  A x <= b);  here A = all-ones (cardinality constraint)
  M        penalty scalar
  s, D     slack variables and their coefficients   A x + D s = b
  x_from, x_to   solution vectors before / after a 1-flip move
  M_est    estimate of the lower bound of M, Eq. (3)
Indices are 0-based in code (the ORLIB file is 1-based).

USAGE
  python qubo_paper_pipeline.py selftest
  python qubo_paper_pipeline.py table1 --file bqp2500.txt --instance 4
  python qubo_paper_pipeline.py table2 --file bqp2500.txt --instance 4 --seconds 500
  python qubo_paper_pipeline.py table3 --file bqp2500.txt --instance 4 --b 250 --seconds 500
  --solver tabu | pr | popl   (popl = Lewis & Kochenberger 2016 solver, default tabu)
  --report penalised | feasible  (penalised = what the paper's tables show; default)
"""
import argparse
import itertools
import time
import numpy as np
import gzip
import pandas as pd


# ------------------------------------------------------------------ data
def load_orlib(path, instance=1):
    """Read Beasley's bqp2500.txt and return the upper-triangular Q of the paper's MIN problem.
    File: #instances, then per instance 'n m' and m lines 'i j v' (1-indexed, i<=j, zeros omitted).
    ORLIB is  max x'Vx  with V symmetric (V_ij = V_ji = v), so the x_i x_j term has coefficient 2v.
        Q[i,i] = -v        Q[i,j] = -2v  (i<j)        (negated: max -> min)
    `instance` is 1-based (instance=4 -> the paper's 'Q4')."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    tok = np.array(data.decode().split(), dtype=float)
    n_inst, pos = int(tok[0]), 1
    for k in range(1, n_inst + 1):
        n, m = int(tok[pos]), int(tok[pos + 1])
        pos += 2
        if k != instance:
            pos += 3 * m
            continue
        t = tok[pos:pos + 3 * m].reshape(m, 3)
        i = np.minimum(t[:, 0], t[:, 1]).astype(int) - 1
        j = np.maximum(t[:, 0], t[:, 1]).astype(int) - 1
        v = t[:, 2]
        Q = np.zeros((n, n))
        Q[i, j] = -v * np.where(i == j, 1.0, 2.0)
        return Q, m
    raise ValueError("instance not found")


def objective(Q, x):
    """x'Qx for upper-triangular Q."""
    x = np.asarray(x, float)
    return float(x @ (Q @ x))


def symmetrize(Q):
    """q = diagonal of Q;  Qs[i,j] = Qs[j,i] = Q[i,j] for i<j  (needed for O(n) flip updates)."""
    U = np.triu(Q, 1)
    return np.diag(Q).copy(), U + U.T


# --------------------------------------------------------- Section 2: M_est
def candidate_vectors(Q, labeling="paper"):
    """Explicit (x_from, x_to) pairs of Remark 1 - slow, for small n / teaching / testing.
    labeling="paper"    : Delta_1 = x_i 0->1 with positive neighbours on,  Delta_2 = x_i 1->0 with negative neighbours on
    labeling="corrected": the same neighbour sets with the flip direction reversed, which is
                          what actually *improves* f (see report)."""
    n = len(Q)
    Qs = symmetrize(Q)[1]
    out = []
    for i in range(n):
        pos = (Qs[i] > 0)
        neg = (Qs[i] < 0)
        # neighbour set for Delta_1 (positive q_ij on)
        base = pos.astype(int)
        base[i] = 0
        a0, a1 = base.copy(), base.copy()
        a1[i] = 1          # x_i = 0 / x_i = 1
        out.append((i, "D1", (a0, a1) if labeling == "paper" else (a1, a0)))
        # neighbour set for Delta_2 (negative q_ij on)
        base = neg.astype(int); base[i] = 0
        b0, b1 = base.copy(), base.copy(); b1[i] = 1
        out.append((i, "D2", (b1, b0) if labeling == "paper" else (b0, b1)))
    return out


def estimate_M_explicit(Q, A, b, mode="strict", labeling="corrected"):
    """Literal, slow version in the style of the illustration (uses objective() differences)."""
    A = np.asarray(A, float)
    ratios = []
    for i, tag, (x_from, x_to) in candidate_vectors(Q, labeling):
        lhs_from, lhs_to = A @ x_from, A @ x_to
        if mode == "strict" and not np.isclose(lhs_from, b):
            continue
        if np.isclose(lhs_to, b):
            continue
        improvement = objective(Q, x_from) - objective(Q, x_to)
        if improvement > 0:
            ratios.append(improvement / (lhs_to - b) ** 2)
    return max(ratios) if ratios else None


def estimate_M(Q, A, b, mode="paper", labeling="paper"):
    """Vectorised M_est, O(n^2).
    mode = "paper"     Eq.(3): numerator / (A x_to - b)^2, skip if x_to feasible (x_from NOT required feasible)
           "strict"    same, but additionally require x_from feasible (Lemma 1 taken literally)
           "numerator" no division (conservative: for a cardinality constraint any 1-flip from a
                       feasible point has (A x_to - b)^2 = 1)
           "diff"      denominator = (A x_to - b)^2 - (A x_from - b)^2, i.e. the true change of the penalty
                       term between the two vectors (must be > 0).  Equals Eq.(2)'s denominator when
                       x_from is feasible.  This variant reproduces the tail of the paper's Table 1.
    labeling = "paper" | "corrected"  (see candidate_vectors)."""
    A = np.asarray(A, float)
    q, Qs = symmetrize(Q)
    P, N = Qs > 0, Qs < 0
    delta1 = q + np.where(P, Qs, 0.0).sum(1)       # max Delta_1 = q_ii + sum_{q_ij>0} q_ij
    delta2 = -q - np.where(N, Qs, 0.0).sum(1)      # max Delta_2 = -q_ii - sum_{q_ij<0} q_ij
    A_pos, A_neg = P @ A, N @ A                    # A . (neighbour set switched on)
    if labeling == "paper":
        pairs = [(delta1, A_pos, A_pos + A), (delta2, A_neg + A, A_neg)]     # (numerator, lhs_from, lhs_to)
    else:
        pairs = [(delta1, A_pos + A, A_pos), (delta2, A_neg, A_neg + A)]
    best = None
    for delta, lhs_from, lhs_to in pairs:
        ok = delta > 0
        if mode == "numerator":
            ratio = delta
        elif mode == "diff":
            den = (lhs_to - b) ** 2 - (lhs_from - b) ** 2
            ok &= den > 0
            ratio = delta / np.maximum(den, 1e-12)
        else:
            ok &= ~np.isclose(lhs_to, b)
            if mode == "strict":
                ok &= np.isclose(lhs_from, b)
            ratio = delta / np.maximum((lhs_to - b) ** 2, 1e-12)
        if ok.any():
            r = float(ratio[ok].max())
            best = r if best is None else max(best, r)
    return best


# ----------------------------------------------- penalty / slack construction
def slack_coefficients(width):
    """D such that sum_k D_k s_k takes exactly the integer values 0..width (Glover et al. [11])."""
    if width <= 0:
        return np.array([])
    k = int(np.floor(np.log2(width))) + 1
    D = [2 ** i for i in range(k - 1)]
    D.append(width - sum(D))
    return np.array(D, float)


def penalized_Q(Q, a, rhs, M):
    """Upper-triangular Q' for  x'Qx + M (a.z - rhs)^2  (constant M*rhs^2 dropped):
         Q'[i,i] = Q[i,i] + M (a_i^2 - 2 rhs a_i)       Q'[i,j] = Q[i,j] + 2 M a_i a_j   (i<j)"""
    Qp = Q + 2.0 * M * np.triu(np.outer(a, a), 1)
    Qp[np.diag_indices_from(Qp)] += M * (a * a - 2.0 * rhs * a)
    return Qp


def _pad(Q, k):
    n = len(Q)
    out = np.zeros((n + k, n + k))
    out[:n, :n] = Q
    return out


def build_equality(Q, A, b, M):
    """Section 2/4.1:  min x'Qx + M (A x - b)^2"""
    return penalized_Q(Q, np.asarray(A, float), float(b), M), np.asarray(A, float), float(b)


def build_inequality_full(Q, A, b, M):
    """Full slack range:  A x + D s = b,   s covers [0, b]"""
    D = slack_coefficients(b)
    a = np.concatenate([np.asarray(A, float), D])
    return penalized_Q(_pad(Q, D.size), a, float(b), M), a, float(b)


def build_partition(Q, A, lo, hi, M):
    """Section 3 partition: A x in [lo, hi]  ->  A x - D s = lo,  s covers [0, hi-lo]"""
    D = slack_coefficients(hi - lo)
    a = np.concatenate([np.asarray(A, float), -D])
    return penalized_Q(_pad(Q, D.size), a, float(lo), M), a, float(lo)


def split_range(b, m):
    """m consecutive integer sub-ranges of [0, b]  (integer version of [(i-1)(b+1)/m, i(b+1)/m))."""
    return [(int(p[0]), int(p[-1])) for p in np.array_split(np.arange(b + 1), m)]


# ------------------------------------------------------------------- solver
def solve(Q_pen, a, rhs, M, A, seconds=10, seed=0):
    """Stand-in for the paper's multistart 1-flip path-relinking solver:
    multistart 1-flip tabu search with O(n) incremental updates.
    Reports the best FEASIBLE point (a.z == rhs, i.e. penalty = 0) as [x'Qx, A.x]."""
    rng = np.random.default_rng(seed)
    n = len(A)
    q, Qs = symmetrize(Q_pen)
    N = q.size
    best = dict(obj=np.inf, lhs=None, x=None)
    pen = [np.inf, 0.0, None]                  # best PENALISED point: (f, a.z, z)
    t_end = time.time() + seconds
    while time.time() < t_end:
        z = (rng.random(N) < 0.5).astype(float)
        g = q + Qs @ z                         # g_i = Q_ii + sum_j Q_ij z_j  ->  Delta_i = (1-2 z_i) g_i
        f = float(z @ (Q_pen @ z))
        total = float(a @ z)
        tabu = np.zeros(N, dtype=int)
        f_run = f
        for it in range(1, 10 * N + 1):
            delta = (1 - 2 * z) * g
            cand = np.where((tabu < it) | (f + delta < f_run - 1e-9), delta, np.inf)
            k = int(cand.argmin())
            if not np.isfinite(cand[k]):
                continue
            step = 1 - 2 * z[k]
            f += delta[k]; z[k] += step; g += Qs[k] * step; total += a[k] * step
            tabu[k] = it + N // 100 + int(rng.integers(0, 10))
            f_run = min(f_run, f)
            if f < pen[0] - 1e-9:
                pen[0], pen[1], pen[2] = f, total, z.copy()
            if abs(total - rhs) < 1e-9:        # feasible
                obj = f + M * rhs * rhs        # undo dropped constant
                if obj < best["obj"]:
                    best.update(obj=obj, lhs=float(A @ z[:n]), x=z[:n].copy())
            if (it & 255) == 0 and time.time() > t_end:
                break
    _finalize(best, pen, M, rhs, A)
    return best


class _Timeout(Exception):
    pass


class _State:
    """Current solution z with incrementally maintained gain vector g, objective f, constraint total."""
    __slots__ = ("z", "g", "f", "tot")

    def __init__(self, z, q, Qs, Q_pen, a):
        self.z = z.astype(float).copy()
        self.g = q + Qs @ self.z                 # g_i = Q_ii + sum_j Q_ij z_j  ->  flip change = (1 - 2 z_i) g_i
        self.f = float(self.z @ (Q_pen @ self.z))
        self.tot = float(a @ self.z)


# def solve_pr(Q_pen, a, rhs, M, A, seconds=10, seed=0, elite_size=10):
#     """Approximation of the paper's solver (Lewis & Kochenberger 2016):
#     probabilistic multistart + greedy 1-flip descent + path relinking.
#       1. start: greedy construction (first time), later sampled from the bit frequencies of an elite set
#       2. greedy 1-flip descent to a local optimum  -> offer it to the elite set
#       3. path relinking from that optimum towards a random elite solution (flip the differing bit with
#          the best gain each step), take the best intermediate point, descend again -> offer to elite
#     The paper gives no further details, so population size, clipping of probabilities and the choice of
#     relinking pairs are my choices.  Returns the best FEASIBLE point as in solve()."""
#     rng = np.random.default_rng(seed)
#     n = len(A)
#     q, Qs = symmetrize(Q_pen)
#     N = q.size
#     best = dict(obj=np.inf, lhs=None, x=None, restarts=0)
#     elite = []                                   # list of (f, z), sorted by f
#     pen = [np.inf, 0.0, None]
#     t_end = time.time() + seconds
#     cnt = [0]

#     def tick():
#         cnt[0] += 1
#         if (cnt[0] & 127) == 0 and time.time() > t_end:
#             raise _Timeout

#     def record(st):
#         if st.f < pen[0] - 1e-9:
#             pen[0], pen[1], pen[2] = st.f, st.tot, st.z.copy()
#         if abs(st.tot - rhs) < 1e-9:
#             obj = st.f + M * rhs * rhs
#             if obj < best["obj"]:
#                 best.update(obj=obj, lhs=float(A @ st.z[:n]), x=st.z[:n].copy())

#     def flip(st, k):
#         step = 1 - 2 * st.z[k]
#         st.f += step * st.g[k]
#         st.z[k] += step
#         st.g += Qs[k] * step
#         st.tot += a[k] * step

#     def descend(st):
#         while True:
#             tick()
#             delta = (1 - 2 * st.z) * st.g
#             k = int(delta.argmin())
#             if delta[k] >= -1e-9:
#                 return
#             flip(st, k)
#             record(st)

#     def relink(st, guide):
#         diff = st.z != guide
#         steps = int(diff.sum())
#         best_f, best_z = np.inf, None
#         for s in range(steps):
#             tick()
#             delta = (1 - 2 * st.z) * st.g
#             k = int(np.where(diff, delta, np.inf).argmin())
#             flip(st, k)
#             diff[k] = False
#             record(st)
#             if 0 < s < steps - 1 and st.f < best_f:      # intermediate points only
#                 best_f, best_z = st.f, st.z.copy()
#         return best_z

#     def offer(st):
#         if any(np.array_equal(st.z, z) for _, z in elite):
#             return
#         elite.append((st.f, st.z.copy()))
#         elite.sort(key=lambda t: t[0])
#         del elite[elite_size:]

#     greedy = (2.0 * q + np.triu(Q_pen, 1).sum(1)) < 0     # paper: contribution estimate decides x_i
#     try:
#         while time.time() < t_end:
#             best["restarts"] += 1
#             if not elite:
#                 z0 = greedy if best["restarts"] == 1 else (rng.random(N) < 0.5)
#             else:
#                 p = np.clip(np.mean([z for _, z in elite], axis=0), 0.05, 0.95)
#                 z0 = rng.random(N) < p
#             st = _State(z0, q, Qs, Q_pen, a)
#             descend(st)
#             offer(st)
#             if len(elite) >= 2:
#                 guide = elite[int(rng.integers(len(elite)))][1]
#                 mid = relink(st, guide)
#                 if mid is not None:
#                     st2 = _State(mid, q, Qs, Q_pen, a)
#                     descend(st2)
#                     offer(st2)
#     except _Timeout:
#         pass
#     _finalize(best, pen, M, rhs, A)
#     return best


def _finalize(best, pen, M, rhs, A):
    """Add the best PENALISED point to the result.  The paper's tables report [x'Qx, A.x] of the best point
    of the penalised objective, which is infeasible when M is too small (e.g. LHS=105 for b=100, M=100)."""
    f, tot, z = pen
    n = len(A)
    best.update(pen_obj=f + M * rhs * rhs - M * (tot - rhs) ** 2,    # x'Qx without the penalty term
                pen_lhs=float(A @ z[:n]), pen_x=z[:n].copy(), pen_feasible=abs(tot - rhs) < 1e-9)


# ------------------------------------------------- Lewis & Kochenberger (2016) solver
def doe_effects(Q_pen, chunk=512):
    """DOE presolve (paper's Appendix).  Saturated two-level fractional-factorial design with
    n_rows = 2^ceil(log2(m+eps)) tests; column j uses the parity of (row & (j+1)), i.e. the main-effect
    columns are the single-bit masks and the other columns are interactions of main effects (XOR; the
    paper's XNOR differs only by a complemented column, which the mirrored design covers).
    Returns, for each variable, the average change of h = -x'Q_pen x when x_i goes 0 -> 1."""
    N = len(Q_pen)
    n_rows = 1 << int(np.ceil(np.log2(N + 1e-9)))
    nbits = n_rows.bit_length() - 1
    masks = np.arange(1, N + 1, dtype=np.int64)
    sum1, cnt1 = np.zeros(N), np.zeros(N)
    sumF, cnt = 0.0, 0
    for start in range(0, n_rows, chunk):
        r = np.arange(start, min(start + chunk, n_rows), dtype=np.int64)[:, None]
        par = np.zeros((r.shape[0], N), dtype=np.int64)
        for k in range(nbits):
            par ^= ((r & masks[None, :]) >> k) & 1
        T = par.astype(float)
        for Tm in (T, 1.0 - T):                          # design and its mirror
            Fh = -np.einsum("ij,ij->i", Tm, Tm @ Q_pen.T)
            sum1 += Fh @ Tm
            cnt1 += Tm.sum(0)
            sumF += Fh.sum()
            cnt += len(Fh)
    return sum1 / cnt1 - (sumF - sum1) / (cnt - cnt1)


def _dist1(e, u, s):
    """Fig. 4 probabilities p(x_i = 1) from the estimated effect (normal-based steps)."""
    p = np.full(e.shape, 0.5)
    p[e > u + s] = 0.7; p[e > u + 2 * s] = 0.8; p[e > u + 3 * s] = 0.95
    p[e < u - s] = 0.35; p[e < u - 2 * s] = 0.25; p[e < u - 3 * s] = 0.05
    return p


def _dist2(e, u, s):
    """Fig. 8 'Dist. 2' (linear ramp).  Values read off the figure."""
    return np.interp((e - u) / s, [-4, -3, -2, -1, 0, 1, 2, 3, 4], [0, .1, .2, .4, .5, .7, .8, .9, 1.0])


def solve_popl(Q_pen, a, rhs, M, A, seconds=10, seed=0, max_iter=20000,
               percent_limit=0.8, aspiration_div=(1.0, 0.0), aspiration_center="zero", doe=None):
    """POPL = DOE presolve + probability oscillation + path loitering  (Lewis & Kochenberger 2016).
      Main loop (Fig. 1): start from all ones; repeat { perturb (Fig. 4/7);  1-flip local search (Fig. 5)
      until no improvement;  on a new best: path-relink to the previous best if > 5-10 bits differ, save
      the solution (3 kept);  every 1000 iterations relink current<->best }; at the end relink best with
      the saved solutions.  Path relinking (Fig. 6) flips the best bit of the difference set; 'loitering'
      adds one local-search step after each path flip (difference set is recomputed each step).
    Internally the paper's maximisation is used with h = -f, so 'gain' > 0 means f decreases.
    NOT specified in the paper, my choices: percent_limit, the two aspiration levels, the relinking cap,
    the 0.9/0.1 split of the time budget between main loop and final relinking.
    aspiration_center='paper' uses the mean DOE effect u as in the paper; for penalised models u is hugely
    negative (random designs violate the constraint badly), so every flip passes the aspiration test and the
    perturbation degenerates to random flips.  'zero' centres the aspiration at 0 instead."""
    rng = np.random.default_rng(seed)
    n = len(A)
    q, Qs = symmetrize(Q_pen)
    N = q.size
    t0 = time.time()
    if doe is None:
        doe = doe_effects(Q_pen)
    t_presolve = time.time() - t0
    u, s_ = float(doe.mean()), float(doe.std())
    p1, p2 = _dist1(doe, u, s_), _dist2(doe, u, s_)
    c = u if aspiration_center == "paper" else 0.0   # 'paper': avg effect u ; 'zero': 0 (see report: u is huge for penalised models)
    asp = np.array([c - aspiration_div[0] * s_, c - aspiration_div[1] * s_])   # [low, high]
    order = np.argsort(-doe)                       # examine variables from largest estimated effect
    best = dict(obj=np.inf, lhs=None, x=None, iters=0, presolve_s=t_presolve)
    pen = [np.inf, 0.0, None]
    t_end = t0 + seconds
    deadline = [t0 + 0.9 * seconds]
    cnt = [0]

    def tick():
        cnt[0] += 1
        if (cnt[0] & 63) == 0 and time.time() > deadline[0]:
            raise _Timeout

    def record(st):
        if st.f < pen[0] - 1e-9:
            pen[0], pen[1], pen[2] = st.f, st.tot, st.z.copy()
        if abs(st.tot - rhs) < 1e-9:
            obj = st.f + M * rhs * rhs
            if obj < best["obj"]:
                best.update(obj=obj, lhs=float(A @ st.z[:n]), x=st.z[:n].copy())

    def flip(st, k):
        step = 1 - 2 * st.z[k]
        st.f += step * st.g[k]
        st.z[k] += step
        st.g += Qs[k] * step
        st.tot += a[k] * step

    def perturb(st):                                # Fig. 7: probability + aspiration oscillation
        p = np.where(rng.random(N) > 0.5, p1, p2)
        want = rng.random(N) < p
        al = asp[(rng.random(N) > 0.5).astype(int)]
        mism = want != (st.z > 0.5)
        for j in order[mism[order]]:
            if -(1 - 2 * st.z[j]) * st.g[j] > al[j]:
                flip(st, j)

    def ls_step(st):                                # Fig. 5: one DOE-guided 1-flip step
        gain = -(1 - 2 * st.z) * st.g
        cand = np.flatnonzero(gain > 1e-9)
        if cand.size == 0:
            return False
        kbest = int(cand[gain[cand].argmax()])
        i, j = rng.choice(cand, 2)
        if doe[i] < doe[j]:
            i = j
        k = int(i) if gain[i] > percent_limit * gain[kbest] else kbest
        flip(st, k)
        record(st)
        return True

    def relink(z_start, z_end):                     # Fig. 6 + path loitering
        st = _State(z_start, q, Qs, Q_pen, a)
        cap = 2 * int((st.z != z_end).sum()) + 5
        for _ in range(cap):
            tick()
            diff = st.z != z_end
            if not diff.any():
                break
            gain = -(1 - 2 * st.z) * st.g
            flip(st, int(np.where(diff, gain, -np.inf).argmax()))
            record(st)
            ls_step(st)

    st = _State(np.ones(N), q, Qs, Q_pen, a)        # paper: start from all ones
    loop_best_f, best_z, saved = np.inf, None, []
    try:
        for it in range(1, max_iter + 1):
            tick()
            best["iters"] = it
            perturb(st)
            while ls_step(st):
                tick()
            record(st)
            if st.f < loop_best_f - 1e-9:
                if best_z is not None and np.count_nonzero(st.z != best_z) > int(rng.integers(5, 11)):
                    relink(st.z, best_z)
                saved = (saved + [st.z.copy()])[-3:]
                loop_best_f, best_z = st.f, st.z.copy()
            if it % 1000 == 0 and best_z is not None:
                relink(st.z, best_z)
    except _Timeout:
        pass
    deadline[0] = t_end
    try:                                            # final: best <-> saved solutions
        if best_z is not None:
            for sz in saved:
                relink(best_z, sz)
    except _Timeout:
        pass
    _finalize(best, pen, M, rhs, A)
    return best


SOLVERS = {"tabu": solve, "popl": solve_popl}


def fmt(r, report="penalised"):
    """report='penalised': [x'Qx, A.x] of the best point of the penalised objective (what the paper tables
    show, so infeasible points can appear);  'feasible': best point with penalty term = 0."""
    if report == "feasible":
        return "no feasible solution" if r["x"] is None else f"[{r['obj']:.0f}, {r['lhs']:.0f}]"
    return f"[{r['pen_obj']:.0f}, {r['pen_lhs']:.0f}]" + ("" if r["pen_feasible"] else "  (infeasible)")


# ------------------------------------------------------- paper's Table 1
PAPER_TABLE1 = {  # dataset -> M_est for b = 100,150,200,250,500,1000,1500,2000,2500
    1: [11090, 15908, 264, 100, 24, 10, 6, 5, 4], 2: [11935, 16372, 212, 91, 24, 10, 6, 4, 3],
    3: [11052, 16374, 202, 90, 24, 10, 6, 5, 4], 4: [10183, 16036, 199, 87, 23, 10, 6, 4, 3],
    5: [11976, 15872, 219, 93, 25, 10, 6, 5, 4], 6: [11010, 15952, 238, 99, 25, 10, 6, 5, 4],
    7: [11214, 15716, 215, 92, 24, 10, 6, 5, 4], 8: [12088, 16990, 215, 93, 24, 10, 6, 5, 4],
    9: [11940, 16180, 199, 91, 25, 10, 6, 5, 4], 10: [11488, 16360, 214, 90, 25, 10, 6, 5, 4]}
B_LIST = [100, 150, 200, 250, 500, 1000, 1500, 2000, 2500]


# --------------------------------------------------------------- commands
def cmd_selftest(_):
    # the 3-variable example from the paper, constraint sum(x) = 1
    Q = np.array([[-1, -4, 6],
                  [0, 2, -2],
                  [0, 0, 4]], float)
    A, b = np.ones(3), 1
    print("Paper example, labeling='paper' (explicit f(x_from) - f(x_to)):")
    for i, tag, (xf, xt) in candidate_vectors(Q, "paper"):
        if i == 0:
            print(f"  x_{i+1} {tag}: {xf} -> {xt}   actual improvement = {objective(Q, xf) - objective(Q, xt):+.0f}")
    print("  -> actual improvement is -5: the paper's 0->1 / 1->0 labels are reversed.")
    print("Corrected labeling:")
    for i, tag, (xf, xt) in candidate_vectors(Q, "corrected"):
        if i == 0:
            print(f"  x_{i+1} {tag}: {xf} -> {xt}   actual improvement = {objective(Q, xf) - objective(Q, xt):+.0f}")
    # vectorised numerators vs explicit objective differences (corrected labeling) on random Q
    rng = np.random.default_rng(1)
    n = 14
    Qr = np.triu(rng.integers(-100, 101, (n, n)) * (rng.random((n, n)) < 0.3)).astype(float)
    for mode in ("numerator", "paper"):
        v = estimate_M(Qr, np.ones(n), 5, mode=mode, labeling="corrected")
        e = estimate_M_explicit(Qr, np.ones(n), 5, mode="paper" if mode == "paper" else "numerator", labeling="corrected") \
            if mode == "paper" else max(
                objective(Qr, xf) - objective(Qr, xt)
                for _, _, (xf, xt) in candidate_vectors(Qr, "corrected"))
        print(f"vectorised vs explicit ({mode}, corrected): {v:.4f} vs {e:.4f}")
    for mode in ("paper", "strict", "numerator"):
        for lab in ("paper", "corrected"):
            print(f"example  mode={mode:9s} labeling={lab:9s} M_est = {estimate_M(Q, A, b, mode, lab)}")


def cmd_table1(a):
    Q, m = load_orlib(a.file, a.instance)
    ref = PAPER_TABLE1.get(a.instance)
    print(f"M_est for instance {a.instance}, {m} edges")
    print(f"{'b':>6} {'paper Tbl1':>11} {'squared':>9} {'sq-corr':>9} {'diff':>9} {'diff-corr':>10} {'numerator':>10}")
    for j, b in enumerate(B_LIST):
        A = np.ones(len(Q))
        r1 = estimate_M(Q, A, b, "paper", "paper")
        r2 = estimate_M(Q, A, b, "paper", "corrected")
        r3 = estimate_M(Q, A, b, "numerator")
        r4 = estimate_M(Q, A, b, "diff", "paper")
        r5 = estimate_M(Q, A, b, "diff", "corrected")
        f = lambda v: "-" if v is None else f"{np.ceil(v):.0f}"
        print(f"{b:>6} {ref[j] if ref else '-':>11} {f(r1):>9} {f(r2):>9} {f(r4):>9} {f(r5):>10} {f(r3):>10}")


def cmd_table2(a):
    Q, m = load_orlib(a.file, a.instance)
    A = np.ones(len(Q))
    lst = []
    for b in (100, 150):
        Me = estimate_M(Q, A, b, "paper", a.labeling)
        for M in (100, 1000, Me): #(10, 100, 1000, 10000, Me)
            Qp, aa, rhs = build_equality(Q, A, b, M)
            r = SOLVERS[a.solver](Qp, aa, rhs, M, A, a.seconds)
            tag = f"M_est={M:.0f}" if M == Me else f"M={M}"
            print(f"b={b:4d} {tag:>12}: {fmt(r, a.report)}", flush=True)
            lst.append({"instance": a.instance, "b": b, "m": m, "M": M, "Mest": True if M == Me else False, "result": fmt(r, a.report)})
    df = pd.DataFrame(lst)
    df.to_csv(f"table2_results_{a.instance}.csv", index=False)

# def cmd_table2(a):
#     lst = []
#     for i in range(1, 11):
#         Q, m = load_orlib(a.file, i)
#         A = np.ones(len(Q))
#         for b in (100, 150):
#             Me = estimate_M(Q, A, b, "paper", a.labeling)
#             for M in (100, 1000, Me): #(10, 100, 1000, 10000, Me)
#                 Qp, aa, rhs = build_equality(Q, A, b, M)
#                 r = SOLVERS[a.solver](Qp, aa, rhs, M, A, a.seconds)
#                 tag = f"M_est={M:.0f}" if M == Me else f"M={M}"
#                 print(f"b={b:4d} {tag:>12}: {fmt(r, a.report)}", flush=True)
#                 lst.extend([{"instance": i, "b": b, "m": m, "M": M, "Mest": True if M == Me else False, "result": fmt(r, a.report)}])
#     df = pd.DataFrame(lst)
#     df.to_csv("table2_results.csv", index=False)


def cmd_table3(a):
    Q, m = load_orlib(a.file, a.instance)
    A = np.ones(len(Q))
    for M in (100, 10000):
        Qp, aa, rhs = build_inequality_full(Q, A, a.b, M)
        r = SOLVERS[a.solver](Qp, aa, rhs, M, A, a.seconds * a.m)
        print(f"M={M:6d} full ({len(aa) - len(A)} slack vars, {a.seconds * a.m}s): {fmt(r, a.report)}", flush=True)
        results = []
        for lo, hi in split_range(a.b, a.m):
            Qp, aa, rhs = build_partition(Q, A, lo, hi, M)
            r = SOLVERS[a.solver](Qp, aa, rhs, M, A, a.seconds)
            key = (0, r["obj"]) if r["x"] is not None else (1, r["pen_obj"])
            results.append((key, r, lo, hi, len(aa) - len(A)))
            print(f"   partition [{lo},{hi}] ({len(aa) - len(A)} slack vars): {fmt(r, a.report)}", flush=True)
        o, r, lo, hi, _ = min(results, key=lambda t: t[0])
        print(f"   best partition [{lo},{hi}]: {fmt(r, a.report)}", flush=True)


def cmd_candidates(a):
    """Diagnostic: which (Delta, k_from, k_to) candidate wins under each denominator rule?
    Use it to find the rule that reproduces the paper's Table 1 value for a given b."""
    Q, m = load_orlib(a.file, a.instance)
    n, b = len(Q), a.b
    q, Qs = symmetrize(Q)
    P, N = Qs > 0, Qs < 0
    d1 = q + np.where(P, Qs, 0.0).sum(1)
    d2 = -q - np.where(N, Qs, 0.0).sum(1)
    kp, kn = P.sum(1), N.sum(1)
    if a.labeling == "paper":
        sets = [("D1", d1, kp, kp + 1), ("D2", d2, kn + 1, kn)]
    else:
        sets = [("D1", d1, kp + 1, kp), ("D2", d2, kn, kn + 1)]
    rules = {
        "squared  (kt-b)^2":            lambda kf, kt: (kt - b) ** 2.0,
        "diff     (kt-b)^2-(kf-b)^2":   lambda kf, kt: (kt - b) ** 2.0 - (kf - b) ** 2.0,
        "absdiff  |(kt-b)^2-(kf-b)^2|": lambda kf, kt: np.abs((kt - b) ** 2.0 - (kf - b) ** 2.0),
        "abs      |kt-b|":              lambda kf, kt: np.abs(kt - b) * 1.0,
        "from_sq  (kf-b)^2":            lambda kf, kt: (kf - b) ** 2.0,
    }
    print(f"instance {a.instance}, b={b}, labeling={a.labeling}")
    for name, rule in rules.items():
        rows = []
        for tag, delta, kf, kt in sets:
            den = rule(kf, kt)
            ok = (delta > 0) & (den > 0)
            for i in np.where(ok)[0]:
                rows.append((delta[i] / den[i], tag, i, delta[i], kf[i], kt[i], den[i]))
        rows.sort(reverse=True)
        print(f"\n{name}")
        print("   ratio      set  node  Delta  k_from  k_to  denominator")
        for r in rows[:a.top]:
            print(f"  {r[0]:9.3f}   {r[1]}  {r[2]:5d}  {r[3]:6.0f}  {r[4]:5d}  {r[5]:5d}  {r[6]:9.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "table1", "table2", "table3", "candidates"])
    ap.add_argument("--file", default="bqp2500.txt")
    ap.add_argument("--instance", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=500)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--solver", default="tabu", choices=["tabu", "popl"])
    ap.add_argument("--report", default="penalised", choices=["penalised", "feasible"])
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--labeling", default="paper", choices=["paper", "corrected"])
    a = ap.parse_args()
    {"selftest": cmd_selftest, "table1": cmd_table1, "table2": cmd_table2, "table3": cmd_table3,
     "candidates": cmd_candidates}[a.cmd](a)