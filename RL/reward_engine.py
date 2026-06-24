def compute_claim_metrics(oss_RE, diag_RE, oss_GEM, diag_GEM):
    """
    Calcola precision, recall e F1 sui claim clinici,
    seguendo la definizione di Jha25 (DocLens claim-level metrics).
    """

    # Reference claims (rule engine)
    R = set(oss_RE + diag_RE)

    # Output claims (GEM)
    O = set(oss_GEM + diag_GEM)

    # ---- Recall (Completezza) ----
    # proporzione di claim veri coperti dal modello
    if len(R) == 0:
        recall = 1.0
    else:
        tp_recall = sum(1 for r in R if r in O)
        recall = tp_recall / len(R)

    # ---- Precision (Factualità) ----
    # proporzione di claim generati supportati dal rule engine
    if len(O) == 0:
        precision = 1.0
    else:
        tp_prec = sum(1 for o in O if o in R)
        precision = tp_prec / len(O)

    # ---- F1 ----
    eps = 1e-8
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def compute_reward_jha25(
    oss_RE, diag_RE, oss_GEM, diag_GEM,
    scale=10.0,
    gating_threshold=0.6
):
    """
    Reward in stile Jha25:
    - precision e recall sui claim
    - F1 claim-based
    - scaling a [0, 10]
    - reward gating (F1 < 0.6 → reward = 0)
    """

    precision, recall, f1 = compute_claim_metrics(
        oss_RE, diag_RE, oss_GEM, diag_GEM
    )

    # ---- Reward gating ----
    if f1 < gating_threshold:
        return 0.0

    # ---- Reward scalato ----
    return scale * f1
