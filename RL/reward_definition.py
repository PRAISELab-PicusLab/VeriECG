# DEFINIZIONI REWARD


# DEFINIZIONE REWARD [Jha25]

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
    if len(R) == 0:
        recall = 1.0
    else:
        tp_recall = sum(1 for r in R if r in O)
        recall = tp_recall / len(R)

    # ---- Precision (Factualita') ----
    if len(O) == 0:
        precision = 1.0
    else:
        tp_prec = sum(1 for o in O if o in R)
        precision = tp_prec / len(O)

    # ---- F1 ----
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    return precision, recall, f1


def compute_reward_jha25(
    oss_RE, diag_RE, oss_GEM, diag_GEM,
    scale=10.0,
    beta=0.5
):
    """
    Reward DENSA con F-beta (niente gating).

    beta < 1 pesa di piu' la PRECISION: con beta=0.5 la precision conta il
    doppio della recall. Scoraggia il fallimento visto nel run RL, in cui il
    modello elencava osservazioni INVENTATE per alzare la recall (irraggiungibile
    "onestamente" perche' la ground truth contiene molti reperti normali che GEM
    non enumera), affondando la precision. Con F-0.5 un referto conciso e corretto
    vale piu' di uno prolisso e impreciso -> il PPO non e' piu' incentivato a
    "barare" elencando.

    Il gating (F1 < 0.6 -> 0) resta rimosso: rendeva la reward sparsa e bloccava
    l'apprendimento (advantage 0 -> nessun gradiente).
    """

    precision, recall, _f1 = compute_claim_metrics(
        oss_RE, diag_RE, oss_GEM, diag_GEM
    )

    b2 = beta * beta
    denom = b2 * precision + recall
    f_beta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0

    return scale * f_beta


# # =====================================================================
# # ===  REWARD STRUTTURATA A 4 QUADRANTI  (diagnosi x osservazioni)  ====
# # ===  ALTERNATIVA a compute_reward_jha25: LE DUE COESISTONO.       ====
# # =====================================================================
# def _prf_gruppo(ref, out, beta):
#     """precision / recall / F-beta su UN gruppo di claim (trattati come set)."""
#     R = set(ref); O = set(out)
#     recall    = 1.0 if len(R) == 0 else sum(1 for r in R if r in O) / len(R)
#     precision = 1.0 if len(O) == 0 else sum(1 for o in O if o in R) / len(O)
#     b2 = beta * beta
#     denom = b2 * precision + recall
#     fbeta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0
#     return precision, recall, fbeta


# def compute_claim_metrics_split(oss_RE, diag_RE, oss_GEM, diag_GEM,
#                                 beta_d=0.5, beta_o=0.25):
#     """
#     Come compute_claim_metrics ma DIAGNOSI e OSSERVAZIONI restano SEPARATE.
#       q_d = F-beta_d sulle diagnosi     (beta_d=0.5: precision-leaning, recall conta)
#       q_o = F-beta_o sulle osservazioni (beta_o=0.25: quasi-precision -> non punisce
#             l'omissione dei tanti reperti normali che il rule engine elenca e GEM no)
#     """
#     p_d, r_d, q_d = _prf_gruppo(diag_RE, diag_GEM, beta_d)
#     p_o, r_o, q_o = _prf_gruppo(oss_RE, oss_GEM, beta_o)
#     return {"q_d": q_d, "q_o": q_o,
#             "prec_diag": p_d, "rec_diag": r_d,
#             "prec_oss": p_o, "rec_oss": r_o}


# def compute_reward_strutturata(oss_RE, diag_RE, oss_GEM, diag_GEM,
#                                R_pos=10.0, P_o=6.0, P_d=3.0,
#                                beta_d=0.5, beta_o=0.25,
#                                strict=False, tau=0.5,
#                                empty_penalty=-9.0, clip=(-9.0, 10.0)):
#     """
#     R = R_pos*q_d*q_o - P_o*(1-q_o) - P_d*(1-q_d)   con P_o > P_d.
#     Ordine voluto (R_pos=10,P_o=6,P_d=3):
#         dx OK obs OK -> +10 | dx NO obs OK -> -3 | dx OK obs NO -> -6 | dx NO obs NO -> -9
#     Osservazioni sbagliate pesano PIU' delle diagnosi sbagliate (evidenza inventata).
#     Stessa firma dei primi 4 arg di compute_reward_jha25 -> DROP-IN.
#     strict=True -> soglia q_d/q_o a tau (variante "strict" del paper DRG-Sapphire).
#     """
#     # guardia anti-exploit: referto degenere (nessun claim) -> peggior quadrante
#     # (in compute_claim_metrics un output vuoto avrebbe precision=1.0: qui NO)
#     if len(set(oss_GEM)) + len(set(diag_GEM)) == 0:
#         return empty_penalty

#     m = compute_claim_metrics_split(oss_RE, diag_RE, oss_GEM, diag_GEM,
#                                     beta_d=beta_d, beta_o=beta_o)
#     q_d, q_o = m["q_d"], m["q_o"]

#     if strict:
#         q_d = 1.0 if q_d >= tau else 0.0
#         q_o = 1.0 if q_o >= tau else 0.0

#     R = R_pos * q_d * q_o - P_o * (1.0 - q_o) - P_d * (1.0 - q_d)
#     lo, hi = clip
#     return max(lo, min(hi, R))


# # =====================================================================
# # ===  SCELTA DELLA REWARD PER IL TRAINING  ==========================
# # =====================================================================
# # >>> SCEGLI QUI, UNA VOLTA SOLA, quale reward usa il training (e la eval):
# REWARD_TRAINING = "strutturata"      # opzioni:  "jha25"  |  "strutturata"
# # ---------------------------------------------------------------------
# # training_rl.py importa 'reward_fn_training' da questo file e lo usa in
# # evaluate_policy e train_rl_multimodal. Entrambe le funzioni restano definite:
# # per cambiare basta la stringa qui sopra.
# _REWARD_DISPONIBILI = {
#     "jha25":       compute_reward_jha25,
#     "strutturata": compute_reward_strutturata,
# }
# if REWARD_TRAINING not in _REWARD_DISPONIBILI:
#     raise ValueError("REWARD_TRAINING=%r non valido. Usa uno di: %s"
#                      % (REWARD_TRAINING, list(_REWARD_DISPONIBILI)))
# reward_fn_training = _REWARD_DISPONIBILI[REWARD_TRAINING]
# print("[REWARD] training + eval useranno:", reward_fn_training.__name__)


# # --- demo dei 4 quadranti: solo se esegui QUESTO file direttamente ---
# def _demo_reward_strutturata():
#     oss_RE, diag_RE = ["OBS_A", "OBS_B"], ["DIAG_X"]
#     casi = {
#         "dx OK  & obs OK ": (["OBS_A", "OBS_B"], ["DIAG_X"]),
#         "dx NO  & obs OK ": (["OBS_A", "OBS_B"], ["DIAG_Y"]),
#         "dx OK  & obs NO ": (["OBS_Z"],          ["DIAG_X"]),
#         "dx NO  & obs NO ": (["OBS_Z"],          ["DIAG_Y"]),
#         "referto vuoto    ": ([],                 []),
#     }
#     print("[REWARD strutturata] demo 4 quadranti (R_pos=10, P_o=6, P_d=3):")
#     for nome, (og, dg) in casi.items():
#         r  = compute_reward_strutturata(oss_RE, diag_RE, og, dg)
#         rs = compute_reward_strutturata(oss_RE, diag_RE, og, dg, strict=True)
#         print(f"   {nome} -> densa={r:+6.2f} | strict={rs:+6.2f}")


# if __name__ == "__main__":
#     _demo_reward_strutturata()


# =====================================================================
# UNICA REWARD ATTIVA: jha25.
# La reward strutturata a 4 quadranti e il codice di selezione qui sopra sono
# COMMENTATI perche' NON testati. 'reward_fn_training' e' il nome che il
# training (evaluate_policy, train_rl_multimodal) usa: qui punta FISSO a
# compute_reward_jha25 -> jha25 e' l'unica reward, attiva di default.
# =====================================================================
reward_fn_training = compute_reward_jha25
