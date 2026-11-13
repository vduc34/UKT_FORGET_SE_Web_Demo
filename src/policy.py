import math

def clip01(x):
    return max(0.0, min(1.0, float(x)))

def normalize_uncertainty(raw_uncertainty, stats):
    q05 = float(stats.get("q05", 0.0))
    q95 = float(stats.get("q95", 1.0))
    den = max(q95 - q05, 1e-8)
    return clip01((float(raw_uncertainty) - q05) / den)

def pair_interference_risk(target_kc_index, intervening_kcs, matrix):
    target = str(int(target_kc_index))
    risks = []
    for k in set(int(x) for x in intervening_kcs):
        if k == int(target_kc_index):
            continue
        try:
            risks.append(float(matrix[target][str(k)]["risk"]))
        except Exception:
            pass
    return max(risks) if risks else 0.0

def derive_gap_for_target(history_rows, target_kc_index):
    """
    history_rows: list[dict] có kc_index, gap_days_from_previous_interaction
    Trả về tổng số ngày kể từ lần cuối target xuất hiện.
    Nếu target chưa từng xuất hiện: tổng elapsed days của history.
    """
    if not history_rows:
        return 0.0, []
    last = None
    for i, row in enumerate(history_rows):
        if int(row["kc_index"]) == int(target_kc_index):
            last = i

    start = 0 if last is None else last + 1
    elapsed = sum(max(0.0, float(r.get("gap_days", 0.0))) for r in history_rows[start:])
    intervening = [int(r["kc_index"]) for r in history_rows[start:]]
    return elapsed, intervening

def forgetting_risk(gap_value, interference_risk, stats):
    scale = max(float(stats.get("gap_scale_q95", 14.0)), 1e-6)
    g = clip01(math.log1p(max(0.0, float(gap_value))) / math.log1p(scale))
    wg = float(stats.get("gap_weight", 0.60))
    wi = float(stats.get("interference_weight", 0.40))
    return clip01(wg*g + wi*float(interference_risk)), g

def decide(p_correct, uncertainty_norm, forgetting, cfg):
    p = clip01(p_correct)
    u = clip01(uncertainty_norm)
    f = clip01(forgetting)

    safe = clip01(
        p
        - float(cfg.get("uncertainty_penalty",0.25))*u
        - float(cfg.get("forgetting_penalty",0.15))*f
    )

    if u >= float(cfg.get("uncertainty_threshold",0.75)):
        return safe, "DIAGNOSE", "Độ bất định cao: cần thêm câu hỏi chẩn đoán trước khi kết luận."
    if f >= float(cfg.get("forgetting_threshold",0.70)):
        return safe, "REVIEW", "Rủi ro quên cao: ưu tiên ôn lại để củng cố kiến thức."
    if safe >= float(cfg.get("mastery_threshold",0.70)):
        return safe, "ADVANCE", "Mức thành thạo an toàn đạt ngưỡng và mô hình đủ chắc chắn."
    return safe, "REVIEW", "Mức thành thạo an toàn chưa đạt ngưỡng."
