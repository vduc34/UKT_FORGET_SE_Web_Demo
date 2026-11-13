import random
import pandas as pd

def default_history(kc_items, n=5):
    rows = []
    for i in range(n):
        item = kc_items[min(i, len(kc_items)-1)]
        rows.append({
            "kc_index": int(item["kc_index"]),
            "KC": f'{item["sequence_id_normalized"]} — {item["name_en"]}',
            "correct": bool(i % 3 != 1),
            "gap_days": 1.0 if i else 0.0,
            "raw_qid": "",
        })
    return rows

def synthetic_history(kc_items, n=10):
    rows = []
    for i in range(n):
        item = random.choice(kc_items)
        rows.append({
            "kc_index": int(item["kc_index"]),
            "KC": f'{item["sequence_id_normalized"]} — {item["name_en"]}',
            "correct": random.random() < 0.60,
            "gap_days": 0.0 if i == 0 else random.choice([1.0,1.0,2.0,7.0,14.0]),
            "raw_qid": "",
        })
    return rows

def real_demo_history(store, n=15):
    if store.demo_history.empty:
        return None
    g = store.demo_history.head(n)
    rows = []
    prev_kc = None
    for i, r in g.iterrows():
        rows.append({
            "kc_index": int(r["kc_index"]),
            "KC": f'{r["kc_code"]} — {r["kc_name_en"]}',
            "correct": bool(int(r["correct"])),
            "gap_days": 0.0 if pd.isna(r.get("gap_days_prev_interaction")) else float(r.get("gap_days_prev_interaction")),
            "raw_qid": str(r["raw_qid"]),
        })
    return rows

def normalize_editor_rows(edited, store):
    rows = []
    for _, r in edited.iterrows():
        item = store.get_kc_by_label(r["KC"])
        raw_qid = str(r.get("Question ID", "")).strip()
        if raw_qid.lower() == "nan":
            raw_qid = ""
        rows.append({
            "kc_index": int(item["kc_index"]),
            "KC": r["KC"],
            "correct": bool(r["Đúng?"]),
            "gap_days": float(r["Khoảng cách (ngày)"]),
            "raw_qid": raw_qid,
        })
    return rows
