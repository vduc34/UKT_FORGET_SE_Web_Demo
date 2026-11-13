from pathlib import Path
import json
import pandas as pd

def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.kc_mapping = load_json(self.root / "kc_mapping.json", {"knowledge_components":[]})
        self.curriculum = load_json(self.root / "curriculum_graph.json", {})
        self.web_config = load_json(self.root / "web_config.json", {})
        self.metrics = load_json(self.root / "metrics.json", {})
        self.uncertainty_stats = load_json(self.root / "uncertainty_stats.json", {})
        self.forgetting_stats = load_json(self.root / "forgetting_stats.json", {})
        self.interference = load_json(self.root / "interference_matrix.json", {})
        self.policy_config = load_json(self.root / "policy_config.json", {
            "mastery_threshold":0.70,
            "uncertainty_threshold":0.75,
            "forgetting_threshold":0.70,
            "uncertainty_penalty":0.25,
            "forgetting_penalty":0.15,
        })

        qmap = self.root / "question_mapping.csv"
        self.question_mapping = pd.read_csv(qmap, dtype={"raw_qid":str}) if qmap.exists() else pd.DataFrame()

        dh = self.root / "demo_history.csv"
        self.demo_history = pd.read_csv(dh, dtype={"raw_qid":str}) if dh.exists() else pd.DataFrame()

    def status(self):
        required = [
            "best.pt","web_config.json","metrics.json","kc_mapping.json",
            "question_mapping.csv","uncertainty_stats.json","forgetting_stats.json",
            "interference_matrix.json","policy_config.json","curriculum_graph.json",
            "demo_history.csv",
        ]
        return {name:(self.root/name).exists() for name in required}

    @property
    def ready_for_model(self):
        return (
            (self.root/"best.pt").exists()
            and bool(self.web_config)
            and not self.question_mapping.empty
        )

    def kc_options(self):
        return [
            f'{x["sequence_id_normalized"]} — {x["name_en"]}'
            for x in self.kc_mapping.get("knowledge_components", [])
        ]

    def get_kc_by_label(self, label):
        code = str(label).split(" — ")[0].strip()
        for x in self.kc_mapping.get("knowledge_components", []):
            if x["sequence_id_normalized"] == code:
                return x
        raise KeyError(label)

    def representative_question(self, kc_index):
        if self.question_mapping.empty:
            return None
        sub = self.question_mapping[self.question_mapping["kc_index"].astype(int) == int(kc_index)]
        if sub.empty:
            return None
        if "interaction_count" in sub:
            sub = sub.sort_values("interaction_count", ascending=False)
        return str(sub.iloc[0]["raw_qid"])

    def questions_for_kc(self, kc_index):
        if self.question_mapping.empty:
            return []
        sub = self.question_mapping[self.question_mapping["kc_index"].astype(int) == int(kc_index)]
        return [str(x) for x in sub["raw_qid"].tolist()]
