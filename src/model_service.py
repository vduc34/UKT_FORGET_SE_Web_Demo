from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .policy import (
    normalize_uncertainty, pair_interference_risk,
    derive_gap_for_target, forgetting_risk, decide
)

class UKTService:
    def __init__(self, project_root, store):
        self.project_root = Path(project_root)
        self.store = store
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if self.store.ready_for_model:
            self._load_model()

    def _load_model(self):
        ukt_root = self.project_root / "UKT"
        if not ukt_root.exists():
            raise FileNotFoundError(
                "Thiếu folder UKT/. Hãy chạy: python setup_demo.py"
            )
        if str(ukt_root) not in sys.path:
            sys.path.insert(0, str(ukt_root))

        from models import init_model

        ckpt = torch.load(
            self.store.root / "best.pt",
            map_location=self.device
        )
        model_config = ckpt.get("model_config") or self.store.web_config["model_config"]
        data_cfg = ckpt.get("data_config") or self.store.web_config["data_config"]
        emb_type = ckpt.get("emb_type") or self.store.web_config.get("emb_type","stoc_qid")

        self.model_config = model_config
        self.data_cfg = data_cfg
        self.emb_type = emb_type
        self.d_model = int(model_config["d_model"])

        self.model = init_model.init_model(
            "ukt", model_config, data_cfg, emb_type
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _raw_to_model_qid(self, raw_qid):
        sub = self.store.question_mapping[
            self.store.question_mapping["raw_qid"].astype(str) == str(raw_qid)
        ]
        if sub.empty:
            raise ValueError(f"Question ID {raw_qid} không tồn tại trong mapping.")
        return int(sub.iloc[0]["model_question_id"])

    def _kc_to_model_cid(self, kc_index):
        return int(kc_index) - 1

    def _representative_raw_qid(self, kc_index):
        q = self.store.representative_question(kc_index)
        if q is None:
            raise ValueError(
                "Chưa có question_mapping.csv từ notebook Colab."
            )
        return q

    def _history_to_encoded(self, history_rows):
        q, c, r = [], [], []
        for row in history_rows:
            kc = int(row["kc_index"])
            raw_qid = row.get("raw_qid")
            if raw_qid is None or str(raw_qid).strip() == "":
                raw_qid = self._representative_raw_qid(kc)
            else:
                # Question nhập tay phải thuộc đúng KC đã chọn.
                sub = self.store.question_mapping[
                    self.store.question_mapping["raw_qid"].astype(str) == str(raw_qid)
                ]
                if sub.empty:
                    raise ValueError(f"Question ID {raw_qid} không tồn tại trong mapping.")
                mapped_kc = int(sub.iloc[0]["kc_index"])
                if mapped_kc != kc:
                    raise ValueError(
                        f"Question ID {raw_qid} thuộc KC{mapped_kc}, "
                        f"không thuộc KC{kc}. Hãy để trống để web tự chọn câu đại diện."
                    )
            q.append(self._raw_to_model_qid(str(raw_qid)))
            c.append(self._kc_to_model_cid(kc))
            r.append(int(bool(row["correct"])))
        return q,c,r

    def _make_candidate_dcur(self, history_rows, candidate_kc_index, candidate_raw_qid):
        if len(history_rows) < 1:
            raise ValueError("Cần ít nhất 1 interaction trong lịch sử.")

        q,c,r = self._history_to_encoded(history_rows)
        cq = self._raw_to_model_qid(candidate_raw_qid)
        cc = self._kc_to_model_cid(candidate_kc_index)

        # History length n; shifted sequence có n-1 historical next items + candidate.
        qseq, cseq, rseq = q, c, r
        sh_q = q[1:] + [cq]
        sh_c = c[1:] + [cc]
        sh_r = r[1:] + [0]  # candidate label unknown; causal mask prevents target leakage.

        n = len(qseq)
        ones = torch.ones((1,n), dtype=torch.bool)
        dcur = {
            "qseqs": torch.tensor([qseq], dtype=torch.long),
            "cseqs": torch.tensor([cseq], dtype=torch.long),
            "rseqs": torch.tensor([rseq], dtype=torch.float),
            "shft_qseqs": torch.tensor([sh_q], dtype=torch.long),
            "shft_cseqs": torch.tensor([sh_c], dtype=torch.long),
            "shft_rseqs": torch.tensor([sh_r], dtype=torch.float),
            "masks": ones.clone(),
            "smasks": ones.clone(),
        }
        return {k:v.to(self.device) for k,v in dcur.items()}

    @torch.no_grad()
    def predict_next(self, history_rows, candidate_kc_index, candidate_raw_qid):
        if self.model is None:
            raise RuntimeError(
                "Model chưa sẵn sàng. Copy web_bundle từ Colab vào artifacts/ "
                "và chạy python setup_demo.py."
            )

        dcur = self._make_candidate_dcur(
            history_rows, candidate_kc_index, candidate_raw_qid
        )
        out = self.model(dcur, qtest=True)
        if not isinstance(out, (tuple,list)) or len(out) < 2:
            raise RuntimeError("UKT qtest=True không trả representation như mong đợi.")

        preds, concat = out[0], out[1]
        p = float(preds[0,-1].detach().cpu().item())

        cov_d = concat[..., self.d_model:2*self.d_model]
        raw_u = float(
            (F.elu(cov_d) + 1.0).mean(dim=-1)[0,-1].detach().cpu().item()
        )
        u = normalize_uncertainty(raw_u, self.store.uncertainty_stats)

        gap, interveners = derive_gap_for_target(history_rows, candidate_kc_index)
        ir = pair_interference_risk(
            candidate_kc_index, interveners, self.store.interference
        )
        fr, gap_risk = forgetting_risk(
            gap, ir, self.store.forgetting_stats
        )

        safe, action, reason = decide(
            p, u, fr, self.store.policy_config
        )

        return {
            "p_correct":p,
            "raw_uncertainty":raw_u,
            "uncertainty_norm":u,
            "gap_value":gap,
            "gap_mode":self.store.forgetting_stats.get("gap_mode","days"),
            "gap_risk":gap_risk,
            "interference_risk":ir,
            "forgetting_risk":fr,
            "safe_mastery":safe,
            "action":action,
            "reason":reason,
        }

    def evaluate_all_kcs(self, history_rows):
        results = []
        for item in self.store.kc_mapping.get("knowledge_components", []):
            k = int(item["kc_index"])
            q = self._representative_raw_qid(k)
            res = self.predict_next(history_rows, k, q)
            res.update({
                "kc_index":k,
                "kc_code":item["sequence_id_normalized"],
                "kc_name_en":item["name_en"],
                "kc_name_vi":item["name_vi"],
                "representative_qid":q,
            })
            results.append(res)
        return pd.DataFrame(results)
