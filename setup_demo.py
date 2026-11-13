from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
UKT = ROOT / "UKT"

if not UKT.exists():
    print("Cloning official UKT repository...")
    subprocess.run(
        ["git","clone","https://github.com/UncertaintyForKnowledgeTracing/UKT.git",str(UKT)],
        check=True
    )
else:
    print("UKT folder exists:", UKT)

required = [
    "best.pt","web_config.json","question_mapping.csv",
    "uncertainty_stats.json","forgetting_stats.json",
    "interference_matrix.json","policy_config.json","demo_history.csv"
]
missing = [x for x in required if not (ROOT/"artifacts"/x).exists()]
if missing:
    print("\nArtifacts chưa đủ:")
    for x in missing:
        print(" -", x)
    print("\nSau khi chạy Colab, copy toàn bộ UKT_FORGET_SE_RESEARCH/web_bundle/* vào artifacts/")
else:
    print("\n✓ Artifacts đầy đủ. Chạy: streamlit run app.py")
