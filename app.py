from pathlib import Path
import random
import pandas as pd
import streamlit as st

from src.artifacts import ArtifactStore
from src.model_service import UKTService
from src.planner import build_course_order_path
from src.utils import default_history, synthetic_history, real_demo_history, normalize_editor_rows

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(
    page_title="UKT × FORGET-SE Learning Path Demo",
    page_icon="🎓",
    layout="wide",
)

@st.cache_resource
def get_store():
    return ArtifactStore(ART)

store = get_store()

@st.cache_resource
def get_service():
    return UKTService(ROOT, store)

# Model có thể chưa đủ artifact; không crash toàn web.
try:
    service = get_service()
    model_error = None
except Exception as e:
    service = None
    model_error = str(e)

kc_items = store.kc_mapping.get("knowledge_components", [])
kc_labels = store.kc_options()

st.title("🎓 UKT × FORGET-SE")
st.caption(
    "Knowledge Tracing có xét uncertainty + forgetting/interference → "
    "adaptive decision → curriculum-aware learning path prototype"
)

with st.sidebar:
    st.header("⚙️ Trạng thái hệ thống")
    status = store.status()
    for k,v in status.items():
        st.write(("✅ " if v else "❌ ") + k)

    if model_error:
        st.warning("Model chưa load được: " + model_error)
    elif service and service.model is not None:
        st.success("UKT model đã sẵn sàng.")
    else:
        st.info("Chưa có best.pt/web artifacts đầy đủ.")

    st.divider()
    st.markdown(
        "**Lưu ý khoa học:** lộ trình ở web dùng **thứ tự curriculum của FORGET-SE**, "
        "không gọi là prerequisite graph do chuyên gia xác nhận."
    )

tab_intro, tab_history, tab_ukt, tab_path, tab_mapping, tab_model = st.tabs([
    "🏠 Giới thiệu & Cách dùng",
    "🧑‍🎓 Lịch sử học tập",
    "🧠 Phân tích UKT",
    "🛤️ Lộ trình",
    "🗺️ Mapping & Dữ liệu",
    "📊 Mô hình & Metrics",
])

with tab_intro:
    st.header("Hệ thống này làm gì?")
    st.markdown("""
**KC (Knowledge Component)** là một đơn vị kiến thức/kỹ năng. FORGET-SE có 10 KC về Software Engineering.

Luồng hoạt động:

`Lịch sử sinh viên → UKT → P(correct) + uncertainty → forgetting/interference risk → DIAGNOSE / REVIEW / ADVANCE → lộ trình`

### Cách sử dụng
1. Mở tab **Lịch sử học tập**.
2. Chọn số lượng interaction, random lịch sử hoặc sửa từng dòng.
3. Sang **Phân tích UKT**, chọn KC/câu hỏi muốn dự đoán tiếp.
4. Xem `P(correct)`, uncertainty, forgetting risk và quyết định.
5. Sang **Lộ trình**, chọn KC mục tiêu để hệ thống đánh giá toàn bộ KC theo curriculum order.

### Ý nghĩa 3 action
- **DIAGNOSE**: model chưa đủ chắc chắn → hỏi thêm câu chẩn đoán.
- **REVIEW**: kiến thức/rủi ro quên chưa an toàn → ôn lại.
- **ADVANCE**: có thể chuyển tiếp.

> Random history chỉ để demo giao diện. Khi có `demo_history.csv`, nút **Random lịch sử thật** dùng prefix từ dữ liệu de-identified đã export ở Colab.
""")

    st.subheader("10 KC")
    if kc_items:
        st.dataframe(pd.DataFrame(kc_items), use_container_width=True, hide_index=True)

with tab_history:
    st.header("Nhập hoặc Random lịch sử người học")
    st.info(
        "Bạn chỉ cần chọn KC, đúng/sai và khoảng cách thời gian. "
        "Question ID có thể để trống; web tự lấy câu đại diện hợp lệ của KC từ question_mapping.csv."
    )

    if "history_rows" not in st.session_state:
        st.session_state.history_rows = default_history(kc_items, 5) if kc_items else []

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        desired_n = st.number_input("Số interaction", min_value=1, max_value=60, value=max(1,len(st.session_state.history_rows)), step=1)
    with c2:
        if st.button("🎲 Random tổng hợp", use_container_width=True):
            st.session_state.history_rows = synthetic_history(kc_items, int(desired_n))
            st.rerun()
    with c3:
        if st.button("📚 Random lịch sử thật", use_container_width=True):
            real = real_demo_history(store, int(desired_n))
            if real is None:
                st.warning("Chưa có demo_history.csv từ Colab.")
            else:
                st.session_state.history_rows = real
                st.rerun()
    with c4:
        col_add,col_del = st.columns(2)
        if col_add.button("➕", help="Thêm 1 interaction", use_container_width=True):
            last = st.session_state.history_rows[-1] if st.session_state.history_rows else None
            item = kc_items[0]
            st.session_state.history_rows.append({
                "kc_index":int(item["kc_index"]),
                "KC":f'{item["sequence_id_normalized"]} — {item["name_en"]}',
                "correct":False,
                "gap_days":1.0,
                "raw_qid":"",
            })
            st.rerun()
        if col_del.button("➖", help="Xoá interaction cuối", use_container_width=True):
            if len(st.session_state.history_rows) > 1:
                st.session_state.history_rows.pop()
                st.rerun()

    # Đồng bộ desired_n nhẹ nhàng
    if int(desired_n) != len(st.session_state.history_rows):
        if int(desired_n) > len(st.session_state.history_rows):
            while len(st.session_state.history_rows) < int(desired_n):
                item = random.choice(kc_items)
                st.session_state.history_rows.append({
                    "kc_index":int(item["kc_index"]),
                    "KC":f'{item["sequence_id_normalized"]} — {item["name_en"]}',
                    "correct":False,
                    "gap_days":1.0,
                    "raw_qid":"",
                })
        else:
            st.session_state.history_rows = st.session_state.history_rows[:int(desired_n)]

    editor_df = pd.DataFrame([{
        "KC":r["KC"],
        "Đúng?":bool(r["correct"]),
        "Khoảng cách (ngày)":float(r.get("gap_days",0)),
        "Question ID":str(r.get("raw_qid","")),
    } for r in st.session_state.history_rows])

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=False,
        num_rows="fixed",
        column_config={
            "KC": st.column_config.SelectboxColumn("KC", options=kc_labels, required=True),
            "Đúng?": st.column_config.CheckboxColumn("Đúng?"),
            "Khoảng cách (ngày)": st.column_config.NumberColumn(
                "Khoảng cách từ interaction trước (ngày)", min_value=0.0, step=1.0
            ),
            "Question ID": st.column_config.TextColumn(
                "Question ID (nâng cao, có thể để trống)"
            )
        },
        key="history_editor",
    )

    st.session_state.history_rows = normalize_editor_rows(edited, store)

    st.caption(
        "Nếu Question ID để trống, model dùng câu đại diện xuất hiện nhiều nhất của KC. "
        "Nếu nhập Question ID thủ công, ID phải tồn tại trong mapping."
    )

with tab_ukt:
    st.header("Phân tích interaction tiếp theo bằng UKT")
    st.markdown("""
**Cách dùng:** chọn KC muốn kiểm tra/học tiếp → chọn một Question ID thuộc KC → bấm **Phân tích bằng UKT**.

Các giá trị:
- **P(correct)**: xác suất model dự đoán sinh viên trả lời đúng.
- **Uncertainty**: proxy bất định từ covariance representation của UKT, đã chuẩn hoá validation.
- **Forgetting risk**: kết hợp khoảng cách thời gian và interference proxy.
- **Safe mastery**: P(correct) sau khi trừ penalty uncertainty/forgetting.
""")

    if not kc_labels:
        st.error("Không có kc_mapping.json.")
    else:
        selected = st.selectbox("KC ứng viên", kc_labels, key="candidate_kc")
        item = store.get_kc_by_label(selected)
        q_options = store.questions_for_kc(item["kc_index"])
        if q_options:
            qid = st.selectbox("Question ID", q_options)
        else:
            qid = st.text_input("Question ID", help="Cần question_mapping.csv từ Colab.")

        if st.button("🧠 Phân tích bằng UKT", type="primary"):
            try:
                if service is None:
                    raise RuntimeError(model_error or "Model chưa sẵn sàng.")
                result = service.predict_next(
                    st.session_state.history_rows,
                    item["kc_index"],
                    qid,
                )
                st.session_state.last_result = result
                st.session_state.last_candidate = item
            except Exception as e:
                st.error(f"Không thể inference: {e}")

        if "last_result" in st.session_state:
            r = st.session_state.last_result
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("P(correct)", f'{r["p_correct"]:.3f}')
            m2.metric("Uncertainty", f'{r["uncertainty_norm"]:.3f}')
            m3.metric("Forgetting risk", f'{r["forgetting_risk"]:.3f}')
            m4.metric("Safe mastery", f'{r["safe_mastery"]:.3f}')

            action = r["action"]
            if action == "ADVANCE":
                st.success(f"🟢 ADVANCE — {r['reason']}")
            elif action == "REVIEW":
                st.warning(f"🟡 REVIEW — {r['reason']}")
            else:
                st.info(f"🔵 DIAGNOSE — {r['reason']}")

            with st.expander("Chi tiết quyết định"):
                st.json(r)

with tab_path:
    st.header("Curriculum-aware Learning Path Prototype")
    st.warning(
        "Đây là lộ trình theo **thứ tự KC được giới thiệu trong course FORGET-SE**. "
        "Nó chưa phải prerequisite graph đã được chuyên gia xác nhận."
    )
    st.markdown("""
**Cách dùng:** chọn KC mục tiêu. Web sẽ:
1. dùng một câu đại diện cho từng KC;
2. gọi UKT để ước lượng trạng thái hiện tại;
3. áp dụng policy;
4. bỏ qua KC có `ADVANCE`, giữ lại KC cần `REVIEW/DIAGNOSE`;
5. trả một path theo curriculum order tới mục tiêu.
""")
    if kc_labels:
        target_label = st.selectbox("KC mục tiêu", kc_labels, index=min(9,len(kc_labels)-1), key="target_path")
        target = store.get_kc_by_label(target_label)

        if st.button("🛤️ Sinh lộ trình cá nhân hoá", type="primary"):
            try:
                if service is None:
                    raise RuntimeError(model_error or "Model chưa sẵn sàng.")
                states = service.evaluate_all_kcs(st.session_state.history_rows)
                path_table, path_codes = build_course_order_path(states, target["kc_index"])
                st.session_state.path_table = path_table
                st.session_state.path_codes = path_codes
            except Exception as e:
                st.error(f"Không thể tạo path: {e}")

        if "path_table" in st.session_state:
            path_table = st.session_state.path_table
            st.dataframe(
                path_table[[
                    "kc_code","kc_name_en","p_correct","uncertainty_norm",
                    "forgetting_risk","safe_mastery","action","path_status"
                ]],
                use_container_width=True,
                hide_index=True
            )
            codes = st.session_state.path_codes
            if codes:
                st.subheader("Lộ trình được đề xuất")
                st.code(" → ".join(codes))
            else:
                st.success("Các KC tới mục tiêu đều đang ở trạng thái ADVANCE theo policy hiện tại.")

with tab_mapping:
    st.header("Mapping sequence_id và thông tin FORGET-SE")
    st.markdown("""
`sequence_id` trong dữ liệu không phải tên chung chung vô nghĩa. Nó đại diện cho **10 KC cụ thể**.

Notebook chuẩn hoá:
`raw sequence_id → KC1..KC10 → model_concept_id 0..9 → tên KC`.

Question ID cũng được map tự động:
`raw qid → model_question_id → KC`.
""")

    kc_df = pd.DataFrame(kc_items)
    if not kc_df.empty:
        st.subheader("KC mapping")
        st.dataframe(kc_df, use_container_width=True, hide_index=True)

    st.subheader("Question mapping")
    if store.question_mapping.empty:
        st.info("Chạy notebook rồi copy question_mapping.csv vào artifacts/.")
    else:
        st.dataframe(store.question_mapping, use_container_width=True, hide_index=True)

    st.subheader("Thiết kế dữ liệu")
    st.markdown("""
- 186 sinh viên đại học.
- 56 câu hỏi.
- 10.873 interactions.
- 10 KC Software Engineering.
- Weekly quizzes được thiết kế có kiểm soát để nghiên cứu **memory decay** và **interference**.
- Quiz 0 đóng vai trò pre-diagnostic; các quiz tiếp theo kiểm soát target/interference; có một khoảng nghỉ 2 tuần phục vụ điều kiện decay.
""")

with tab_model:
    st.header("Model, checkpoint và metrics")
    st.markdown("""
**Cách đọc tab này**
- `best.pt`: trọng số UKT tốt nhất theo validation AUC.
- `latest.pt`: chỉ nằm trên Google Drive để resume training, không cần cho web.
- `web_config.json`: kiến trúc/model/data config để dựng lại UKT.
- `metrics.json`: kết quả validation.
""")
    if store.metrics:
        cols = st.columns(min(5, max(1,len(store.metrics))))
        for i,(k,v) in enumerate(store.metrics.items()):
            if isinstance(v,(int,float)):
                cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v,float) else str(v))
    else:
        st.info("Chưa có metrics.json.")

    with st.expander("web_config.json"):
        st.json(store.web_config or {})
