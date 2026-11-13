def build_course_order_path(kc_state_df, target_kc_index):
    """
    Demo path theo curriculum order của FORGET-SE.
    Không gọi đây là expert prerequisite graph.
    """
    target = int(target_kc_index)
    df = kc_state_df[kc_state_df["kc_index"].astype(int) <= target].copy()
    df = df.sort_values("kc_index")

    recommended = []
    for _, row in df.iterrows():
        if row["action"] == "ADVANCE":
            status = "Có thể bỏ qua / đã đủ tốt"
        elif row["action"] == "DIAGNOSE":
            status = "Cần kiểm tra chẩn đoán"
            recommended.append(str(row["kc_code"]))
        else:
            status = "Cần ôn tập"
            recommended.append(str(row["kc_code"]))
        df.loc[row.name, "path_status"] = status

    # Target vẫn được thêm nếu trước đó tất cả prerequisite/order KC được skip.
    target_code = f"KC{target}"
    if target_code not in recommended:
        target_row = df[df["kc_index"].astype(int) == target]
        if not target_row.empty and target_row.iloc[0]["action"] != "ADVANCE":
            recommended.append(target_code)

    return df, recommended
