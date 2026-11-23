import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import explode_skills
from src.skills import SKILL_GROUP, SKILLS, extract_skills, highlight_skills, skills_matrix
from src.state import get_data
from src.ui import SERIES, fmt_int, fmt_pct, kpi_cards, page_header, style_fig

page_header("code", "Trích xuất kỹ năng", "Text mining dựa trên từ điển + biểu thức chính quy: nhận diện và chuẩn hóa tên kỹ năng/công nghệ "
            "từ mô tả công việc, tạo bảng đặc trưng (feature table).")

li, hn, _ = get_data()

kpi_cards([
    ("Kỹ năng trong từ điển", len(SKILLS), f"{len(set(SKILL_GROUP.values()))} nhóm"),
    ("Biến thể (mẫu regex)", sum(len(p) for _, _, p in SKILLS)),
    ("TB kỹ năng/tin (LinkedIn)", f"{li['n_skills'].mean():.1f}", fmt_pct((li["n_skills"] > 0).mean()) + " tin có ≥1"),
    ("TB kỹ năng/tin (HN)", f"{hn['n_skills'].mean():.1f}", fmt_pct((hn["n_skills"] > 0).mean()) + " tin có ≥1"),
])

t1, t2, t3, t4 = st.tabs([":material/science: Thử trích xuất", ":material/menu_book: Từ điển kỹ năng",
                         ":material/grid_on: Bảng đặc trưng", ":material/search: Tra cứu theo kỹ năng"])

with t1:
    src = st.radio("Văn bản", ["Lấy ngẫu nhiên 1 tin LinkedIn", "Lấy ngẫu nhiên 1 tin Hacker News", "Tự nhập"], horizontal=True)
    if src.startswith("Tự"):
        text = st.text_area("Dán mô tả công việc (tiếng Anh)", height=180,
                            value="We are hiring a Senior Backend Engineer (Golang/Python) to build microservices on AWS with "
                                  "Docker, k8s and Terraform. Experience with Postgres, Redis, Kafka and CI/CD (GitHub Actions) "
                                  "is a plus. Bonus: LLM / RAG experience.")
    else:
        if st.button("Lấy tin khác", icon=":material/shuffle:") or "sample_seed" not in st.session_state:
            st.session_state["sample_seed"] = int(pd.Timestamp.now().value % 100000)
        df = li if "LinkedIn" in src else hn
        row = df.sample(1, random_state=st.session_state["sample_seed"]).iloc[0]
        text = row["desc_excerpt"]
        st.caption(f"**{row.get('title', row.get('position', ''))}** · {row.get('company_name', '')} · "
                   f"(đoạn trích {len(text)} ký tự đầu)")
    found = extract_skills(text)
    c1, c2 = st.columns([2, 1])
    c1.markdown("**Văn bản (kỹ năng được tô đậm):**")
    c1.markdown(highlight_skills(text))
    c2.markdown(f"**Kết quả: {len(found)} kỹ năng**")
    c2.markdown("".join(f'<span class="pill">{s}</span>' for s in found) or "_(không tìm thấy)_", unsafe_allow_html=True)
    c2.dataframe(pd.DataFrame({"Kỹ năng": found, "Nhóm": [SKILL_GROUP[s] for s in found]}), hide_index=True, width="stretch")

with t2:
    st.markdown("Mỗi kỹ năng có một **tên chuẩn** và nhiều **biến thể** được gộp về tên chuẩn đó (chuẩn hóa tên kỹ năng).")
    dic = pd.DataFrame([{"Kỹ năng chuẩn": n, "Nhóm": g, "Biến thể (regex)": " | ".join(p)} for n, g, p in SKILLS])
    grp = st.multiselect("Lọc theo nhóm", sorted(dic["Nhóm"].unique()))
    st.dataframe(dic[dic["Nhóm"].isin(grp)] if grp else dic, width="stretch", hide_index=True, height=480)
    st.caption("Ranh giới từ được kiểm tra để tránh nhận nhầm (vd “Java” không khớp “JavaScript”, “Excel” không khớp “excellent”, "
               "“R”/“C”/“Go” chỉ khớp khi đứng trong danh sách như “Python, R, SQL”).")

with t3:
    st.markdown("Bảng đặc trưng nhị phân: mỗi dòng là một tin, mỗi cột là một kỹ năng (1 = có yêu cầu). Dùng cho EDA, mô hình lương và chuỗi thời gian.")
    which = st.radio("Nguồn", ["LinkedIn", "Hacker News"], horizontal=True, key="ft_src")
    df = li if which == "LinkedIn" else hn
    m = skills_matrix(df["skills"].head(2000))
    meta = df[["title", "role", "experience", "salary_usd"]] if which == "LinkedIn" else df[["month", "position", "role", "country"]]
    st.dataframe(pd.concat([meta.head(2000).reset_index(drop=True), m.reset_index(drop=True)], axis=1).head(300),
                 width="stretch", height=380)
    st.caption(f"Hiển thị 300 dòng đầu; bảng đầy đủ có {fmt_int(len(df))} dòng × {skills_matrix(df['skills'].head(3000)).shape[1]}+ cột kỹ năng. "
               "Tải về ở trang Xuất kết quả.")

with t4:
    allsk = explode_skills(li)["skill"].value_counts()
    sk = st.selectbox("Chọn kỹ năng", allsk.index.tolist())
    sub = li[li["skills"].str.split(";").map(lambda xs: sk in xs)]
    sal = sub["salary_usd"].dropna()
    kpi_cards([("Số tin LinkedIn", fmt_int(len(sub)), fmt_pct(len(sub) / len(li))),
               ("Lương trung vị", f"${sal.median()/1000:,.0f}k" if len(sal) else "–", f"{fmt_int(len(sal))} tin có lương"),
               ("Vị trí phổ biến", sub["role"].value_counts().index[0] if len(sub) else "–"),
               ("Tin HN nhắc tới", fmt_int(hn["skills"].str.split(";").map(lambda xs: sk in xs).sum()))])
    co = explode_skills(sub)["skill"].value_counts().drop(sk, errors="ignore").head(12)
    fig = px.bar(x=co.values / max(len(sub), 1) * 100, y=co.index, orientation="h", labels={"x": "% tin", "y": ""},
                 title=f"Kỹ năng thường đi kèm {sk}", color_discrete_sequence=[SERIES[4]])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 400), width="stretch")
    st.dataframe(sub[["title", "company_name", "role", "experience", "location", "salary_usd"]].head(50), width="stretch",
                 hide_index=True)
