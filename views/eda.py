import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import EXP_ORDER, explode_skills
from src.skills import SKILL_GROUP
from src.ui import SEQ_BLUE, SERIES, fmt_int, fmt_pct, fmt_usd, kpi_cards, page_header, style_fig
from src.state import get_data

page_header("🔎", "Phân tích khám phá (EDA)", "Kỹ năng/công nghệ phổ biến, mức lương theo vị trí và kinh nghiệm, phân bố theo khu vực.")

li, hn, _ = get_data()
sal = li[li["salary_usd"].notna()]
ex = explode_skills(li, ["role", "salary_usd"])
top_skill = ex["skill"].value_counts()

kpi_cards([
    ("Tin CNTT", fmt_int(len(li))),
    ("Lương trung vị", fmt_usd(sal["salary_usd"].median()), f"{fmt_int(len(sal))} tin có lương"),
    ("Kỹ năng phổ biến nhất", top_skill.index[0], fmt_pct(top_skill.iloc[0] / len(li)) + " số tin"),
    ("Vị trí nhiều tin nhất", li["role"].value_counts().index[0]),
    ("Tỉ lệ cho làm từ xa", fmt_pct(li["remote"].mean())),
])

# ---------------- Insight tự động ----------------
role_med = sal.groupby("role")["salary_usd"].agg(["median", "size"]).query("size >= 30").sort_values("median")
state_cnt = li.loc[~li["state"].isin(["Toàn quốc", "Khác"]), "state"].value_counts()
sk_sal = ex.dropna(subset=["salary_usd"]).groupby("skill")["salary_usd"].agg(["median", "size"]).query("size >= 80")
hn_recent = hn[hn["month"] >= hn["month"].max() - pd.DateOffset(months=11)]
ins = [
    f"**{top_skill.index[0]}**, **{top_skill.index[1]}** và **{top_skill.index[2]}** là 3 kỹ năng được nhắc nhiều nhất trong tin CNTT trên LinkedIn.",
    f"Vị trí trả lương cao nhất (trung vị) là **{role_med.index[-1]}** ({fmt_usd(role_med['median'].iloc[-1])}), "
    f"thấp nhất là **{role_med.index[0]}** ({fmt_usd(role_med['median'].iloc[0])}).",
    f"Kỹ năng gắn với lương trung vị cao nhất: **{sk_sal['median'].idxmax()}** ({fmt_usd(sk_sal['median'].max())}).",
    f"Bang có nhiều tin nhất: **{state_cnt.index[0]}** ({fmt_int(state_cnt.iloc[0])} tin), tiếp theo là {state_cnt.index[1]} và {state_cnt.index[2]}.",
    f"Trên Hacker News, 12 tháng gần nhất có trung bình **{len(hn_recent) / 12:.0f} tin/tháng**, "
    f"so với **{len(hn[hn['month'].dt.year == 2018]) / 12:.0f} tin/tháng** năm 2018.",
]
ins = [re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", i) for i in ins]
st.markdown('<div class="card"><h4>💡 Insight nổi bật</h4>' + "".join(f"<div>• {i}</div>" for i in ins) + "</div>",
            unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs(["🧩 Kỹ năng", "💵 Lương", "🗺️ Khu vực", "👔 Vị trí & cấp bậc", "🟧 Hacker News"])

with t1:
    c1, c2 = st.columns([1.3, 1])
    n = c1.slider("Số kỹ năng hiển thị", 10, 40, 20)
    d = (top_skill.head(n) / len(li) * 100).rename("pct").reset_index()
    d["Nhóm"] = d["skill"].map(SKILL_GROUP)
    fig = px.bar(d.iloc[::-1], x="pct", y="skill", color="Nhóm", orientation="h",
                 labels={"pct": "% số tin nhắc tới", "skill": ""}, title=f"Top {n} kỹ năng được yêu cầu (LinkedIn)")
    fig.update_yaxes(categoryorder="total ascending")
    c1.plotly_chart(style_fig(fig, 30 * n // 1 + 60), width="stretch")
    g = ex.assign(group=ex["skill"].map(SKILL_GROUP)).groupby("group").size().sort_values()
    fig = px.pie(values=g.values, names=g.index, hole=0.5, title="Cơ cấu nhu cầu theo nhóm kỹ năng")
    fig.update_traces(textinfo="label+percent", textposition="inside", insidetextorientation="horizontal")
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style_fig(fig, 360), width="stretch")
    nsk = li["n_skills"].clip(upper=15)
    fig = px.histogram(nsk, nbins=16, title="Số kỹ năng yêu cầu trong mỗi tin", labels={"value": "Số kỹ năng"})
    fig.update_layout(showlegend=False, yaxis_title="Số tin")
    c2.plotly_chart(style_fig(fig, 260), width="stretch")

    st.markdown("##### Kỹ năng hay đi cùng nhau (đồng xuất hiện)")
    from src.skills import skills_matrix
    top12 = top_skill.head(14).index
    m = skills_matrix(li["skills"])[top12]
    co = (m.T @ m) / np.diag(m.T @ m)[:, None] * 100
    fig = px.imshow(co.round(0), text_auto=True, color_continuous_scale=SEQ_BLUE, aspect="auto",
                    labels=dict(color="%"), title="Trong các tin có kỹ năng ở hàng, bao nhiêu % cũng yêu cầu kỹ năng ở cột")
    st.plotly_chart(style_fig(fig, 520), width="stretch")

with t2:
    c1, c2 = st.columns(2)
    fig = px.histogram(sal, x="salary_usd", nbins=60, title="Phân bố lương (USD/năm)", labels={"salary_usd": "Lương"})
    fig.add_vline(sal["salary_usd"].median(), line_dash="dash", annotation_text=f"Trung vị {fmt_usd(sal['salary_usd'].median())}")
    fig.update_layout(yaxis_title="Số tin")
    c1.plotly_chart(style_fig(fig, 340), width="stretch")
    exp = sal[sal["experience"] != "Không rõ"]
    fig = px.box(exp, x="experience", y="salary_usd", category_orders={"experience": EXP_ORDER}, points=False,
                 title="Lương theo cấp bậc kinh nghiệm", labels={"experience": "", "salary_usd": "USD/năm"})
    c2.plotly_chart(style_fig(fig, 340), width="stretch")
    rm = role_med.reset_index()
    fig = px.bar(rm, x="median", y="role", orientation="h", text=rm["median"].map(fmt_usd),
                 title="Lương trung vị theo vị trí (≥ 30 tin có lương)", labels={"median": "USD/năm", "role": ""},
                 color_discrete_sequence=[SERIES[0]])
    st.plotly_chart(style_fig(fig, 480), width="stretch")
    sk = sk_sal.sort_values("median").tail(20).reset_index()
    fig = px.bar(sk, x="median", y="skill", orientation="h", text=sk["median"].map(fmt_usd),
                 title="Top 20 kỹ năng gắn với lương trung vị cao nhất (≥ 80 tin)", labels={"median": "USD/năm", "skill": ""},
                 color_discrete_sequence=[SERIES[2]])
    st.plotly_chart(style_fig(fig, 560), width="stretch")
    st.caption("Lưu ý: đây là tương quan, không phải nhân quả. Kỹ năng gắn với vị trí cấp cao sẽ có lương cao hơn.")

with t3:
    sc = state_cnt.rename_axis("state").reset_index(name="n")
    smed = sal.groupby("state")["salary_usd"].median()
    sc["Lương trung vị"] = sc["state"].map(smed)
    fig = px.choropleth(sc, locations="state", locationmode="USA-states", color="n", scope="usa",
                        color_continuous_scale=SEQ_BLUE, hover_data={"Lương trung vị": ":,.0f"},
                        title="Số tin tuyển dụng CNTT theo bang", labels={"n": "Số tin"})
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    c1, c2 = st.columns(2)
    top_city = li["city"].value_counts().head(15).rename_axis("city").reset_index(name="n")
    fig = px.bar(top_city.iloc[::-1], x="n", y="city", orientation="h", title="Top 15 thành phố nhiều tin",
                 labels={"n": "Số tin", "city": ""})
    c1.plotly_chart(style_fig(fig, 460), width="stretch")
    sm = sc.dropna().query("n >= 100").sort_values("Lương trung vị").tail(15)
    fig = px.bar(sm, x="Lương trung vị", y="state", orientation="h", title="Bang có lương trung vị cao nhất (≥ 100 tin)",
                 labels={"state": ""}, color_discrete_sequence=[SERIES[2]])
    c2.plotly_chart(style_fig(fig, 460), width="stretch")
    wr = li.groupby("work_type").size().sort_values()
    fig = px.pie(values=wr.values, names=wr.index, hole=0.55, title="Hình thức làm việc")
    fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5))
    st.plotly_chart(style_fig(fig, 320), width="stretch")

with t4:
    c1, c2 = st.columns(2)
    r = li["role"].value_counts().iloc[::-1]
    fig = px.bar(x=r.values, y=r.index, orientation="h", title="Số tin theo nhóm vị trí", labels={"x": "Số tin", "y": ""})
    c1.plotly_chart(style_fig(fig, 520), width="stretch")
    ct = pd.crosstab(li["role"], li["experience"], normalize="index").reindex(columns=EXP_ORDER).fillna(0) * 100
    fig = px.imshow(ct.round(0), text_auto=True, color_continuous_scale=SEQ_BLUE, aspect="auto",
                    title="Cơ cấu cấp bậc trong từng vị trí (%)", labels=dict(color="%", x="", y=""))
    c2.plotly_chart(style_fig(fig, 520), width="stretch")
    role = st.selectbox("Xem kỹ năng hàng đầu của vị trí", li["role"].value_counts().index)
    sub = ex[ex["role"] == role]["skill"].value_counts().head(15)
    nrole = (li["role"] == role).sum()
    fig = px.bar(x=sub.values / nrole * 100, y=sub.index, orientation="h", title=f"Top kỹ năng cho {role}",
                 labels={"x": "% số tin của vị trí", "y": ""}, color_discrete_sequence=[SERIES[1]])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 440), width="stretch")

with t5:
    c1, c2 = st.columns(2)
    mcount = hn.groupby("month").size()
    fig = px.line(x=mcount.index, y=mcount.values, markers=True, title="Số tin tuyển dụng mỗi tháng trên Hacker News",
                  labels={"x": "", "y": "Số tin"})
    c1.plotly_chart(style_fig(fig, 340), width="stretch")
    rem = hn.groupby(hn["month"].dt.to_period("Q").dt.to_timestamp())["remote"].mean() * 100
    fig = px.area(x=rem.index, y=rem.values, title="Tỉ lệ tin cho làm từ xa (remote) theo quý",
                  labels={"x": "", "y": "% tin"}, color_discrete_sequence=[SERIES[2]])
    c2.plotly_chart(style_fig(fig, 340), width="stretch")
    cty = hn["country"].value_counts().head(12).iloc[::-1]
    fig = px.bar(x=cty.values, y=cty.index, orientation="h", title="Quốc gia tuyển dụng nhiều nhất", labels={"x": "Số tin", "y": ""})
    c1.plotly_chart(style_fig(fig, 420), width="stretch")
    hsk = explode_skills(hn)["skill"].value_counts().head(15).iloc[::-1] / len(hn) * 100
    fig = px.bar(x=hsk.values, y=hsk.index, orientation="h", title="Top 15 kỹ năng trên HN (2018–2023)",
                 labels={"x": "% số tin", "y": ""}, color_discrete_sequence=[SERIES[1]])
    c2.plotly_chart(style_fig(fig, 420), width="stretch")
