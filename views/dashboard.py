import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import EXP_ORDER, explode_skills
from src.skills import SKILL_GROUP
from src.state import get_data
from src.ui import SEQ_BLUE, SERIES, fmt_int, fmt_pct, fmt_usd, kpi_cards, page_header, style_fig

page_header("dashboard", "Dashboard thị trường tuyển dụng CNTT", "Lọc theo vị trí, cấp bậc, khu vực, kỹ năng để xem bức tranh tổng thể.")

li, hn, _ = get_data()

with st.sidebar:
    st.markdown("### :material/tune: Bộ lọc")
    roles = st.multiselect("Vị trí", li["role"].value_counts().index.tolist())
    exps = st.multiselect("Cấp bậc", [e for e in EXP_ORDER if e in li["experience"].unique()])
    states = st.multiselect("Bang (LinkedIn)", li["state"].value_counts().index.tolist())
    sk = st.multiselect("Phải có kỹ năng", explode_skills(li)["skill"].value_counts().index.tolist())
    remote_only = st.toggle("Chỉ tin cho làm từ xa")
    yr = st.slider("Giai đoạn (Hacker News)", int(hn["month"].dt.year.min()), int(hn["month"].dt.year.max()),
                   (int(hn["month"].dt.year.min()), int(hn["month"].dt.year.max())))

f = li
if roles:
    f = f[f["role"].isin(roles)]
if exps:
    f = f[f["experience"].isin(exps)]
if states:
    f = f[f["state"].isin(states)]
if remote_only:
    f = f[f["remote"]]
if sk:
    f = f[f["skills"].str.split(";").map(lambda xs: all(s in xs for s in sk))]
h = hn[hn["month"].dt.year.between(*yr)]
if roles:
    h = h[h["role"].isin(roles)]
if remote_only:
    h = h[h["remote"]]
if sk:
    h = h[h["skills"].str.split(";").map(lambda xs: all(s in xs for s in sk))]

if f.empty:
    st.warning("Không có tin nào khớp bộ lọc. Hãy nới lỏng bộ lọc.")
    st.stop()

sal = f["salary_usd"].dropna()
top = explode_skills(f)["skill"].value_counts()
kpi_cards([
    ("Số tin (LinkedIn)", fmt_int(len(f)), fmt_pct(len(f) / len(li)) + " tổng"),
    ("Số công ty", fmt_int(f["company_name"].nunique())),
    ("Lương trung vị", fmt_usd(sal.median()) if len(sal) else "–", f"P25–P75: {fmt_usd(sal.quantile(.25))}–{fmt_usd(sal.quantile(.75))}"
     if len(sal) else ""),
    ("Kỹ năng hot nhất", top.index[0] if len(top) else "–"),
    ("Tin HN trong giai đoạn", fmt_int(len(h)), f"{yr[0]}–{yr[1]}"),
])

c1, c2 = st.columns([1.2, 1])
d = (top.head(15) / len(f) * 100).rename("pct").reset_index()
d["Nhóm"] = d["skill"].map(SKILL_GROUP)
fig = px.bar(d.iloc[::-1], x="pct", y="skill", orientation="h", color="Nhóm", title="Top 15 kỹ năng (% tin)",
             labels={"pct": "% tin", "skill": ""})
fig.update_yaxes(categoryorder="total ascending")
c1.plotly_chart(style_fig(fig, 470), width="stretch")
r = f["role"].value_counts()
fig = px.pie(values=r.values, names=r.index, hole=0.5, title="Cơ cấu vị trí")
fig.update_traces(textposition="inside", textinfo="percent")
fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle"))
c2.plotly_chart(style_fig(fig, 470), width="stretch")

c1, c2 = st.columns(2)
if len(sal) >= 10:
    fs = f.dropna(subset=["salary_usd"])
    fs = fs[fs["experience"] != "Không rõ"]
    fig = px.violin(fs, x="experience", y="salary_usd", box=True, category_orders={"experience": EXP_ORDER},
                    title="Phân bố lương theo cấp bậc", labels={"experience": "", "salary_usd": "USD/năm"},
                    color_discrete_sequence=[SERIES[0]])
    c1.plotly_chart(style_fig(fig, 380), width="stretch")
else:
    c1.info("Quá ít tin có lương để vẽ phân bố.")
sc = f.loc[~f["state"].isin(["Toàn quốc", "Khác"]), "state"].value_counts().rename_axis("state").reset_index(name="n")
fig = px.choropleth(sc, locations="state", locationmode="USA-states", color="n", scope="usa", color_continuous_scale=SEQ_BLUE,
                    title="Phân bố theo bang", labels={"n": "Số tin"})
c2.plotly_chart(style_fig(fig, 380), width="stretch")

m = h.groupby("month").size()
fig = px.area(x=m.index, y=m.values, title="Số tin Hacker News mỗi tháng (theo bộ lọc)", labels={"x": "", "y": "Số tin"},
              color_discrete_sequence=[SERIES[1]])
st.plotly_chart(style_fig(fig, 320), width="stretch")

st.markdown("##### :material/list_alt: Danh sách tin (LinkedIn)")
st.dataframe(f[["title", "company_name", "role", "experience", "location", "salary_usd", "skills"]].head(500), width="stretch",
             hide_index=True, column_config={"salary_usd": st.column_config.NumberColumn("Lương (USD/năm)", format="$%,.0f")})
