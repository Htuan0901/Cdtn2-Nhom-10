import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.state import get_data
from src.ui import SERIES, fmt_int, fmt_pct, kpi_cards, page_header, style_fig

page_header("🧹", "Làm sạch dữ liệu", "Nhật ký từng bước xử lý: số dòng trước/sau, lý do loại bỏ, kiểm tra chất lượng sau làm sạch.")

li, hn, log = get_data()
st.page_link("views/cleaning_interactive.py", label="Làm sạch tương tác trên bộ dữ liệu phân tích (thiếu, trùng, ngoại lệ, chuẩn hóa, chuyển kiểu, nhật ký, hoàn tác)",
             icon=":material/tune:")

st.markdown("""
<div class="pipeline">
<span class="node">Xóa trùng</span><span class="arrow">→</span><span class="node">Xử lý thiếu</span><span class="arrow">→</span>
<span class="node">Lọc tin CNTT</span><span class="arrow">→</span><span class="node">Chuẩn hóa lương USD/năm</span><span class="arrow">→</span>
<span class="node">Chuẩn hóa địa điểm/cấp bậc/tháng</span><span class="arrow">→</span><span class="node">Trích xuất kỹ năng & vị trí</span>
</div>""", unsafe_allow_html=True)

if log is None or log.empty:
    st.info("Chưa có nhật ký làm sạch. Hãy chạy pipeline ở trang Nhập dữ liệu.")
    st.stop()

for src in log["Nguồn"].unique():
    lg = log[log["Nguồn"] == src].reset_index(drop=True)
    first, last = lg.iloc[0]["Trước"], lg.iloc[-1]["Sau"]
    st.markdown(f"### {'🔗' if src == 'LinkedIn' else '🟧'} {src}")
    kpi_cards([("Số dòng gốc", fmt_int(first)), ("Sau làm sạch", fmt_int(last)),
               ("Bị loại", fmt_int(first - last), fmt_pct((first - last) / first), "down"),
               ("Tỉ lệ giữ lại", fmt_pct(last / first))])
    steps = lg.iloc[1:-1]
    steps = steps[steps["Bị loại"] > 0]
    fig = go.Figure(go.Waterfall(
        x=["Gốc"] + steps["Bước"].tolist() + ["Còn lại"],
        measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        y=[first] + (-steps["Bị loại"]).tolist() + [0],
        text=[fmt_int(first)] + [f"-{fmt_int(v)}" for v in steps["Bị loại"]] + [fmt_int(last)],
        textposition="outside", connector={"line": {"color": "#c9ced8"}},
        decreasing={"marker": {"color": "#e34948"}}, totals={"marker": {"color": SERIES[0]}},
        increasing={"marker": {"color": SERIES[2]}}))
    fig.update_layout(title=f"Số dòng qua từng bước – {src}", showlegend=False)
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    st.dataframe(lg, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="localized") for c in ["Trước", "Sau", "Bị loại"]})

st.markdown("### ✅ Kiểm tra chất lượng sau làm sạch")
t1, t2 = st.tabs(["LinkedIn", "Hacker News"])


def _quality(df: pd.DataFrame):
    q = pd.DataFrame({"Kiểu dữ liệu": df.dtypes.astype(str), "Giá trị thiếu": df.isna().sum(),
                      "% thiếu": (df.isna().mean() * 100).round(1), "Số giá trị khác nhau": df.nunique()})
    st.dataframe(q, width="stretch")


with t1:
    _quality(li)
    st.caption("`salary_usd` thiếu là bình thường: chỉ ~1/3 tin LinkedIn công bố lương. Các phân tích lương chỉ dùng tin có lương.")
with t2:
    _quality(hn)
    st.caption("`salary_usd` của HN được tách từ nội dung tin bằng regex (ví dụ “$120k – $160k”), nên phần lớn là thiếu.")

with st.expander("📖 Giải thích các quy tắc làm sạch"):
    st.markdown("""
- **Lọc tin CNTT (LinkedIn):** giữ tin có tiêu đề chứa từ khóa CNTT (software, developer, data, cloud, devops, security, IT support…)
  **hoặc** được LinkedIn gắn nhãn chức năng *Information Technology*. Hacker News vốn là cộng đồng công nghệ nên giữ toàn bộ.
- **Lương:** dùng `normalized_salary` (LinkedIn đã quy đổi giờ/tháng → năm), chỉ giữ USD, loại giá trị < 15.000 hoặc > 600.000 USD/năm.
- **Địa điểm:** tách mã bang từ chuỗi “Thành phố, ST”; tin chỉ ghi “United States” xếp vào “Toàn quốc”.
- **Cấp bậc:** map *Entry level, Mid-Senior level…* sang nhãn tiếng Việt; thiếu thì là “Không rõ”.
- **Tháng (HN):** ghép tên tháng và năm của thread “Who is hiring” → ngày đầu tháng.
- **Kỹ năng:** từ điển ~80 kỹ năng, mỗi kỹ năng có nhiều biến thể (vd *k8s → Kubernetes*, *golang → Go*, *postgres → PostgreSQL*).
""")
