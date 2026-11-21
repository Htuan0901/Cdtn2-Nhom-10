"""Kiểm thử: upload CSV (hợp lệ, rỗng, encoding, dấu phân cách), schema, làm sạch, trạng thái phiên."""
import numpy as np
import pandas as pd
import pytest

from src.analytics import state
from src.analytics.data import cleaning, loader, schema


# ------------------------------------------------------------------ Upload
def test_upload_valid_csv():
    raw = "a,b,c\n1,2,x\n3,4,y\n".encode("utf-8")
    res = loader.load_file(raw, "t.csv")
    assert res.df.shape == (2, 3)
    assert res.delimiter == ","
    assert res.meta["n_rows"] == 2


def test_upload_utf8_bom_and_semicolon_vietnamese():
    raw = "tên;lương;tỉnh\nAn;10;Hà Nội\nBình;12;Đà Nẵng\n".encode("utf-8-sig")
    res = loader.load_file(raw, "vn.csv")
    assert list(res.df.columns) == ["tên", "lương", "tỉnh"]   # BOM không dính vào tên cột đầu
    assert res.delimiter == ";"
    assert res.df.iloc[0]["tỉnh"] == "Hà Nội"


def test_upload_tab_delimited():
    res = loader.load_file(b"x\ty\n1\t2\n3\t4\n", "t.tsv")
    assert res.df.shape == (2, 2) and res.delimiter == "\t"


def test_upload_empty_file():
    with pytest.raises(loader.DataLoadError, match="rỗng"):
        loader.load_file(b"", "empty.csv")


def test_upload_whitespace_only():
    with pytest.raises(loader.DataLoadError):
        loader.load_file(b"   \n  \n", "blank.csv")


def test_upload_header_only():
    with pytest.raises(loader.DataLoadError, match="tiêu đề"):
        loader.load_file(b"a,b,c\n", "header.csv")


def test_upload_binary_rejected():
    with pytest.raises(loader.DataLoadError):
        loader.load_file(b"\x00\x01\x02binary\x00", "bin.csv")


def test_upload_wrong_extension():
    with pytest.raises(loader.DataLoadError, match="không hỗ trợ"):
        loader.load_file(b"a,b\n1,2", "x.xlsx")


def test_duplicate_column_names_are_renamed():
    res = loader.load_file(b"a,a,\n1,2,3\n", "dup.csv")
    assert len(set(res.df.columns)) == 3


def test_dataset_id_changes_with_content():
    a = loader.load_file(b"a\n1\n", "x.csv").dataset_id
    b = loader.load_file(b"a\n2\n", "x.csv").dataset_id
    assert a != b


# ------------------------------------------------------------------ Schema
def test_schema_roles(job_df, numeric_df):
    r = {p.name: p.role for p in schema.profile_columns(job_df)}
    assert r["id"] == "id"
    assert r["year"] == "year"
    assert r["job_title"] == "categorical"
    r2 = {p.name: p.role for p in schema.profile_columns(numeric_df)}
    assert r2["x1"] == "numeric" and r2["date"] == "datetime"
    assert "date" in schema.time_candidates(numeric_df)
    assert "id" not in schema.numeric_columns(job_df)   # ID không được coi là biến số phân tích


def test_schema_string_dates_detected():
    df = pd.DataFrame({"ngay_do": ["01/02/2023", "02/02/2023", "03/02/2023"] * 10, "v": range(30)})
    assert schema.infer_role(df["ngay_do"], "ngay_do") == "datetime"


def test_wrong_type_numeric_as_text_converted():
    df = pd.DataFrame({"v": ["1,5", "2", "abc", "4 "]})
    out, step = cleaning.convert_type(df, "v", "numeric")
    assert pd.api.types.is_numeric_dtype(out["v"])
    assert out["v"].tolist()[:2] == [1.5, 2.0]
    assert np.isnan(out["v"].iloc[2])
    assert "1 giá trị" in step.detail


# ------------------------------------------------------------------ Làm sạch
def test_missing_values_and_impute(numeric_df):
    n_miss = numeric_df["x3"].isna().sum()
    assert n_miss > 0
    out, step = cleaning.impute(numeric_df, ["x3"], "median")
    assert out["x3"].isna().sum() == 0
    assert step.cells_changed == n_miss
    assert numeric_df["x3"].isna().sum() == n_miss       # dữ liệu gốc không bị sửa
    out2, _ = cleaning.impute(numeric_df, ["group"], "mean")   # mean trên cột chữ ⇒ bỏ qua an toàn
    assert out2["group"].equals(numeric_df["group"])


def test_impute_mode_and_drop_rows(numeric_df):
    out, _ = cleaning.impute(numeric_df, ["x3"], "drop_rows")
    assert len(out) == len(numeric_df) - numeric_df["x3"].isna().sum()


def test_duplicates_ignore_id(job_df):
    dup = job_df.iloc[:5].copy()
    dup["id"] = range(10_000, 10_005)        # ID khác nhưng nội dung trùng
    df = pd.concat([job_df, dup], ignore_index=True)
    assert cleaning.count_duplicates(df, ignore_id=True) >= 5
    out, step = cleaning.drop_duplicates(df, ignore_id=True)
    assert len(out) == len(df) - cleaning.count_duplicates(df, ignore_id=True)
    assert step.rows_before - step.rows_after == len(df) - len(out)


def test_outliers_iqr_cap_and_remove():
    df = pd.DataFrame({"v": list(range(1, 101)) + [10_000]})
    rep = cleaning.detect_outliers(df, ["v"], "iqr")
    assert rep.iloc[0]["Số ngoại lệ"] == 1
    capped, _ = cleaning.handle_outliers(df, ["v"], "iqr", "cap")
    assert capped["v"].max() < 10_000 and len(capped) == len(df)
    removed, _ = cleaning.handle_outliers(df, ["v"], "iqr", "remove")
    assert len(removed) == len(df) - 1
    kept, _ = cleaning.handle_outliers(df, ["v"], "zscore", "keep")
    assert kept.equals(df)


def test_scaling(numeric_df):
    out, _ = cleaning.scale_columns(numeric_df, ["x1"], "standard")
    assert abs(out["x1_std"].mean()) < 1e-9 and abs(out["x1_std"].std(ddof=0) - 1) < 1e-6
    out2, _ = cleaning.scale_columns(numeric_df, ["x1"], "minmax")
    assert out2["x1_minmax"].min() == 0 and out2["x1_minmax"].max() == 1


def test_invalid_values_placeholders():
    df = pd.DataFrame({"a": ["x", "N/A", " y ", "Đang cập nhật", None], "b": [1.0, np.inf, -2, 3, 4]})
    rep = cleaning.invalid_value_report(df)
    assert not rep.empty
    out, step = cleaning.fix_invalid_values(df, negative_to_nan_cols=["b"])
    assert out["a"].isna().sum() == 3 and out["a"].iloc[2] == "y"
    assert out["b"].isna().sum() == 2


def test_datetime_parts():
    df = pd.DataFrame({"d": ["2024-01-15", "2024-03-20", "bad"]})
    out, _ = cleaning.extract_datetime_parts(df, "d")
    assert out["d_month"].tolist()[:2] == [1, 3]


def test_quality_compare(numeric_df):
    out, _ = cleaning.impute(numeric_df, ["x3"], "mean")
    cmp = cleaning.compare_quality(numeric_df, out)
    row = cmp[cmp["Chỉ số"] == "Ô bị thiếu"].iloc[0]
    assert row["Trước"] > 0 and row["Sau"] == 0


# ------------------------------------------------------------------ Trạng thái phiên
def test_state_new_dataset_resets_results(numeric_df):
    ss = {}
    state.init_state(ss)
    state.set_dataset(ss, numeric_df, {"dataset_id": "A", "name": "a.csv"})
    state.put_result(ss, "ml", {"x": 1})
    assert state.get_result(ss, "ml") == {"x": 1}
    state.set_dataset(ss, numeric_df.head(10), {"dataset_id": "B", "name": "b.csv"})
    assert state.get_result(ss, "ml") is None            # không dùng nhầm kết quả dataset cũ


def test_state_undo(numeric_df):
    ss = {}
    state.init_state(ss)
    state.set_dataset(ss, numeric_df, {"dataset_id": "A"})
    out, step = cleaning.impute(numeric_df, ["x3"], "mean")
    state.apply_step(ss, out, step)
    assert len(ss["ws_clean_log"]) == 1 and ss["df"]["x3"].isna().sum() == 0
    assert state.undo(ss)
    assert ss["df"]["x3"].isna().sum() > 0 and ss["ws_clean_log"] == []
