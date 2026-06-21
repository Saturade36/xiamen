# -*- coding: utf-8 -*-
"""
功能：
1. 数据初步描述：
   - 数据规模检查
   - 字段类型检查
   - 数据字典
   - 基础统计摘要
   - 区县-年份记录数检查

2. 数据预处理：
   - 字段清理
   - 类型转换
   - 缺失值识别与处理
   - 重复值检查
   - 坐标有效性检查
   - 夜光值质量控制
   - IQR 和 3σ 异常值标记
   - 对数变换
   - Z-score 标准化
   - Min-Max 归一化
   - 经纬度转 UTM 50N 坐标
   - 构建像元级多年特征

3. 探索性数据分析 EDA：
   - 直方图
   - 箱线图
   - 区县年度均值折线图
   - 空间散点分布图
   - 2013、2018、2024 夜光图
   - 2024 相对 2013 夜光变化图
   - 散点图
   - 相关矩阵
   - PCA 二维可视化
   - 近似 Moran's I 空间自相关
   - 自动生成 EDA 发现草稿

适用数据：
xiamen_viirs_pixel_samples_2013_2024.csv

建议运行环境：
Anaconda / Python 3.9+
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 0. 全局参数
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

INPUT_CSV = PROJECT_DIR / "xiamen_viirs_pixel_samples_2013_2024.csv"

# 如果你把 CSV 放在 data 文件夹下，代码也会自动查找
if not INPUT_CSV.exists():
    INPUT_CSV = PROJECT_DIR / "data" / "xiamen_viirs_pixel_samples_2013_2024.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs_pixel_samples"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report_text"

for folder in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ID_COL = "district"
YEAR_COL = "year"
LON_COL = "lon"
LAT_COL = "lat"

CORE_COLS = [
    "district",
    "year",
    "lon",
    "lat",
    "rad",
    "median_rad",
    "cf_cvg",
    "scale_m",
    "source_dataset",
]

NUMERIC_COLS = [
    "year",
    "lon",
    "lat",
    "rad",
    "median_rad",
    "cf_cvg",
    "scale_m",
]

FIELD_META = {
    "district": {
        "中文含义": "厦门市区县名称",
        "单位": "无",
        "说明": "像元所属行政区，包括思明区、湖里区、海沧区、集美区、同安区、翔安区",
    },
    "year": {
        "中文含义": "年份",
        "单位": "年",
        "说明": "2013—2024 年",
    },
    "lon": {
        "中文含义": "像元中心点经度",
        "单位": "度",
        "说明": "WGS84 坐标系下的经度",
    },
    "lat": {
        "中文含义": "像元中心点纬度",
        "单位": "度",
        "说明": "WGS84 坐标系下的纬度",
    },
    "rad": {
        "中文含义": "夜光辐亮度",
        "单位": "nW/cm²/sr",
        "说明": "VIIRS average_masked 波段，对应主要夜光强度指标",
    },
    "median_rad": {
        "中文含义": "中位夜光辐亮度",
        "单位": "nW/cm²/sr",
        "说明": "VIIRS median_masked 波段，可作为稳健性参考",
    },
    "cf_cvg": {
        "中文含义": "有效观测次数",
        "单位": "次",
        "说明": "云等条件筛选后的有效观测次数，用于评价数据质量",
    },
    "scale_m": {
        "中文含义": "采样尺度",
        "单位": "m",
        "说明": "GEE sampleRegions 使用的空间尺度，通常为 500 m",
    },
    "source_dataset": {
        "中文含义": "数据源",
        "单位": "无",
        "说明": "2013—2021 为 VIIRS 年度 V21，2022—2024 为 VIIRS 年度 V22",
    },
    "system:index": {
        "中文含义": "GEE 系统索引",
        "单位": "无",
        "说明": "GEE 导出的系统字段，分析时可删除",
    },
    ".geo": {
        "中文含义": "GEE 几何字段",
        "单位": "无",
        "说明": "当前文件中通常为空 MultiPoint，分析时可删除",
    },
}


# ============================================================
# 1. 基础工具函数
# ============================================================

def set_chinese_font():
    """
    设置中文字体。Windows 下优先使用微软雅黑或黑体。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    尝试使用不同编码读取 CSV。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{path}\n"
            f"请确认 xiamen_viirs_pixel_samples_2013_2024.csv 与脚本在同一文件夹，"
            f"或放在 data 文件夹中。"
        )

    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"成功读取 CSV，编码：{enc}")
            return df
        except Exception as e:
            last_error = e

    raise last_error


def save_csv(df: pd.DataFrame, path: Path):
    """
    保存 CSV，使用 utf-8-sig 方便 Excel 打开中文。
    """
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_excel(sheets: dict, path: Path):
    """
    多个表保存到一个 Excel 文件。
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def save_fig(filename: str):
    """
    保存当前图像。
    """
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理字段名。
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_divide(a, b):
    """
    安全除法，避免除以 0。
    """
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


# ============================================================
# 2. 数据初步描述
# ============================================================

def generate_data_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据规模和结构概览。
    """
    overview = {
        "记录数": len(df),
        "字段数": df.shape[1],
        "区县数量": df[ID_COL].nunique() if ID_COL in df.columns else np.nan,
        "年份数量": df[YEAR_COL].nunique() if YEAR_COL in df.columns else np.nan,
        "最小年份": df[YEAR_COL].min() if YEAR_COL in df.columns else np.nan,
        "最大年份": df[YEAR_COL].max() if YEAR_COL in df.columns else np.nan,
        "经度最小值": df[LON_COL].min() if LON_COL in df.columns else np.nan,
        "经度最大值": df[LON_COL].max() if LON_COL in df.columns else np.nan,
        "纬度最小值": df[LAT_COL].min() if LAT_COL in df.columns else np.nan,
        "纬度最大值": df[LAT_COL].max() if LAT_COL in df.columns else np.nan,
    }

    return pd.DataFrame([overview])


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成数据字典：
    字段名称、类型、单位、取值范围、缺失情况。
    """
    rows = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_rate = missing_count / len(df) if len(df) > 0 else np.nan

        meta = FIELD_META.get(col, {})
        meaning = meta.get("中文含义", "需人工补充")
        unit = meta.get("单位", "需人工补充")
        note = meta.get("说明", "")

        if pd.api.types.is_numeric_dtype(s):
            min_value = s.min()
            max_value = s.max()
            q25 = s.quantile(0.25)
            median = s.quantile(0.5)
            q75 = s.quantile(0.75)
            examples = ""
        else:
            min_value = ""
            max_value = ""
            q25 = ""
            median = ""
            q75 = ""
            examples = "，".join(s.dropna().astype(str).unique()[:8])

        rows.append({
            "字段名": col,
            "中文含义": meaning,
            "数据类型": str(s.dtype),
            "单位": unit,
            "非空记录数": int(s.notna().sum()),
            "缺失记录数": missing_count,
            "缺失率": round(missing_rate, 6),
            "唯一值数量": int(s.nunique(dropna=True)),
            "最小值": min_value,
            "25%分位数": q25,
            "中位数": median,
            "75%分位数": q75,
            "最大值": max_value,
            "示例取值": examples,
            "说明": note,
        })

    return pd.DataFrame(rows)


def create_basic_statistics(df: pd.DataFrame) -> dict:
    """
    基础统计摘要。
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        numeric_desc = pd.DataFrame()
    else:
        numeric_desc = (
            numeric_df
            .describe(percentiles=[0.25, 0.5, 0.75])
            .T
            .reset_index()
            .rename(columns={"index": "字段名"})
        )

        missing = df.isna().mean().reset_index()
        missing.columns = ["字段名", "缺失率"]
        numeric_desc = numeric_desc.merge(missing, on="字段名", how="left")

    district_year_count = pd.DataFrame()
    if ID_COL in df.columns and YEAR_COL in df.columns:
        district_year_count = (
            df.groupby([ID_COL, YEAR_COL])
            .size()
            .reset_index(name="记录数")
            .sort_values([ID_COL, YEAR_COL])
        )

    by_district = pd.DataFrame()
    if ID_COL in df.columns and "rad" in df.columns:
        by_district = (
            df.groupby(ID_COL)
            .agg(
                记录数=("rad", "size"),
                平均夜光强度=("rad", "mean"),
                夜光中位数=("rad", "median"),
                夜光标准差=("rad", "std"),
                夜光最小值=("rad", "min"),
                夜光最大值=("rad", "max"),
                平均有效观测次数=("cf_cvg", "mean") if "cf_cvg" in df.columns else ("rad", "size"),
            )
            .reset_index()
        )

    by_year = pd.DataFrame()
    if YEAR_COL in df.columns and "rad" in df.columns:
        by_year = (
            df.groupby(YEAR_COL)
            .agg(
                记录数=("rad", "size"),
                平均夜光强度=("rad", "mean"),
                夜光中位数=("rad", "median"),
                夜光标准差=("rad", "std"),
                夜光最小值=("rad", "min"),
                夜光最大值=("rad", "max"),
                平均有效观测次数=("cf_cvg", "mean") if "cf_cvg" in df.columns else ("rad", "size"),
            )
            .reset_index()
            .sort_values(YEAR_COL)
        )

    return {
        "numeric_describe": numeric_desc,
        "district_year_count": district_year_count,
        "by_district": by_district,
        "by_year": by_year,
    }


def export_initial_description(df_raw: pd.DataFrame):
    """
    输出数据初步描述结果。
    """
    overview = generate_data_overview(df_raw)
    data_dictionary = create_data_dictionary(df_raw)
    stats = create_basic_statistics(df_raw)

    save_csv(overview, TABLE_DIR / "01_data_overview.csv")
    save_csv(data_dictionary, TABLE_DIR / "02_data_dictionary.csv")
    save_csv(stats["numeric_describe"], TABLE_DIR / "03_basic_statistics_numeric.csv")
    save_csv(stats["district_year_count"], TABLE_DIR / "04_district_year_count.csv")
    save_csv(stats["by_district"], TABLE_DIR / "05_basic_statistics_by_district.csv")
    save_csv(stats["by_year"], TABLE_DIR / "06_basic_statistics_by_year.csv")

    save_excel(
        {
            "data_overview": overview,
            "data_dictionary": data_dictionary,
            "numeric_statistics": stats["numeric_describe"],
            "district_year_count": stats["district_year_count"],
            "by_district": stats["by_district"],
            "by_year": stats["by_year"],
        },
        TABLE_DIR / "initial_description_summary.xlsx"
    )


# ============================================================
# 3. 数据预处理
# ============================================================

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    字段类型转换。
    """
    df = df.copy()

    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str).str.strip()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if YEAR_COL in df.columns:
        df[YEAR_COL] = df[YEAR_COL].round().astype("Int64")

    if "source_dataset" in df.columns:
        df["source_dataset"] = df["source_dataset"].astype(str).str.strip()

    return df


def create_missing_report(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值处理前后对比。
    """
    before_report = pd.DataFrame({
        "字段名": before.columns,
        "缺失数_处理前": before.isna().sum().values,
        "缺失率_处理前": before.isna().mean().values,
    })

    after_report = pd.DataFrame({
        "字段名": after.columns,
        "缺失数_处理后": after.isna().sum().values,
        "缺失率_处理后": after.isna().mean().values,
    })

    return before_report.merge(after_report, on="字段名", how="outer")


def infer_missing_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    简单识别缺失模式：
    - 只做描述性判断，正式报告中应结合数据来源说明。
    """
    rows = []

    for col in df.columns:
        missing_rate = df[col].isna().mean()

        if missing_rate == 0:
            pattern = "无缺失"
            reason = "该字段没有缺失值"
        else:
            pattern = "需结合业务判断"
            reason = "脚本只检测缺失比例，是否为 MCAR/MAR/MNAR 需结合年份、区县和数据生成过程解释"

            if YEAR_COL in df.columns:
                by_year = df.groupby(YEAR_COL)[col].apply(lambda s: s.isna().mean())
                if by_year.max() - by_year.min() > 0.05:
                    pattern = "可能 MAR"
                    reason = "缺失率在不同年份之间差异较明显，可能与年份或数据源有关"

            if ID_COL in df.columns:
                by_district = df.groupby(ID_COL)[col].apply(lambda s: s.isna().mean())
                if by_district.max() - by_district.min() > 0.05:
                    pattern = "可能 MAR"
                    reason = "缺失率在不同区县之间差异较明显，可能与空间位置或行政区有关"

        rows.append({
            "字段名": col,
            "缺失率": missing_rate,
            "缺失模式初步判断": pattern,
            "说明": reason,
        })

    return pd.DataFrame(rows)


def detect_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查重复记录。
    对像元样本而言，district + year + lon + lat 理论上应唯一。
    """
    subset = [c for c in [ID_COL, YEAR_COL, LON_COL, LAT_COL] if c in df.columns]

    if len(subset) < 4:
        return pd.DataFrame()

    duplicate_mask = df.duplicated(subset=subset, keep=False)
    return df.loc[duplicate_mask].copy()


def clean_invalid_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    删除关键字段缺失、坐标异常、夜光值异常的记录。
    注意：
    - rad < 0 通常不应保留；
    - 高夜光值不直接删除，只在异常值检测中标记。
    """
    df = df.copy()

    invalid_reasons = []

    required = [ID_COL, YEAR_COL, LON_COL, LAT_COL, "rad"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"缺少必要字段：{col}")

    mask_missing_required = df[required].isna().any(axis=1)
    invalid_reasons.append(("关键字段缺失", mask_missing_required))

    mask_invalid_lonlat = ~(
        df[LON_COL].between(-180, 180) &
        df[LAT_COL].between(-90, 90)
    )
    invalid_reasons.append(("经纬度范围异常", mask_invalid_lonlat))

    mask_negative_rad = df["rad"] < 0
    invalid_reasons.append(("rad 小于 0", mask_negative_rad))

    invalid_mask = pd.Series(False, index=df.index)
    reason_text = pd.Series("", index=df.index, dtype="object")

    for reason, mask in invalid_reasons:
        invalid_mask = invalid_mask | mask.fillna(False)
        reason_text.loc[mask.fillna(False)] = reason_text.loc[mask.fillna(False)] + reason + ";"

    invalid_records = df.loc[invalid_mask].copy()
    if not invalid_records.empty:
        invalid_records["剔除原因"] = reason_text.loc[invalid_mask].values

    df_clean = df.loc[~invalid_mask].copy().reset_index(drop=True)

    return df_clean, invalid_records


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值填充。
    像元夜光主指标 rad 不建议随意填补，前面已经删除 rad 缺失记录。
    这里主要处理 cf_cvg、median_rad、scale_m 等辅助字段。
    """
    df = df.copy()

    if "median_rad" in df.columns:
        df["median_rad"] = df["median_rad"].fillna(df["rad"])

    if "cf_cvg" in df.columns:
        # 先按年份中位数填充，再用全局中位数兜底
        if YEAR_COL in df.columns:
            df["cf_cvg"] = df.groupby(YEAR_COL)["cf_cvg"].transform(
                lambda s: s.fillna(s.median())
            )
        df["cf_cvg"] = df["cf_cvg"].fillna(df["cf_cvg"].median())

    if "scale_m" in df.columns:
        df["scale_m"] = df["scale_m"].fillna(df["scale_m"].median())

    if "source_dataset" in df.columns:
        df["source_dataset"] = df["source_dataset"].fillna("Unknown")

    return df


def detect_outliers(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    IQR 与 3σ 异常值识别。
    只标记，不删除。
    """
    df = df.copy()
    report_rows = []
    detail_frames = []

    for col in cols:
        if col not in df.columns:
            continue

        x = df[col].dropna()
        if len(x) < 10:
            continue

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1

        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr

        mean_val = x.mean()
        std_val = x.std()

        if std_val == 0 or pd.isna(std_val):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (df[col] - mean_val) / std_val

        iqr_flag = ((df[col] < lower_iqr) | (df[col] > upper_iqr)).fillna(False)
        z_flag = (z.abs() > 3).fillna(False)

        df[f"{col}_iqr_outlier"] = iqr_flag.astype(int)
        df[f"{col}_z_outlier"] = z_flag.astype(int)

        report_rows.append({
            "字段名": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "IQR下界": lower_iqr,
            "IQR上界": upper_iqr,
            "均值": mean_val,
            "标准差": std_val,
            "IQR异常数量": int(iqr_flag.sum()),
            "3σ异常数量": int(z_flag.sum()),
            "IQR异常率": float(iqr_flag.mean()),
            "3σ异常率": float(z_flag.mean()),
        })

        flag_mask = iqr_flag | z_flag

        base_cols = [c for c in [ID_COL, YEAR_COL, LON_COL, LAT_COL, col] if c in df.columns]
        detail = df.loc[flag_mask, base_cols].copy()
        if not detail.empty:
            detail["异常字段"] = col
            detail["IQR异常"] = iqr_flag.loc[flag_mask].values
            detail["3σ异常"] = z_flag.loc[flag_mask].values
            detail["Z_score"] = z.loc[flag_mask].values
            detail_frames.append(detail)

    report = pd.DataFrame(report_rows)

    if detail_frames:
        details = pd.concat(detail_frames, ignore_index=True)
    else:
        details = pd.DataFrame()

    return df, report, details


def add_projection_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 WGS84 经纬度转换为 WGS84 / UTM Zone 50N，EPSG:32650。
    厦门位于 118E 附近，适合使用 UTM 50N。
    """
    df = df.copy()

    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32650",
            always_xy=True
        )

        x_utm, y_utm = transformer.transform(
            df[LON_COL].values,
            df[LAT_COL].values
        )

        df["x_utm50n_m"] = x_utm
        df["y_utm50n_m"] = y_utm
        df["projected_crs"] = "EPSG:32650"

    except Exception as e:
        print("提示：pyproj 未安装或投影转换失败，跳过 UTM 坐标生成。")
        print(f"原因：{e}")

    return df


def add_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据变换和基础特征工程。
    """
    df = df.copy()

    # 对数变换，缓解右偏分布
    if "rad" in df.columns:
        df["log1p_rad"] = np.log1p(df["rad"].clip(lower=0))

    if "median_rad" in df.columns:
        df["log1p_median_rad"] = np.log1p(df["median_rad"].clip(lower=0))

    # 平均夜光与中位夜光差异
    if "rad" in df.columns and "median_rad" in df.columns:
        df["rad_minus_median"] = df["rad"] - df["median_rad"]
        df["rad_div_median_plus001"] = df["rad"] / (df["median_rad"] + 0.001)

    # 按年度计算标准化夜光，用于跨年比较
    if YEAR_COL in df.columns and "rad" in df.columns:
        df["rad_year_zscore"] = df.groupby(YEAR_COL)["rad"].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() != 0 else 0
        )

    # 构造像元 ID。经纬度保留 6 位，避免浮点字符串过长
    df["lon_round6"] = df[LON_COL].round(6)
    df["lat_round6"] = df[LAT_COL].round(6)
    df["pixel_id"] = (
        df[ID_COL].astype(str) + "_" +
        df["lon_round6"].astype(str) + "_" +
        df["lat_round6"].astype(str)
    )

    return df


def add_scaled_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加 Z-score 和 Min-Max 标准化字段。
    原始字段不覆盖。
    """
    df = df.copy()

    scale_cols = [
        "rad",
        "median_rad",
        "cf_cvg",
        "log1p_rad",
        "rad_minus_median",
    ]

    scale_cols = [c for c in scale_cols if c in df.columns]

    if not scale_cols:
        return df

    temp = df[scale_cols].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    temp_imputed = imputer.fit_transform(temp)

    z_scaler = StandardScaler()
    z_values = z_scaler.fit_transform(temp_imputed)

    mm_scaler = MinMaxScaler()
    mm_values = mm_scaler.fit_transform(temp_imputed)

    for i, col in enumerate(scale_cols):
        df[f"z_{col}"] = z_values[:, i]
        df[f"minmax_{col}"] = mm_values[:, i]

    return df


def make_pixel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建像元级多年特征。
    每个像元一行，用于后续 K-Means / 层次聚类，也用于 EDA 的 PCA。
    """
    required = ["pixel_id", ID_COL, LON_COL, LAT_COL, YEAR_COL, "rad"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"构建像元特征缺少字段：{col}")

    id_cols = ["pixel_id", ID_COL, LON_COL, LAT_COL]

    extra_first_cols = []
    for col in ["x_utm50n_m", "y_utm50n_m"]:
        if col in df.columns:
            extra_first_cols.append(col)

    first_info = (
        df.groupby("pixel_id")
        .agg(
            district=(ID_COL, "first"),
            lon=(LON_COL, "first"),
            lat=(LAT_COL, "first"),
            **{col: (col, "first") for col in extra_first_cols}
        )
        .reset_index()
    )

    pivot = df.pivot_table(
        index="pixel_id",
        columns=YEAR_COL,
        values="rad",
        aggfunc="mean"
    )

    # 确保年份按顺序排列
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    year_values = [int(y) for y in pivot.columns]
    pivot.columns = [f"rad_{int(y)}" for y in pivot.columns]

    feat = first_info.merge(
        pivot.reset_index(),
        on="pixel_id",
        how="left"
    )

    rad_cols = [f"rad_{y}" for y in year_values if f"rad_{y}" in feat.columns]

    feat["mean_rad_2013_2024"] = feat[rad_cols].mean(axis=1)
    feat["median_rad_2013_2024"] = feat[rad_cols].median(axis=1)
    feat["std_rad_2013_2024"] = feat[rad_cols].std(axis=1)
    feat["min_rad_2013_2024"] = feat[rad_cols].min(axis=1)
    feat["max_rad_2013_2024"] = feat[rad_cols].max(axis=1)
    feat["range_rad_2013_2024"] = feat["max_rad_2013_2024"] - feat["min_rad_2013_2024"]
    feat["cv_rad_2013_2024"] = safe_divide(
        feat["std_rad_2013_2024"],
        feat["mean_rad_2013_2024"]
    )

    if "rad_2013" in feat.columns and "rad_2024" in feat.columns:
        feat["change_2024_minus_2013"] = feat["rad_2024"] - feat["rad_2013"]
        feat["growth_rate_2024_vs_2013"] = (
            feat["change_2024_minus_2013"] / (feat["rad_2013"] + 0.001)
        )
    else:
        feat["change_2024_minus_2013"] = np.nan
        feat["growth_rate_2024_vs_2013"] = np.nan

    # 线性趋势斜率：每个像元 2013—2024 夜光随年份变化的斜率
    years_array = np.array(year_values, dtype=float)

    def calc_slope(row):
        values = row[rad_cols].values.astype(float)
        valid = ~np.isnan(values)

        if valid.sum() < 2:
            return np.nan

        slope = np.polyfit(years_array[valid], values[valid], 1)[0]
        return slope

    feat["trend_slope_rad_per_year"] = feat.apply(calc_slope, axis=1)

    # 对像元特征做标准化
    feature_scale_cols = [
        "mean_rad_2013_2024",
        "std_rad_2013_2024",
        "range_rad_2013_2024",
        "cv_rad_2013_2024",
        "change_2024_minus_2013",
        "growth_rate_2024_vs_2013",
        "trend_slope_rad_per_year",
    ]

    feature_scale_cols = [c for c in feature_scale_cols if c in feat.columns]

    temp = feat[feature_scale_cols].replace([np.inf, -np.inf], np.nan)
    temp = temp.fillna(temp.median(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(temp)

    for i, col in enumerate(feature_scale_cols):
        feat[f"z_{col}"] = scaled[:, i]

    return feat


def preprocess_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    完整预处理流程。
    返回：
    - 清洗后的像元-年份数据
    - 像元级多年特征数据
    - 各类报告
    """
    df = df_raw.copy()

    # 字段名清理
    df = clean_column_names(df)

    # 删除 GEE 辅助字段
    drop_cols = [c for c in ["system:index", ".geo"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 将常见空值字符串转为 NaN
    df = df.replace(["", " ", "NULL", "null", "None", "none", "NaN", "nan"], np.nan)

    # 类型转换
    df = convert_types(df)

    # 预处理前缺失模式报告
    missing_pattern = infer_missing_pattern(df)

    # 重复记录检查
    duplicate_records = detect_duplicate_records(df)

    # 删除完全重复记录
    subset = [c for c in [ID_COL, YEAR_COL, LON_COL, LAT_COL] if c in df.columns]
    if len(subset) == 4:
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    before_clean = df.copy()

    # 删除关键字段缺失、坐标异常、rad<0 的记录
    df, invalid_records = clean_invalid_records(df)

    # 缺失值填充
    df = fill_missing_values(df)

    # 异常值标记：不删除
    df, outlier_report, outlier_details = detect_outliers(
        df,
        cols=["rad", "median_rad", "cf_cvg"]
    )

    # 经纬度转 UTM 50N 坐标
    df = add_projection_coordinates(df)

    # 特征工程
    df = add_transform_features(df)

    # 标准化 / 归一化
    df = add_scaled_columns(df)

    # 年份转 int，避免图表显示 2013.0
    if YEAR_COL in df.columns:
        df[YEAR_COL] = df[YEAR_COL].astype(int)

    # 构建像元级多年特征
    pixel_features = make_pixel_features(df)

    # 缺失处理前后报告
    missing_report = create_missing_report(before_clean, df)

    reports = {
        "missing_pattern": missing_pattern,
        "missing_report": missing_report,
        "duplicate_records": duplicate_records,
        "invalid_records": invalid_records,
        "outlier_report": outlier_report,
        "outlier_details": outlier_details,
    }

    return df, pixel_features, reports


def export_preprocessing_results(
    df_processed: pd.DataFrame,
    pixel_features: pd.DataFrame,
    reports: dict
):
    """
    输出预处理结果。
    """
    save_csv(df_processed, TABLE_DIR / "07_processed_pixel_year_data.csv")
    save_csv(pixel_features, TABLE_DIR / "08_pixel_multi_year_features.csv")

    save_csv(reports["missing_pattern"], TABLE_DIR / "09_missing_pattern_judgement.csv")
    save_csv(reports["missing_report"], TABLE_DIR / "10_missing_value_report.csv")
    save_csv(reports["duplicate_records"], TABLE_DIR / "11_duplicate_records.csv")
    save_csv(reports["invalid_records"], TABLE_DIR / "12_invalid_records_removed.csv")
    save_csv(reports["outlier_report"], TABLE_DIR / "13_outlier_report_iqr_zscore.csv")
    save_csv(reports["outlier_details"], TABLE_DIR / "14_outlier_details.csv")

    save_excel(
        {
            "processed_pixel_year": df_processed,
            "pixel_features": pixel_features,
            "missing_pattern": reports["missing_pattern"],
            "missing_report": reports["missing_report"],
            "duplicates": reports["duplicate_records"],
            "invalid_records": reports["invalid_records"],
            "outlier_report": reports["outlier_report"],
            "outlier_details": reports["outlier_details"],
        },
        TABLE_DIR / "preprocessing_summary.xlsx"
    )


# ============================================================
# 4. EDA 绘图函数
# ============================================================

def plot_histogram(df: pd.DataFrame, col: str, title: str, xlabel: str, filename: str):
    if col not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df[col].dropna(), bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("频数")
    save_fig(filename)


def plot_box_by_year(df: pd.DataFrame, col: str, title: str, ylabel: str, filename: str):
    if col not in df.columns or YEAR_COL not in df.columns:
        return

    years = sorted(df[YEAR_COL].dropna().unique())
    data = [df.loc[df[YEAR_COL] == y, col].dropna() for y in years]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=years, showmeans=True, showfliers=False)
    plt.title(title)
    plt.xlabel("年份")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    save_fig(filename)


def plot_box_by_district(df: pd.DataFrame, col: str, title: str, ylabel: str, filename: str):
    if col not in df.columns or ID_COL not in df.columns:
        return

    districts = sorted(df[ID_COL].dropna().unique())
    data = [df.loc[df[ID_COL] == d, col].dropna() for d in districts]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=districts, showmeans=True, showfliers=False)
    plt.title(title)
    plt.xlabel("区县")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    save_fig(filename)


def plot_line_mean_by_district(df: pd.DataFrame, col: str, title: str, ylabel: str, filename: str):
    if col not in df.columns or ID_COL not in df.columns or YEAR_COL not in df.columns:
        return

    grouped = (
        df.groupby([ID_COL, YEAR_COL])[col]
        .mean()
        .reset_index()
        .sort_values([ID_COL, YEAR_COL])
    )

    plt.figure(figsize=(10, 6))

    for district, g in grouped.groupby(ID_COL):
        plt.plot(g[YEAR_COL], g[col], marker="o", label=district)

    plt.title(title)
    plt.xlabel("年份")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(filename)


def plot_spatial_year(
    df: pd.DataFrame,
    year: int,
    value_col: str,
    title: str,
    filename: str,
    sample_max: int = 50000
):
    """
    绘制某一年像元空间散点图。
    使用经纬度进行可视化，不用于精确距离计算。
    """
    required = [LON_COL, LAT_COL, YEAR_COL, value_col]
    if not all(c in df.columns for c in required):
        return

    sub = df.loc[df[YEAR_COL] == year, [LON_COL, LAT_COL, value_col]].dropna().copy()

    if sub.empty:
        return

    if len(sub) > sample_max:
        sub = sub.sample(sample_max, random_state=42)

    plt.figure(figsize=(8, 7))
    sc = plt.scatter(
        sub[LON_COL],
        sub[LAT_COL],
        c=sub[value_col],
        s=4,
        alpha=0.8
    )
    plt.colorbar(sc, label=value_col)
    plt.title(title)
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.grid(alpha=0.2)
    save_fig(filename)


def plot_spatial_pixel_feature(
    pixel_features: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    sample_max: int = 50000
):
    """
    绘制像元级多年特征空间分布图。
    """
    required = [LON_COL, LAT_COL, value_col]
    if not all(c in pixel_features.columns for c in required):
        return

    sub = pixel_features[[LON_COL, LAT_COL, value_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if sub.empty:
        return

    if len(sub) > sample_max:
        sub = sub.sample(sample_max, random_state=42)

    plt.figure(figsize=(8, 7))
    sc = plt.scatter(
        sub[LON_COL],
        sub[LAT_COL],
        c=sub[value_col],
        s=4,
        alpha=0.8
    )
    plt.colorbar(sc, label=value_col)
    plt.title(title)
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.grid(alpha=0.2)
    save_fig(filename)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    sample_max: int = 30000
):
    if x_col not in df.columns or y_col not in df.columns:
        return

    sub = df[[x_col, y_col, ID_COL]].replace([np.inf, -np.inf], np.nan).dropna()

    if sub.empty:
        return

    if len(sub) > sample_max:
        sub = sub.sample(sample_max, random_state=42)

    plt.figure(figsize=(8, 6))

    if ID_COL in sub.columns:
        for district, g in sub.groupby(ID_COL):
            plt.scatter(g[x_col], g[y_col], s=8, alpha=0.5, label=district)
        plt.legend()
    else:
        plt.scatter(sub[x_col], sub[y_col], s=8, alpha=0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    save_fig(filename)


def plot_correlation_matrix(df: pd.DataFrame, cols: list[str], filename: str) -> pd.DataFrame:
    """
    相关矩阵。
    """
    cols = [c for c in cols if c in df.columns]

    if len(cols) < 2:
        return pd.DataFrame()

    corr = df[cols].replace([np.inf, -np.inf], np.nan).corr(method="pearson")

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("夜光像元指标相关矩阵")
    save_fig(filename)

    corr_out = corr.reset_index().rename(columns={"index": "字段名"})
    save_csv(corr_out, TABLE_DIR / "15_correlation_matrix.csv")

    return corr


def plot_record_count_heatmap(df: pd.DataFrame):
    """
    区县-年份样本数量热力图，用于检查数据完整性。
    """
    if ID_COL not in df.columns or YEAR_COL not in df.columns:
        return

    count_table = df.pivot_table(
        index=ID_COL,
        columns=YEAR_COL,
        values="rad",
        aggfunc="count"
    )

    plt.figure(figsize=(10, 5))
    im = plt.imshow(count_table.values)
    plt.colorbar(im, label="记录数")

    plt.xticks(range(len(count_table.columns)), count_table.columns, rotation=45)
    plt.yticks(range(len(count_table.index)), count_table.index)

    for i in range(count_table.shape[0]):
        for j in range(count_table.shape[1]):
            plt.text(j, i, int(count_table.values[i, j]), ha="center", va="center", fontsize=8)

    plt.title("区县—年份像元样本数量矩阵")
    plt.xlabel("年份")
    plt.ylabel("区县")
    save_fig("record_count_heatmap_district_year.png")

    save_csv(count_table.reset_index(), TABLE_DIR / "16_record_count_district_year_matrix.csv")


def plot_pca_pixel_features(pixel_features: pd.DataFrame) -> pd.DataFrame:
    """
    PCA 二维可视化。
    """
    feature_cols = [
        "mean_rad_2013_2024",
        "std_rad_2013_2024",
        "range_rad_2013_2024",
        "cv_rad_2013_2024",
        "change_2024_minus_2013",
        "growth_rate_2024_vs_2013",
        "trend_slope_rad_per_year",
    ]

    feature_cols = [c for c in feature_cols if c in pixel_features.columns]

    if len(feature_cols) < 2:
        return pd.DataFrame()

    X = pixel_features[feature_cols].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pixel_features[["pixel_id", ID_COL, LON_COL, LAT_COL]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(9, 7))

    for district, g in pca_df.groupby(ID_COL):
        plt.scatter(g["PC1"], g["PC2"], s=8, alpha=0.5, label=district)

    plt.title(f"PCA 二维可视化：PC1={explained[0]:.2%}, PC2={explained[1]:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig("pca_pixel_features_2d.png")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_loading", "PC2_loading"],
        index=feature_cols
    ).reset_index().rename(columns={"index": "字段名"})

    explained_df = pd.DataFrame({
        "主成分": ["PC1", "PC2"],
        "解释方差比例": explained,
    })

    save_csv(pca_df, TABLE_DIR / "17_pca_scores_pixel_features.csv")
    save_csv(loadings, TABLE_DIR / "18_pca_loadings_pixel_features.csv")
    save_csv(explained_df, TABLE_DIR / "19_pca_explained_variance.csv")

    return pca_df


def compute_morans_i_knn(
    df: pd.DataFrame,
    year: int,
    value_col: str = "rad",
    k: int = 8,
    sample_max: int = 5000
) -> pd.DataFrame:
    """
    近似 Global Moran's I。
    使用 KNN 邻接，避免完整距离矩阵过大。

    注意：
    - 这里是 EDA 阶段的近似空间自相关指标；
    - 正式空间统计可在 GeoDa、ArcGIS、PySAL 中进一步验证。
    """
    coord_cols = None

    if "x_utm50n_m" in df.columns and "y_utm50n_m" in df.columns:
        coord_cols = ["x_utm50n_m", "y_utm50n_m"]
    elif LON_COL in df.columns and LAT_COL in df.columns:
        coord_cols = [LON_COL, LAT_COL]

    if coord_cols is None:
        return pd.DataFrame()

    sub = df.loc[df[YEAR_COL] == year, coord_cols + [value_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(sub) <= k + 1:
        return pd.DataFrame()

    if len(sub) > sample_max:
        sub = sub.sample(sample_max, random_state=42)

    coords = sub[coord_cols].values
    values = sub[value_col].values.astype(float)

    x = values - values.mean()
    denominator = np.sum(x ** 2)

    if denominator == 0:
        moran_i = np.nan
    else:
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

        # 去掉自身，即第一列
        neighbor_indices = indices[:, 1:]

        numerator = 0.0
        s0 = 0.0

        for i in range(len(values)):
            for j in neighbor_indices[i]:
                numerator += x[i] * x[j]
                s0 += 1.0

        n = len(values)
        moran_i = (n / s0) * (numerator / denominator)

    out = pd.DataFrame([{
        "year": year,
        "value_col": value_col,
        "k_neighbors": k,
        "sample_size": len(sub),
        "moran_i_approx": moran_i,
        "coordinate_used": ",".join(coord_cols),
    }])

    return out


def run_eda(df: pd.DataFrame, pixel_features: pd.DataFrame) -> dict:
    """
    执行完整 EDA。
    """
    # 单变量分布
    plot_histogram(
        df,
        "rad",
        "像元夜光辐亮度分布",
        "rad，nW/cm²/sr",
        "hist_rad.png"
    )

    plot_histogram(
        df,
        "log1p_rad",
        "log1p(rad) 分布",
        "log1p(rad)",
        "hist_log1p_rad.png"
    )

    plot_histogram(
        df,
        "cf_cvg",
        "有效观测次数 cf_cvg 分布",
        "cf_cvg，次",
        "hist_cf_cvg.png"
    )

    # 箱线图
    plot_box_by_year(
        df,
        "rad",
        "不同年份像元夜光强度箱线图",
        "rad，nW/cm²/sr",
        "box_rad_by_year.png"
    )

    plot_box_by_district(
        df,
        "rad",
        "不同区县像元夜光强度箱线图",
        "rad，nW/cm²/sr",
        "box_rad_by_district.png"
    )

    # 时间序列
    plot_line_mean_by_district(
        df,
        "rad",
        "厦门六区平均像元夜光强度变化：2013—2024",
        "平均 rad，nW/cm²/sr",
        "line_mean_rad_by_district.png"
    )

    plot_line_mean_by_district(
        df,
        "cf_cvg",
        "厦门六区平均有效观测次数变化：2013—2024",
        "平均 cf_cvg，次",
        "line_cf_cvg_by_district.png"
    )

    # 空间分布图
    for y in [2013, 2018, 2024]:
        plot_spatial_year(
            df,
            y,
            "rad",
            f"厦门市 {y} 年 VIIRS 像元夜光空间分布",
            f"spatial_rad_{y}.png"
        )

    plot_spatial_pixel_feature(
        pixel_features,
        "mean_rad_2013_2024",
        "厦门市 2013—2024 多年平均夜光空间分布",
        "spatial_mean_rad_2013_2024.png"
    )

    plot_spatial_pixel_feature(
        pixel_features,
        "change_2024_minus_2013",
        "厦门市 2024 相对 2013 夜光变化空间分布",
        "spatial_change_2024_minus_2013.png"
    )

    plot_spatial_pixel_feature(
        pixel_features,
        "trend_slope_rad_per_year",
        "厦门市 2013—2024 夜光趋势斜率空间分布",
        "spatial_trend_slope_rad_per_year.png"
    )

    # 双变量散点图
    plot_scatter(
        df,
        "median_rad",
        "rad",
        "median_rad 与 rad 的关系",
        "median_rad，nW/cm²/sr",
        "rad，nW/cm²/sr",
        "scatter_median_rad_vs_rad.png"
    )

    plot_scatter(
        df,
        "cf_cvg",
        "rad",
        "cf_cvg 与 rad 的关系",
        "cf_cvg，次",
        "rad，nW/cm²/sr",
        "scatter_cf_cvg_vs_rad.png"
    )

    plot_scatter(
        pixel_features,
        "mean_rad_2013_2024",
        "change_2024_minus_2013",
        "多年平均夜光与 2013—2024 变化量关系",
        "多年平均 rad，nW/cm²/sr",
        "2024-2013 rad 变化量，nW/cm²/sr",
        "scatter_mean_rad_vs_change.png"
    )

    # 相关矩阵
    corr_cols = [
        "rad",
        "median_rad",
        "cf_cvg",
        "log1p_rad",
        "rad_minus_median",
        "rad_year_zscore",
    ]
    corr_raw = plot_correlation_matrix(
        df,
        corr_cols,
        "correlation_matrix_pixel_year.png"
    )

    corr_feature_cols = [
        "mean_rad_2013_2024",
        "std_rad_2013_2024",
        "range_rad_2013_2024",
        "cv_rad_2013_2024",
        "change_2024_minus_2013",
        "growth_rate_2024_vs_2013",
        "trend_slope_rad_per_year",
    ]
    corr_features = plot_correlation_matrix(
        pixel_features,
        corr_feature_cols,
        "correlation_matrix_pixel_features.png"
    )

    # 样本数量矩阵
    plot_record_count_heatmap(df)

    # PCA
    pca_df = plot_pca_pixel_features(pixel_features)

    # Moran's I
    moran_list = []
    for y in [2013, 2018, 2024]:
        moran_y = compute_morans_i_knn(df, year=y, value_col="rad", k=8, sample_max=5000)
        if not moran_y.empty:
            moran_list.append(moran_y)

    if moran_list:
        moran_result = pd.concat(moran_list, ignore_index=True)
    else:
        moran_result = pd.DataFrame()

    save_csv(moran_result, TABLE_DIR / "20_morans_i_approx.csv")

    return {
        "corr_raw": corr_raw,
        "corr_features": corr_features,
        "pca_df": pca_df,
        "moran_result": moran_result,
    }


# ============================================================
# 5. 自动生成 EDA 发现草稿
# ============================================================

def get_top_corr_pair(corr: pd.DataFrame) -> dict:
    """
    提取相关矩阵中绝对值最大的非对角相关对。
    """
    if corr is None or corr.empty:
        return {}

    cols = list(corr.columns)
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.isna(r):
                continue
            pairs.append({
                "变量1": cols[i],
                "变量2": cols[j],
                "相关系数": r,
                "绝对相关系数": abs(r),
            })

    if not pairs:
        return {}

    pairs = sorted(pairs, key=lambda x: x["绝对相关系数"], reverse=True)
    return pairs[0]


def generate_report_text(
    df: pd.DataFrame,
    pixel_features: pd.DataFrame,
    eda_results: dict
):
    """
    根据实际统计结果自动生成 EDA 初步发现草稿。
    """
    lines = []

    lines.append("# 厦门市 VIIRS 像元样本数据：初步描述、预处理与 EDA 发现草稿\n")
    lines.append("以下内容由脚本自动生成，可作为课程报告“数据介绍、预处理、探索性数据分析”部分的素材。")
    lines.append("正式写入报告前，应结合地图、统计公报和厦门岛内—岛外发展背景进行人工核验。\n")

    # 1. 数据规模
    n_rows = len(df)
    n_cols = df.shape[1]
    n_district = df[ID_COL].nunique() if ID_COL in df.columns else np.nan
    n_year = df[YEAR_COL].nunique() if YEAR_COL in df.columns else np.nan
    n_pixel = pixel_features["pixel_id"].nunique() if "pixel_id" in pixel_features.columns else np.nan
    year_min = df[YEAR_COL].min() if YEAR_COL in df.columns else np.nan
    year_max = df[YEAR_COL].max() if YEAR_COL in df.columns else np.nan

    lines.append("## 1. 数据规模与结构\n")
    lines.append(
        f"- 清洗后的像元—年份样本共有 {n_rows:,} 条记录，包含 {n_cols} 个字段。"
    )
    lines.append(
        f"- 数据覆盖 {n_district} 个区县、{n_year} 个年份，时间范围为 {year_min}—{year_max} 年。"
    )
    lines.append(
        f"- 依据经纬度构造的稳定像元数量约为 {n_pixel:,} 个，可用于后续像元尺度聚类和空间格局识别。"
    )
    lines.append("- 数据同时包含空间属性 lon/lat、时间属性 year 和夜光属性 rad，满足地理空间时序数据分析需求。\n")

    # 2. 夜光强度区县差异
    lines.append("## 2. 区县夜光强度差异\n")

    district_mean = (
        df.groupby(ID_COL)["rad"]
        .mean()
        .sort_values(ascending=False)
    )

    if not district_mean.empty:
        top_district = district_mean.index[0]
        bottom_district = district_mean.index[-1]
        top_value = district_mean.iloc[0]
        bottom_value = district_mean.iloc[-1]

        lines.append(
            f"- 2013—2024 年像元平均夜光强度最高的区县为 {top_district}，"
            f"平均 rad 约为 {top_value:.3f} nW/cm²/sr。"
        )
        lines.append(
            f"- 平均夜光强度最低的区县为 {bottom_district}，"
            f"平均 rad 约为 {bottom_value:.3f} nW/cm²/sr。"
        )
        lines.append(
            "- 这说明厦门市内部存在明显夜光强度差异，可进一步结合岛内核心城区和岛外新城建设背景解释。\n"
        )

    # 3. 时间变化
    lines.append("## 3. 夜光时间变化\n")

    if 2013 in df[YEAR_COL].unique() and 2024 in df[YEAR_COL].unique():
        mean_2013 = df.loc[df[YEAR_COL] == 2013].groupby(ID_COL)["rad"].mean()
        mean_2024 = df.loc[df[YEAR_COL] == 2024].groupby(ID_COL)["rad"].mean()

        change = (mean_2024 - mean_2013).dropna().sort_values(ascending=False)

        if not change.empty:
            grow_district = change.index[0]
            grow_value = change.iloc[0]

            lines.append(
                f"- 从区县平均值看，2013—2024 年夜光增长量最大的区县为 {grow_district}，"
                f"平均 rad 增加约 {grow_value:.3f} nW/cm²/sr。"
            )
            lines.append(
                "- 该现象可作为判断城市建设强度、产业外溢和岛外空间扩展的重要线索。\n"
            )

    # 4. 像元变化
    lines.append("## 4. 像元尺度变化特征\n")

    if "change_2024_minus_2013" in pixel_features.columns:
        pos_ratio = (pixel_features["change_2024_minus_2013"] > 0).mean()
        neg_ratio = (pixel_features["change_2024_minus_2013"] < 0).mean()

        lines.append(
            f"- 在具有 2013 和 2024 年记录的像元中，夜光增强像元比例约为 {pos_ratio:.2%}，"
            f"夜光减弱像元比例约为 {neg_ratio:.2%}。"
        )
        lines.append(
            "- 夜光增强区的空间位置可结合 2024 相对 2013 夜光变化图进一步判断是否集中在岛外新城、产业园区或交通走廊附近。\n"
        )

    # 5. 相关性
    lines.append("## 5. 指标相关性\n")

    top_pair = get_top_corr_pair(eda_results.get("corr_features"))

    if top_pair:
        lines.append(
            f"- 像元多年特征中，{top_pair['变量1']} 与 {top_pair['变量2']} 的相关性较强，"
            f"Pearson 相关系数约为 {top_pair['相关系数']:.3f}。"
        )
        lines.append(
            "- 相关矩阵可用于判断后续聚类建模中是否存在冗余特征，必要时可通过 PCA 或特征筛选降低维度。\n"
        )

    # 6. 空间自相关
    moran_result = eda_results.get("moran_result")
    lines.append("## 6. 空间自相关\n")

    if moran_result is not None and not moran_result.empty:
        for _, row in moran_result.iterrows():
            lines.append(
                f"- {int(row['year'])} 年 rad 的近似 Moran's I 为 {row['moran_i_approx']:.4f}，"
                f"使用 {int(row['k_neighbors'])} 近邻，样本量为 {int(row['sample_size'])}。"
            )

        lines.append(
            "- Moran's I 若为正，通常说明夜光高值或低值像元存在空间集聚；若接近 0，则空间随机性较强。"
        )
        lines.append(
            "- 该结果为近似计算，正式报告中可作为 EDA 辅助指标，不宜过度解释。\n"
        )
    else:
        lines.append("- 未生成 Moran's I 结果，可能是坐标字段或样本量不足。\n")

    # 7. 数据质量问题
    lines.append("## 7. 数据质量与局限性\n")
    lines.append("- VIIRS 夜光数据可能受到港口灯光、道路照明、机场灯光、临时施工灯光和近海船舶灯光影响。")
    lines.append("- 本脚本对异常值采用“标记而不直接删除”的策略，因为极高夜光值可能具有真实地理意义。")
    lines.append("- 当前分析只使用夜光像元样本，若要完成“夜光与经济关联”，还需补充 GDP、人口等统计数据。")
    lines.append("- 当前空间图使用经纬度散点展示，适合 EDA 可视化；涉及距离、面积或空间邻接分析时，应优先使用 UTM 50N 投影坐标。\n")

    out_text = "\n".join(lines)

    out_path = REPORT_DIR / "eda_findings_draft.md"
    out_path.write_text(out_text, encoding="utf-8")

    return out_text


# ============================================================
# 6. 主函数
# ============================================================

def main():
    set_chinese_font()

    print("=" * 70)
    print("厦门 VIIRS 像元样本 CSV：数据初步描述 + 预处理 + EDA")
    print("=" * 70)
    print(f"输入文件：{INPUT_CSV}")
    print(f"输出目录：{OUTPUT_DIR}")

    # 1. 读取数据
    df_raw = read_csv_safely(INPUT_CSV)
    df_raw = clean_column_names(df_raw)

    print(f"原始数据规模：{df_raw.shape[0]:,} 行 × {df_raw.shape[1]} 列")
    print("原始字段：")
    print(df_raw.columns.tolist())

    # 2. 数据初步描述
    export_initial_description(df_raw)
    print("已输出：数据概览、数据字典、基础统计表。")

    # 3. 数据预处理
    df_processed, pixel_features, reports = preprocess_data(df_raw)
    export_preprocessing_results(df_processed, pixel_features, reports)

    print(f"清洗后像元—年份数据规模：{df_processed.shape[0]:,} 行 × {df_processed.shape[1]} 列")
    print(f"像元级多年特征数据规模：{pixel_features.shape[0]:,} 行 × {pixel_features.shape[1]} 列")
    print("已输出：预处理数据、缺失值报告、重复值报告、异常值报告、像元多年特征。")

    # 4. EDA
    eda_results = run_eda(df_processed, pixel_features)
    print("已输出：EDA 图表、相关矩阵、PCA 结果、Moran's I 近似结果。")

    # 5. 自动生成报告文字
    report_text = generate_report_text(df_processed, pixel_features, eda_results)
    print("已输出：EDA 发现草稿。")

    # 6. 运行日志
    run_log = f"""
运行完成。

输入文件：
{INPUT_CSV}

输出目录：
{OUTPUT_DIR}

主要输出：
1. tables/01_data_overview.csv
2. tables/02_data_dictionary.csv
3. tables/03_basic_statistics_numeric.csv
4. tables/07_processed_pixel_year_data.csv
5. tables/08_pixel_multi_year_features.csv
6. tables/13_outlier_report_iqr_zscore.csv
7. tables/15_correlation_matrix.csv
8. tables/17_pca_scores_pixel_features.csv
9. tables/20_morans_i_approx.csv
10. figures/ 下所有 EDA 图片
11. report_text/eda_findings_draft.md

说明：
- 异常值仅标记，不直接删除。
- 空间可视化使用 lon/lat 散点图。
- 如果安装 pyproj，会额外生成 EPSG:32650 的 UTM 坐标字段。
- 当前脚本不做 K-Means / 层次聚类建模，只完成初步描述、预处理和 EDA。
"""
    (OUTPUT_DIR / "run_log.txt").write_text(run_log, encoding="utf-8")

    print("=" * 70)
    print("全部完成。请查看 outputs_pixel_samples 文件夹。")
    print("=" * 70)


if __name__ == "__main__":
    main()