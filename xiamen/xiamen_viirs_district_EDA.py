# -*- coding: utf-8 -*-
"""
用途：
1. 对 xiamen_viirs_district_2013_2024.csv 进行数据初步描述
2. 完成数据预处理：类型转换、缺失值、异常值、数据变换、特征工程、标准化
3. 完成探索性数据分析 EDA：直方图、箱线图、时间序列、散点图、相关矩阵、PCA
4. 输出可用于课程报告的表格、图片和文字发现

适用环境：
Anaconda / Python 3.9+

建议安装：
conda install pandas numpy matplotlib scikit-learn openpyxl

注意：
本脚本针对“区县年度统计 CSV”。
如果后续做 K-Means / 层次聚类，建议再使用像元样本 CSV：
xiamen_viirs_pixel_samples_2013_2024.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# ============================================================
# 0. 参数设置
# ============================================================

warnings.filterwarnings("ignore")

# 项目根目录：默认是脚本所在目录
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# 输入 CSV 路径
# 方案 1：CSV 放在 data 文件夹下
INPUT_CSV = PROJECT_DIR / "data" / "xiamen_viirs_district_2013_2024.csv"

# 方案 2：CSV 与本脚本放在同一目录
if not INPUT_CSV.exists():
    INPUT_CSV = PROJECT_DIR / "xiamen_viirs_district_2013_2024.csv"

# 输出目录
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report_text"

for folder in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 关键字段
ID_COL = "district"
YEAR_COL = "year"

CORE_NUMERIC_COLS = [
    "mean_rad",
    "total_light",
    "lit_area_km2",
    "valid_area_km2",
    "mean_median_rad",
    "cf_cvg_mean",
    "rad_min",
    "rad_max",
]

# 用于 EDA 与预处理的核心指标
ANALYSIS_COLS = [
    "mean_rad",
    "total_light",
    "lit_area_km2",
    "valid_area_km2",
    "mean_median_rad",
    "cf_cvg_mean",
    "rad_max",
]

# 字段说明，用于自动生成数据字典
FIELD_META = {
    "district": {
        "中文含义": "厦门市区县名称",
        "单位": "无",
        "说明": "包括思明区、湖里区、海沧区、集美区、同安区、翔安区",
    },
    "year": {
        "中文含义": "年份",
        "单位": "年",
        "说明": "2013—2024 年",
    },
    "mean_rad": {
        "中文含义": "平均夜光强度",
        "单位": "nW/cm²/sr",
        "说明": "区县范围内 average_masked 波段的平均值",
    },
    "total_light": {
        "中文含义": "夜光总量",
        "单位": "nW/cm²/sr × km²",
        "说明": "夜光辐亮度乘像元面积后在区县内求和",
    },
    "lit_area_km2": {
        "中文含义": "发光面积",
        "单位": "km²",
        "说明": "夜光强度大于设定阈值的像元面积总和",
    },
    "valid_area_km2": {
        "中文含义": "有效统计面积",
        "单位": "km²",
        "说明": "通过质量控制后的有效像元面积总和",
    },
    "mean_median_rad": {
        "中文含义": "中位夜光波段均值",
        "单位": "nW/cm²/sr",
        "说明": "median_masked 波段在区县内的平均值，用于稳健性参考",
    },
    "cf_cvg_mean": {
        "中文含义": "平均有效观测次数",
        "单位": "次",
        "说明": "cf_cvg 波段的区县平均值，用于反映观测质量",
    },
    "rad_min": {
        "中文含义": "最小夜光强度",
        "单位": "nW/cm²/sr",
        "说明": "区县内夜光最小值",
    },
    "rad_max": {
        "中文含义": "最大夜光强度",
        "单位": "nW/cm²/sr",
        "说明": "区县内夜光最大值",
    },
    "source_dataset": {
        "中文含义": "数据源",
        "单位": "无",
        "说明": "NOAA/VIIRS/DNB/ANNUAL_V21 或 V22",
    },
    "scale_m": {
        "中文含义": "统计尺度",
        "单位": "m",
        "说明": "GEE reduceRegions 使用的空间尺度",
    },
    "lit_threshold": {
        "中文含义": "发光面积阈值",
        "单位": "nW/cm²/sr",
        "说明": "rad 大于该阈值的像元计入发光面积",
    },
    "min_cf_cvg": {
        "中文含义": "最小 cf_cvg 阈值",
        "单位": "次",
        "说明": "GEE 中质量控制参数",
    },
}


# ============================================================
# 1. 通用工具函数
# ============================================================

def set_chinese_font():
    """
    设置中文字体。
    Windows 常用 Microsoft YaHei / SimHei。
    如果机器没有这些字体，英文和数字仍可正常显示。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_csv(df: pd.DataFrame, path: Path):
    """
    保存 CSV。utf-8-sig 方便 Excel 直接打开中文不乱码。
    """
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_excel(df_dict: dict, path: Path):
    """
    保存多个 DataFrame 到一个 Excel 文件。
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, data in df_dict.items():
            safe_name = sheet_name[:31]
            data.to_excel(writer, sheet_name=safe_name, index=False)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理字段名：去除首尾空格。
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def save_fig(fig_name: str):
    """
    保存当前 matplotlib 图像。
    """
    plt.tight_layout()
    out_path = FIG_DIR / fig_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def safe_divide(a, b):
    """
    安全除法，避免除以 0。
    """
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


# ============================================================
# 2. 读取数据
# ============================================================

def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{input_csv}\n"
            f"请将 xiamen_viirs_district_2013_2024.csv 放到脚本同目录或 data 文件夹。"
        )

    df = pd.read_csv(input_csv)
    df = clean_column_names(df)

    # 将常见空值统一为 np.nan
    df = df.replace(["", " ", "NULL", "null", "None", "none", "NaN", "nan"], np.nan)

    return df


# ============================================================
# 3. 数据初步描述
# ============================================================

def generate_data_overview(df: pd.DataFrame) -> dict:
    """
    生成基础概览信息。
    """
    overview = {
        "记录数": len(df),
        "字段数": df.shape[1],
        "区县数量": df[ID_COL].nunique() if ID_COL in df.columns else np.nan,
        "年份数量": df[YEAR_COL].nunique() if YEAR_COL in df.columns else np.nan,
        "最小年份": df[YEAR_COL].min() if YEAR_COL in df.columns else np.nan,
        "最大年份": df[YEAR_COL].max() if YEAR_COL in df.columns else np.nan,
    }
    return overview


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成数据字典：
    字段名、类型、含义、单位、取值范围、缺失情况。
    """
    rows = []

    for col in df.columns:
        series = df[col]
        missing_count = series.isna().sum()
        missing_rate = missing_count / len(df) if len(df) > 0 else np.nan

        is_numeric = pd.api.types.is_numeric_dtype(series)

        meta = FIELD_META.get(col, {})
        meaning = meta.get("中文含义", "需人工补充")
        unit = meta.get("单位", "需人工补充")
        note = meta.get("说明", "")

        if is_numeric:
            min_val = series.min()
            max_val = series.max()
            unique_count = series.nunique(dropna=True)
            sample_values = ""
        else:
            min_val = ""
            max_val = ""
            unique_count = series.nunique(dropna=True)
            sample_values = ", ".join(series.dropna().astype(str).unique()[:6])

        rows.append({
            "字段名": col,
            "中文含义": meaning,
            "数据类型": str(series.dtype),
            "单位": unit,
            "非空记录数": series.notna().sum(),
            "缺失记录数": missing_count,
            "缺失率": round(missing_rate, 4),
            "唯一值数量": unique_count,
            "最小值": min_val,
            "最大值": max_val,
            "示例取值": sample_values,
            "说明": note,
        })

    data_dict = pd.DataFrame(rows)
    return data_dict


def basic_statistics(df: pd.DataFrame) -> dict:
    """
    生成基础统计表。
    """
    numeric_df = df.select_dtypes(include=[np.number])

    describe_numeric = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    describe_numeric = describe_numeric.reset_index().rename(columns={"index": "字段名"})

    # 添加缺失率
    missing_rate = df.isna().mean().reset_index()
    missing_rate.columns = ["字段名", "缺失率"]
    describe_numeric = describe_numeric.merge(missing_rate, on="字段名", how="left")

    # 按区县统计
    by_district = pd.DataFrame()
    if ID_COL in df.columns:
        existing_cols = [c for c in ANALYSIS_COLS if c in df.columns]
        if existing_cols:
            by_district = df.groupby(ID_COL)[existing_cols].agg(["count", "mean", "median", "std", "min", "max"])
            by_district.columns = ["_".join(col).strip() for col in by_district.columns.values]
            by_district = by_district.reset_index()

    # 按年份统计
    by_year = pd.DataFrame()
    if YEAR_COL in df.columns:
        existing_cols = [c for c in ANALYSIS_COLS if c in df.columns]
        if existing_cols:
            by_year = df.groupby(YEAR_COL)[existing_cols].agg(["count", "mean", "median", "std", "min", "max"])
            by_year.columns = ["_".join(col).strip() for col in by_year.columns.values]
            by_year = by_year.reset_index()

    # 区县-年份完整性检查
    completeness = pd.DataFrame()
    if ID_COL in df.columns and YEAR_COL in df.columns:
        completeness = (
            df.groupby(ID_COL)[YEAR_COL]
            .agg(["count", "min", "max", "nunique"])
            .reset_index()
            .rename(columns={
                "count": "记录数",
                "min": "最小年份",
                "max": "最大年份",
                "nunique": "年份数量",
            })
        )

    return {
        "numeric_describe": describe_numeric,
        "by_district": by_district,
        "by_year": by_year,
        "district_year_completeness": completeness,
    }


def export_initial_description(df: pd.DataFrame):
    """
    输出数据初步描述结果。
    """
    overview = generate_data_overview(df)
    overview_df = pd.DataFrame([overview])

    data_dict = create_data_dictionary(df)
    stats = basic_statistics(df)

    save_csv(overview_df, TABLE_DIR / "01_data_overview.csv")
    save_csv(data_dict, TABLE_DIR / "02_data_dictionary.csv")
    save_csv(stats["numeric_describe"], TABLE_DIR / "03_basic_statistics_numeric.csv")

    if not stats["by_district"].empty:
        save_csv(stats["by_district"], TABLE_DIR / "04_basic_statistics_by_district.csv")

    if not stats["by_year"].empty:
        save_csv(stats["by_year"], TABLE_DIR / "05_basic_statistics_by_year.csv")

    if not stats["district_year_completeness"].empty:
        save_csv(stats["district_year_completeness"], TABLE_DIR / "06_district_year_completeness.csv")

    save_excel(
        {
            "data_overview": overview_df,
            "data_dictionary": data_dict,
            "numeric_describe": stats["numeric_describe"],
            "by_district": stats["by_district"],
            "by_year": stats["by_year"],
            "district_year_check": stats["district_year_completeness"],
        },
        TABLE_DIR / "initial_description_summary.xlsx"
    )


# ============================================================
# 4. 数据预处理
# ============================================================

def preprocess_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    数据预处理：
    1. 删除无分析价值字段
    2. 类型转换
    3. 重复记录检查
    4. 缺失值处理
    5. 异常值识别
    6. 数据变换
    7. 特征工程
    8. 标准化
    """
    df = df_raw.copy()

    # 4.1 删除 GEE 导出的无分析价值字段
    drop_cols = [c for c in ["system:index", ".geo"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 4.2 类型转换
    if YEAR_COL in df.columns:
        df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")

    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str).str.strip()

    for col in CORE_NUMERIC_COLS + ["lit_threshold", "min_cf_cvg", "scale_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4.3 排序
    if ID_COL in df.columns and YEAR_COL in df.columns:
        df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

    # 4.4 重复值检查
    duplicate_report = pd.DataFrame()
    if ID_COL in df.columns and YEAR_COL in df.columns:
        duplicate_mask = df.duplicated(subset=[ID_COL, YEAR_COL], keep=False)
        duplicate_report = df.loc[duplicate_mask].copy()
        # 如果存在重复，保留第一条
        df = df.drop_duplicates(subset=[ID_COL, YEAR_COL], keep="first").reset_index(drop=True)

    # 4.5 缺失值报告
    missing_before = pd.DataFrame({
        "字段名": df.columns,
        "缺失数_处理前": df.isna().sum().values,
        "缺失率_处理前": df.isna().mean().values,
    })

    # 对核心数值字段按区县进行时间插值
    numeric_existing = [c for c in CORE_NUMERIC_COLS if c in df.columns]

    if ID_COL in df.columns and YEAR_COL in df.columns and numeric_existing:
        df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

        for c in numeric_existing:
            df[c] = (
                df.groupby(ID_COL)[c]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )

    # 若仍有数值缺失，用中位数填充
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_all:
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols_all] = imputer.fit_transform(df[numeric_cols_all])

    # 非数值缺失，用 Unknown 填充
    object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in object_cols:
        df[c] = df[c].fillna("Unknown")

    missing_after = pd.DataFrame({
        "字段名": df.columns,
        "缺失数_处理后": df.isna().sum().values,
        "缺失率_处理后": df.isna().mean().values,
    })

    missing_report = missing_before.merge(missing_after, on="字段名", how="outer")

    # 4.6 异常值识别：IQR + Z-score
    # 检查关键字段是否仍然存在
    print("预处理后、异常值检测前字段：")
    print(df.columns.tolist())

    if ID_COL not in df.columns:
        raise KeyError(f"关键字段 {ID_COL} 在预处理过程中丢失，请检查 groupby 或字段名。")

    if YEAR_COL not in df.columns:
        raise KeyError(f"关键字段 {YEAR_COL} 在预处理过程中丢失，请检查年份字段。")
    outlier_report = detect_outliers(df, numeric_existing)

    # 4.7 数据变换与特征工程
    df = feature_engineering(df)

    # 4.8 标准化
    df = add_scaled_features(df)

    # 将年份重新转为整数，避免图表显示为 2013.0
    if YEAR_COL in df.columns:
        df[YEAR_COL] = df[YEAR_COL].round().astype(int)
    reports = {
        "duplicate_report": duplicate_report,
        "missing_report": missing_report,
        "outlier_report": outlier_report,
    }

    return df, reports


def detect_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    异常值识别：
    - IQR 法
    - 3σ / Z-score 法

    注意：
    夜光高值可能代表真实港口、机场、商业中心等强夜光区域。
    因此本脚本默认只标记异常，不直接删除。
    """
    rows = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        x = df[col].dropna()
        if len(x) < 4:
            continue

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mean_val = x.mean()
        std_val = x.std()

        if std_val == 0 or pd.isna(std_val):
            z_scores = pd.Series(0, index=df.index)
        else:
            z_scores = (df[col] - mean_val) / std_val

        iqr_flag = (df[col] < lower) | (df[col] > upper)
        z_flag = z_scores.abs() > 3

        flag_mask = (iqr_flag | z_flag).fillna(False)

        base_cols = [x for x in [ID_COL, YEAR_COL, col] if x in df.columns]

        flagged = df.loc[flag_mask, base_cols].copy()
        if flagged.empty:
            continue

        flagged = flagged.rename(columns={col: "异常值"})
        flagged["字段名"] = col
        flagged["IQR下界"] = lower
        flagged["IQR上界"] = upper
        flagged["是否IQR异常"] = iqr_flag.loc[flagged.index].values
        flagged["Z_score"] = z_scores.loc[flagged.index].values
        flagged["是否3σ异常"] = z_flag.loc[flagged.index].values

        rows.append(flagged)

    if rows:
        outlier_report = pd.concat(rows, ignore_index=True)
    else:
        outlier_report = pd.DataFrame(columns=[
            ID_COL, YEAR_COL, "字段名", "异常值",
            "IQR下界", "IQR上界", "是否IQR异常", "Z_score", "是否3σ异常"
        ])

    return outlier_report


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：
    - 发光面积占比
    - 单位有效面积夜光总量
    - 夜光极差
    - 平均值与中位波段差异
    - 对数变换
    - 同比变化
    - 相对 2013 年变化
    """
    df = df.copy()

    # 发光面积占比
    if "lit_area_km2" in df.columns and "valid_area_km2" in df.columns:
        df["lit_area_ratio"] = safe_divide(df["lit_area_km2"], df["valid_area_km2"])

    # 单位有效面积夜光总量
    if "total_light" in df.columns and "valid_area_km2" in df.columns:
        df["total_light_per_km2"] = safe_divide(df["total_light"], df["valid_area_km2"])

    # 夜光极差
    if "rad_max" in df.columns and "rad_min" in df.columns:
        df["rad_range"] = df["rad_max"] - df["rad_min"]

    # mean_rad 与 median_rad 的差异
    if "mean_rad" in df.columns and "mean_median_rad" in df.columns:
        df["mean_minus_median_rad"] = df["mean_rad"] - df["mean_median_rad"]

    # 对数变换：减弱偏态分布影响
    for c in ["mean_rad", "total_light", "lit_area_km2", "valid_area_km2", "rad_max"]:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))

    # 年度同比变化和同比增长率
    if ID_COL in df.columns and YEAR_COL in df.columns:
        df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

        yoy_cols = ["mean_rad", "total_light", "lit_area_km2", "rad_max"]
        for c in yoy_cols:
            if c in df.columns:
                df[f"{c}_yoy_change"] = df.groupby(ID_COL)[c].diff()
                df[f"{c}_yoy_growth_rate"] = df.groupby(ID_COL)[c].pct_change()
                df[f"{c}_yoy_growth_rate"] = df[f"{c}_yoy_growth_rate"].replace([np.inf, -np.inf], np.nan)

        # 相对 2013 年变化
        base_year = 2013
        for c in yoy_cols:
            if c in df.columns:
                base = (
                    df.loc[df[YEAR_COL] == base_year, [ID_COL, c]]
                    .rename(columns={c: f"{c}_{base_year}"})
                )
                df = df.merge(base, on=ID_COL, how="left")
                df[f"{c}_change_vs_{base_year}"] = df[c] - df[f"{c}_{base_year}"]
                df[f"{c}_growth_vs_{base_year}"] = safe_divide(
                    df[f"{c}_change_vs_{base_year}"],
                    df[f"{c}_{base_year}"]
                )

    # 清理 inf
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加 Z-score 和 Min-Max 标准化字段。
    不覆盖原始字段。
    """
    df = df.copy()

    scale_cols = [
        "mean_rad",
        "total_light",
        "lit_area_km2",
        "valid_area_km2",
        "mean_median_rad",
        "cf_cvg_mean",
        "rad_max",
        "lit_area_ratio",
        "total_light_per_km2",
        "rad_range",
    ]

    scale_cols = [c for c in scale_cols if c in df.columns]

    if not scale_cols:
        return df

    temp = df[scale_cols].copy()
    temp = temp.replace([np.inf, -np.inf], np.nan)

    # 用中位数填充标准化输入中的缺失
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


def export_preprocessing_results(df_processed: pd.DataFrame, reports: dict):
    """
    输出预处理结果。
    """
    save_csv(df_processed, TABLE_DIR / "07_processed_district_data.csv")

    save_csv(reports["missing_report"], TABLE_DIR / "08_missing_value_report.csv")
    save_csv(reports["outlier_report"], TABLE_DIR / "09_outlier_report_iqr_zscore.csv")

    if not reports["duplicate_report"].empty:
        save_csv(reports["duplicate_report"], TABLE_DIR / "10_duplicate_records.csv")

    save_excel(
        {
            "processed_data": df_processed,
            "missing_report": reports["missing_report"],
            "outlier_report": reports["outlier_report"],
            "duplicate_records": reports["duplicate_report"],
        },
        TABLE_DIR / "preprocessing_summary.xlsx"
    )


# ============================================================
# 5. EDA 绘图
# ============================================================

def plot_histogram(df: pd.DataFrame, col: str, title: str, xlabel: str):
    if col not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df[col].dropna(), bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("频数")
    save_fig(f"hist_{col}.png")


def plot_box_by_district(df: pd.DataFrame, col: str, title: str, ylabel: str):
    if col not in df.columns or ID_COL not in df.columns:
        return

    districts = sorted(df[ID_COL].dropna().unique())
    data = [df.loc[df[ID_COL] == d, col].dropna() for d in districts]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=districts, showmeans=True)
    plt.title(title)
    plt.xlabel("区县")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    save_fig(f"box_{col}_by_district.png")


def plot_line_by_district(df: pd.DataFrame, col: str, title: str, ylabel: str):
    if col not in df.columns or ID_COL not in df.columns or YEAR_COL not in df.columns:
        return

    plt.figure(figsize=(9, 5))

    for district, g in df.groupby(ID_COL):
        g = g.sort_values(YEAR_COL)
        plt.plot(g[YEAR_COL], g[col], marker="o", label=district)

    plt.title(title)
    plt.xlabel("年份")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(f"line_{col}_by_district.png")


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str):
    if x_col not in df.columns or y_col not in df.columns:
        return

    plt.figure(figsize=(7, 5))

    if ID_COL in df.columns:
        for district, g in df.groupby(ID_COL):
            plt.scatter(g[x_col], g[y_col], label=district, alpha=0.8)
        plt.legend()
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    save_fig(f"scatter_{x_col}_vs_{y_col}.png")


def plot_correlation_matrix(df: pd.DataFrame, cols: list[str]):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()

    corr = df[cols].corr(method="pearson")

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("夜光指标相关系数矩阵")
    save_fig("correlation_matrix.png")

    save_csv(corr.reset_index().rename(columns={"index": "字段名"}), TABLE_DIR / "11_correlation_matrix.csv")

    return corr


def plot_pca(df: pd.DataFrame, feature_cols: list[str]):
    """
    多变量分析：PCA 二维可视化。
    """
    feature_cols = [c for c in feature_cols if c in df.columns]

    if len(feature_cols) < 2:
        return pd.DataFrame()

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pca_df = df[[ID_COL, YEAR_COL]].copy() if ID_COL in df.columns and YEAR_COL in df.columns else pd.DataFrame()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))

    if ID_COL in pca_df.columns:
        for district, g in pca_df.groupby(ID_COL):
            plt.scatter(g["PC1"], g["PC2"], label=district, alpha=0.8)
        plt.legend()
    else:
        plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.8)

    plt.title(f"PCA 二维可视化：PC1={explained[0]:.2%}, PC2={explained[1]:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    save_fig("pca_2d_scatter.png")

    # PCA 载荷
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_loading", "PC2_loading"],
        index=feature_cols
    ).reset_index().rename(columns={"index": "字段名"})

    explained_df = pd.DataFrame({
        "主成分": ["PC1", "PC2"],
        "解释方差比例": explained,
    })

    save_csv(pca_df, TABLE_DIR / "12_pca_scores.csv")
    save_csv(loadings, TABLE_DIR / "13_pca_loadings.csv")
    save_csv(explained_df, TABLE_DIR / "14_pca_explained_variance.csv")

    return pca_df


def export_pivot_tables(df: pd.DataFrame):
    """
    输出区县-年份透视表，方便报告制表。
    """
    if ID_COL not in df.columns or YEAR_COL not in df.columns:
        return

    for col in ["mean_rad", "total_light", "lit_area_km2", "lit_area_ratio"]:
        if col in df.columns:
            pivot = df.pivot_table(index=YEAR_COL, columns=ID_COL, values=col, aggfunc="mean")
            pivot = pivot.reset_index()
            save_csv(pivot, TABLE_DIR / f"pivot_{col}_year_by_district.csv")


def run_eda(df: pd.DataFrame):
    """
    执行 EDA。
    """
    # 单变量分析：直方图
    plot_histogram(
        df,
        "mean_rad",
        "厦门六区平均夜光强度分布",
        "平均夜光强度 nW/cm²/sr"
    )

    plot_histogram(
        df,
        "total_light",
        "厦门六区夜光总量分布",
        "夜光总量 nW/cm²/sr × km²"
    )

    plot_histogram(
        df,
        "lit_area_km2",
        "厦门六区发光面积分布",
        "发光面积 km²"
    )

    # 单变量 + 分组：箱线图
    plot_box_by_district(
        df,
        "mean_rad",
        "不同区县平均夜光强度箱线图",
        "平均夜光强度 nW/cm²/sr"
    )

    plot_box_by_district(
        df,
        "total_light",
        "不同区县夜光总量箱线图",
        "夜光总量 nW/cm²/sr × km²"
    )

    plot_box_by_district(
        df,
        "lit_area_km2",
        "不同区县发光面积箱线图",
        "发光面积 km²"
    )

    # 时间序列 EDA
    plot_line_by_district(
        df,
        "mean_rad",
        "厦门六区平均夜光强度时间变化：2013—2024",
        "平均夜光强度 nW/cm²/sr"
    )

    plot_line_by_district(
        df,
        "total_light",
        "厦门六区夜光总量时间变化：2013—2024",
        "夜光总量 nW/cm²/sr × km²"
    )

    plot_line_by_district(
        df,
        "lit_area_km2",
        "厦门六区发光面积时间变化：2013—2024",
        "发光面积 km²"
    )

    if "mean_rad_growth_vs_2013" in df.columns:
        plot_line_by_district(
            df,
            "mean_rad_growth_vs_2013",
            "厦门六区平均夜光强度相对 2013 年增长率",
            "相对 2013 年增长率"
        )

    # 双变量分析：散点图
    plot_scatter(
        df,
        "mean_rad",
        "total_light",
        "平均夜光强度与夜光总量关系",
        "平均夜光强度 nW/cm²/sr",
        "夜光总量 nW/cm²/sr × km²"
    )

    plot_scatter(
        df,
        "lit_area_km2",
        "total_light",
        "发光面积与夜光总量关系",
        "发光面积 km²",
        "夜光总量 nW/cm²/sr × km²"
    )

    if "valid_area_km2" in df.columns:
        plot_scatter(
            df,
            "valid_area_km2",
            "total_light",
            "有效统计面积与夜光总量关系",
            "有效统计面积 km²",
            "夜光总量 nW/cm²/sr × km²"
        )

    # 相关矩阵
    corr_cols = [
        "mean_rad",
        "total_light",
        "lit_area_km2",
        "valid_area_km2",
        "mean_median_rad",
        "cf_cvg_mean",
        "rad_max",
        "lit_area_ratio",
        "total_light_per_km2",
        "rad_range",
    ]
    corr = plot_correlation_matrix(df, corr_cols)

    # 多变量分析：PCA
    pca_features = [
        "mean_rad",
        "total_light",
        "lit_area_km2",
        "valid_area_km2",
        "cf_cvg_mean",
        "rad_max",
        "lit_area_ratio",
        "total_light_per_km2",
        "rad_range",
    ]
    pca_df = plot_pca(df, pca_features)

    # 透视表
    export_pivot_tables(df)

    return corr, pca_df


# ============================================================
# 6. 趋势分析与自动生成报告文字
# ============================================================

def compute_district_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算各区县 2013—2024 趋势：
    - 首年值
    - 末年值
    - 总变化量
    - 总增长率
    - 线性趋势斜率
    """
    if ID_COL not in df.columns or YEAR_COL not in df.columns:
        return pd.DataFrame()

    indicators = ["mean_rad", "total_light", "lit_area_km2"]

    rows = []

    for district, g in df.groupby(ID_COL):
        g = g.sort_values(YEAR_COL)

        for col in indicators:
            if col not in g.columns:
                continue

            valid = g[[YEAR_COL, col]].dropna()
            if len(valid) < 2:
                continue

            start_year = int(valid[YEAR_COL].min())
            end_year = int(valid[YEAR_COL].max())

            start_val = valid.loc[valid[YEAR_COL] == start_year, col].iloc[0]
            end_val = valid.loc[valid[YEAR_COL] == end_year, col].iloc[0]

            change = end_val - start_val
            growth_rate = np.nan if start_val == 0 else change / start_val

            # 线性趋势斜率
            slope = np.polyfit(valid[YEAR_COL].astype(float), valid[col].astype(float), 1)[0]

            rows.append({
                "district": district,
                "indicator": col,
                "start_year": start_year,
                "end_year": end_year,
                "start_value": start_val,
                "end_value": end_val,
                "change": change,
                "growth_rate": growth_rate,
                "trend_slope_per_year": slope,
            })

    trend_df = pd.DataFrame(rows)
    save_csv(trend_df, TABLE_DIR / "15_district_trend_summary.csv")
    return trend_df


def generate_report_findings(df: pd.DataFrame, trend_df: pd.DataFrame, corr: pd.DataFrame):
    """
    自动生成可放进报告草稿的 EDA 发现。
    该文本不是最终结论，需要你结合图表和地图人工核验。
    """
    lines = []

    lines.append("# EDA 初步发现草稿\n")
    lines.append("以下文字由脚本根据区县年度夜光统计表自动生成，可作为报告“探索性数据分析”部分的初稿。")
    lines.append("正式报告中应结合空间图、统计公报和实际城市发展背景进一步解释。\n")

    # 数据规模
    n_rows = len(df)
    n_districts = df[ID_COL].nunique() if ID_COL in df.columns else np.nan
    n_years = df[YEAR_COL].nunique() if YEAR_COL in df.columns else np.nan

    lines.append("## 1. 数据规模与结构\n")
    lines.append(f"- 本区县统计表共包含 {n_rows} 条记录，涉及 {n_districts} 个区县和 {n_years} 个年份。")
    lines.append("- 每条记录表示某一区县在某一年度的 VIIRS 夜光统计指标。")
    lines.append("- 该表适用于区县尺度趋势分析、相关分析和后续 GDP 回归；若进行像元级聚类，应使用像元样本 CSV。\n")

    # 缺失情况
    missing_total = int(df.isna().sum().sum())
    lines.append("## 2. 缺失值情况\n")
    lines.append(f"- 预处理后数据表剩余缺失值总数为 {missing_total}。")
    lines.append("- 对核心数值字段，脚本采用按区县时间序列线性插值和中位数填充的方式处理缺失值。")
    lines.append("- 若正式报告中发现 GDP 或人口缺失，GDP 不建议随意插值，应优先查找统计年鉴或统计公报补齐。\n")

    # 夜光强度最高区县
    if "mean_rad" in df.columns:
        mean_by_district = (
            df.groupby(ID_COL)["mean_rad"]
            .mean()
            .sort_values(ascending=False)
        )

        top_d = mean_by_district.index[0]
        top_v = mean_by_district.iloc[0]

        lines.append("## 3. 平均夜光强度差异\n")
        lines.append(f"- 从 2013—2024 年平均值看，{top_d} 的平均夜光强度最高，均值约为 {top_v:.3f} nW/cm²/sr。")
        lines.append("- 这说明不同区县之间存在明显夜光强度差异，后续可结合岛内—岛外空间结构解释。\n")

    # 增长最快区县
    if not trend_df.empty:
        mean_rad_trend = trend_df[trend_df["indicator"] == "mean_rad"].copy()
        if not mean_rad_trend.empty:
            mean_rad_trend = mean_rad_trend.sort_values("growth_rate", ascending=False)
            grow_d = mean_rad_trend.iloc[0]["district"]
            grow_rate = mean_rad_trend.iloc[0]["growth_rate"]

            lines.append("## 4. 夜光增长差异\n")
            lines.append(f"- 以平均夜光强度相对首年变化衡量，{grow_d} 的增长率最高，约为 {grow_rate:.2%}。")
            lines.append("- 该结果可作为“岛外新城建设、产业外溢或交通基础设施扩展是否推动夜光增长”的数据依据。\n")

    # 相关性
    if corr is not None and not corr.empty:
        corr_pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_pairs.append({
                    "变量1": cols[i],
                    "变量2": cols[j],
                    "相关系数": corr.iloc[i, j],
                    "绝对相关系数": abs(corr.iloc[i, j]),
                })

        corr_pairs = pd.DataFrame(corr_pairs).sort_values("绝对相关系数", ascending=False)
        if not corr_pairs.empty:
            r = corr_pairs.iloc[0]
            lines.append("## 5. 指标相关性\n")
            lines.append(
                f"- 在夜光指标中，{r['变量1']} 与 {r['变量2']} 的相关性较强，"
                f"Pearson 相关系数约为 {r['相关系数']:.3f}。"
            )
            lines.append("- 后续如果加入 GDP 和人口字段，应重点检查 total_light、mean_rad、lit_area_km2 与 GDP 的相关性。\n")

    # PCA
    lines.append("## 6. 多变量结构\n")
    lines.append("- PCA 图用于观察不同区县年份样本是否在多维夜光指标上形成分组。")
    lines.append("- 如果同一区县样本在 PCA 图上聚集，说明其夜光特征具有较强稳定性；如果不同年份沿某一方向移动，说明存在明显时间演化趋势。\n")

    # 限制
    lines.append("## 7. 当前数据限制\n")
    lines.append("- 当前 CSV 是区县年度聚合表，记录数较少，不能单独支撑像元级聚类。")
    lines.append("- 当前 CSV 不含有效几何边界，无法单独绘制严格的空间分布地图。空间图应使用 GeoTIFF 或区县边界 Shapefile / GeoJSON。")
    lines.append("- 夜光高值可能受到港口、机场、道路、施工灯光和近海船舶灯光影响，异常值不应简单删除，应结合地图人工核验。\n")

    out_text = "\n".join(lines)

    out_path = REPORT_DIR / "eda_findings_draft.md"
    out_path.write_text(out_text, encoding="utf-8")

    return out_text


# ============================================================
# 7. 主函数
# ============================================================

def main():
    set_chinese_font()

    print("============================================================")
    print("厦门 VIIRS 区县年度夜光数据：初步描述 + 预处理 + EDA")
    print("============================================================")
    print(f"输入文件：{INPUT_CSV}")
    print(f"输出目录：{OUTPUT_DIR}")

    # 1. 读取数据
    df_raw = load_data(INPUT_CSV)
    print(f"原始数据规模：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

    # 2. 数据初步描述
    export_initial_description(df_raw)
    print("已输出：数据概览、数据字典、基础统计表。")

    # 3. 数据预处理
    df_processed, reports = preprocess_data(df_raw)
    export_preprocessing_results(df_processed, reports)
    print("已输出：预处理后数据、缺失值报告、异常值报告。")

    # 4. EDA
    corr, pca_df = run_eda(df_processed)
    print("已输出：EDA 图表和相关矩阵、PCA 结果。")

    # 5. 趋势分析
    trend_df = compute_district_trends(df_processed)
    print("已输出：区县趋势汇总表。")

    # 6. 自动生成报告文字
    report_text = generate_report_findings(df_processed, trend_df, corr)
    print("已输出：EDA 初步发现草稿。")

    # 7. 运行完成提示
    print("============================================================")
    print("运行完成。请查看 outputs 文件夹：")
    print(f"- 表格：{TABLE_DIR}")
    print(f"- 图像：{FIG_DIR}")
    print(f"- 报告文字草稿：{REPORT_DIR}")
    print("============================================================")


if __name__ == "__main__":
    main()