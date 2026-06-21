# -*- coding: utf-8 -*-
"""
用途：
基于厦门市 2013—2024 年区县 VIIRS 夜光统计表 + GDP/人口 Excel，
比较 Linear Regression 与 Ridge Regression 对区县 GDP 的解释能力。

输入优先级：
1. outputs/tables/07_processed_district_data.csv
2. xiamen_viirs_district_2013_2024.csv

经济数据：
经济数据(1).xlsx

主要输出：
1. 合并后的建模数据表
2. Linear Regression 与 Ridge Regression 评价指标
3. 预测结果表
4. 回归系数表
5. 相关矩阵图
6. GDP 与夜光散点图
7. 实际值 vs 预测值图
8. 残差图
9. 模型指标对比图
10. 报告文字草稿

说明：
- 默认采用 2013—2021 年作为训练集，2022—2024 年作为测试集。
- 因变量为 GDP，单位为亿元。
- 默认对 GDP 做 log1p 变换训练，再反变换回亿元计算 R²、RMSE、MAE。
- 自变量主要包括夜光总量、平均夜光强度、发光面积、人口等。
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# 0. 参数设置
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# 优先使用预处理后的区县年度夜光表
VIIRS_CSV_CANDIDATES = [
    PROJECT_DIR / "outputs" / "tables" / "07_processed_district_data.csv",
    PROJECT_DIR / "outputs_district" / "tables" / "07_processed_district_data.csv",
    PROJECT_DIR / "data_processed" / "07_processed_district_data.csv",
    PROJECT_DIR / "xiamen_viirs_district_2013_2024.csv",
    PROJECT_DIR / "data" / "xiamen_viirs_district_2013_2024.csv",
]

ECON_EXCEL_CANDIDATES = [
    PROJECT_DIR / "经济数据.xlsx",
    PROJECT_DIR / "data" / "经济数据.xlsx",
]

OUTPUT_DIR = PROJECT_DIR / "outputs_regression"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report_text"

for folder in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ID_COL = "district"
YEAR_COL = "year"

TRAIN_START_YEAR = 2013
TRAIN_END_YEAR = 2021
TEST_START_YEAR = 2022
TEST_END_YEAR = 2024

# 是否对 GDP 做 log1p 变换。
# 小样本 + GDP 右偏时，建议 True。
USE_LOG_TARGET = True

# 是否把 year 作为模型自变量。
# 默认 False：更强调夜光与人口对 GDP 的解释，而不是简单拟合时间趋势。
INCLUDE_YEAR_FEATURE = False

# 是否把区县作为哑变量加入模型。
# 默认 False：更强调夜光和人口指标。若想控制区县固定差异，可改为 True，但样本较少时要谨慎。
INCLUDE_DISTRICT_DUMMIES = False


# ============================================================
# 1. 基础工具函数
# ============================================================

def set_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def find_existing_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"成功读取 CSV：{path}")
            print(f"编码：{enc}")
            return df
        except Exception:
            pass

    raise ValueError(f"CSV 读取失败，请检查编码：{path}")


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_excel(sheets: dict, path: Path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def save_fig(filename: str):
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_district_name(s):
    if pd.isna(s):
        return np.nan

    s = str(s).strip()
    s = s.replace("厦门市", "")
    s = s.replace(" ", "")
    s = s.replace("\u3000", "")

    return s


def safe_divide(a, b):
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


# ============================================================
# 2. 读取区县夜光 CSV
# ============================================================

def load_viirs_district_data() -> pd.DataFrame:
    csv_path = find_existing_file(VIIRS_CSV_CANDIDATES)

    if csv_path is None:
        raise FileNotFoundError(
            "未找到区县年度夜光 CSV。\n"
            "请确认 outputs/tables/07_processed_district_data.csv 或 "
            "xiamen_viirs_district_2013_2024.csv 位于脚本同目录或 data 文件夹。"
        )

    df = read_csv_safely(csv_path)
    df = clean_columns(df)

    # 兼容字段名
    if ID_COL not in df.columns:
        for cand in ["district_std", "区县", "行政区", "name", "adname"]:
            if cand in df.columns:
                df = df.rename(columns={cand: ID_COL})
                break

    if YEAR_COL not in df.columns:
        for cand in ["年份", "Year", "YEAR"]:
            if cand in df.columns:
                df = df.rename(columns={cand: YEAR_COL})
                break

    if ID_COL not in df.columns or YEAR_COL not in df.columns:
        raise KeyError("夜光 CSV 必须包含 district 和 year 字段。")

    df[ID_COL] = df[ID_COL].apply(clean_district_name)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").round().astype("Int64")

    # 删除 GEE 辅助字段
    df = df.drop(columns=[c for c in ["system:index", ".geo"] if c in df.columns], errors="ignore")

    # 常见数值字段转换
    numeric_candidates = [
        "mean_rad",
        "total_light",
        "lit_area_km2",
        "valid_area_km2",
        "mean_median_rad",
        "cf_cvg_mean",
        "rad_min",
        "rad_max",
        "scale_m",
        "min_cf_cvg",
        "lit_threshold",
        "lit_area_ratio",
        "total_light_per_km2",
        "rad_range",
        "mean_minus_median_rad",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[YEAR_COL].between(2013, 2024)].copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    # 基础特征补充
    df = add_viirs_features(df)

    print(f"夜光数据记录数：{len(df)}")
    print(f"夜光数据区县：{sorted(df[ID_COL].dropna().unique())}")
    print(f"夜光数据年份：{df[YEAR_COL].min()}—{df[YEAR_COL].max()}")

    return df


def add_viirs_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lit_area_ratio" not in df.columns:
        if "lit_area_km2" in df.columns and "valid_area_km2" in df.columns:
            df["lit_area_ratio"] = safe_divide(df["lit_area_km2"], df["valid_area_km2"])

    if "total_light_per_km2" not in df.columns:
        if "total_light" in df.columns and "valid_area_km2" in df.columns:
            df["total_light_per_km2"] = safe_divide(df["total_light"], df["valid_area_km2"])

    if "rad_range" not in df.columns:
        if "rad_max" in df.columns and "rad_min" in df.columns:
            df["rad_range"] = df["rad_max"] - df["rad_min"]

    if "mean_minus_median_rad" not in df.columns:
        if "mean_rad" in df.columns and "mean_median_rad" in df.columns:
            df["mean_minus_median_rad"] = df["mean_rad"] - df["mean_median_rad"]

    # 对数变换特征
    log_cols = [
        "mean_rad",
        "total_light",
        "lit_area_km2",
        "valid_area_km2",
        "mean_median_rad",
        "rad_max",
        "total_light_per_km2",
    ]

    for col in log_cols:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))

    # 相对 2013 年变化，用于分析，不作为默认回归特征
    if "total_light" in df.columns:
        base = df[df[YEAR_COL] == 2013][[ID_COL, "total_light"]].rename(
            columns={"total_light": "total_light_2013"}
        )
        df = df.merge(base, on=ID_COL, how="left")
        df["total_light_change_vs_2013"] = df["total_light"] - df["total_light_2013"]
        df["total_light_growth_vs_2013"] = safe_divide(
            df["total_light_change_vs_2013"],
            df["total_light_2013"]
        )

    if "mean_rad" in df.columns:
        base = df[df[YEAR_COL] == 2013][[ID_COL, "mean_rad"]].rename(
            columns={"mean_rad": "mean_rad_2013"}
        )
        df = df.merge(base, on=ID_COL, how="left")
        df["mean_rad_change_vs_2013"] = df["mean_rad"] - df["mean_rad_2013"]
        df["mean_rad_growth_vs_2013"] = safe_divide(
            df["mean_rad_change_vs_2013"],
            df["mean_rad_2013"]
        )

    return df


# ============================================================
# 3. 读取经济 Excel
# ============================================================

def find_column_by_keywords(columns, keywords):
    for col in columns:
        c = str(col).lower()
        for kw in keywords:
            if kw.lower() in c:
                return col
    return None


def load_economic_data() -> pd.DataFrame:
    excel_path = find_existing_file(ECON_EXCEL_CANDIDATES)

    if excel_path is None:
        raise FileNotFoundError(
            "未找到经济数据 Excel。\n"
            "请将 经济数据(1).xlsx 放到脚本同目录或 data 文件夹。"
        )

    print(f"读取经济数据：{excel_path}")

    xl = pd.ExcelFile(excel_path)
    candidate_frames = []

    for sheet in xl.sheet_names:
        temp = pd.read_excel(excel_path, sheet_name=sheet)
        temp = clean_columns(temp)

        if temp.empty or temp.shape[1] < 3:
            continue

        cols = temp.columns.tolist()

        district_col = find_column_by_keywords(cols, ["district", "区县", "行政区", "地区"])
        year_col = find_column_by_keywords(cols, ["year", "年份"])
        gdp_col = find_column_by_keywords(cols, ["gdp", "生产总值", "地区生产总值"])
        pop_col = find_column_by_keywords(cols, ["population", "常住人口", "人口"])

        if district_col is not None and year_col is not None and gdp_col is not None:
            rename_map = {
                district_col: ID_COL,
                year_col: YEAR_COL,
                gdp_col: "gdp_yi_yuan",
            }

            if pop_col is not None:
                rename_map[pop_col] = "population_wan"

            econ = temp.rename(columns=rename_map).copy()

            keep_cols = [ID_COL, YEAR_COL, "gdp_yi_yuan"]
            if "population_wan" in econ.columns:
                keep_cols.append("population_wan")

            econ = econ[keep_cols].copy()
            econ["source_sheet"] = sheet

            candidate_frames.append(econ)

    if not candidate_frames:
        raise ValueError(
            "没有在 Excel 中找到包含 district/year/GDP 的有效工作表。\n"
            "建议表头使用：district, year, gdp(亿元), population(万人)。"
        )

    econ = pd.concat(candidate_frames, ignore_index=True)

    econ[ID_COL] = econ[ID_COL].apply(clean_district_name)
    econ[YEAR_COL] = pd.to_numeric(econ[YEAR_COL], errors="coerce")
    econ["gdp_yi_yuan"] = pd.to_numeric(econ["gdp_yi_yuan"], errors="coerce")

    if "population_wan" in econ.columns:
        econ["population_wan"] = pd.to_numeric(econ["population_wan"], errors="coerce")
    else:
        econ["population_wan"] = np.nan

    econ = econ.dropna(subset=[ID_COL, YEAR_COL, "gdp_yi_yuan"]).copy()
    econ[YEAR_COL] = econ[YEAR_COL].round().astype(int)

    # 只保留 2013—2024，与夜光数据保持一致；Excel 中 2025 会自动剔除
    econ = econ[econ[YEAR_COL].between(2013, 2024)].copy()

    # 去重
    econ = econ.drop_duplicates(subset=[ID_COL, YEAR_COL], keep="first").reset_index(drop=True)

    # 构造人均 GDP，只作为描述或后续分析，不作为默认回归自变量
    econ["gdp_per_capita_yuan"] = np.where(
        econ["population_wan"].notna() & (econ["population_wan"] > 0),
        econ["gdp_yi_yuan"] * 10000 / econ["population_wan"],
        np.nan
    )

    print(f"经济数据记录数：{len(econ)}")
    print(f"经济数据区县：{sorted(econ[ID_COL].dropna().unique())}")
    print(f"经济数据年份：{econ[YEAR_COL].min()}—{econ[YEAR_COL].max()}")

    return econ


# ============================================================
# 4. 合并数据与建模特征
# ============================================================

def merge_viirs_and_economic(viirs: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    df = viirs.merge(
        econ,
        on=[ID_COL, YEAR_COL],
        how="inner",
        validate="one_to_one"
    )

    if df.empty:
        raise ValueError(
            "夜光数据与经济数据合并后为空。请检查 district 和 year 是否一致。"
        )

    # 经济字段对数变换
    df["log1p_gdp"] = np.log1p(df["gdp_yi_yuan"].clip(lower=0))

    if "population_wan" in df.columns:
        df["log1p_population_wan"] = np.log1p(df["population_wan"].clip(lower=0))

    # 年份索引，可选
    df["year_index"] = df[YEAR_COL] - df[YEAR_COL].min()

    df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

    save_csv(df, TABLE_DIR / "01_merged_regression_dataset.csv")

    print(f"合并后建模数据记录数：{len(df)}")
    print(df[[ID_COL, YEAR_COL, "gdp_yi_yuan"]].head())

    return df


def choose_feature_columns(df: pd.DataFrame) -> list:
    """
    选择回归自变量。
    默认不使用 gdp_per_capita，避免用 GDP 派生变量预测 GDP。
    """
    candidate_features = [
        # 夜光核心指标
        "log1p_total_light",
        "log1p_mean_rad",
        "log1p_lit_area_km2",
        "log1p_valid_area_km2",
        "log1p_mean_median_rad",
        "log1p_rad_max",

        # 夜光结构与质量指标
        "lit_area_ratio",
        "total_light_per_km2",
        "rad_range",
        "mean_minus_median_rad",
        "cf_cvg_mean",

        # 社会经济辅助变量
        "log1p_population_wan",
    ]

    if INCLUDE_YEAR_FEATURE:
        candidate_features.append("year_index")

    feature_cols = []

    for col in candidate_features:
        if col not in df.columns:
            print(f"跳过缺失特征：{col}")
            continue

        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        if s.notna().sum() == 0:
            print(f"跳过全为空特征：{col}")
            continue

        feature_cols.append(col)

    if len(feature_cols) < 2:
        raise ValueError(f"可用于回归的特征太少：{feature_cols}")

    print("回归使用特征：")
    for col in feature_cols:
        print(f"  - {col}")

    save_csv(pd.DataFrame({"feature_used": feature_cols}), TABLE_DIR / "02_regression_features_used.csv")

    return feature_cols


# ============================================================
# 5. 模型训练与评价
# ============================================================

def make_train_test_split(df: pd.DataFrame):
    train_mask = df[YEAR_COL].between(TRAIN_START_YEAR, TRAIN_END_YEAR)
    test_mask = df[YEAR_COL].between(TEST_START_YEAR, TEST_END_YEAR)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("训练集或测试集为空，请检查年份范围。")

    print(f"训练集：{TRAIN_START_YEAR}—{TRAIN_END_YEAR}，记录数 {len(train_df)}")
    print(f"测试集：{TEST_START_YEAR}—{TEST_END_YEAR}，记录数 {len(test_df)}")

    return train_df, test_df


def inverse_target(y_model_scale):
    if USE_LOG_TARGET:
        return np.expm1(y_model_scale)

    return y_model_scale


def prepare_target(df: pd.DataFrame):
    if USE_LOG_TARGET:
        return np.log1p(df["gdp_yi_yuan"].values)

    return df["gdp_yi_yuan"].values


def evaluate_regression(y_true_original, y_pred_original, model_name, subset_name):
    r2 = r2_score(y_true_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae = mean_absolute_error(y_true_original, y_pred_original)

    return {
        "model": model_name,
        "subset": subset_name,
        "R2": r2,
        "RMSE_yi_yuan": rmse,
        "MAE_yi_yuan": mae,
        "n": len(y_true_original),
    }


def train_models(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    y_train_model = prepare_target(train_df)
    y_test_model = prepare_target(test_df)

    y_train_original = train_df["gdp_yi_yuan"].values
    y_test_original = test_df["gdp_yi_yuan"].values

    alphas = np.logspace(-3, 3, 80)

    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=alphas, cv=5))
        ])
    }

    metrics = []
    prediction_frames = []
    coef_frames = []

    for model_name, pipe in models.items():
        print(f"训练模型：{model_name}")

        pipe.fit(X_train, y_train_model)

        pred_train_model = pipe.predict(X_train)
        pred_test_model = pipe.predict(X_test)

        pred_train = inverse_target(pred_train_model)
        pred_test = inverse_target(pred_test_model)

        # 防止反变换后出现负值
        pred_train = np.clip(pred_train, a_min=0, a_max=None)
        pred_test = np.clip(pred_test, a_min=0, a_max=None)

        metrics.append(evaluate_regression(y_train_original, pred_train, model_name, "train"))
        metrics.append(evaluate_regression(y_test_original, pred_test, model_name, "test"))

        # 预测结果表
        train_pred_df = train_df[[ID_COL, YEAR_COL, "gdp_yi_yuan"]].copy()
        train_pred_df["subset"] = "train"
        train_pred_df["model"] = model_name
        train_pred_df["gdp_pred_yi_yuan"] = pred_train
        train_pred_df["residual_yi_yuan"] = train_pred_df["gdp_yi_yuan"] - train_pred_df["gdp_pred_yi_yuan"]
        train_pred_df["abs_error_yi_yuan"] = train_pred_df["residual_yi_yuan"].abs()

        test_pred_df = test_df[[ID_COL, YEAR_COL, "gdp_yi_yuan"]].copy()
        test_pred_df["subset"] = "test"
        test_pred_df["model"] = model_name
        test_pred_df["gdp_pred_yi_yuan"] = pred_test
        test_pred_df["residual_yi_yuan"] = test_pred_df["gdp_yi_yuan"] - test_pred_df["gdp_pred_yi_yuan"]
        test_pred_df["abs_error_yi_yuan"] = test_pred_df["residual_yi_yuan"].abs()

        prediction_frames.append(train_pred_df)
        prediction_frames.append(test_pred_df)

        # 回归系数
        model_obj = pipe.named_steps["model"]
        coef = model_obj.coef_

        coef_df = pd.DataFrame({
            "model": model_name,
            "feature": feature_cols,
            "coefficient_on_scaled_features": coef
        })

        if model_name == "Ridge Regression":
            coef_df["ridge_alpha"] = model_obj.alpha_
        else:
            coef_df["ridge_alpha"] = np.nan

        coef_frames.append(coef_df)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    coef_df = pd.concat(coef_frames, ignore_index=True)

    save_csv(metrics_df, TABLE_DIR / "03_regression_metrics.csv")
    save_csv(predictions_df, TABLE_DIR / "04_regression_predictions.csv")
    save_csv(coef_df, TABLE_DIR / "05_regression_coefficients.csv")

    save_excel(
        {
            "metrics": metrics_df,
            "predictions": predictions_df,
            "coefficients": coef_df,
            "train_data": train_df,
            "test_data": test_df,
        },
        TABLE_DIR / "regression_summary.xlsx"
    )

    return models, metrics_df, predictions_df, coef_df


# ============================================================
# 6. 可视化
# ============================================================

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list):
    corr_cols = feature_cols + ["gdp_yi_yuan", "population_wan"]
    corr_cols = [c for c in corr_cols if c in df.columns]

    if len(corr_cols) < 2:
        return

    corr = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("夜光指标、人口与 GDP 相关矩阵")
    save_fig("01_correlation_matrix_regression.png")

    save_csv(corr.reset_index().rename(columns={"index": "field"}), TABLE_DIR / "06_correlation_matrix_regression.csv")


def plot_gdp_vs_nightlight(df: pd.DataFrame):
    if "total_light" not in df.columns:
        return

    plt.figure(figsize=(8, 6))

    for district, g in df.groupby(ID_COL):
        plt.scatter(g["total_light"], g["gdp_yi_yuan"], label=district, alpha=0.8)

    plt.title("GDP 与夜光总量散点图")
    plt.xlabel("夜光总量，nW/cm²/sr × km²")
    plt.ylabel("GDP，亿元")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig("02_scatter_gdp_vs_total_light.png")


def plot_metrics_comparison(metrics_df: pd.DataFrame):
    test_metrics = metrics_df[metrics_df["subset"] == "test"].copy()

    # R2 对比
    plt.figure(figsize=(7, 5))
    plt.bar(test_metrics["model"], test_metrics["R2"])
    plt.title("测试集 R² 对比")
    plt.xlabel("模型")
    plt.ylabel("R²")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    save_fig("03_model_comparison_r2.png")

    # RMSE / MAE 对比
    x = np.arange(len(test_metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, test_metrics["RMSE_yi_yuan"], width, label="RMSE")
    plt.bar(x + width / 2, test_metrics["MAE_yi_yuan"], width, label="MAE")
    plt.title("测试集 RMSE 与 MAE 对比")
    plt.xlabel("模型")
    plt.ylabel("误差，亿元")
    plt.xticks(x, test_metrics["model"], rotation=15)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    save_fig("04_model_comparison_rmse_mae.png")


def plot_actual_vs_predicted(predictions_df: pd.DataFrame):
    test_df = predictions_df[predictions_df["subset"] == "test"].copy()

    plt.figure(figsize=(7, 7))

    for model_name, g in test_df.groupby("model"):
        plt.scatter(g["gdp_yi_yuan"], g["gdp_pred_yi_yuan"], label=model_name, alpha=0.8)

    min_val = min(test_df["gdp_yi_yuan"].min(), test_df["gdp_pred_yi_yuan"].min())
    max_val = max(test_df["gdp_yi_yuan"].max(), test_df["gdp_pred_yi_yuan"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="1:1 reference")
    plt.title("测试集 GDP 实际值 vs 预测值")
    plt.xlabel("实际 GDP，亿元")
    plt.ylabel("预测 GDP，亿元")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig("05_actual_vs_predicted_test.png")


def plot_residuals(predictions_df: pd.DataFrame):
    test_df = predictions_df[predictions_df["subset"] == "test"].copy()

    for model_name, g in test_df.groupby("model"):
        plt.figure(figsize=(8, 5))
        plt.scatter(g["gdp_pred_yi_yuan"], g["residual_yi_yuan"], alpha=0.8)
        plt.axhline(0, linestyle="--")
        plt.title(f"{model_name} 测试集残差图")
        plt.xlabel("预测 GDP，亿元")
        plt.ylabel("残差，实际值 - 预测值，亿元")
        plt.grid(alpha=0.3)

        safe_name = model_name.lower().replace(" ", "_")
        save_fig(f"06_residuals_{safe_name}.png")


def plot_prediction_time_series(predictions_df: pd.DataFrame):
    test_df = predictions_df[predictions_df["subset"] == "test"].copy()

    for model_name, g_model in test_df.groupby("model"):
        plt.figure(figsize=(10, 6))

        for district, g in g_model.groupby(ID_COL):
            g = g.sort_values(YEAR_COL)
            plt.plot(g[YEAR_COL], g["gdp_yi_yuan"], marker="o", linestyle="-", label=f"{district} 实际")
            plt.plot(g[YEAR_COL], g["gdp_pred_yi_yuan"], marker="x", linestyle="--", label=f"{district} 预测")

        plt.title(f"{model_name}：2022—2024 年各区 GDP 实际值与预测值")
        plt.xlabel("年份")
        plt.ylabel("GDP，亿元")
        plt.legend(fontsize=8, ncol=2)
        plt.grid(alpha=0.3)

        safe_name = model_name.lower().replace(" ", "_")
        save_fig(f"07_timeseries_prediction_{safe_name}.png")


def plot_coefficients(coef_df: pd.DataFrame):
    pivot = coef_df.pivot_table(
        index="feature",
        columns="model",
        values="coefficient_on_scaled_features",
        aggfunc="first"
    ).fillna(0)

    features = pivot.index.tolist()
    x = np.arange(len(features))
    width = 0.35

    plt.figure(figsize=(11, 6))

    if "Linear Regression" in pivot.columns:
        plt.bar(x - width / 2, pivot["Linear Regression"], width, label="Linear Regression")

    if "Ridge Regression" in pivot.columns:
        plt.bar(x + width / 2, pivot["Ridge Regression"], width, label="Ridge Regression")

    plt.axhline(0, linestyle="--")
    plt.title("线性回归与岭回归标准化系数对比")
    plt.xlabel("特征")
    plt.ylabel("标准化回归系数")
    plt.xticks(x, features, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    save_fig("08_coefficient_comparison.png")


def run_visualizations(df, feature_cols, metrics_df, predictions_df, coef_df):
    plot_correlation_matrix(df, feature_cols)
    plot_gdp_vs_nightlight(df)
    plot_metrics_comparison(metrics_df)
    plot_actual_vs_predicted(predictions_df)
    plot_residuals(predictions_df)
    plot_prediction_time_series(predictions_df)
    plot_coefficients(coef_df)


# ============================================================
# 7. 报告文字草稿
# ============================================================

def generate_report_text(df, feature_cols, metrics_df, coef_df):
    test_metrics = metrics_df[metrics_df["subset"] == "test"].copy()
    best_row = test_metrics.sort_values("RMSE_yi_yuan").iloc[0]

    lines = []

    lines.append("# 线性回归与岭回归对比结果草稿\n")
    lines.append("本部分可作为课程报告“回归分析”和“模型评估”章节的初稿。正式写入报告前，应结合图表和厦门各区产业结构进行人工解释。\n")

    lines.append("## 1. 数据与建模对象\n")
    lines.append(f"- 本次回归使用区县年度样本，共 {len(df)} 条记录。")
    lines.append(f"- 时间范围为 {df[YEAR_COL].min()}—{df[YEAR_COL].max()} 年，研究对象包括 {df[ID_COL].nunique()} 个区。")
    lines.append("- 因变量为区县 GDP，单位为亿元。")
    lines.append("- 自变量包括夜光总量、平均夜光强度、发光面积、有效统计面积、夜光极值、人口等指标。")
    lines.append(f"- 训练集为 {TRAIN_START_YEAR}—{TRAIN_END_YEAR} 年，测试集为 {TEST_START_YEAR}—{TEST_END_YEAR} 年。")
    if USE_LOG_TARGET:
        lines.append("- 建模时对 GDP 使用 log1p 变换，预测后反变换为亿元进行 R²、RMSE 和 MAE 评价。")
    lines.append("")

    lines.append("## 2. 使用的特征\n")
    for col in feature_cols:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("## 3. 模型评价\n")
    for _, row in test_metrics.iterrows():
        lines.append(
            f"- {row['model']}：测试集 R²={row['R2']:.4f}，"
            f"RMSE={row['RMSE_yi_yuan']:.2f} 亿元，"
            f"MAE={row['MAE_yi_yuan']:.2f} 亿元。"
        )

    lines.append("")
    lines.append(
        f"- 按测试集 RMSE 判断，表现较好的模型为 {best_row['model']}。"
    )
    lines.append("- 若岭回归优于普通线性回归，说明夜光变量之间可能存在较强相关性，正则化有助于降低多重共线性影响。")
    lines.append("- 若普通线性回归优于岭回归，说明当前样本和特征下线性模型已经足够，岭回归的收缩可能降低了拟合能力。")
    lines.append("")

    lines.append("## 4. 残差解释建议\n")
    lines.append("- 残差较大的区县不宜简单解释为模型错误，应结合港口经济、产业结构、服务业集聚、交通照明、建设用地扩张等因素讨论。")
    lines.append("- 夜光强度不完全等于经济产出。港口、机场、道路照明和近海灯光可能增强夜光，但不一定同步提高 GDP。")
    lines.append("- 区县样本数量较少，回归结果更适合作为夜光与经济关联的验证，而不是用于高精度预测。")

    report_path = REPORT_DIR / "regression_report_draft.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 8. 主函数
# ============================================================

def main():
    set_chinese_font()

    print("=" * 80)
    print("厦门 VIIRS 夜光与 GDP：Linear Regression vs Ridge Regression")
    print("=" * 80)

    # 1. 读取数据
    viirs = load_viirs_district_data()
    econ = load_economic_data()

    # 2. 合并数据
    df = merge_viirs_and_economic(viirs, econ)

    # 3. 选择特征
    feature_cols = choose_feature_columns(df)

    # 4. 时间切分
    train_df, test_df = make_train_test_split(df)

    # 5. 训练与评价
    models, metrics_df, predictions_df, coef_df = train_models(train_df, test_df, feature_cols)

    # 6. 可视化
    run_visualizations(df, feature_cols, metrics_df, predictions_df, coef_df)

    # 7. 报告文字
    generate_report_text(df, feature_cols, metrics_df, coef_df)

    # 8. 运行日志
    run_log = f"""
运行完成。

推荐输入：
outputs/tables/07_processed_district_data.csv

备用输入：
xiamen_viirs_district_2013_2024.csv

经济数据：
经济数据(1).xlsx

训练集：
{TRAIN_START_YEAR}—{TRAIN_END_YEAR}

测试集：
{TEST_START_YEAR}—{TEST_END_YEAR}

是否对 GDP 做 log1p 变换：
{USE_LOG_TARGET}

是否使用 year 特征：
{INCLUDE_YEAR_FEATURE}

是否使用 district 哑变量：
{INCLUDE_DISTRICT_DUMMIES}

使用特征：
{feature_cols}

主要输出：
1. tables/01_merged_regression_dataset.csv
2. tables/03_regression_metrics.csv
3. tables/04_regression_predictions.csv
4. tables/05_regression_coefficients.csv
5. tables/regression_summary.xlsx
6. figures/01_correlation_matrix_regression.png
7. figures/02_scatter_gdp_vs_total_light.png
8. figures/03_model_comparison_r2.png
9. figures/04_model_comparison_rmse_mae.png
10. figures/05_actual_vs_predicted_test.png
11. figures/06_residuals_linear_regression.png
12. figures/06_residuals_ridge_regression.png
13. figures/08_coefficient_comparison.png
14. report_text/regression_report_draft.md
"""

    (OUTPUT_DIR / "run_log.txt").write_text(run_log, encoding="utf-8")

    print("=" * 80)
    print("回归分析完成。")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()