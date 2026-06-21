# -*- coding: utf-8 -*-
"""
xiamen_viirs_pixel_samples_clusterprj_fixed.py

基于厦门市 NPP-VIIRS 像元多年夜光特征进行：
1. K-Means 聚类
2. 层次聚类
3. 两种聚类方法定量对比
4. 聚类结果可视化
5. 聚类剖面解释

推荐输入：
08_pixel_multi_year_features.csv

备用输入：
xiamen_viirs_pixel_samples_2013_2024.csv

修复内容：
- 不再强制要求 cf_cvg_mean 字段；
- 聚类剖面表根据实际存在字段动态生成；
- 若缺少 log1p_mean_rad、year_count、变化量等字段，自动补充；
- 若已有 x_utm50n_m / y_utm50n_m，则直接使用；
- 若没有投影坐标，则从 lon / lat 转换到 EPSG:32650；
- 默认不把投影坐标纳入聚类特征，只用于空间可视化。
"""

from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics import pairwise_distances_argmin_min


# ============================================================
# 0. 参数设置
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

FEATURE_CANDIDATES = [
    PROJECT_DIR / "08_pixel_multi_year_features.csv",
    PROJECT_DIR / "outputs_pixel_samples" / "tables" / "08_pixel_multi_year_features.csv",
    PROJECT_DIR / "outputs_pixel_no_projection" / "tables" / "08_pixel_multi_year_features_no_projection.csv",
    PROJECT_DIR / "data" / "08_pixel_multi_year_features.csv",
]

RAW_CANDIDATES = [
    PROJECT_DIR / "xiamen_viirs_pixel_samples_2013_2024.csv",
    PROJECT_DIR / "data" / "xiamen_viirs_pixel_samples_2013_2024.csv",
]

OUTPUT_DIR = PROJECT_DIR / "outputs_cluster_projected_fixed"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report_text"

for folder in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ID_COL = "district"
YEAR_COL = "year"
LON_COL = "lon"
LAT_COL = "lat"
X_COL = "x_utm50n_m"
Y_COL = "y_utm50n_m"

K_RANGE = [3, 4, 5, 6]

# 默认 False：聚类解释“夜光发展类型”
# 如果改为 True：聚类会同时受到空间位置影响
USE_PROJECTED_COORDS_IN_CLUSTERING = False

MAX_FULL_HIERARCHICAL_N = 12000
HIERARCHICAL_SAMPLE_N = 10000
EVAL_SAMPLE_N = 10000
DENDROGRAM_SAMPLE_N = 1500

RANDOM_STATE = 42


# ============================================================
# 1. 基础工具
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
    if not path.exists():
        raise FileNotFoundError(f"未找到文件：{path}")

    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"成功读取：{path}")
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


def safe_divide(a, b):
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)


def compute_sse(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    sse = 0.0

    for lab in np.unique(labels):
        pts = X_scaled[labels == lab]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        sse += ((pts - center) ** 2).sum()

    return float(sse)


def sample_for_metric(X: np.ndarray, labels: np.ndarray, sample_n: int):
    n = X.shape[0]

    if n <= sample_n:
        return X, labels

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(n, size=sample_n, replace=False)

    return X[idx], labels[idx]


# ============================================================
# 2. 坐标转换
# ============================================================

def add_utm50n_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 lon / lat 从 EPSG:4326 转换为 EPSG:32650。
    若已有 x_utm50n_m / y_utm50n_m，则直接使用。
    """
    df = df.copy()

    if X_COL in df.columns and Y_COL in df.columns:
        df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
        df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

        if df[X_COL].notna().sum() > 0 and df[Y_COL].notna().sum() > 0:
            df["projected_crs"] = "EPSG:32650"
            print("检测到已有 UTM 50N 坐标字段，直接使用。")
            return df

    if LON_COL not in df.columns or LAT_COL not in df.columns:
        raise KeyError("缺少 lon / lat 字段，无法转换坐标系。")

    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError(
            "未安装 pyproj。请在 Anaconda Prompt 中运行：\n"
            "conda install -c conda-forge pyproj"
        )

    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32650",
        always_xy=True
    )

    valid = df[LON_COL].notna() & df[LAT_COL].notna()

    x = np.full(len(df), np.nan)
    y = np.full(len(df), np.nan)

    x_valid, y_valid = transformer.transform(
        df.loc[valid, LON_COL].values,
        df.loc[valid, LAT_COL].values
    )

    x[valid.values] = x_valid
    y[valid.values] = y_valid

    df[X_COL] = x
    df[Y_COL] = y
    df["projected_crs"] = "EPSG:32650"

    print("已完成坐标转换：EPSG:4326 -> EPSG:32650")

    return df


# ============================================================
# 3. 原始像元表备用构造多年特征
# ============================================================

def load_raw_pixel_samples(raw_path: Path) -> pd.DataFrame:
    df = read_csv_safely(raw_path)
    df = clean_columns(df)

    drop_cols = [c for c in ["system:index", ".geo"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = df.replace(["", " ", "NULL", "null", "None", "none", "NaN", "nan"], np.nan)

    required = [ID_COL, YEAR_COL, LON_COL, LAT_COL, "rad"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"原始像元表缺少必要字段：{missing}")

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df["rad"] = pd.to_numeric(df["rad"], errors="coerce")

    if "median_rad" in df.columns:
        df["median_rad"] = pd.to_numeric(df["median_rad"], errors="coerce")
    else:
        df["median_rad"] = df["rad"]

    if "cf_cvg" in df.columns:
        df["cf_cvg"] = pd.to_numeric(df["cf_cvg"], errors="coerce")
    else:
        df["cf_cvg"] = np.nan

    before_n = len(df)

    df = df.dropna(subset=[ID_COL, YEAR_COL, LON_COL, LAT_COL, "rad"]).copy()
    df[YEAR_COL] = df[YEAR_COL].round().astype(int)

    df = df[
        df[LON_COL].between(-180, 180) &
        df[LAT_COL].between(-90, 90)
    ].copy()

    df = df[df["rad"] >= 0].copy()

    df["median_rad"] = df["median_rad"].fillna(df["rad"])

    if df["cf_cvg"].notna().sum() > 0:
        df["cf_cvg"] = df.groupby(YEAR_COL)["cf_cvg"].transform(
            lambda s: s.fillna(s.median())
        )
        df["cf_cvg"] = df["cf_cvg"].fillna(df["cf_cvg"].median())

    df = df.drop_duplicates(
        subset=[ID_COL, YEAR_COL, LON_COL, LAT_COL],
        keep="first"
    ).reset_index(drop=True)

    df["lon_round6"] = df[LON_COL].round(6)
    df["lat_round6"] = df[LAT_COL].round(6)
    df["pixel_id"] = (
        df[ID_COL].astype(str) + "_" +
        df["lon_round6"].astype(str) + "_" +
        df["lat_round6"].astype(str)
    )

    after_n = len(df)

    print(f"原始像元—年份表清洗前：{before_n:,} 行")
    print(f"原始像元—年份表清洗后：{after_n:,} 行")
    print(f"像元数量：{df['pixel_id'].nunique():,}")

    return df


def calculate_trend_slope(years: np.ndarray, values: np.ndarray) -> float:
    valid = ~np.isnan(values)

    if valid.sum() < 2:
        return np.nan

    return float(np.polyfit(years[valid], values[valid], 1)[0])


def build_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = add_utm50n_coordinates(df)

    first_info = (
        df.groupby("pixel_id")
        .agg(
            district=(ID_COL, "first"),
            lon=(LON_COL, "first"),
            lat=(LAT_COL, "first"),
            x_utm50n_m=(X_COL, "first"),
            y_utm50n_m=(Y_COL, "first"),
            year_count=(YEAR_COL, "nunique"),
        )
        .reset_index()
    )

    rad_wide = df.pivot_table(
        index="pixel_id",
        columns=YEAR_COL,
        values="rad",
        aggfunc="mean"
    )
    rad_wide = rad_wide.reindex(sorted(rad_wide.columns), axis=1)

    year_list = [int(y) for y in rad_wide.columns]
    rad_wide.columns = [f"rad_{int(y)}" for y in rad_wide.columns]

    median_wide = df.pivot_table(
        index="pixel_id",
        columns=YEAR_COL,
        values="median_rad",
        aggfunc="mean"
    )
    median_wide = median_wide.reindex(sorted(median_wide.columns), axis=1)
    median_wide.columns = [f"median_rad_{int(y)}" for y in median_wide.columns]

    feat = first_info.merge(rad_wide.reset_index(), on="pixel_id", how="left")
    feat = feat.merge(median_wide.reset_index(), on="pixel_id", how="left")

    if "cf_cvg" in df.columns and df["cf_cvg"].notna().sum() > 0:
        cf_stats = (
            df.groupby("pixel_id")
            .agg(
                cf_cvg_mean=("cf_cvg", "mean"),
                cf_cvg_min=("cf_cvg", "min"),
            )
            .reset_index()
        )
        feat = feat.merge(cf_stats, on="pixel_id", how="left")

    feat = ensure_feature_columns(feat)

    return feat


# ============================================================
# 4. 特征表读取与字段补全
# ============================================================

def get_rad_year_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c.startswith("rad_"):
            suffix = c.replace("rad_", "")
            if suffix.isdigit():
                cols.append(c)

    cols = sorted(cols, key=lambda x: int(x.replace("rad_", "")))
    return cols


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据已有 rad_年份 字段补充缺失的多年特征。
    """
    df = df.copy()

    if "pixel_id" not in df.columns:
        if all(c in df.columns for c in [ID_COL, LON_COL, LAT_COL]):
            df["pixel_id"] = (
                df[ID_COL].astype(str) + "_" +
                pd.to_numeric(df[LON_COL], errors="coerce").round(6).astype(str) + "_" +
                pd.to_numeric(df[LAT_COL], errors="coerce").round(6).astype(str)
            )
        else:
            raise KeyError("特征表缺少 pixel_id，且无法通过 district/lon/lat 构造。")

    if ID_COL not in df.columns:
        df[ID_COL] = "Unknown"

    for c in [LON_COL, LAT_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rad_cols = get_rad_year_columns(df)

    if len(rad_cols) > 0:
        for c in rad_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if "year_count" not in df.columns:
            df["year_count"] = df[rad_cols].notna().sum(axis=1)

        if "mean_rad_2013_2024" not in df.columns:
            df["mean_rad_2013_2024"] = df[rad_cols].mean(axis=1)

        if "median_rad_2013_2024" not in df.columns:
            df["median_rad_2013_2024"] = df[rad_cols].median(axis=1)

        if "std_rad_2013_2024" not in df.columns:
            df["std_rad_2013_2024"] = df[rad_cols].std(axis=1)

        if "min_rad_2013_2024" not in df.columns:
            df["min_rad_2013_2024"] = df[rad_cols].min(axis=1)

        if "max_rad_2013_2024" not in df.columns:
            df["max_rad_2013_2024"] = df[rad_cols].max(axis=1)

        if "range_rad_2013_2024" not in df.columns:
            df["range_rad_2013_2024"] = (
                df["max_rad_2013_2024"] - df["min_rad_2013_2024"]
            )

        if "cv_rad_2013_2024" not in df.columns:
            df["cv_rad_2013_2024"] = safe_divide(
                df["std_rad_2013_2024"],
                df["mean_rad_2013_2024"] + 0.001
            )

        if "rad_2013" in df.columns and "rad_2024" in df.columns:
            if "change_2024_minus_2013" not in df.columns:
                df["change_2024_minus_2013"] = df["rad_2024"] - df["rad_2013"]

            if "growth_rate_2024_vs_2013" not in df.columns:
                df["growth_rate_2024_vs_2013"] = (
                    df["change_2024_minus_2013"] / (df["rad_2013"] + 0.001)
                )

        if "trend_slope_rad_per_year" not in df.columns:
            years = np.array([int(c.replace("rad_", "")) for c in rad_cols], dtype=float)

            def slope_row(row):
                values = row[rad_cols].values.astype(float)
                return calculate_trend_slope(years, values)

            df["trend_slope_rad_per_year"] = df.apply(slope_row, axis=1)

    if "mean_rad_2013_2024" in df.columns and "log1p_mean_rad" not in df.columns:
        df["log1p_mean_rad"] = np.log1p(
            pd.to_numeric(df["mean_rad_2013_2024"], errors="coerce").clip(lower=0)
        )

    # 如果 cf_cvg_mean 不存在，不强制创建；后续动态跳过
    if "cf_cvg_mean" in df.columns:
        df["cf_cvg_mean"] = pd.to_numeric(df["cf_cvg_mean"], errors="coerce")

    # 去除重复 pixel_id
    df = df.drop_duplicates(subset=["pixel_id"], keep="first").reset_index(drop=True)

    return df


def load_or_build_feature_table() -> pd.DataFrame:
    feature_path = find_existing_file(FEATURE_CANDIDATES)

    if feature_path is not None:
        print("使用推荐输入：像元多年特征表")
        feature_df = read_csv_safely(feature_path)
        feature_df = clean_columns(feature_df)

        feature_df = ensure_feature_columns(feature_df)
        feature_df = add_utm50n_coordinates(feature_df)

        print(f"读取特征表记录数：{len(feature_df):,}")
        print("当前特征表字段：")
        print(feature_df.columns.tolist())

        return feature_df

    raw_path = find_existing_file(RAW_CANDIDATES)

    if raw_path is None:
        raise FileNotFoundError(
            "未找到 08_pixel_multi_year_features.csv，也未找到 xiamen_viirs_pixel_samples_2013_2024.csv。"
        )

    print("未找到 08 特征表，改用原始像元—年份表重新构造特征。")
    raw_df = load_raw_pixel_samples(raw_path)
    feature_df = build_features_from_raw(raw_df)
    feature_df = add_utm50n_coordinates(feature_df)

    return feature_df


# ============================================================
# 5. 聚类输入矩阵
# ============================================================

def prepare_clustering_matrix(feature_df: pd.DataFrame):
    """
    默认仅使用夜光时序特征，不使用坐标。
    """
    candidate_features = [
        "log1p_mean_rad",
        "mean_rad_2013_2024",
        "std_rad_2013_2024",
        "range_rad_2013_2024",
        "cv_rad_2013_2024",
        "change_2024_minus_2013",
        "growth_rate_2024_vs_2013",
        "trend_slope_rad_per_year",
        "cf_cvg_mean",
    ]

    if USE_PROJECTED_COORDS_IN_CLUSTERING:
        candidate_features += [X_COL, Y_COL]

    feature_cols = []

    for c in candidate_features:
        if c not in feature_df.columns:
            print(f"跳过缺失特征：{c}")
            continue

        s = pd.to_numeric(feature_df[c], errors="coerce")
        valid_count = s.replace([np.inf, -np.inf], np.nan).notna().sum()

        if valid_count == 0:
            print(f"跳过全为空特征：{c}")
            continue

        feature_cols.append(c)

    if len(feature_cols) < 3:
        raise ValueError(f"可用于聚类的特征太少：{feature_cols}")

    X_raw = feature_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    # 缩尾处理，降低极端增长率影响
    X_winsor = X_raw.copy()
    for c in X_winsor.columns:
        X_winsor[c] = winsorize_series(X_winsor[c], 0.01, 0.99)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_winsor)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=[f"z_{c}" for c in feature_cols]
    )

    feature_info = pd.DataFrame({
        "feature_used": feature_cols,
        "note": [
            "投影坐标特征" if c in [X_COL, Y_COL] else "夜光时序或质量特征"
            for c in feature_cols
        ],
    })

    save_csv(feature_info, TABLE_DIR / "01_features_used_for_clustering.csv")
    save_csv(X_scaled_df, TABLE_DIR / "02_scaled_clustering_matrix.csv")

    print("聚类使用特征：")
    for c in feature_cols:
        print(f"  - {c}")

    return X_scaled, X_scaled_df, feature_cols


# ============================================================
# 6. 聚类建模
# ============================================================

def fit_kmeans(X_scaled: np.ndarray, k: int):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=30,
        max_iter=500
    )
    labels = model.fit_predict(X_scaled)

    return model, labels


def fit_hierarchical(X_scaled: np.ndarray, k: int, feature_df: pd.DataFrame):
    n = X_scaled.shape[0]

    if n <= MAX_FULL_HIERARCHICAL_N:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )
        labels = model.fit_predict(X_scaled)

        info = {
            "mode": "full",
            "sample_size": n,
        }

        return model, labels, info

    temp = feature_df[[ID_COL]].copy()
    temp["_row_id"] = np.arange(n)

    sample_ids = []
    per_group_n = max(1, math.ceil(HIERARCHICAL_SAMPLE_N / temp[ID_COL].nunique()))

    for _, g in temp.groupby(ID_COL):
        take_n = min(len(g), per_group_n)
        sample_ids.extend(
            g.sample(take_n, random_state=RANDOM_STATE)["_row_id"].tolist()
        )

    if len(sample_ids) > HIERARCHICAL_SAMPLE_N:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_ids = rng.choice(sample_ids, size=HIERARCHICAL_SAMPLE_N, replace=False).tolist()

    sample_ids = np.array(sample_ids)
    X_sample = X_scaled[sample_ids]

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )
    sample_labels = model.fit_predict(X_sample)

    centers = []
    for lab in range(k):
        pts = X_sample[sample_labels == lab]
        centers.append(pts.mean(axis=0))

    centers = np.vstack(centers)

    labels, _ = pairwise_distances_argmin_min(X_scaled, centers)

    info = {
        "mode": "sample_then_nearest_centroid",
        "sample_size": len(sample_ids),
    }

    return model, labels, info


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray, method: str, k: int):
    if len(np.unique(labels)) < 2:
        return {
            "method": method,
            "n_clusters": k,
            "silhouette_score": np.nan,
            "calinski_harabasz_score": np.nan,
            "davies_bouldin_score": np.nan,
            "SSE": np.nan,
            "eval_sample_size": 0,
        }

    X_eval, y_eval = sample_for_metric(X_scaled, labels, EVAL_SAMPLE_N)

    return {
        "method": method,
        "n_clusters": k,
        "silhouette_score": silhouette_score(X_eval, y_eval),
        "calinski_harabasz_score": calinski_harabasz_score(X_eval, y_eval),
        "davies_bouldin_score": davies_bouldin_score(X_eval, y_eval),
        "SSE": compute_sse(X_scaled, labels),
        "eval_sample_size": len(X_eval),
    }


def run_clustering(feature_df: pd.DataFrame, X_scaled: np.ndarray):
    evaluation_rows = []
    all_labels = pd.DataFrame({
        "pixel_id": feature_df["pixel_id"].values
    })
    hier_info_rows = []

    for k in K_RANGE:
        print(f"训练 K-Means，k={k}")
        _, kmeans_labels = fit_kmeans(X_scaled, k)

        eval_kmeans = evaluate_clustering(
            X_scaled,
            kmeans_labels,
            method="KMeans",
            k=k
        )
        eval_kmeans["hierarchical_mode"] = ""
        eval_kmeans["hierarchical_sample_size"] = ""
        evaluation_rows.append(eval_kmeans)

        all_labels[f"kmeans_k{k}"] = kmeans_labels

        print(f"训练层次聚类，k={k}")
        _, hier_labels, hier_info = fit_hierarchical(X_scaled, k, feature_df)

        eval_hier = evaluate_clustering(
            X_scaled,
            hier_labels,
            method="Hierarchical_Ward",
            k=k
        )
        eval_hier["hierarchical_mode"] = hier_info["mode"]
        eval_hier["hierarchical_sample_size"] = hier_info["sample_size"]
        evaluation_rows.append(eval_hier)

        all_labels[f"hierarchical_k{k}"] = hier_labels

        hier_info_rows.append({
            "n_clusters": k,
            **hier_info,
        })

    evaluation_df = pd.DataFrame(evaluation_rows)
    hier_info_df = pd.DataFrame(hier_info_rows)

    mean_sil = (
        evaluation_df
        .groupby("n_clusters")["silhouette_score"]
        .mean()
        .reset_index()
        .sort_values("silhouette_score", ascending=False)
    )

    final_k = int(mean_sil.iloc[0]["n_clusters"])

    print(f"推荐最终类别数：k={final_k}")

    return evaluation_df, all_labels, hier_info_df, final_k


# ============================================================
# 7. 聚类结果剖面
# ============================================================

def merge_labels(feature_df: pd.DataFrame, all_labels: pd.DataFrame, final_k: int):
    result = feature_df.copy()
    result = result.merge(all_labels, on="pixel_id", how="left")

    result["kmeans_cluster"] = result[f"kmeans_k{final_k}"]
    result["hierarchical_cluster"] = result[f"hierarchical_k{final_k}"]

    return result


def assign_cluster_type(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.copy()

    if "mean_rad" not in profile.columns:
        profile["mean_rad"] = np.nan

    if "change_2024_minus_2013" not in profile.columns:
        profile["change_2024_minus_2013"] = np.nan

    mean_series = profile["mean_rad"].fillna(profile["mean_rad"].median())
    change_series = profile["change_2024_minus_2013"].fillna(
        profile["change_2024_minus_2013"].median()
    )

    mean_q1 = mean_series.quantile(0.33)
    mean_q2 = mean_series.quantile(0.67)

    change_q1 = change_series.quantile(0.33)
    change_q2 = change_series.quantile(0.67)

    types = []

    for _, row in profile.iterrows():
        mean_val = row["mean_rad"]
        change_val = row["change_2024_minus_2013"]

        if pd.isna(mean_val):
            mean_val = mean_series.median()

        if pd.isna(change_val):
            change_val = change_series.median()

        if mean_val >= mean_q2 and change_val >= change_q2:
            t = "高亮增长型"
        elif mean_val >= mean_q2 and change_val < change_q2:
            t = "高亮稳定型"
        elif mean_val < mean_q1 and change_val >= change_q2:
            t = "低亮快速增长型"
        elif mean_val < mean_q1 and change_val < change_q1:
            t = "低亮稳定或减弱型"
        elif change_val >= change_q2:
            t = "中等增长型"
        else:
            t = "中等稳定型"

        types.append(t)

    profile["cluster_type_suggestion"] = types

    return profile


def create_cluster_profile(result_df: pd.DataFrame, label_col: str, method_name: str):
    """
    动态生成聚类剖面表。
    修复点：不再强制要求 cf_cvg_mean。
    """
    agg_dict = {
        "pixel_count": ("pixel_id", "count"),
    }

    def add_agg(out_name, source_col, func="mean"):
        if source_col in result_df.columns:
            agg_dict[out_name] = (source_col, func)

    add_agg("district_count", ID_COL, "nunique")
    add_agg("mean_lon", LON_COL, "mean")
    add_agg("mean_lat", LAT_COL, "mean")
    add_agg("mean_x_utm50n_m", X_COL, "mean")
    add_agg("mean_y_utm50n_m", Y_COL, "mean")

    add_agg("mean_rad", "mean_rad_2013_2024", "mean")
    add_agg("median_rad", "median_rad_2013_2024", "mean")
    add_agg("std_rad", "std_rad_2013_2024", "mean")
    add_agg("rad_range", "range_rad_2013_2024", "mean")
    add_agg("cv_rad", "cv_rad_2013_2024", "mean")
    add_agg("change_2024_minus_2013", "change_2024_minus_2013", "mean")
    add_agg("growth_rate_2024_vs_2013", "growth_rate_2024_vs_2013", "mean")
    add_agg("trend_slope", "trend_slope_rad_per_year", "mean")

    # 可选字段：存在才统计
    add_agg("cf_cvg_mean", "cf_cvg_mean", "mean")
    add_agg("year_count_mean", "year_count", "mean")

    profile = (
        result_df
        .groupby(label_col)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={label_col: "cluster"})
    )

    profile["method"] = method_name
    profile = assign_cluster_type(profile)

    return profile


# ============================================================
# 8. 可视化
# ============================================================

def plot_evaluation_metrics(evaluation_df: pd.DataFrame):
    metrics = [
        ("silhouette_score", "轮廓系数 Silhouette，越高越好", "evaluation_silhouette.png"),
        ("calinski_harabasz_score", "Calinski-Harabasz 指数，越高越好", "evaluation_calinski_harabasz.png"),
        ("davies_bouldin_score", "Davies-Bouldin 指数，越低越好", "evaluation_davies_bouldin.png"),
        ("SSE", "SSE，越低越好", "evaluation_sse.png"),
    ]

    for metric, title, filename in metrics:
        plt.figure(figsize=(8, 5))

        for method, g in evaluation_df.groupby("method"):
            g = g.sort_values("n_clusters")
            plt.plot(g["n_clusters"], g[metric], marker="o", label=method)

        plt.title(title)
        plt.xlabel("聚类类别数 k")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(alpha=0.3)
        save_fig(filename)


def plot_pca_clusters(result_df: pd.DataFrame, X_scaled: np.ndarray, label_col: str, title: str, filename: str):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)

    keep_cols = ["pixel_id", ID_COL, LON_COL, LAT_COL, X_COL, Y_COL, label_col]
    keep_cols = [c for c in keep_cols if c in result_df.columns]

    tmp = result_df[keep_cols].copy()
    tmp["PC1"] = pcs[:, 0]
    tmp["PC2"] = pcs[:, 1]

    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(tmp["PC1"], tmp["PC2"], c=tmp[label_col], s=5, alpha=0.7)
    plt.colorbar(sc, label="Cluster")
    plt.title(f"{title}\nPC1={explained[0]:.2%}, PC2={explained[1]:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    save_fig(filename)

    save_csv(tmp, TABLE_DIR / filename.replace(".png", ".csv"))


def plot_spatial_clusters_projected(result_df: pd.DataFrame, label_col: str, title: str, filename: str, sample_max=80000):
    required = [X_COL, Y_COL, label_col]
    if not all(c in result_df.columns for c in required):
        print(f"缺少字段，跳过空间图：{required}")
        return

    plot_df = result_df[required].dropna().copy()

    if len(plot_df) > sample_max:
        plot_df = plot_df.sample(sample_max, random_state=RANDOM_STATE)

    plt.figure(figsize=(8, 7))
    sc = plt.scatter(
        plot_df[X_COL],
        plot_df[Y_COL],
        c=plot_df[label_col],
        s=5,
        alpha=0.8
    )
    plt.colorbar(sc, label="Cluster")
    plt.title(title)
    plt.xlabel("X，UTM Zone 50N，m")
    plt.ylabel("Y，UTM Zone 50N，m")
    plt.grid(alpha=0.2)
    save_fig(filename)


def plot_spatial_comparison(result_df: pd.DataFrame, final_k: int, sample_max=80000):
    required = [X_COL, Y_COL, "kmeans_cluster", "hierarchical_cluster"]
    if not all(c in result_df.columns for c in required):
        print("缺少字段，跳过空间对比图。")
        return

    plot_df = result_df[required].dropna().copy()

    if len(plot_df) > sample_max:
        plot_df = plot_df.sample(sample_max, random_state=RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(
        plot_df[X_COL],
        plot_df[Y_COL],
        c=plot_df["kmeans_cluster"],
        s=4,
        alpha=0.8
    )
    axes[0].set_title(f"K-Means 聚类空间分布，k={final_k}")
    axes[0].set_xlabel("X，UTM Zone 50N，m")
    axes[0].set_ylabel("Y，UTM Zone 50N，m")
    axes[0].grid(alpha=0.2)

    sc2 = axes[1].scatter(
        plot_df[X_COL],
        plot_df[Y_COL],
        c=plot_df["hierarchical_cluster"],
        s=4,
        alpha=0.8
    )
    axes[1].set_title(f"层次聚类空间分布，k={final_k}")
    axes[1].set_xlabel("X，UTM Zone 50N，m")
    axes[1].set_ylabel("Y，UTM Zone 50N，m")
    axes[1].grid(alpha=0.2)

    fig.colorbar(sc1, ax=axes[0], label="Cluster")
    fig.colorbar(sc2, ax=axes[1], label="Cluster")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "spatial_kmeans_vs_hierarchical_projected.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_count(result_df: pd.DataFrame, label_col: str, title: str, filename: str):
    count_df = (
        result_df[label_col]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    count_df.columns = ["cluster", "pixel_count"]

    plt.figure(figsize=(7, 5))
    plt.bar(count_df["cluster"].astype(str), count_df["pixel_count"])
    plt.title(title)
    plt.xlabel("聚类类别")
    plt.ylabel("像元数量")
    save_fig(filename)

    save_csv(count_df, TABLE_DIR / filename.replace(".png", ".csv"))


def plot_cluster_profile_heatmap(profile: pd.DataFrame, method_name: str, filename: str):
    candidate_cols = [
        "mean_rad",
        "std_rad",
        "rad_range",
        "cv_rad",
        "change_2024_minus_2013",
        "growth_rate_2024_vs_2013",
        "trend_slope",
        "cf_cvg_mean",
        "year_count_mean",
    ]

    cols = [c for c in candidate_cols if c in profile.columns]

    if len(cols) < 2:
        print(f"{method_name} 聚类剖面可用字段太少，跳过热力图。")
        return

    tmp = profile[["cluster"] + cols].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    tmp[cols] = tmp[cols].fillna(tmp[cols].median(numeric_only=True))

    scaler = StandardScaler()
    values = scaler.fit_transform(tmp[cols])

    plt.figure(figsize=(11, 5))
    im = plt.imshow(values, aspect="auto")
    plt.colorbar(im, label="标准化簇均值")

    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(tmp)), [f"Cluster {c}" for c in tmp["cluster"]])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8)

    plt.title(f"{method_name} 聚类剖面热力图")
    save_fig(filename)


def plot_dendrogram_sample(X_scaled: np.ndarray):
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram

        n = X_scaled.shape[0]

        if n > DENDROGRAM_SAMPLE_N:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(n, size=DENDROGRAM_SAMPLE_N, replace=False)
            X_d = X_scaled[idx]
        else:
            X_d = X_scaled

        Z = linkage(X_d, method="ward")

        plt.figure(figsize=(12, 6))
        dendrogram(
            Z,
            truncate_mode="level",
            p=5,
            no_labels=True
        )
        plt.title(f"层次聚类树状图抽样展示，样本量={len(X_d)}")
        plt.xlabel("样本")
        plt.ylabel("Ward 距离")
        save_fig("hierarchical_dendrogram_sample.png")

    except Exception as e:
        print("树状图生成失败，已跳过。")
        print(f"原因：{e}")


# ============================================================
# 9. 报告文字草稿
# ============================================================

def generate_report_text(
    feature_df: pd.DataFrame,
    result_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    final_k: int,
    kmeans_profile: pd.DataFrame,
    hier_profile: pd.DataFrame,
    feature_cols: list
):
    lines = []

    lines.append("# K-Means 与层次聚类对比结果草稿\n")
    lines.append("本部分用于课程报告“数据挖掘建模”和“模型评估”章节。正式写入报告前，应结合空间分布图和厦门城市发展背景进行人工修正。\n")

    lines.append("## 1. 数据与坐标处理\n")
    lines.append(f"- 本次聚类使用像元多年夜光特征表，共 {len(feature_df):,} 个像元。")
    lines.append("- 原始坐标为 GCS_WGS_1984 经纬度坐标，脚本已确认或转换为 WGS84 / UTM Zone 50N，EPSG:32650。")
    lines.append("- 转换后的 x_utm50n_m、y_utm50n_m 用于空间分布图绘制。")

    if USE_PROJECTED_COORDS_IN_CLUSTERING:
        lines.append("- 本次聚类将投影坐标纳入聚类特征，因此结果同时受到夜光特征和空间位置影响。")
    else:
        lines.append("- 本次聚类未将坐标纳入聚类特征，聚类主要反映夜光发展类型；投影坐标仅用于空间可视化。")

    lines.append("\n## 2. 聚类特征\n")
    for c in feature_cols:
        lines.append(f"- {c}")

    lines.append("\n## 3. 类别数选择与评价\n")
    lines.append(f"- 脚本比较了 k={K_RANGE} 下 K-Means 和层次聚类的表现。")
    lines.append(f"- 根据两种方法平均轮廓系数，推荐最终类别数为 k={final_k}。")
    lines.append("- 轮廓系数和 Calinski-Harabasz 指数越高越好，Davies-Bouldin 指数和 SSE 越低越好。\n")

    best_rows = evaluation_df[evaluation_df["n_clusters"] == final_k]

    for _, row in best_rows.iterrows():
        lines.append(
            f"- {row['method']}：Silhouette={row['silhouette_score']:.4f}，"
            f"Calinski-Harabasz={row['calinski_harabasz_score']:.2f}，"
            f"Davies-Bouldin={row['davies_bouldin_score']:.4f}，"
            f"SSE={row['SSE']:.2f}。"
        )

    lines.append("\n## 4. K-Means 聚类类型解释\n")
    for _, row in kmeans_profile.sort_values("cluster").iterrows():
        mean_rad = row.get("mean_rad", np.nan)
        change = row.get("change_2024_minus_2013", np.nan)
        slope = row.get("trend_slope", np.nan)

        lines.append(
            f"- Cluster {int(row['cluster'])}：{row['cluster_type_suggestion']}，"
            f"像元数 {int(row['pixel_count'])}，"
            f"多年平均夜光 {mean_rad:.3f}，"
            f"2024 相对 2013 变化量 {change:.3f}，"
            f"趋势斜率 {slope:.4f}。"
        )

    lines.append("\n## 5. 层次聚类类型解释\n")
    for _, row in hier_profile.sort_values("cluster").iterrows():
        mean_rad = row.get("mean_rad", np.nan)
        change = row.get("change_2024_minus_2013", np.nan)
        slope = row.get("trend_slope", np.nan)

        lines.append(
            f"- Cluster {int(row['cluster'])}：{row['cluster_type_suggestion']}，"
            f"像元数 {int(row['pixel_count'])}，"
            f"多年平均夜光 {mean_rad:.3f}，"
            f"2024 相对 2013 变化量 {change:.3f}，"
            f"趋势斜率 {slope:.4f}。"
        )

    lines.append("\n## 6. 图表引用建议\n")
    lines.append("- `evaluation_silhouette.png`：说明不同类别数和不同方法的轮廓系数。")
    lines.append("- `evaluation_sse.png`：说明 K 值变化下的 SSE 变化。")
    lines.append("- `pca_kmeans_clusters.png` 和 `pca_hierarchical_clusters.png`：展示聚类结果在 PCA 空间中的分布。")
    lines.append("- `spatial_kmeans_vs_hierarchical_projected.png`：对比两种方法的空间分类结果。")
    lines.append("- `profile_heatmap_kmeans.png` 和 `profile_heatmap_hierarchical.png`：解释不同类别的夜光发展特征。")

    out_text = "\n".join(lines)

    (REPORT_DIR / "clustering_report_draft_projected_fixed.md").write_text(out_text, encoding="utf-8")


# ============================================================
# 10. 主函数
# ============================================================

def main():
    set_chinese_font()

    print("=" * 80)
    print("厦门 VIIRS 夜光像元：K-Means 与层次聚类对比，投影坐标版")
    print("=" * 80)

    # 1. 读取或构造特征表
    feature_df = load_or_build_feature_table()

    required_base = ["pixel_id", ID_COL, LON_COL, LAT_COL]
    missing_base = [c for c in required_base if c not in feature_df.columns]
    if missing_base:
        raise KeyError(f"特征表缺少必要字段：{missing_base}")

    # 2. 坐标转换或确认
    feature_df = add_utm50n_coordinates(feature_df)

    # 3. 再次补充特征
    feature_df = ensure_feature_columns(feature_df)

    save_csv(feature_df, TABLE_DIR / "00_pixel_multi_year_features_projected_used.csv")

    print(f"建模像元数量：{len(feature_df):,}")
    print(f"字段数量：{feature_df.shape[1]}")

    # 4. 准备聚类矩阵
    X_scaled, X_scaled_df, feature_cols = prepare_clustering_matrix(feature_df)

    # 5. 聚类对比
    evaluation_df, all_labels, hier_info_df, final_k = run_clustering(feature_df, X_scaled)

    save_csv(evaluation_df, TABLE_DIR / "03_clustering_evaluation_all_k.csv")
    save_csv(all_labels, TABLE_DIR / "04_all_cluster_labels.csv")
    save_csv(hier_info_df, TABLE_DIR / "05_hierarchical_mode_info.csv")

    # 6. 合并标签
    result_df = merge_labels(feature_df, all_labels, final_k)
    save_csv(result_df, TABLE_DIR / "06_final_pixel_clusters_projected.csv")

    # 7. 聚类剖面
    kmeans_profile = create_cluster_profile(result_df, "kmeans_cluster", "KMeans")
    hier_profile = create_cluster_profile(result_df, "hierarchical_cluster", "Hierarchical_Ward")

    save_csv(kmeans_profile, TABLE_DIR / "07_kmeans_cluster_profile.csv")
    save_csv(hier_profile, TABLE_DIR / "08_hierarchical_cluster_profile.csv")

    save_excel(
        {
            "features_used": pd.DataFrame({"feature": feature_cols}),
            "evaluation": evaluation_df,
            "final_clusters": result_df,
            "kmeans_profile": kmeans_profile,
            "hierarchical_profile": hier_profile,
        },
        TABLE_DIR / "clustering_summary_projected_fixed.xlsx"
    )

    # 8. 可视化
    plot_evaluation_metrics(evaluation_df)

    plot_pca_clusters(
        result_df,
        X_scaled,
        "kmeans_cluster",
        f"K-Means 聚类 PCA 可视化，k={final_k}",
        "pca_kmeans_clusters.png"
    )

    plot_pca_clusters(
        result_df,
        X_scaled,
        "hierarchical_cluster",
        f"层次聚类 PCA 可视化，k={final_k}",
        "pca_hierarchical_clusters.png"
    )

    plot_spatial_clusters_projected(
        result_df,
        "kmeans_cluster",
        f"K-Means 聚类空间分布，k={final_k}，EPSG:32650",
        "spatial_kmeans_clusters_projected.png"
    )

    plot_spatial_clusters_projected(
        result_df,
        "hierarchical_cluster",
        f"层次聚类空间分布，k={final_k}，EPSG:32650",
        "spatial_hierarchical_clusters_projected.png"
    )

    plot_spatial_comparison(result_df, final_k)

    plot_cluster_count(
        result_df,
        "kmeans_cluster",
        f"K-Means 各类别像元数量，k={final_k}",
        "count_kmeans_clusters.png"
    )

    plot_cluster_count(
        result_df,
        "hierarchical_cluster",
        f"层次聚类各类别像元数量，k={final_k}",
        "count_hierarchical_clusters.png"
    )

    plot_cluster_profile_heatmap(
        kmeans_profile,
        "K-Means",
        "profile_heatmap_kmeans.png"
    )

    plot_cluster_profile_heatmap(
        hier_profile,
        "Hierarchical Ward",
        "profile_heatmap_hierarchical.png"
    )

    plot_dendrogram_sample(X_scaled)

    # 9. 报告草稿
    generate_report_text(
        feature_df,
        result_df,
        evaluation_df,
        final_k,
        kmeans_profile,
        hier_profile,
        feature_cols
    )

    # 10. 日志
    run_log = f"""
运行完成。

输入优先级：
1. 08_pixel_multi_year_features.csv
2. xiamen_viirs_pixel_samples_2013_2024.csv

实际建模像元数：
{len(feature_df):,}

最终推荐类别数：
k = {final_k}

坐标转换：
EPSG:4326 -> EPSG:32650
字段：
x_utm50n_m, y_utm50n_m

是否将投影坐标纳入聚类特征：
{USE_PROJECTED_COORDS_IN_CLUSTERING}

聚类特征：
{feature_cols}

主要输出：
1. tables/03_clustering_evaluation_all_k.csv
2. tables/06_final_pixel_clusters_projected.csv
3. tables/07_kmeans_cluster_profile.csv
4. tables/08_hierarchical_cluster_profile.csv
5. tables/clustering_summary_projected_fixed.xlsx
6. figures/evaluation_silhouette.png
7. figures/evaluation_sse.png
8. figures/pca_kmeans_clusters.png
9. figures/pca_hierarchical_clusters.png
10. figures/spatial_kmeans_vs_hierarchical_projected.png
11. figures/profile_heatmap_kmeans.png
12. figures/profile_heatmap_hierarchical.png
13. report_text/clustering_report_draft_projected_fixed.md

本版修复：
- 不再强制要求 cf_cvg_mean。
- 聚类剖面根据实际字段动态生成。
- 如果部分特征不存在，会自动跳过或补充。
"""

    (OUTPUT_DIR / "run_log.txt").write_text(run_log, encoding="utf-8")

    print("=" * 80)
    print("聚类对比完成。")
    print(f"推荐类别数：k={final_k}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()