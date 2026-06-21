import time
import requests
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pathlib import Path

# =========================
# 1. 基础参数
# =========================
AMAP_KEY = "a5077ef12dc4c425ed3f26c544bb71ab"
CITY_ADCODE = "350200"  # 厦门市
BASE_URL = "https://restapi.amap.com/v3/config/district"

# 保存到桌面
desktop_candidates = [
    Path.home() / "Desktop",
    Path.home() / "桌面",
    Path.home() / "OneDrive" / "Desktop",
    Path.home() / "OneDrive" / "桌面",
]

desktop_dir = None
for p in desktop_candidates:
    if p.exists():
        desktop_dir = p
        break

if desktop_dir is None:
    desktop_dir = Path.home() / "Desktop"

out_dir = desktop_dir / "xiamen_boundary"
out_dir.mkdir(parents=True, exist_ok=True)

print("输出文件夹：", out_dir.resolve())

# =========================
# 2. 请求函数
# =========================
def amap_district_query(keywords, subdistrict=0, extensions="base", max_retries=5):
    """
    调用高德行政区域查询接口。
    加入重试机制，避免 QPS 超限导致程序中断。
    """
    params = {
        "key": AMAP_KEY,
        "keywords": keywords,
        "subdistrict": subdistrict,
        "extensions": extensions,
        "output": "JSON"
    }

    for attempt in range(1, max_retries + 1):
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("status") == "1":
            return data

        info = data.get("info", "")
        infocode = data.get("infocode", "")

        # QPS 超限：等待后重试
        if info in ["CUQPS_HAS_EXCEEDED_THE_LIMIT", "CKQPS_HAS_EXCEEDED_THE_LIMIT"] or infocode in ["10020", "10021"]:
            wait_seconds = 3 * attempt
            print(f"请求过快，等待 {wait_seconds} 秒后重试。keywords={keywords}, 第 {attempt}/{max_retries} 次")
            time.sleep(wait_seconds)
            continue

        # 其他错误直接报错
        raise RuntimeError(f"高德API请求失败：{data}")

    raise RuntimeError(f"多次重试后仍然失败：keywords={keywords}")


# =========================
# 3. 先查询厦门市下辖区列表
# =========================
city_data = amap_district_query(
    keywords=CITY_ADCODE,
    subdistrict=1,
    extensions="base"
)

city_info = city_data["districts"][0]
districts = city_info["districts"]

print("厦门市下辖区：")
for d in districts:
    print(d["name"], d["adcode"])


# =========================
# 4. 解析高德 polyline 为 Shapely 几何
# =========================
def parse_amap_polyline(polyline):
    """
    高德 polyline 格式通常为：
    lng,lat;lng,lat;...|lng,lat;lng,lat;...
    其中 | 表示多块面。
    """
    geom_list = []

    for part in polyline.split("|"):
        coords = []

        for pair in part.split(";"):
            pair = pair.strip()
            if not pair:
                continue

            lng, lat = pair.split(",")
            coords.append((float(lng), float(lat)))

        if len(coords) < 3:
            continue

        # 闭合多边形
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        polygon = Polygon(coords)

        # 修复可能的无效几何
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if not polygon.is_empty:
            if polygon.geom_type == "Polygon":
                geom_list.append(polygon)
            elif polygon.geom_type == "MultiPolygon":
                geom_list.extend(list(polygon.geoms))

    if not geom_list:
        return None

    return unary_union(geom_list)


# =========================
# 5. 逐区查询边界
# =========================
features = []

for d in districts:
    district_name = d["name"]
    adcode = d["adcode"]

    print(f"正在获取边界：{district_name} {adcode}")

    # 每次请求前暂停，避免触发 QPS 限制
    time.sleep(5)

    detail = amap_district_query(
        keywords=adcode,
        subdistrict=0,
        extensions="all"
    )

    info = detail["districts"][0]
    polyline = info.get("polyline", "")

    if not polyline:
        print(f"警告：{district_name} 没有返回 polyline")
        continue

    geom = parse_amap_polyline(polyline)

    if geom is None:
        print(f"警告：{district_name} 边界解析失败")
        continue

    features.append({
        "district": district_name,
        "adcode": adcode,
        "city": "厦门市",
        "citycode": "350200",
        "source": "AMap District API",
        "geometry": geom
    })


# =========================
# 6. 生成 GeoDataFrame
# =========================
gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

# 只保留需要字段
gdf = gdf[["district", "adcode", "city", "citycode", "source", "geometry"]]

print(gdf)
print(gdf.crs)


# =========================
# 7. 保存 GeoJSON 和 Shapefile
# =========================
geojson_path = out_dir / "xiamen_districts_amap.geojson"
shp_path = out_dir / "xiamen_districts_amap.shp"

gdf.to_file(geojson_path, driver="GeoJSON")
gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

print("GeoJSON 已保存：", geojson_path.resolve())
print("Shapefile 已保存：", shp_path.resolve())