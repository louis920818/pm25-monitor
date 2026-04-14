import streamlit as st
import pandas as pd
import requests
import folium

from streamlit_folium import st_folium
import numpy as np
from datetime import date, datetime
from sklearn.ensemble import RandomForestRegressor
import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings('ignore')

# ==========================================
# 網頁配置
# ==========================================
st.set_page_config(page_title="全台 PM2.5 監測系統", layout="wide")
st.title("全台 PM2.5 監測 x 預測 x 天氣因子整合系統")

# ==========================================
# 金鑰設定（從 .env 或環境變數讀取）
# ==========================================
CSV_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pm2.5Data Set')
MOENV_API_KEY      = st.secrets.get('MOENV_API_KEY', '') or os.environ.get('MOENV_API_KEY', '')
CWA_API_KEY        = st.secrets.get('CWA_API_KEY', '') or os.environ.get('CWA_API_KEY', '')
TELEGRAM_BOT_TOKEN = st.secrets.get('TELEGRAM_BOT_TOKEN', '') or os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID   = st.secrets.get('TELEGRAM_CHAT_ID', '') or os.environ.get('TELEGRAM_CHAT_ID', '')

COUNTY_COORDS = {
    '基隆市': [25.1276, 121.7391], '台北市': [25.0329, 121.5654], '新北市': [25.0115, 121.4615],
    '桃園市': [24.9936, 121.3009], '新竹縣': [24.8383, 121.0142], '新竹市': [24.8138, 120.9675],
    '苗栗縣': [24.5601, 120.8126], '台中市': [24.1477, 120.6736], '彰化縣': [24.0517, 120.5392],
    '南投縣': [23.9609, 120.9718], '雲林縣': [23.7092, 120.4313], '嘉義縣': [23.4518, 120.2554],
    '嘉義市': [23.4800, 120.4491], '台南市': [22.9997, 120.2270], '高雄市': [22.6272, 120.3014],
    '屏東縣': [22.6687, 120.4861], '宜蘭縣': [24.7021, 121.7321], '花蓮縣': [23.9871, 121.6015],
    '台東縣': [22.7972, 121.1444], '澎湖縣': [23.5711, 119.5793], '金門縣': [24.4410, 118.3280],
    '連江縣': [26.1505, 119.9334]
}
COUNTY_MAPPING = {
    '臺北市': '台北市', '臺中市': '台中市', '臺南市': '台南市',
    '臺東縣': '台東縣', '連江縣': '連江縣'
}

# ==========================================
# AQI 等級輔助函式（依台灣 EPA 標準）
# ==========================================
def get_aqi_info(pm25):
    """回傳 (hex顏色, 等級名稱, Folium圖釘顏色)"""
    if pm25 <= 15.4:
        return '#00e400', '良好',           'green'
    elif pm25 <= 35.4:
        return '#e6c000', '普通',           'beige'
    elif pm25 <= 54.4:
        return '#ff7e00', '對敏感族群不健康', 'orange'
    elif pm25 <= 150.4:
        return '#ff0000', '對所有族群不健康', 'red'
    elif pm25 <= 250.4:
        return '#8f3f97', '非常不健康',      'purple'
    else:
        return '#7e0023', '危害',           'darkred'

# ==========================================
# 從 CWA 取得即時天氣資料（風速、濕度）
# ==========================================
@st.cache_data(ttl=1800)  # 快取 30 分鐘
def get_cwa_weather():
    """從氣象署 API 取得各縣市平均風速與濕度"""
    if not CWA_API_KEY:
        return {}
    try:
        url = (
            f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001"
            f"?Authorization={CWA_API_KEY}&elementName=WDSD,HUMD"
        )
        resp = requests.get(url, timeout=10, verify=False).json()
        stations = resp.get('records', {}).get('Station', [])

        county_data = {}
        for s in stations:
            county = s.get('GeoInfo', {}).get('CountyName', '')
            county = COUNTY_MAPPING.get(county, county)
            elements = {e['ElementName']: e['ElementValue'] for e in s.get('WeatherElement', [])}
            try:
                wind  = float(elements.get('WDSD', -99))
                humid = float(elements.get('HUMD', -99)) * 100
                if wind < 0 or humid < 0:
                    continue
                if county not in county_data:
                    county_data[county] = {'wind': [], 'humidity': []}
                county_data[county]['wind'].append(wind)
                county_data[county]['humidity'].append(humid)
            except (ValueError, TypeError):
                continue

        return {
            county: {
                'WindSpeed': float(np.mean(v['wind'])),
                'Humidity':  float(np.mean(v['humidity']))
            }
            for county, v in county_data.items()
        }
    except Exception:
        return {}

# ==========================================
# CSV 讀取（從本機資料夾讀取）
# ==========================================
@st.cache_data
def load_csv():
    """自動讀取資料夾內所有 CSV 並合併"""
    csv_files = [
        os.path.join(CSV_FOLDER, f)
        for f in os.listdir(CSV_FOLDER)
        if f.lower().endswith('.csv')
    ]
    dfs = []
    for path in csv_files:
        try:
            dfs.append(pd.read_csv(path))
        except Exception:
            continue

    if not dfs:
        raise FileNotFoundError(f"在 {CSV_FOLDER} 找不到任何 CSV 檔案")

    df = pd.concat(dfs, ignore_index=True)

    if 'datacreationdate' in df.columns:
        df['time'] = pd.to_datetime(df['datacreationdate'], errors='coerce')
    elif 'publishtime' in df.columns:
        df['time'] = pd.to_datetime(df['publishtime'], errors='coerce')
    else:
        raise ValueError("找不到時間欄位 (datacreationdate 或 publishtime)")

    df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
    df = df.dropna(subset=['pm25', 'time'])
    df = df[(df['pm25'] >= 0) & (df['pm25'] <= 500)]
    df = df.drop_duplicates()
    return df

# ==========================================
# AI 模型訓練（使用真實時間特徵）
# ==========================================
@st.cache_resource
def train_ai_model():
    """訓練隨機森林預測模型"""
    try:
        df = load_csv()
        df['hour']      = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.dayofweek
        df['month']     = df['time'].dt.month

        # 天氣因子：優先使用 CWA 真實資料，否則以合理預設值替代
        weather = get_cwa_weather()
        np.random.seed(42)
        df['WindSpeed'] = df['county'].map(
            lambda c: weather.get(COUNTY_MAPPING.get(c, c), {}).get('WindSpeed', None)
        ).fillna(pd.Series(np.random.uniform(0.5, 5.0, len(df)), index=df.index))
        df['Humidity'] = df['county'].map(
            lambda c: weather.get(COUNTY_MAPPING.get(c, c), {}).get('Humidity', None)
        ).fillna(pd.Series(np.random.uniform(50, 100, len(df)), index=df.index))

        features = ['hour', 'dayofweek', 'month', 'WindSpeed', 'Humidity']
        model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(df[features], df['pm25'])

        county_base = df.groupby('county')['pm25'].mean().to_dict()
        return model, county_base
    except Exception as e:
        st.error(f"模型訓練失敗: {e}")
        return None, None

ai_model, county_averages = train_ai_model()

# ==========================================
# 核心邏輯：資料分流器
# ==========================================
def get_final_data(selected_date, selected_hour):
    today = date.today()

    # --- 模式 A: 未來（AI 預測）---
    if selected_date > today:
        if ai_model is None or county_averages is None:
            st.error("AI 模型未載入，無法進行預測。請確認 CSV 檔案路徑是否正確。")
            return None

        st.sidebar.warning("AI 預測模式")
        weather = get_cwa_weather()
        predictions = []
        for county in COUNTY_COORDS:
            w = weather.get(county, {})
            wind = w.get('WindSpeed', float(np.random.uniform(1.0, 4.0)))
            hum  = w.get('Humidity',  float(np.random.uniform(60, 90)))

            test_X = pd.DataFrame(
                [[selected_hour, selected_date.weekday(), selected_date.month, wind, hum]],
                columns=['hour', 'dayofweek', 'month', 'WindSpeed', 'Humidity']
            )
            base = county_averages.get(county, county_averages.get('台北市', 20))
            pred = ai_model.predict(test_X)[0] * (base / max(np.mean(list(county_averages.values())), 1))
            pred = max(0.0, pred)
            predictions.append({
                'county_zh': county,
                'pm25': round(pred, 1),
                'WindSpeed': round(wind, 1),
                'Humidity': round(hum, 1),
                'type': 'AI 預測值'
            })
        return pd.DataFrame(predictions)

    # --- 模式 B: 今天（即時 API）---
    elif selected_date == today:
        st.sidebar.success("即時監測模式")
        if not MOENV_API_KEY:
            st.error("未設定 MOENV_API_KEY，無法取得即時資料。")
            return None
        try:
            url = f"https://data.moenv.gov.tw/api/v2/aqx_p_432?api_key={MOENV_API_KEY}"
            resp = requests.get(url, timeout=10, verify=False).json()
            records = resp['records'] if isinstance(resp, dict) and 'records' in resp else resp
            df = pd.DataFrame(records)
            pm_col = 'pm2.5' if 'pm2.5' in df.columns else 'pm25'
            df['pm25'] = pd.to_numeric(df[pm_col], errors='coerce')
            df['county_zh'] = df['county'].replace(COUNTY_MAPPING)
            result = df.groupby('county_zh').agg({'pm25': 'mean'}).reset_index()
            result['pm25'] = result['pm25'].round(1)
            result['type'] = '即時觀測值'
            return result
        except Exception as e:
            st.error(f"即時 API 取得失敗: {e}")
            return None

    # --- 模式 C: 過去（歷史 CSV，找不到則自動從 API 下載）---
    else:
        st.sidebar.info("歷史資料模式 (CSV)")
        try:
            df = load_csv()
            df_day = df[
                (df['time'].dt.date == selected_date) &
                (df['time'].dt.hour == selected_hour)
            ].copy()

            # CSV 有資料 → 直接回傳
            if not df_day.empty:
                df_day['county_zh'] = df_day['county'].replace(COUNTY_MAPPING)
                result = df_day.groupby('county_zh').agg({'pm25': 'mean'}).reset_index()
                result['pm25'] = result['pm25'].round(1)
                result['type'] = '歷史觀測值'
                return result

            # CSV 沒有資料 → 嘗試從 MOENV aqx_p_02 下載歷史資料
            if not MOENV_API_KEY:
                min_date = df['time'].dt.date.min()
                max_date = df['time'].dt.date.max()
                st.error(
                    f"找不到 **{selected_date} {selected_hour:02d}:00** 的資料。\n\n"
                    f"目前 CSV 資料範圍：**{min_date}　～　{max_date}**\n\n"
                    "未設定 MOENV_API_KEY，無法自動下載。請在 `.env` 填入 API 金鑰。"
                )
                return None

            date_str = selected_date.strftime('%Y-%m-%d')
            hour_str = f"{selected_hour:02d}"

            with st.spinner(f"雲端無 {selected_date} 的資料，嘗試從環境部 API 自動下載..."):
                # aqx_p_02 = 空氣品質監測時值（支援歷史查詢）
                # MOENV API v2 已停止支援 GTE/LTE，改用 EQ 精確比對整點時間
                filters = f"datacreationdate,EQ,{date_str} {hour_str}:00"
                url = (
                    f"https://data.moenv.gov.tw/api/v2/aqx_p_02"
                    f"?api_key={MOENV_API_KEY}"
                    f"&filters={filters}"
                    f"&limit=1000"
                )
                try:
                    resp = requests.get(url, timeout=20, verify=False)
                except requests.exceptions.Timeout:
                    st.error("API 請求逾時，請稍後再試。")
                    return None
                except requests.exceptions.RequestException as e:
                    st.error(f"API 請求失敗: {e}")
                    return None

                # 檢查回應是否有效
                if resp.status_code != 200:
                    st.error(f"API 回傳錯誤狀態碼: {resp.status_code}")
                    return None
                if not resp.text.strip():
                    st.error(
                        f"環境部 API 無 {date_str} {hour_str}:00 的歷史資料。\n\n"
                        "可能原因：該日期資料尚未收錄，或超出 API 保存範圍。\n\n"
                        "請至 [環境部開放資料平台](https://data.moenv.gov.tw/dataset/detail/aqx_p_02) "
                        "手動下載 CSV 並上傳至 Google Drive。"
                    )
                    return None

                try:
                    data_json = resp.json()
                except ValueError:
                    st.error("API 回傳格式異常（無法解析 JSON）。")
                    st.code(resp.text[:1000], language=None)  # 顯示實際回傳內容供除錯
                    return None

                records = data_json.get('records', data_json) if isinstance(data_json, dict) else data_json
                if not records:
                    st.warning(
                        f"環境部 API 查無 {date_str} {hour_str}:00 的資料。\n\n"
                        "請至 [環境部開放資料平台](https://data.moenv.gov.tw/dataset/detail/aqx_p_02) "
                        "手動下載 CSV 並上傳至 Google Drive。"
                    )
                    return None

                new_df = pd.DataFrame(records)
                if new_df.empty:
                    st.warning("API 回傳空資料集。")
                    return None

                # 儲存到本機
                save_path = os.path.join(CSV_FOLDER, f"PM2.5_{date_str}.csv")
                if os.path.exists(save_path):
                    existing_df = pd.read_csv(save_path)
                    new_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
                new_df.to_csv(save_path, index=False)

                load_csv.clear()
                st.sidebar.success(f"已自動下載並儲存 {date_str} {hour_str}:00 的資料！")

                # 處理欄位並回傳
                pm_col = next((c for c in new_df.columns if c.lower() in ('pm2.5', 'pm25')), None)
                if pm_col is None:
                    st.error(f"API 回傳資料中找不到 PM2.5 欄位，現有欄位：{list(new_df.columns)}")
                    return None

                new_df['pm25'] = pd.to_numeric(new_df[pm_col], errors='coerce')
                new_df['county_zh'] = new_df['county'].replace(COUNTY_MAPPING)
                result = new_df.groupby('county_zh').agg({'pm25': 'mean'}).reset_index()
                result['pm25'] = result['pm25'].round(1)
                result['type'] = '歷史觀測值 (API 自動下載)'
                return result

        except Exception as e:
            st.error(f"歷史資料讀取失敗: {e}")
            return None

# ==========================================
# UI 介面
# ==========================================
today = date.today()

st.sidebar.header("參數設定")
target_date = st.sidebar.date_input("選擇觀測/預測日期", value=today)

if target_date == today:
    target_hour = datetime.now().hour
    st.sidebar.info(f"即時模式：目前時間 {target_hour:02d}:00")
else:
    target_hour = st.sidebar.slider("選擇時間（0～23 點）", min_value=0, max_value=23, value=12, step=1)

# 即時模式提供刷新按鈕
if target_date == today:
    if st.sidebar.button("刷新即時資料"):
        st.cache_data.clear()
        st.rerun()

data = get_final_data(target_date, target_hour)

if data is not None:
    # --- 頂部看板 ---
    st.markdown("### 系統數據總覽")
    c1, c2, c3, c4 = st.columns(4)
    avg_pm25 = data['pm25'].mean()
    _, level, _ = get_aqi_info(avg_pm25)
    max_row = data.loc[data['pm25'].idxmax()]

    c1.metric("查詢時間",        f"{target_date} {target_hour:02d}:00")
    c2.metric("全台平均 PM2.5",  f"{avg_pm25:.1f} µg/m³")
    c3.metric("空氣品質等級",    level)
    c4.metric("最高濃度縣市",    f"{max_row['county_zh']} ({max_row['pm25']} µg/m³)")

    # --- 地圖 ---
    st.markdown("### 空氣品質動態分佈圖")

    # AQI 圖例 HTML（顏色與 Folium 圖釘一致）
    # green=#72b026, beige=#ffcb92, orange=#f69730, red=#d63e2a, purple=#d152b8, darkred=#a23336, gray=#575757
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:8px;
                border:1px solid #ccc; font-size:13px; line-height:2;">
        <b>PM2.5 等級</b><br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#72b026;vertical-align:middle;margin-right:6px;"></span>≤15.4　良好<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#ffcb92;vertical-align:middle;margin-right:6px;"></span>≤35.4　普通<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#f69730;vertical-align:middle;margin-right:6px;"></span>≤54.4　對敏感族群不健康<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#d63e2a;vertical-align:middle;margin-right:6px;"></span>≤150.4　對所有族群不健康<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#d152b8;vertical-align:middle;margin-right:6px;"></span>≤250.4　非常不健康<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#a23336;vertical-align:middle;margin-right:6px;"></span>&gt;250.4　危害<br>
        <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#575757;vertical-align:middle;margin-right:6px;"></span>無資料
    </div>
    """

    # 建立資料查詢字典：county_zh → row（排除 pm25 為 NaN 的資料）
    data_dict = {
        row['county_zh']: row
        for _, row in data.iterrows()
        if pd.notna(row['pm25'])
    }

    m = folium.Map(location=[23.6978, 120.9605], zoom_start=7)
    m.get_root().html.add_child(folium.Element(legend_html))

    # 確保全部 22 個縣市都有圖釘
    for name, coords in COUNTY_COORDS.items():
        if name in data_dict:
            row = data_dict[name]
            val = row['pm25']
            hex_color, level_name, pin_color = get_aqi_info(val)

            extra = ""
            if 'WindSpeed' in row and pd.notna(row.get('WindSpeed')):
                extra = f"<br>風速: {row['WindSpeed']} m/s　濕度: {row['Humidity']}%"

            popup_html = (
                f"<b>{name}</b><br>"
                f"PM2.5: <b>{val} µg/m³</b><br>"
                f"等級: <span style='color:{hex_color};font-weight:bold'>{level_name}</span>"
                f"{extra}"
            )
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{name}　{val} µg/m³　{level_name}",
                icon=folium.Icon(color=pin_color, icon='map-marker', prefix='fa')
            ).add_to(m)
        else:
            # 無資料縣市（API 未回傳或值為 NaN）：灰色圖釘
            folium.Marker(
                location=coords,
                popup=folium.Popup(f"<b>{name}</b><br>目前無資料", max_width=150),
                tooltip=f"{name}　無資料",
                icon=folium.Icon(color='gray', icon='map-marker', prefix='fa')
            ).add_to(m)

    st_folium(m, width="100%", height=520, returned_objects=[])

    # --- 各縣市 PM2.5 長條圖 ---
    st.markdown("### 各縣市 PM2.5 濃度比較")
    chart_data = data.set_index('county_zh')[['pm25']].sort_values('pm25', ascending=False)
    st.bar_chart(chart_data)

    # --- 告警機制 ---
    st.sidebar.markdown("---")
    st.sidebar.header("告警機制")

    threshold = st.sidebar.slider("告警閾值 (µg/m³)", min_value=10, max_value=100, value=35, step=5)

    if st.sidebar.button("發送告警至手機"):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            st.sidebar.error("未設定 Telegram 環境變數，無法發送通知。")
        else:
            high_risk = data[data['pm25'] > threshold].sort_values('pm25', ascending=False)
            if not high_risk.empty:
                lines = "\n".join([
                    f"📍 {r['county_zh']}: {r['pm25']} µg/m³ ({get_aqi_info(r['pm25'])[1]})"
                    for _, r in high_risk.iterrows()
                ])
                msg = (
                    f"【PM2.5 預警通報】\n"
                    f"時間：{target_date} {target_hour:02d}:00\n"
                    f"資料來源：{data['type'].iloc[0]}\n"
                    f"超標測站（>{threshold} µg/m³）：\n{lines}\n"
                    f"全台平均：{avg_pm25:.1f} µg/m³"
                )
                try:
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                        timeout=10,
                        verify=False
                    )
                    st.sidebar.success(f"告警已發送！共 {len(high_risk)} 個縣市超標。")
                except Exception as e:
                    st.sidebar.error(f"發送失敗: {e}")
            else:
                st.sidebar.info(f"所有縣市 PM2.5 均在 {threshold} µg/m³ 以下，無須告警。")

    # --- 原始資料表 ---
    with st.expander("檢視原始數據表"):
        st.dataframe(data, use_container_width=True)

else:
    if target_date > today:
        st.error("AI 預測失敗，請確認 CSV 檔案路徑與模型訓練是否成功。")
    elif target_date == today:
        st.error("即時資料取得失敗，請確認 MOENV_API_KEY 是否正確設定。")
