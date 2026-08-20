import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="VN Stock AI Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 1. SECURITY & API CONFIGURATION
# ----------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets!")
    st.stop()

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

# ----------------------------------------------------
# 2. DATA ENGINE VỚI CACHE & FAILOVER
# ----------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def get_stock_data(symbol: str):
    """
    Lấy dữ liệu nến 60 phiên gần nhất.
    1. Ưu tiên: TCBS Endpoint
    2. Fallback: vnstock library
    """
    sym = symbol.upper().strip()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    df = None

    # --- CÁCH 1: TCBS API ---
    try:
        url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={sym}&type=stock&resolution=D"
        res = requests.get(url, headers=headers, timeout=5).json()
        if "data" in res and len(res["data"]) >= 20:
            df = pd.DataFrame(res["data"])
            df = df.rename(columns={"tradingDate": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
            # Chuẩn hóa về đơn vị đồng
            for col in ["open", "high", "low", "close"]:
                if df[col].iloc[-1] < 1000:
                    df[col] = df[col] * 1000
    except Exception:
        df = None

    # --- CÁCH 2: FALLBACK QUA VNSTOCK NẾU TCBS THẤT BẠI ---
    if df is None or df.empty or len(df) < 20:
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=sym, source='VCI')
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
            raw_df = stock.quote.history(start=start_date, end=end_date)
            if raw_df is not None and not raw_df.empty:
                df = raw_df.rename(columns={"time": "date", "match_price": "close"}).copy()
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
                for col in ["open", "high", "low", "close"]:
                    if df[col].iloc[-1] < 1000:
                        df[col] = df[col] * 1000
        except Exception as e:
            return {"success": False, "error": f"Lỗi truy xuất toàn bộ nguồn dữ liệu: {str(e)}"}

    if df is None or len(df) < 20:
        return {"success": False, "error": f"Không đủ dữ liệu lịch sử để phân tích cho mã {sym}."}

    # --- TÍNH TOÁN CÁC CHỈ BÁO KỸ THUẬT CƠ BẢN ---
    df["SMA20"] = df["close"].rolling(window=20).mean()
    df["SMA50"] = df["close"].rolling(window=50).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    curr_price = float(latest["close"])
    change = curr_price - float(prev["close"])
    pct_change = (change / float(prev["close"])) * 100

    return {
        "success": True,
        "symbol": sym,
        "current_price": curr_price,
        "change": change,
        "percent_change": pct_change,
        "volume": int(latest["volume"]),
        "avg_vol_20": int(df["volume"].tail(20).mean()),
        "rsi": round(float(latest["RSI"]), 1) if not pd.isna(latest["RSI"]) else 50.0,
        "sma20": round(float(latest["SMA20"]), 0) if not pd.isna(latest["SMA20"]) else curr_price,
        "sma50": round(float(latest["SMA50"]), 0) if not pd.isna(latest["SMA50"]) else curr_price,
        "support_20": float(df["low"].tail(20).min()),
        "resistance_20": float(df["high"].tail(20).max()),
        "history_df": df.tail(40)
    }

# ----------------------------------------------------
# 3. AI REASONING ENGINE (GEMINI 3.6 / FLASH)
# ----------------------------------------------------
def get_ai_trading_plan(data: dict) -> dict:
    prompt = f"""
Bạn là Chuyên gia Tư vấn Đầu tư Chứng khoán Cao cấp (CMT/CFA). Hãy phân tích kỹ thuật và đưa ra khuyến nghị trading kỷ luật, thực chiến cho mã {data['symbol']}.

[DỮ LIỆU THỊ TRƯỜNG HIỆN TẠI]:
- Giá hiện tại: {data['current_price']:,.0f} VNĐ ({data['percent_change']:+.2f}%)
- Khối lượng phiên: {data['volume']:,} CP | Khối lượng TB 20 phiên: {data['avg_vol_20']:,} CP
- RSI (14): {data['rsi']}
- SMA20: {data['sma20']:,.0f} | SMA50: {data['sma50']:,.0f}
- Vùng Hỗ trợ 20 phiên: {data['support_20']:,.0f} VNĐ
- Vùng Kháng cự 20 phiên: {data['resistance_20']:,.0f} VNĐ

[YÊU CẦU ĐẦU RA]:
Trả về DUY NHẤT 1 JSON Object hợp lệ (không markdown, không bọc ```json) theo format sau:
{{
  "action": "MUA MỚI" | "MUA GIA TĂNG" | "NẮM GIỮ" | "BÁN HẠ TỶ TRỌNG" | "BÁN CẮT LỖ" | "THEO DÕI",
  "buy_zone": "Mức giá hoặc khoảng giá mua tối ưu (VNĐ)",
  "target_price": "Mục tiêu chốt lời ngắn - trung hạn (kèm % kỳ vọng)",
  "stop_loss": "Mức giá cắt lỗ nghiêm ngặt (kèm % rủi ro)",
  "risk_reward_ratio": "Tỷ lệ R:R (VD: 1:2.5)",
  "trend_weekly": "TĂNG" | "GIẢM" | "ĐI NGANG (TÍCH LŨY)",
  "trend_monthly": "TĂNG" | "GIẢM" | "ĐI NGANG (TÍCH LŨY)",
  "catalysts": [
    "Đánh giá chi tiết về dòng tiền (Volume so với TB 20 phiên)",
    "Đánh giá cấu trúc xu hướng giá so với SMA20, SMA50 và chỉ số RSI",
    "Kế hoạch đi lệnh và quản trị rủi ro cụ thể"
  ],
  "capital_allocation": "Tỷ trọng khuyến nghị giải ngân (% tổng tài khoản)"
}}
"""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
    }

    try:
        resp = requests.post(GEMINI_URL, json=payload, timeout=25)
        res_json = resp.json()
        if "error" in res_json:
            return {"error": res_json["error"].get("message", "Lỗi Gemini API")}
        text_resp = res_json["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_resp)
    except Exception as e:
        return {"error": f"Không thể xử lý khuyến nghị từ AI: {str(e)}"}

# ----------------------------------------------------
# 4. GIAO DIỆN STREAMLIT APPLICATION
# ----------------------------------------------------
st.title("📈 Trợ Lý Tư Vấn Xu Hướng Chứng Khoán VN")
st.caption("Dữ liệu trực tiếp kết hợp thuật toán phân tích kỹ thuật và AI Gemini")

# Sidebar cấu hình
with st.sidebar:
    st.header("⚙️ Tra Cứu Cổ Phiếu")
    ticker_input = st.text_input("Nhập mã CK (3 chữ cái):", value="HPG").upper().strip()
    cache_ttl_info = st.info("⏱️ Dữ liệu được cache 120s để tối ưu tốc độ và tránh nghẽn lệnh.")
    submit_btn = st.button("🚀 Phân Tích Kỹ Thuật", type="primary", use_container_width=True)

if ticker_input:
    with st.spinner(f"Đang đồng bộ dữ liệu sàn và phân tích mã {ticker_input}..."):
        market_data = get_stock_data(ticker_input)

        if not market_data.get("success"):
            st.error(market_data.get("error", "Lỗi không xác định."))
            st.stop()

        ai_plan = get_ai_trading_plan(market_data)
        if "error" in ai_plan:
            st.error(ai_plan["error"])
            st.stop()

        # Top KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Giá Khớp Lệnh", f"{market_data['current_price']:,.0f} đ", f"{market_data['percent_change']:+.2f}%")
        m2.metric("Khối Lượng Khớp", f"{market_data['volume']:,} CP")
        m3.metric("RSI (14)", f"{market_data['rsi']}")
        m4.metric("Hỗ Trợ (20P)", f"{market_data['support_20']:,.0f} đ")
        m5.metric("Kháng Cự (20P)", f"{market_data['resistance_20']:,.0f} đ")

        st.markdown("---")

        # Bảng Action Plan
        st.subheader("🎯 Kế Hoạch Giao Dịch Khuyến Nghị")
        
        action = ai_plan.get("action", "THEO DÕI")
        badge_color = {
            "MUA MỚI": "#28a745",
            "MUA GIA TĂNG": "#17a2b8",
            "NẮM GIỮ": "#007bff",
            "BÁN HẠ TỶ TRỌNG": "#fd7e14",
            "BÁN CẮT LỖ": "#dc3545",
            "THEO DÕI": "#6c757d"
        }.get(action, "#6c757d")

        col_act, col_buy, col_tp, col_sl = st.columns(4)
        col_act.markdown(f"**Khuyến Nghị Hành Động**<br><h2 style='color:{badge_color}; margin:0;'>{action}</h2>", unsafe_allow_html=True)
        col_buy.markdown(f"**Vùng Mua Tối Ưu**<br><h4>{ai_plan.get('buy_zone')}</h4>", unsafe_allow_html=True)
        col_tp.markdown(f"**Mục Tiêu Giá (Target)**<br><h4 style='color:#28a745;'>{ai_plan.get('target_price')}</h4>", unsafe_allow_html=True)
        col_sl.markdown(f"**Cắt Lỗ (Stoploss)**<br><h4 style='color:#dc3545;'>{ai_plan.get('stop_loss')}</h4>", unsafe_allow_html=True)

        st.markdown("---")

        # Chi tiết xu hướng & Luận điểm
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("📊 Xu Hướng Đa Khung Thời Gian")
            trend_table = pd.DataFrame([
                {"Khung thời gian": "Ngắn hạn (Tuần)", "Dự báo": ai_plan.get("trend_weekly")},
                {"Khung thời gian": "Trung hạn (Tháng)", "Dự báo": ai_plan.get("trend_monthly")},
                {"Khung thời gian": "Tỷ Lệ Risk:Reward", "Dự báo": ai_plan.get("risk_reward_ratio")},
                {"Khung thời gian": "Tỷ trọng giải ngân đề xuất", "Dự báo": ai_plan.get("capital_allocation")}
            ])
            st.table(trend_table)

        with right_col:
            st.subheader("💡 Luận Điểm & Kỷ Luật Giao Dịch")
            for c in ai_plan.get("catalysts", []):
                st.markdown(f"- {c}")

        # Biểu đồ nến tương tác Plotly
        st.subheader("📉 Biểu Đồ Kỹ Thuật 40 Phiên Gần Nhất")
        df_hist = market_data["history_df"]
        fig = go.Figure(data=[
            go.Candlestick(
                x=df_hist['date'],
                open=df_hist['open'],
                high=df_hist['high'],
                low=df_hist['low'],
                close=df_hist['close'],
                name='Giá Nến'
            ),
            go.Scatter(x=df_hist['date'], y=df_hist['SMA20'], line=dict(color='orange', width=1.5), name='SMA 20'),
            go.Scatter(x=df_hist['date'], y=df_hist['SMA50'], line=dict(color='blue', width=1.5), name='SMA 50')
        ])
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)