import time
import re

# ----------------------------------------------------
# 3. AI REASONING ENGINE (CÓ CACHE VÀ RETRY CHỐNG TIMEOUT)
# ----------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_ai_trading_plan(data: dict) -> dict:
    prompt = f"""
Bạn là Chuyên gia Tư vấn Đầu tư Chứng khoán Cao cấp (CMT/CFA). Hãy phân tích kỹ thuật và đưa ra khuyến nghị trading kỷ luật, thực chiến cho mã {data['symbol']}.

[DỮ LIỆU THỊ TRƯỜNG]:
- Giá hiện tại: {data['current_price']:,.0f} VNĐ ({data['percent_change']:+.2f}%)
- Khối lượng phiên: {data['volume']:,} CP | Khối lượng TB 20 phiên: {data['avg_vol_20']:,} CP
- RSI (14): {data['rsi']}
- SMA20: {data['sma20']:,.0f} | SMA50: {data['sma50']:,.0f}
- Vùng Hỗ trợ 20 phiên: {data['support_20']:,.0f} VNĐ
- Vùng Kháng cự 20 phiên: {data['resistance_20']:,.0f} VNĐ

[YÊU CẦU ĐẦU RA]:
Trả về DUY NHẤT 1 JSON Object hợp lệ (không markdown bọc ngoài) theo format:
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

    # Cơ chế Retry 2 lần với Timeout 60s
    max_retries = 2
    for attempt in range(max_retries):
        try:
            resp = requests.post(GEMINI_URL, json=payload, timeout=60)
            res_json = resp.json()
            
            if "error" in res_json:
                return {"error": res_json["error"].get("message", "Lỗi Gemini API")}
                
            text_resp = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Làm sạch chuỗi JSON nếu có ký tự markdown
            clean_text = re.sub(r"^```json\s*|\s*```$", "", text_resp, flags=re.MULTILINE).strip()
            return json.loads(clean_text)
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)  # Đợi 2s rồi thử lại
                continue
            return {"error": "Máy chủ AI phản hồi quá chậm (Timeout). Vui lòng bấm 'Phân Tích Kỹ Thuật' lại lần nữa."}
        except Exception as e:
            return {"error": f"Không thể xử lý khuyến nghị từ AI: {str(e)}"}
