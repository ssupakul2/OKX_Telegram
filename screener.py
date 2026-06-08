import os
import time
import logging
import requests
import pandas as pd
import numpy as np

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==========================================
# Environment Variables
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "FLOKI", "SHIB", "EIGEN", "OP", "DOGE", "NEAR",
    "TRX", "AVAX", "SUI"
]

# ==========================================
# Constants & Hyperparameters
# ==========================================
API_RATE_LIMIT_DELAY = 0.35
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2.0
RSI_PERIOD = 14
EMA_SHORT = 50
EMA_LONG = 200
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 70

# --- RSI Recovery & Pullback Configuration ---
RSI_RECOVERY_THRESHOLD = 45
RSI_PULLBACK_THRESHOLD = 55
RSI_RECOVERY_LOOKBACK = 5

# --- Divergence Configuration ---
RSI_BULL_DIV_MAX = 45
RSI_BEAR_DIV_MIN = 55
LOOKBACK_BARS = 15
LOOKBACK_SKIP_BARS = 3

# --- Trend Continuity Configuration ---
TREND_SLOPE_BARS = 5          
TREND_MIN_CONSECUTIVE = 3     

# --- RSI Bounce Configuration ---
RSI_BOUNCE_CONFIRM_BARS = 1   
RSI_BOUNCE_MIN_RISE = 1.5     

# --- Order Block (SMC) Configuration ---
OB_LOOKBACK = 20              
OB_IMBALANCE_RATIO = 1.1      

# --- Take Profit Tiers ---
TP_TIERS = {
    "major":  {"tp1": 0.08, "tp2": 0.12, "sl_buffer": 0.02},
    "mid":    {"tp1": 0.12, "tp2": 0.15, "sl_buffer": 0.025},
    "small":  {"tp1": 0.18, "tp2": 0.25, "sl_buffer": 0.03},
}

COIN_TIER = {
    "BTC": "major", "ETH": "major",
    "BNB": "mid",   "SOL": "mid",   "XRP": "mid",
    "ADA": "mid",   "NEAR": "mid",  "OP": "mid",
    "AVAX": "mid",  "SUI": "mid",   "TRX": "mid",
    "FLOKI": "small","SHIB": "small","EIGEN": "small","DOGE": "small",
}


# ==========================================
# Telegram Integration
# ==========================================
def send_telegram_messages(chunks: list) -> None:
    token = str(TELEGRAM_BOT_TOKEN or "").strip()
    chat_id = str(TELEGRAM_CHAT_ID or "").strip()

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID ไม่ได้ตั้งค่าใน Environment Variables")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for idx, chunk in enumerate(chunks, start=1):
        if not chunk.strip():
            continue

        payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"Telegram ส่งสำเร็จ (ส่วน {idx}/{len(chunks)})")
            else:
                logger.warning(f"Telegram ส่งล้มเหลว (ส่วน {idx}): {resp.text}")
        except Exception as e:
            logger.error(f"Exception ขณะส่ง Telegram (ส่วน {idx}): {e}")

        if idx < len(chunks):
            time.sleep(0.5)


# ==========================================
# OKX Data Fetching (Direct 4H Candles)
# ==========================================
def get_historical_data(coin: str) -> pd.DataFrame | None:
    url = "https://www.okx.com/api/v5/market/candles"
    inst_id = f"{coin}-USDT"
    
    params = {
        "instId": inst_id,
        "bar": "4H",      # OKX บังคับใช้ตัวพิมพ์เล็กสำหรับหน่วยเวลาชั่วโมง
        "limit": "100"    # ลิมิตสูงสุดต่อ 1 Request ของ OKX (100 แท่ง เพียงพอต่อการหา EMA200 ด้วยวิธี EWM)
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            data = resp.json()

            if data.get("code") == "0" and "data" in data:
                raw_candles = data["data"]
                if not raw_candles:
                    logger.warning(f"{coin}: OKX ไม่มีข้อมูลตอบกลับใน Array")
                    return None
                
                columns = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
                df = pd.DataFrame(raw_candles, columns=columns)
                
                for col in ["open", "high", "low", "close", "volCcy"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # OKX ส่งข้อมูลเรียงจาก ใหม่ -> เก่า ต้องเรียงลำดับดัชนีเวลาใหม่จาก เก่า -> ใหม่ เพื่อนำไปคำนวณ Indicator
                df["time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
                df.set_index("time", inplace=True)
                df.sort_index(ascending=True, inplace=True)
                
                # ในฝั่ง Spot ของ OKX คอลัมน์ volCcy คือจำนวนมูลค่าเงินฝั่ง Quote currency (USDT)
                df.rename(columns={"volCcy": "volumeto"}, inplace=True)
                df = df[["open", "high", "low", "close", "volumeto"]]

                logger.info(f"{coin}: OKX ดึงข้อมูล 4H สำเร็จ ({len(df)} แท่ง)")
                return df
            else:
                logger.warning(f"{coin} attempt {attempt}: OKX API ตอบกลับผิดปกติ – {data.get('msg')}")

        except requests.exceptions.Timeout:
            logger.warning(f"{coin} attempt {attempt}: Request timeout")
        except Exception as e:
            logger.warning(f"{coin} attempt {attempt}: {e}")

        if attempt < API_MAX_RETRIES:
            time.sleep(API_RETRY_DELAY * attempt)

    logger.error(f"{coin}: ดึงข้อมูลจาก OKX ล้มเหลวทั้ง {API_MAX_RETRIES} ครั้ง")
    return None


# ==========================================
# Technical Indicators
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    df["EMA_50"] = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=EMA_LONG, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

    df["VOL_MA20"] = df["volumeto"].rolling(20).mean()

    return df


# ==========================================
# Advanced Analysis Modules
# ==========================================
def analyze_trend_continuity(df: pd.DataFrame) -> dict:
    result = {
        "ema50_slope_pct": 0.0,
        "ema200_slope_pct": 0.0,
        "ema50_trending_up": False,
        "ema200_trending_up": False,
        "consecutive_up": 0,
        "consecutive_down": 0,
        "trend_strength": "sideways",
        "trend_label": "↔️ ไม่ชัดเจน",
    }

    n = TREND_SLOPE_BARS
    if len(df) < n + 2:
        return result

    ema50_now  = df["EMA_50"].iloc[-1]
    ema50_prev = df["EMA_50"].iloc[-(n + 1)]
    ema200_now  = df["EMA_200"].iloc[-1]
    ema200_prev = df["EMA_200"].iloc[-(n + 1)]

    slope50  = ((ema50_now  - ema50_prev)  / ema50_prev)  * 100 if ema50_prev  != 0 else 0
    slope200 = ((ema200_now - ema200_prev) / ema200_prev) * 100 if ema200_prev != 0 else 0

    result["ema50_slope_pct"]  = round(slope50,  4)
    result["ema200_slope_pct"] = round(slope200, 4)
    result["ema50_trending_up"]  = slope50  > 0
    result["ema200_trending_up"] = slope200 > 0

    closes = df["close"].iloc[-20:]
    diffs  = closes.diff().iloc[1:]

    up_streak = 0
    dn_streak = 0
    for val in reversed(diffs.values):
        if val > 0:
            if dn_streak == 0:
                up_streak += 1
            else:
                break
        elif val < 0:
            if up_streak == 0:
                dn_streak += 1
            else:
                break
        else:
            break

    result["consecutive_up"]   = up_streak
    result["consecutive_down"] = dn_streak

    both_up   = result["ema50_trending_up"] and result["ema200_trending_up"]
    both_down = (not result["ema50_trending_up"]) and (not result["ema200_trending_up"])
    strong_streak = TREND_MIN_CONSECUTIVE

    if both_up and up_streak >= strong_streak:
        strength = "strong_up"
        label = f"🚀 ขาขึ้นต่อเนื่องแข็งแกร่ง ({up_streak} แท่ง, EMA ชันขึ้นทั้งคู่)"
    elif result["ema50_trending_up"] and up_streak >= 1:
        strength = "moderate_up"
        label = f"📈 ขาขึ้นปานกลาง ({up_streak} แท่ง, EMA50 ชันขึ้น)"
    elif both_down and dn_streak >= strong_streak:
        strength = "strong_down"
        label = f"🔻 ขาลงต่อเนื่องแข็งแกร่ง ({dn_streak} แท่ง, EMA ชันลงทั้งคู่)"
    elif (not result["ema50_trending_up"]) and dn_streak >= 1:
        strength = "moderate_down"
        label = f"📉 ขาลงปานกลาง ({dn_streak} แท่ง, EMA50 ชันลง)"
    else:
        strength = "sideways"
        label = "↔️ Sideways / แนวโน้มไม่ชัด"

    result["trend_strength"] = strength
    result["trend_label"]    = label

    return result


def analyze_rsi_bounce(df: pd.DataFrame) -> dict:
    window = LOOKBACK_BARS

    result = {
        "touched_oversold": False,
        "rsi_low": None,
        "rsi_rise": 0.0,
        "consecutive_rise": 0,
        "below_midline": False,
        "quality": "none",
        "quality_label": "⬜ ไม่มีสัญญาณดีดกลับ",
        "entry_timing": "",
    }

    if len(df) < window + RSI_BOUNCE_CONFIRM_BARS + 2:
        return result

    rsi_series = df["RSI"].iloc[-(window + 1):-1]
    rsi_curr   = df["RSI"].iloc[-1]

    rsi_min = rsi_series.min()
    touched_oversold = rsi_min <= RSI_OVERSOLD

    result["touched_oversold"] = touched_oversold
    result["rsi_low"] = round(rsi_min, 2)

    if not touched_oversold:
        return result

    rsi_rise = rsi_curr - rsi_min
    result["rsi_rise"] = round(rsi_rise, 2)

    recent_rsi = df["RSI"].iloc[-(RSI_BOUNCE_CONFIRM_BARS + 3):]
    rsi_diffs  = recent_rsi.diff().iloc[1:]
    consec = 0
    for val in reversed(rsi_diffs.values):
        if val > 0:
            consec += 1
        else:
            break
    result["consecutive_rise"] = consec

    below_midline = rsi_curr < 50
    result["below_midline"] = below_midline

    recent_recovery_zone = df["RSI"].iloc[-RSI_RECOVERY_LOOKBACK:]
    has_recovered = (recent_recovery_zone >= RSI_RECOVERY_THRESHOLD).any()

    score = 0
    if rsi_rise >= RSI_BOUNCE_MIN_RISE:     score += 1
    if consec >= RSI_BOUNCE_CONFIRM_BARS:   score += 1
    if below_midline or has_recovered:       score += 1

    if score == 3:
        quality = "strong"
        label   = (
            f"✅ ดีดกลับแข็งแกร่ง (จากต่ำสุด {result['rsi_low']:.1f} → ขึ้น {rsi_rise:.1f} จุด, "
            f"{consec} แท่งติด, ยืนยันโซนฟื้นตัว)"
        )
        timing  = "⭐ จังหวะเข้าซื้อดีที่สุด: RSI ดีดกลับจาก Oversold อย่างมีคุณภาพและผ่านเกณฑ์ฟื้นตัว"
    elif score == 2:
        quality = "moderate"
        label   = (
            f"🟡 ดีดกลับปานกลาง (จากต่ำสุด {result['rsi_low']:.1f} → ขึ้น {rsi_rise:.1f} จุด, "
            f"{consec} แท่งติด)"
        )
        timing  = "⚡ พิจารณาเข้าซื้อได้ แต่ควรรอยืนยันแท่งเพิ่มเติม"
    elif score == 1:
        quality = "weak"
        label   = f"🟠 ดีดกลับอ่อน (ขึ้นเพียง {rsi_rise:.1f} จุด, {consec} แท่งติด)"
        timing  = "⚠️ ยังไม่แนะนำ: สัญญาณดีดกลับยังไม่ชัดเจนพอ"
    else:
        quality = "none"
        label   = f"⬜ RSI แตะ Oversold แต่ยังไม่ดีดกลับ (ต่ำสุด {result['rsi_low']:.1f})"
        timing  = "🚫 ยังไม่ควรเข้า: รอให้ RSI ดีดกลับก่อน"

    result["quality"]       = quality
    result["quality_label"] = label
    result["entry_timing"]  = timing

    return result


def find_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> dict:
    ob_result = {
        "bullish_ob_price": None,
        "bearish_ob_price": None,
        "has_bullish_ob": False,
        "has_bearish_ob": False
    }
    
    if len(df) < lookback + 5:
        return ob_result

    body_sizes = (df["close"] - df["open"]).abs()
    avg_body = body_sizes.rolling(20).mean().iloc[-1]

    curr_close = df["close"].iloc[-1]
    curr_open = df["open"].iloc[-1]
    curr_body = abs(curr_close - curr_open)

    past_df = df.iloc[-(lookback + 1):-(LOOKBACK_SKIP_BARS)]
    recent_high = past_df["high"].max()
    recent_low = past_df["low"].min()

    if curr_close > recent_high and curr_body > (avg_body * OB_IMBALANCE_RATIO):
        for i in range(2, min(15, len(df))):
            idx = -i
            p_open = df["open"].iloc[idx]
            p_close = df["close"].iloc[idx]
            p_low = df["low"].iloc[idx]
            
            if p_close < p_open: 
                subsequent_lows = df["low"].iloc[idx+1:]
                if not (subsequent_lows < p_low).any(): 
                    ob_result["has_bullish_ob"] = True
                    ob_result["bullish_ob_price"] = p_low
                    break

    elif curr_close < recent_low and curr_body > (avg_body * OB_IMBALANCE_RATIO):
        for i in range(2, min(15, len(df))):
            idx = -i
            p_open = df["open"].iloc[idx]
            p_close = df["close"].iloc[idx]
            p_high = df["high"].iloc[idx]
            
            if p_close > p_open: 
                subsequent_highs = df["high"].iloc[idx+1:]
                if not (subsequent_highs > p_high).any():
                    ob_result["has_bearish_ob"] = True
                    ob_result["bearish_ob_price"] = p_high
                    break

    return ob_result


def check_bullish_divergence(df: pd.DataFrame) -> bool:
    if len(df) < LOOKBACK_BARS + 2:
        return False

    prev_window = df.iloc[-(LOOKBACK_BARS + 1) : -(LOOKBACK_SKIP_BARS)]
    if len(prev_window) == 0:
        return False
        
    min_low_idx = prev_window["low"].argmin()
    prev_low_price = prev_window["low"].iloc[min_low_idx]
    prev_low_rsi   = prev_window["RSI"].iloc[min_low_idx]

    if prev_low_rsi > RSI_BULL_DIV_MAX:
        return False

    curr_price = df["low"].iloc[-1]
    curr_rsi   = df["RSI"].iloc[-1]

    return (curr_price < prev_low_price) and (curr_rsi > prev_low_rsi)


def is_volume_confirmed(row: pd.Series) -> bool:
    if pd.isna(row.get("VOL_MA20")) or row["VOL_MA20"] == 0:
        return False
    return row["volumeto"] > row["VOL_MA20"]


def format_price(price: float) -> str:
    if price < 0.0001:
        return f"{price:.8f}"
    elif price < 0.001:
        return f"{price:.6f}"
    elif price < 1:
        return f"{price:.4f}"
    else:
        return f"{price:.2f}"


# ==========================================
# Market Scanner
# ==========================================
def scan_market():
    buy_signals   = []
    sell_signals  = []
    bullish_coins = 0
    bearish_coins = 0
    total_valid_coins  = 0
    coin_trends_summary = []

    for coin in COINS:
        df = get_historical_data(coin)
        time.sleep(API_RATE_LIMIT_DELAY)

        # ปรับเกณฑ์ความลึกขั้นต่ำลงมาที่ 45 แท่ง เพื่อให้รองรับข้อจำกัดข้อมูล 100 แท่งจาก OKX API v5
        if df is None or len(df) < 45: 
            logger.warning(f"{coin}: ข้อมูลแท่ง 4H ไม่พอคำนวณ (มีเพียง {len(df) if df is not None else 0} แท่ง) – ข้ามเหรียญนี้")
            continue

        df = calculate_indicators(df)
        row = df.iloc[-1]

        current_price = row["close"]
        rsi           = row["RSI"]
        ema_50        = row["EMA_50"]
        ema_200       = row["EMA_200"]
        vol_confirmed = is_volume_confirmed(row)

        total_valid_coins += 1
        is_divergence = check_bullish_divergence(df)
        rsi_rounded   = round(rsi, 2)

        trend_info = analyze_trend_continuity(df)
        bounce_info = analyze_rsi_bounce(df)
        ob_info = find_order_blocks(df)

        tier    = COIN_TIER.get(coin, "mid")
        tp1_pct = TP_TIERS[tier]["tp1"]
        tp2_pct = TP_TIERS[tier]["tp2"]
        sl_buf  = TP_TIERS[tier]["sl_buffer"]
        vol_tag = " 🔊" if vol_confirmed else ""

        signal_type = ""

        if current_price > ema_200:
            coin_trend = "🟢 ขาขึ้น (Above EMA 200)"
            bullish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🟢 ขาขึ้น (RSI: {rsi_rounded}) | {trend_info['trend_label']}"
            )

            if current_price > (ema_50 * 0.98) and (rsi <= RSI_OVERSOLD or rsi <= RSI_PULLBACK_THRESHOLD):
                if bounce_info["quality"] in ["strong", "moderate"]:
                    signal_type = f"RSI Pullback & Rebound 📉{vol_tag}"
                elif rsi <= RSI_OVERSOLD:
                    signal_type = f"RSI Oversold + Pullback 📉{vol_tag}"
                    
            if is_divergence and not signal_type:
                signal_type = f"Bullish Divergence 📈{vol_tag}"
                
            if ob_info["has_bullish_ob"] and not signal_type:
                signal_type = f"Bullish OB Breakout (SMC) 🚀{vol_tag}"

        else:
            coin_trend = "🔴 ขาลง (Below EMA 200)"
            bearish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🔴 ขาลง (RSI: {rsi_rounded}) | {trend_info['trend_label']}"
            )

            if rsi <= RSI_OVERSOLD:
                signal_type = f"RSI Oversold (ขาลง-เสี่ยงสูง) 📉{vol_tag}"
            elif is_divergence:
                signal_type = f"Bullish Divergence (สวนเทรนด์) 📈{vol_tag}"
            elif ob_info["has_bullish_ob"]:
                signal_type = f"Bullish OB (สวนเทรนด์-ระวัง) 🚀{vol_tag}"

        if signal_type:
            entry_min      = format_price(current_price * 0.97)
            entry_max      = format_price(current_price * 1.00)
            target_tp1     = format_price(current_price * (1 + tp1_pct))
            target_tp2     = format_price(current_price * (1 + tp2_pct))
            sl_val         = ema_200 * (1 - sl_buf) if current_price > ema_200 else current_price * (1 - sl_buf)
            stop_loss      = format_price(sl_val)

            buy_signals.append(
                {
                    "coin":          coin,
                    "trend":         coin_trend,
                    "price":         format_price(current_price),
                    "rsi":           rsi_rounded,
                    "type":          signal_type,
                    "ema_50":        format_price(ema_50),
                    "ema_200":       format_price(ema_200),
                    "entry":         f"${entry_min} - ${entry_max}",
                    "tp1":           f"${target_tp1} (+{tp1_pct*100:.0f}%)",
                    "tp2":           f"${target_tp2} (+{tp2_pct*100:.0f}%)",
                    "sl":            f"${stop_loss}",
                    "vol_confirmed": vol_confirmed,
                    "trend_info":    trend_info,
                    "bounce_info":   bounce_info,
                    "ob_info":       ob_info,
                }
            )

        if rsi >= RSI_OVERBOUGHT or ob_info["has_bearish_ob"]:
            tp_min      = format_price(current_price * 1.00)
            tp_max      = format_price(current_price * (1 + tp1_pct * 0.4))
            exit_val    = ema_50 if current_price > ema_50 else current_price * (1 - sl_buf)
            safety_exit = format_price(exit_val)

            sell_signals.append(
                {
                    "coin":          coin,
                    "trend":         coin_trend,
                    "price":         format_price(current_price),
                    "rsi":           rsi_rounded,
                    "ema_50":        format_price(ema_50),
                    "ema_200":       format_price(ema_200),
                    "tp_zone":       f"${tp_min} - ${tp_max}",
                    "exit":          f"${safety_exit}",
                    "vol_confirmed": vol_confirmed,
                    "trend_info":    trend_info,
                    "ob_info":       ob_info,
                }
            )

    if total_valid_coins > 0:
        bullish_ratio = (bullish_coins / total_valid_coins) * 100
        summary_msg = f"📊 <b>[Market Trend Summary via OKX]</b>\n"
        summary_msg += f"📈 ขาขึ้น: {bullish_coins} เหรียญ | 📉 ขาลง: {bearish_coins} เหรียญ\n"

        if bullish_ratio >= 65:
            summary_msg += "🔥 ภาพรวม: <b>🟢 ขาขึ้นชัดเจน (Strong Bullish)</b>\n<i>กลยุทธ์: เน้นดักซื้อเมื่อเกิดการย่อตัว (Buy on Dip)</i>"
        elif bullish_ratio >= 40:
            summary_msg += "🔥 ภาพรวม: <b>🟡 ไซด์เวย์ / เลือกทาง (Sideways)</b>\n<i>กลยุทธ์: ตลาดก้ำกึ่ง ควรเลือกเทรดเฉพาะตัวที่มีสัญญาณชัดเจน</i>"
        else:
            summary_msg += "🔥 ภาพรวม: <b>🔴 ขาลง / พักฐานแรง (Bearish)</b>\n<i>กลยุทธ์: ตลาดมีความเสี่ยงสูง เน้นถือเงินสดหรือลดขนาดไม้ลง</i>"

        summary_msg += "\n\n📋 <b>สรุปแนวโน้มรายเหรียญ:</b>\n"
        summary_msg += "\n".join(coin_trends_summary)
    else:
        summary_msg = "⚠️ ไม่สามารถดึงข้อมูลเหรียญเพื่อวิเคราะห์ภาพรวมได้"

    return buy_signals, sell_signals, summary_msg


# ==========================================
# Message Builder
# ==========================================
def build_messages(buy_list: list, sell_list: list, market_summary: str) -> list:
    message_blocks = []
    message_blocks.append(market_summary)

    if buy_list:
        buy_header = "🎯 <b>[OKX Screener 4H - สัญญาณช้อนซื้อ]</b>"
        current_block = buy_header

        for opt in buy_list:
            vol_note = (
                "\n🔊 Volume: <b>ยืนยันสัญญาณ (สูงกว่า MA20)</b>"
                if opt["vol_confirmed"]
                else "\n🔇 Volume: ไม่ยืนยัน (ต่ำกว่า MA20)"
            )

            ti = opt["trend_info"]
            bi = opt["bounce_info"]
            ob = opt["ob_info"]

            trend_block = (
                f"\n📐 <b>แนวโน้มต่อเนื่อง:</b> {ti['trend_label']}"
                f"\n   EMA50 slope: {ti['ema50_slope_pct']:+.3f}% | EMA200 slope: {ti['ema200_slope_pct']:+.3f}%"
            )

            bounce_block = (
                f"\n🔄 <b>RSI Bounce Check:</b> {bi['quality_label']}"
                + (f"\n   {bi['entry_timing']}" if bi["entry_timing"] else "")
            )

            ob_block = ""
            if ob.get("has_bullish_ob"):
                ob_price_formatted = format_price(ob["bullish_ob_price"])
                ob_block = f"\n🛡️ <b>Smart Money OB Support:</b> แนวรับราคาก้อนใหญ่ย้อนหลังที่ ${ob_price_formatted}"

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b>"
                f"\n📊 เทรนด์: {opt['trend']}"
                f"\n🚨 รูปแบบ: <b>{opt['type']}</b>"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']}"
                f"\n📉 RSI (4H): {opt['rsi']}"
                f"\n📈 เส้น EMA 50 / 200: ${opt['ema_50']} / ${opt['ema_200']}"
                f"{vol_note}"
                f"{trend_block}"
                f"{bounce_block}"
                f"{ob_block}"
                f"\n🟢 ช่วงเข้าซื้อ: <code>{opt['entry']}</code>"
                f"\n💰 เป้าหมายขาย 1 (TP1): <code>{opt['tp1']}</code>"
                f"\n💰 เป้าหมายขาย 2 (TP2): <code>{opt['tp2']}</code>"
                f"\n❌ จุดตัดขาดทุน (SL): <code>{opt['sl']}</code>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = buy_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    if sell_list:
        sell_header = (
            "⚠️ <b>[OKX Screener 4H - เตือนโซนทำกำไร / แนวต้านยักษ์]</b>\n"
            "<i>คำแนะนำ: ราคาถึงแนวต้านหรือซื้อมากเกินไป ควรพิจารณาแบ่งขายทำกำไร</i>"
        )
        current_block = sell_header

        for opt in sell_list:
            vol_note = (
                "\n🔊 Volume: <b>ยืนยันแรงซื้อ (ระวังเกิดการพักตัวแรง)</b>"
                if opt["vol_confirmed"]
                else "\n🔇 Volume: ไม่ผิดปกติ"
            )

            ti = opt["trend_info"]
            ob = opt["ob_info"]
            
            trend_block = (
                f"\n📐 <b>แนวโน้มต่อเนื่อง:</b> {ti['trend_label']}"
                f"\n   EMA50 slope: {ti['ema50_slope_pct']:+.3f}% | EMA200 slope: {ti['ema200_slope_pct']:+.3f}%"
            )

            ob_block = ""
            if ob.get("has_bearish_ob"):
                ob_price_formatted = format_price(ob["bearish_ob_price"])
                ob_block = f"\n🚨 <b>Smart Money Bearish OB:</b> ตรวจพบกำแพงขายของสถาบันที่ ${ob_price_formatted}"

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b>"
                f"\n📊 เทรนด์: {opt['trend']}"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']}"
                f"\n📈 RSI (4H): {opt['rsi']}"
                f"\n📈 เส้น EMA 50 / 200: ${opt['ema_50']} / ${opt['ema_200']}"
                f"{vol_note}"
                f"{trend_block}"
                f"{ob_block}"
                f"\n🔴 ช่วงราคาที่ควรทยอยขาย: <code>{opt['tp_zone']}</code>"
                f"\n❌ จุดล็อกกำไรหลุดตรงนี้ต้องหนี (Safety Exit): <code>{opt['exit']}</code>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = sell_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    if not buy_list and not sell_list:
        message_blocks.append(
            "\n=========================\n😴 <i>ตลาดนิ่งสนิท: ไม่มีสัญญาณซื้อ/ขายที่เข้าเงื่อนไขใหม่ในรอบนี้</i>"
        )

    return message_blocks


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    logger.info("เริ่มต้นใช้งาน OKX Crypto Screener 4H (SMC Order Block + Rebound v5.2)...")

    buy_list, sell_list, market_summary = scan_market()
    logger.info(f"สแกนระบบเสร็จสมบูรณ์ → พบสัญญาณซื้อ: {len(buy_list)} ตัว | พบสัญญาณขาย/ระวัง: {len(sell_list)} ตัว")

    final_messages = build_messages(buy_list, sell_list, market_summary)
    send_telegram_messages(final_messages)

    logger.info("บอททำงานและส่งข้อมูลรายงานตลาดผ่านข้อมูล OKX เสร็จสมบูรณ์!")
