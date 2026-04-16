cat << 'EOF' > app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="標準圓比對評分系統", page_icon="🎯")

def get_comparison_score(image):
    img = np.array(image)
    h, w = img.shape[:2]
    
    # 1. 隔離邊框 (12% 邊距)
    margin_h, margin_w = int(h * 0.12), int(w * 0.12)
    roi_img = img[margin_h:h-margin_h, margin_w:w-margin_w]
    
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 8)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    res_img = img.copy()
    
    if contours:
        # 找出區域內最大的輪廓
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            # 座標還原
            c[:, :, 0] += margin_w
            c[:, :, 1] += margin_h
            
            # 使用凸包建立平滑邊界
            hull = cv2.convexHull(c)
            
            # --- 關鍵：計算「最小外接圓」作為標準 ---
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            
            # 繪製標準紅圓 (這是 100 分的基準)
            cv2.circle(res_img, center, radius, (255, 0, 0), 3)
            
            # 繪製學生真實筆跡 (綠線)
            cv2.drawContours(res_img, [c], -1, (0, 255, 0), 6)
            
            # --- 評分邏輯：計算筆跡與標準圓的偏差 ---
            # 計算學生的輪廓面積與標準圓面積的比例
            student_area = cv2.contourArea(hull)
            perfect_circle_area = np.pi * (radius ** 2)
            
            # 基礎圓度
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * student_area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # 綜合得分：考慮形狀圓度以及是否填滿標準圓
            area_ratio = student_area / perfect_circle_area if perfect_circle_area > 0 else 0
            
            # 分數計算 (結合圓度與面積契合度，並增加嚴格係數)
            final_score = (circularity * 0.6 + area_ratio * 0.4) * 100
            
            # 針對之前 100 分過多的情況進行修正
            if final_score > 98: 
                # 只有非常完美的圓能拿 98 以上
                final_score = 95 + (circularity - 0.9) * 50
            
            return round(min(100, final_score), 1), res_img
            
    return None, res_img

st.title("🎯 標準圓比對評分系統 (V9.0)")
st.write("紅色：標準完美圓 | 綠色：學生筆跡")

img_file = st.camera_input(" ")

if img_file:
    score, res = get_comparison_score(Image.open(img_file))
    if score:
        st.header(f"評定得分：{score} 分")
        st.image(res, caption="紅線為系統計算出的標準圓，綠線為學生筆跡")
        st.info("💡 評分標準：綠色線條越貼合紅色圓圈，分數越高。")
    else:
        st.warning("未能偵測到圓圈，請確保圓圈完整且位於中央。")
EOF
