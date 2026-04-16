import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="標準圓比對評分系統", page_icon="🎯")

def get_comparison_score(image):
    # 1. 讀取影像並轉為 OpenCV 格式
    img = np.array(image)
    h, w = img.shape[:2]
    
    # 【優化 1】針對手機解析度調整縮放，確保運算穩定
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_resized = img.copy()
    
    h2, w2 = img_resized.shape[:2]
    margin_h, margin_w = int(h2 * 0.1), int(w2 * 0.1)
    roi_img = img_resized[margin_h:h2-margin_h, margin_w:w2-margin_w]
    
    # 2. 影像預處理
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    
    # 【優化 2】強化對比度，解決手機拍攝光線不足問題
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 增加模糊度以濾除紙張紋路雜訊
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 【優化 3】放寬二值化門檻，抓取更淡的鉛筆線條
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 4)
    
    # 稍微膨脹線條，避免斷線
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_img = img_resized.copy()
    
    if contours:
        # 篩選掉太小的點，保留主要圓圈
        valid_cnts = [cnt for cnt in contours if cv2.contourArea(cnt) > 1500]
        if valid_cnts:
            c = max(valid_cnts, key=cv2.contourArea)
            
            # 座標還原
            c[:, :, 0] += margin_w
            c[:, :, 1] += margin_h
            
            # 建立凸包與標準圓
            hull = cv2.convexHull(c)
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            
            # 繪製視覺反饋
            cv2.circle(res_img, center, radius, (255, 0, 0), 3) # 紅色標準圓
            cv2.drawContours(res_img, [c], -1, (0, 255, 0), 5)  # 綠色學生筆跡
            
            # --- 評分核心 ---
            student_area = cv2.contourArea(hull)
            perfect_circle_area = np.pi * (radius ** 2)
            perimeter = cv2.arcLength(c, True)
            
            # 圓度 (Circularity)
            circularity = (4 * np.pi * student_area) / (perimeter ** 2) if perimeter > 0 else 0
            # 覆蓋率 (Area Ratio)
            area_ratio = student_area / perfect_circle_area if perfect_circle_area > 0 else 0
            
            # 【優化 4】更直覺的分數映射：手繪圓很難超過 0.9，我們將 0.85 視為優秀
            raw_score = (circularity * 0.7 + area_ratio * 0.3)
            # 將分數拉伸，讓 0.7~0.9 區間對應到 60~95 分
            if raw_score > 0.7:
                final_score = 60 + (raw_score - 0.7) * 150
            else:
                final_score = raw_score * 85
                
            return round(min(100, final_score), 1), res_img
            
    return None, res_img

st.title("🎯 專業圓圈評分系統 V9.5")
st.write("手機版已優化：請確保環境光線充足，避免陰影遮擋。")

img_file = st.camera_input(" ")

if img_file:
    score, res = get_comparison_score(Image.open(img_file))
    if score:
        st.subheader(f"評定得分：{score} 分")
        st.image(res, caption="辨識結果：紅線為標準圓，綠線為學生筆跡")
        if score >= 90: st.balloons()
    else:
        st.warning("偵測失敗！請讓圓圈靠近中心，並檢查光線是否均勻。")
