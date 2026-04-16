import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="標準圓比對評分系統", page_icon="🎯")

def get_comparison_score(image):
    img = np.array(image)
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.12), int(w * 0.12)
    roi_img = img[margin_h:h-margin_h, margin_w:w-margin_w]
    
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 8)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_img = img.copy()
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            c[:, :, 0] += margin_w
            c[:, :, 1] += margin_h
            hull = cv2.convexHull(c)
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            
            cv2.circle(res_img, center, radius, (255, 0, 0), 3)
            cv2.drawContours(res_img, [c], -1, (0, 255, 0), 6)
            
            student_area = cv2.contourArea(hull)
            perfect_circle_area = np.pi * (radius ** 2)
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * student_area) / (perimeter ** 2) if perimeter > 0 else 0
            area_ratio = student_area / perfect_circle_area if perfect_circle_area > 0 else 0
            
            final_score = (circularity * 0.6 + area_ratio * 0.4) * 100
            if final_score > 98: 
                final_score = 95 + (circularity - 0.9) * 50
            
            return round(min(100, final_score), 1), res_img
    return None, res_img

st.title("🎯 標準圓比對評分系統 (V9.0)")
st.write("紅色：標準圓 | 綠色：學生筆跡")

img_file = st.camera_input(" ")

if img_file:
    score, res = get_comparison_score(Image.open(img_file))
    if score:
        st.header(f"評定得分：{score} 分")
        st.image(res, caption="紅線為標準基準，綠線為偵測筆跡")
    else:
        st.warning("未能偵測到圓圈，請確保圓圈完整且位於中央。")
