# Image Processing 2024 Fall (NYCU) - Lab1 to Lab3

## Overview
本專案包含《Image Processing》課程的三個實驗，涵蓋影像增強（Image Enhancement）、影像復原（Image Restoration）、連通元件分析（Connected Component Analysis）、與顏色校正（Color Correction）等主題，並使用 Python 及 OpenCV 進行實作。

---

## Lab1 - Image Enhancement Using Spatial Filters
**目標**:  
使用空間域（Spatial Domain）濾波器來進行影像增強，包括平滑（Smoothing）與銳化（Sharpening）。  

**實作內容**:
- **Padding Function**: 設定影像邊界填補（Zero Padding）。
- **Convolution Function**: 實作 2D 卷積運算。
- **Gaussian Filter**: 使用不同 `sigma` 與 `kernel size` 進行平滑處理。
- **Median Filter**: 以不同 `kernel size` 進行雜訊去除。
- **Laplacian Sharpening**: 使用兩種不同的 Laplacian 核進行影像銳化。

---

## Lab2 - Image Enhancement & Image Restoration
**目標**:  
透過伽瑪校正（Gamma Correction）與直方圖均衡（Histogram Equalization）進行影像增強，並使用維納濾波（Wiener Filtering）與受限最小平方濾波（Constrained Least Squares Filtering）進行影像復原。  

**實作內容**:
- **Gamma Correction**: 應用不同 `gamma` 值調整影像對比與亮度。
- **Histogram Equalization**: 透過灰階直方圖均衡提升影像對比。
- **Motion Blur PSF（Point Spread Function）生成**: 模擬運動模糊效應。
- **Wiener Filtering**: 利用頻域方法去除模糊影像。
- **Constrained Least Squares Filtering**: 結合 Laplacian 正則化進行影像復原。

---

## Lab3 - Connected Component Analysis & Color Correction
**目標**:  
使用連通元件分析（Connected Component Analysis）對影像中的區域進行標記，並利用顏色校正（Color Correction）技術修正影像的色偏問題。  

**實作內容**:
### 連通元件分析（Connected Component Analysis）
- **二值化影像處理**: 轉換影像為二值圖。
- **Two-Pass Algorithm**: 以兩次掃描方式標記連通元件。
- **Seed Filling Algorithm**: 以遞迴方式標記連通元件。
- **4-Connectivity vs 8-Connectivity**: 比較不同連通性對標記結果的影響。
- **Color Mapping**: 使用隨機顏色為不同區域著色。

### 顏色校正（Color Correction）
- **White Patch Algorithm**: 以影像最亮區域為基準進行白平衡校正。
- **Gray-World Algorithm**: 依據灰世界假設調整影像顏色分佈。

---
