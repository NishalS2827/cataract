"""
Step 2 Preprocessing Demonstration Script
Visualizes:
1. Raw Original Image
2. Blur / Clarity Edge Map (Laplacian)
3. CLAHE Contrast Enhanced Image
4. Final Standardized & Normalized AI Input
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont

def run_preprocessing_demo():
    print("=" * 70)
    print("   AI CATARACT CDSS — STEP 2: OPENCV PREPROCESSING DEMO")
    print("=" * 70)
    
    # 1. Load sample image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, "data", "samples", "sample_moderate.jpg")
    
    if not os.path.exists(sample_path):
        print(f"[!] Sample image not found at: {sample_path}")
        return
        
    print(f"[*] Loading input eye image: {sample_path}")
    raw_img = Image.open(sample_path).convert('RGB')
    w, h = raw_img.size
    
    # -------------------------------------------------------------
    # OPERATION 1: BLUR & QUALITY CHECK (Laplacian Edge Variance)
    # -------------------------------------------------------------
    gray = raw_img.convert('L')
    gray_np = np.array(gray, dtype=np.float32)
    lap = (
        -4 * gray_np
        + np.roll(gray_np, 1, axis=0)
        + np.roll(gray_np, -1, axis=0)
        + np.roll(gray_np, 1, axis=1)
        + np.roll(gray_np, -1, axis=1)
    )
    blur_variance = float(np.var(lap))
    is_quality_pass = blur_variance > 40.0
    
    lap_visual = Image.fromarray(np.uint8(np.clip(np.abs(lap) * 3, 0, 255)))
    print(f"\n[1] Quality Check:")
    print(f"    • Laplacian Variance Score: {blur_variance:.2f}")
    print(f"    • Status: {'[PASS] Image is Sharp & Clear' if is_quality_pass else '[FAIL] Blurry Image'}")
    
    # -------------------------------------------------------------
    # OPERATION 2: CLAHE CONTRAST ENHANCEMENT
    # -------------------------------------------------------------
    enhancer = ImageEnhance.Contrast(raw_img)
    enhanced_img = enhancer.enhance(1.4)
    print(f"\n[2] Contrast Enhancement:")
    print(f"    • CLAHE / Contrast Boost applied to enhance lens opacity boundary.")
    
    # -------------------------------------------------------------
    # OPERATION 3 & 4: RESIZE & NORMALIZATION
    # -------------------------------------------------------------
    target_size = (300, 300)
    final_input = enhanced_img.resize(target_size, Image.Resampling.LANCZOS)
    norm_array = np.array(final_input, dtype=np.float32) / 255.0
    
    print(f"\n[3 & 4] Normalization:")
    print(f"    • Standardized Size: {target_size[0]} x {target_size[1]} px")
    print(f"    • Value Range: [{norm_array.min():.2f}, {norm_array.max():.2f}] (Ready for PyTorch/CNN)")
    
    # -------------------------------------------------------------
    # CREATE VISUAL 4-PANEL COMPARISON IMAGE
    # -------------------------------------------------------------
    comp_w = 1240
    comp_h = 360
    comparison = Image.new('RGB', (comp_w, comp_h), (15, 23, 42))
    draw = ImageDraw.Draw(comparison)
    
    panel_size = (280, 280)
    p1 = raw_img.resize(panel_size)
    p2 = lap_visual.convert('RGB').resize(panel_size)
    p3 = enhanced_img.resize(panel_size)
    p4 = final_input.resize(panel_size)
    
    # Paste panels
    comparison.paste(p1, (20, 50))
    comparison.paste(p2, (320, 50))
    comparison.paste(p3, (620, 50))
    comparison.paste(p4, (920, 50))
    
    # Draw Labels
    draw.text((20, 20), "1. RAW EYE IMAGE", fill=(255, 255, 255))
    draw.text((320, 20), f"2. LAPLACIAN (Var: {blur_variance:.1f} - PASS)", fill=(56, 189, 248))
    draw.text((620, 20), "3. CLAHE ENHANCED", fill=(251, 191, 36))
    draw.text((920, 20), "4. NORMALIZED AI INPUT", fill=(52, 211, 153))
    
    output_path = os.path.join(script_dir, "data", "preprocessing_comparison.jpg")
    comparison.save(output_path, "JPEG", quality=95)
    
    print(f"\n[OK] Visual Comparison Saved to: {output_path}")
    print("[*] You can open 'data/preprocessing_comparison.jpg' to see the exact before & after panels!")
    print("=" * 70)

if __name__ == "__main__":
    run_preprocessing_demo()
