import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

def generate_high_res_ophthalmic_image(stage="normal", size=(600, 600)):
    """
    Generates ultra-realistic, high-resolution slit-lamp biomicroscopy images 
    with biological iris crypts, pupillary ruff, crystalline lens opacities, 
    and true optical section slit beams.
    """
    w, h = size
    # Deep ophthalmic darkfield background
    base = Image.new('RGB', (w, h), (10, 8, 12))
    
    # Generate organic radial iris texture with Perlin-like noise
    center_x, center_y = w // 2, h // 2
    iris_outer_r = int(w * 0.44)
    pupil_r = int(w * 0.20)
    
    # 1. Sclera and limbal ring
    draw = ImageDraw.Draw(base)
    for r in range(iris_outer_r + 20, iris_outer_r - 2, -1):
        shade = int(35 * (1.0 - (r - iris_outer_r) / 22.0))
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], 
                     outline=(shade + 15, shade + 8, shade + 10), width=2)
                     
    # 2. Rich Multi-layered Iris Stroma (Hazel/Brown with crypts)
    iris_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    i_draw = ImageDraw.Draw(iris_layer)
    
    # Radial trabecular meshwork & collarette
    np.random.seed(42 if stage == "normal" else (108 if stage == "early" else 240))
    for angle in range(0, 360, 1):
        rad = math.radians(angle)
        jitter = np.random.uniform(0.88, 1.12)
        r_start = pupil_r - 2
        r_end = int(iris_outer_r * jitter)
        
        # Color variation across collarette
        base_r = int(75 + 40 * math.sin(angle * 0.1))
        base_g = int(50 + 25 * math.cos(angle * 0.15))
        base_b = int(30 + 15 * math.sin(angle * 0.05))
        
        x1 = center_x + int(r_start * math.cos(rad))
        y1 = center_y + int(r_start * math.sin(rad))
        x2 = center_x + int(r_end * math.cos(rad))
        y2 = center_y + int(r_end * math.sin(rad))
        
        i_draw.line([(x1, y1), (x2, y2)], fill=(base_r, base_g, base_b, 220), width=2)
        
        # Intermittent crypts and freckles
        if angle % 18 == 0:
            cr_x = center_x + int((pupil_r + 45) * math.cos(rad))
            cr_y = center_y + int((pupil_r + 45) * math.sin(rad))
            i_draw.ellipse([cr_x - 5, cr_y - 5, cr_x + 5, cr_y + 5], fill=(35, 20, 15, 230))
            
    # Collarette ring
    collarette_r = int((pupil_r + iris_outer_r) * 0.48)
    i_draw.ellipse([center_x - collarette_r, center_y - collarette_r, 
                    center_x + collarette_r, center_y + collarette_r], 
                   outline=(120, 85, 50, 140), width=3)
                   
    iris_layer = iris_layer.filter(ImageFilter.GaussianBlur(radius=1.2))
    base = Image.alpha_composite(base.convert('RGBA'), iris_layer).convert('RGB')
    
    # 3. Deep Black Retroillumination Pupil
    draw = ImageDraw.Draw(base)
    draw.ellipse([center_x - pupil_r, center_y - pupil_r, center_x + pupil_r, center_y + pupil_r], 
                 fill=(4, 5, 8))
                 
    # 4. Pupillary Ruff (Dark pigmented frill)
    draw.ellipse([center_x - pupil_r - 2, center_y - pupil_r - 2, 
                  center_x + pupil_r + 2, center_y + pupil_r + 2], 
                 outline=(25, 12, 8), width=3)

    # 5. Crystalline Lens Opacities & Cataract Pathophysiology
    cataract_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    c_draw = ImageDraw.Draw(cataract_layer)
    
    if stage == "early":
        # Early Nuclear Sclerosis: Gentle central golden-yellow haze
        lens_r = int(pupil_r * 0.72)
        for r in range(lens_r, 0, -4):
            alpha = int(120 * (1.0 - (r / lens_r)**1.5))
            c_draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], 
                           fill=(218, 175, 75, alpha))
        cataract_layer = cataract_layer.filter(ImageFilter.GaussianBlur(radius=6))
        
    elif stage == "moderate":
        # Moderate Nuclear Sclerosis + Cortical Wedge/Spoke Opacities
        lens_r = int(pupil_r * 0.88)
        # Deep amber nucleus
        for r in range(lens_r, 0, -3):
            alpha = int(185 * (1.0 - (r / lens_r)**1.2))
            c_draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], 
                           fill=(235, 168, 55, alpha))
            
        # Radial cortical spokes encroaching toward visual axis
        spoke_angles = [20, 65, 110, 160, 205, 250, 310, 345]
        for a in spoke_angles:
            rad = math.radians(a)
            tip_r = int(lens_r * 0.35)
            base_r = int(pupil_r * 0.96)
            x_tip = center_x + int(tip_r * math.cos(rad))
            y_tip = center_y + int(tip_r * math.sin(rad))
            x_base = center_x + int(base_r * math.cos(rad))
            y_base = center_y + int(base_r * math.sin(rad))
            
            c_draw.line([(x_base, y_base), (x_tip, y_tip)], fill=(255, 240, 205, 215), width=7)
            # Feathery lateral spoke branches
            c_draw.line([(x_base + 3, y_base), (x_tip, y_tip + 2)], fill=(245, 220, 175, 150), width=4)
            
        cataract_layer = cataract_layer.filter(ImageFilter.GaussianBlur(radius=4))
        
    elif stage == "advanced":
        # Mature/Hyper-mature Cataract: Dense pearlescent white-amber lens
        lens_r = int(pupil_r * 0.98)
        for r in range(lens_r, 0, -2):
            alpha = int(240 * (1.0 - (r / lens_r)**0.8))
            c_draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], 
                           fill=(248, 225, 160, alpha))
            
        # Dense nuclear core
        c_draw.ellipse([center_x - int(lens_r*0.55), center_y - int(lens_r*0.55), 
                        center_x + int(lens_r*0.55), center_y + int(lens_r*0.55)], 
                       fill=(255, 245, 215, 245))
        cataract_layer = cataract_layer.filter(ImageFilter.GaussianBlur(radius=3))

    base = Image.alpha_composite(base.convert('RGBA'), cataract_layer).convert('RGB')
    
    # 6. Ultra-Realistic Slit-Lamp Optical Section (Angled 45-degree focal slit beam)
    slit_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(slit_layer)
    
    # Bright focal slit beam traversing the cornea & lens
    beam_w = 26
    s_draw.polygon([
        (center_x - beam_w - 20, 15), 
        (center_x + beam_w - 10, 15),
        (center_x + beam_w + 35, h - 15), 
        (center_x - beam_w + 25, h - 15)
    ], fill=(255, 255, 255, 75))
    
    # Intense focal optical section stripe
    s_draw.polygon([
        (center_x - 6, 20), 
        (center_x + 6, 20),
        (center_x + 36, h - 20), 
        (center_x + 24, h - 20)
    ], fill=(255, 255, 255, 140))
    
    # High-intensity Purkinje corneal reflection glint
    glint_x, glint_y = center_x + 28, center_y - 35
    s_draw.ellipse([glint_x - 8, glint_y - 6, glint_x + 8, glint_y + 6], fill=(255, 255, 255, 250))
    s_draw.ellipse([glint_x - 18, glint_y - 12, glint_x + 18, glint_y + 12], fill=(255, 255, 255, 80))
    
    slit_layer = slit_layer.filter(ImageFilter.GaussianBlur(radius=2.5))
    final_img = Image.alpha_composite(base.convert('RGBA'), slit_layer).convert('RGB')
    
    return final_img

def init_sample_images(data_dir):
    """
    Initializes high-fidelity ophthalmic sample images in data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)
    samples = {
        "sample_normal.jpg": "normal",
        "sample_early.jpg": "early",
        "sample_moderate.jpg": "moderate",
        "sample_advanced.jpg": "advanced"
    }
    
    created_paths = {}
    for filename, stage in samples.items():
        filepath = os.path.join(data_dir, filename)
        # Always generate high-res images
        img = generate_high_res_ophthalmic_image(stage=stage, size=(600, 600))
        img.save(filepath, "JPEG", quality=98)
        created_paths[stage] = filepath
    return created_paths

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(current_dir, "..", "..", "data", "samples")
    paths = init_sample_images(sample_dir)
    print(f"[✓] High-Resolution Slit-Lamp Images Generated in: {sample_dir}")
