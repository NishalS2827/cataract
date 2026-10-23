"""
Test AI Model Inference Pipeline
Evaluates input eye image and produces class probabilities using the trained model.
"""

import os
import sys
from PIL import Image
import numpy as np
from backend.app.ai_engine import analyze_cataract_image, preprocess_image

def test_inference(custom_path=None):
    print("=" * 70)
    print("   AI CATARACT CDSS - MODEL INFERENCE & EVALUATION TEST")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If a specific image path is given
    if custom_path:
        target_images = [custom_path]
    else:
        sample_dir = os.path.join(script_dir, "data", "samples")
        target_images = [
            os.path.join(sample_dir, "sample_normal.jpg"),
            os.path.join(sample_dir, "sample_early.jpg"),
            os.path.join(sample_dir, "sample_moderate.jpg"),
            os.path.join(sample_dir, "sample_advanced.jpg")
        ]
        # Also include one cataract and one normal image from test dataset if available
        test_cataract_sample = os.path.join(script_dir, "data", "dataset", "test", "cataract")
        test_normal_sample = os.path.join(script_dir, "data", "dataset", "test", "normal")
        
        if os.path.exists(test_cataract_sample):
            files = [f for f in os.listdir(test_cataract_sample) if f.endswith('.jpg')]
            if files:
                target_images.append(os.path.join(test_cataract_sample, files[0]))
                
        if os.path.exists(test_normal_sample):
            files = [f for f in os.listdir(test_normal_sample) if f.endswith('.jpg')]
            if files:
                target_images.append(os.path.join(test_normal_sample, files[0]))
    
    evaluated = 0
    for img_path in target_images:
        if not os.path.exists(img_path):
            continue
            
        filename = os.path.basename(img_path)
        folder = os.path.basename(os.path.dirname(img_path))
        
        try:
            img = Image.open(img_path)
            preprocessed = preprocess_image(img)
            res = analyze_cataract_image(preprocessed)
            evaluated += 1
            
            print(f"\n[IMAGE]: {folder}/{filename}")
            print(f"  - Detection       : {res['detection']}")
            print(f"  - Confidence      : {res['confidence']}%")
            print(f"  - Severity Stage  : {res['severity']}")
            print(f"  - Opacity Density : {res['opacity_percentage']}%")
            print(f"  - AI Engine Used  : {res.get('engine', 'Clinical Feature Engine')}")
            print("-" * 50)
        except Exception as e:
            print(f"[-] Error evaluating {img_path}: {e}")
            
    print(f"\n[OK] Evaluated {evaluated} images successfully!")
    print("=" * 70)

if __name__ == "__main__":
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_inference(cli_arg)
