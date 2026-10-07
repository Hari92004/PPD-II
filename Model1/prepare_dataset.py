"""
SIH 2026 - Model 1 (2D Camera Vision Perception)
Dataset Generator & Preprocessor for Indian Road Traffic
Formats RGB images and normalized bounding box labels for 7 Indian road classes.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

CLASSES = [
    'auto_rickshaw',
    'cow_cattle',
    'pedestrian',
    'two_wheeler',
    'pothole',
    'truck_bus',
    'car'
]

def generate_model1_dataset(base_dir=r"D:\SIH26\Model1\dataset",
                            train_count=120, val_count=30,
                            img_size=(640, 640)):
    print("=" * 65)
    print("   MODEL 1: PREPARING 2D CAMERA INDIAN TRAFFIC DATASET")
    print("=" * 65)
    
    train_img_dir = os.path.join(base_dir, "images", "train")
    val_img_dir = os.path.join(base_dir, "images", "val")
    train_lbl_dir = os.path.join(base_dir, "labels", "train")
    val_lbl_dir = os.path.join(base_dir, "labels", "val")
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        
    w, h = img_size
    splits = [('train', train_count, train_img_dir, train_lbl_dir),
              ('val', val_count, val_img_dir, val_lbl_dir)]
              
    for split_name, count, img_dir, lbl_dir in splits:
        print(f"[DATASET] Generating {count} {split_name} frames with YOLO labels...")
        for i in range(count):
            # Base Indian road canvas: Asphalt surface (dark gray) with dirt/soil shoulder
            img = Image.new("RGB", (w, h), color=(55, 55, 58))
            draw = ImageDraw.Draw(img)
            
            # Draw roadside dirt edges (unstructured road boundaries)
            shoulder_left = random.randint(60, 100)
            shoulder_right = random.randint(w - 100, w - 60)
            draw.rectangle([0, 0, shoulder_left, h], fill=(130, 95, 60)) # Brown dirt left
            draw.rectangle([shoulder_right, 0, w, h], fill=(130, 95, 60)) # Dirt right
            
            # Subtle road asphalt texture / lane-free variations
            for _ in range(15):
                tx = random.randint(shoulder_left, shoulder_right)
                ty = random.randint(0, h)
                draw.line([(tx, ty), (tx + random.randint(20, 60), ty)], fill=(45, 45, 48), width=2)
            
            labels = []
            num_objects = random.randint(2, 5)
            
            for _ in range(num_objects):
                class_id = random.randint(0, 6)
                
                if class_id == 0: # Auto-Rickshaw (Yellow-green compact)
                    bw = random.randint(70, 110)
                    bh = random.randint(90, 130)
                    bx = random.randint(shoulder_left + 10, shoulder_right - bw - 10)
                    by = random.randint(120, h - bh - 20)
                    # Draw yellow body + green lower half + black roof
                    draw.rectangle([bx, by, bx + bw, by + bh], fill=(235, 190, 20))
                    draw.rectangle([bx, by + int(bh*0.6), bx + bw, by + bh], fill=(30, 130, 40))
                    draw.rectangle([bx + 5, by + 5, bx + bw - 5, by + int(bh*0.35)], fill=(30, 30, 30))
                    
                elif class_id == 1: # Cow / Cattle (White/brown/black spot cattle)
                    bw = random.randint(80, 120)
                    bh = random.randint(65, 95)
                    bx = random.randint(shoulder_left, shoulder_right - bw)
                    by = random.randint(150, h - bh - 30)
                    draw.ellipse([bx, by, bx + bw, by + bh], fill=(215, 210, 200))
                    draw.ellipse([bx + 10, by + 10, bx + int(bw*0.4), by + int(bh*0.6)], fill=(120, 75, 40))
                    draw.rectangle([bx + int(bw*0.7), by - 15, bx + bw + 10, by + int(bh*0.5)], fill=(200, 195, 185))
                    
                elif class_id == 2: # Pedestrian (walking / crossing)
                    bw = random.randint(30, 50)
                    bh = random.randint(70, 110)
                    bx = random.randint(20, w - bw - 20)
                    by = random.randint(100, h - bh - 20)
                    # Head + Torso
                    draw.ellipse([bx + int(bw*0.25), by, bx + int(bw*0.75), by + int(bh*0.25)], fill=(220, 180, 140))
                    draw.rectangle([bx, by + int(bh*0.25), bx + bw, by + bh], fill=(40, 70, 180))
                    
                elif class_id == 3: # Two-Wheeler (Motorcycle/Scooter with rider)
                    bw = random.randint(45, 75)
                    bh = random.randint(80, 120)
                    bx = random.randint(shoulder_left + 10, shoulder_right - bw - 10)
                    by = random.randint(130, h - bh - 20)
                    draw.rectangle([bx, by + int(bh*0.4), bx + bw, by + bh], fill=(180, 30, 30))
                    draw.ellipse([bx + int(bw*0.2), by, bx + int(bw*0.8), by + int(bh*0.35)], fill=(30, 30, 30))
                    
                elif class_id == 4: # Pothole (dark road surface crater / crack)
                    bw = random.randint(60, 120)
                    bh = random.randint(40, 70)
                    bx = random.randint(shoulder_left + 20, shoulder_right - bw - 20)
                    by = random.randint(200, h - bh - 20)
                    draw.ellipse([bx, by, bx + bw, by + bh], fill=(22, 22, 22), outline=(120, 115, 100), width=2)
                    
                elif class_id == 5: # Truck / Bus (Heavy commercial vehicle)
                    bw = random.randint(130, 190)
                    bh = random.randint(140, 200)
                    bx = random.randint(shoulder_left + 10, shoulder_right - bw - 10)
                    by = random.randint(80, h - bh - 40)
                    draw.rectangle([bx, by, bx + bw, by + bh], fill=(210, 90, 20), outline=(250, 240, 40), width=3)
                    draw.rectangle([bx + 15, by + 15, bx + bw - 15, by + int(bh*0.4)], fill=(40, 90, 160))
                    
                else: # Car (Sedan / Hatchback)
                    bw = random.randint(90, 140)
                    bh = random.randint(70, 110)
                    bx = random.randint(shoulder_left + 10, shoulder_right - bw - 10)
                    by = random.randint(100, h - bh - 30)
                    draw.rectangle([bx, by, bx + bw, by + bh], fill=(210, 210, 220))
                    draw.rectangle([bx + 15, by + 10, bx + bw - 15, by + int(bh*0.5)], fill=(40, 40, 45))
                
                # Compute normalized YOLO coordinates: [class_id, x_center, y_center, width, height]
                x_center = (bx + bw / 2.0) / w
                y_center = (by + bh / 2.0) / h
                norm_w = bw / float(w)
                norm_h = bh / float(h)
                
                # Clip to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.01, min(1.0, norm_w))
                norm_h = max(0.01, min(1.0, norm_h))
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                
            # Save Image and Label
            img_file = os.path.join(img_dir, f"frame_{split_name}_{i:04d}.jpg")
            lbl_file = os.path.join(lbl_dir, f"frame_{split_name}_{i:04d}.txt")
            
            img.save(img_file, format="JPEG", quality=95)
            with open(lbl_file, 'w') as f:
                f.write("\n".join(labels))
                
    # Write dataset.yaml configuration file
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    yaml_content = f"""# SIH 2026: Model 1 Indian Road Traffic Dataset Config
path: {base_dir.replace(os.sep, '/')}
train: images/train
val: images/val

names:
  0: auto_rickshaw
  1: cow_cattle
  2: pedestrian
  3: two_wheeler
  4: pothole
  5: truck_bus
  6: car
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"\n[SUCCESS] Dataset Generated: {train_count} Train | {val_count} Val")
    print(f"[SUCCESS] Config Saved: {yaml_path}")
    print("=" * 65)

if __name__ == "__main__":
    generate_model1_dataset()
