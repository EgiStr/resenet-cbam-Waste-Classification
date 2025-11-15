import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data():
    base_dir = 'data/DATASET'
    output_dir = 'data'
    classes = {'O': 'organic', 'R': 'recyclable'}

    for phase in ['TRAIN', 'TEST']:
        for cls_code, cls_name in classes.items():
            src_dir = os.path.join(base_dir, phase, cls_code)
            if os.path.exists(src_dir):
                images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
                # Split 80/20 for train/val
                train_files, val_files = train_test_split(images, test_size=0.2, random_state=42)
                for split, files in [('train', train_files), ('val', val_files)]:
                    dst_dir = os.path.join(output_dir, split, cls_name)
                    os.makedirs(dst_dir, exist_ok=True)
                    for file in files:
                        src_file = os.path.join(src_dir, file)
                        dst_file = os.path.join(dst_dir, file)
                        if not os.path.exists(dst_file):
                            shutil.copy(src_file, dst_file)
    print("Data prepared!")

if __name__ == "__main__":
    prepare_data()