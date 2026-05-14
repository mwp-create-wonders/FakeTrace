import os
import sys
import glob
from PIL import Image
from tqdm import tqdm

def convert_to_jpeg(input_path, output_path, quality=100):
    try:
        img = Image.open(input_path)
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode == 'P':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(output_path, 'JPEG', quality=quality, subsampling=0)
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def process_directory(base_dir):
    subdirs = ['masks', 'tampered']
    
    for subdir in subdirs:
        current_dir = os.path.join(base_dir, subdir)
        if not os.path.exists(current_dir):
            print(f"Directory not found: {current_dir}")
            continue
        
        print(f"\nProcessing {current_dir}...")
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tif']:
            image_files.extend(glob.glob(os.path.join(current_dir, ext)))
        
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} image files")
        
        for img_path in tqdm(image_files):
            dir_name = os.path.dirname(img_path)
            file_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(file_name)[0]
            
            new_name = f"{name_without_ext}.jpg"
            new_path = os.path.join(dir_name, new_name)
            
            if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg'):
                if img_path != new_path:
                    os.rename(img_path, new_path)
            else:
                success = convert_to_jpeg(img_path, new_path)
                if success:
                    os.remove(img_path)
    
    print("\nConversion completed successfully!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python convert_to_jpeg.py <combined_dataset_dir>")
        print("Example: python convert_to_jpeg.py /path/to/combined_dataset")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found - {base_dir}")
        sys.exit(1)
    
    process_directory(base_dir)