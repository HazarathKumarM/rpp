import os
from PIL import Image, ImageChops

def compare_images(img1_path, img2_path, diff_path):
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    diff = ImageChops.difference(img1, img2)
    
    if diff.getbbox():
        # Create a new image to highlight differences
        diff_highlight = Image.new('RGB', img1.size, (255, 255, 255))  # Start with a white image
        pixels1 = img1.load()
        pixels2 = img2.load()
        pixels_diff = diff_highlight.load()
        
        for y in range(img1.height):
            for x in range(img1.width):
                if pixels1[x, y] != pixels2[x, y]:
                    pixels_diff[x, y] = (255, 0, 0)  # Mark differences in red
        
        diff_highlight.save(diff_path)
        print(f"Differences found and saved to {diff_path}")
    else:
        print(f"No differences found between {img1_path} and {img2_path}")

def compare_directories(dir1, dir2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(dir1):
        relative_path = os.path.relpath(root, dir1)
        corresponding_dir2 = os.path.join(dir2, relative_path)
        output_subdir = os.path.join(output_dir, relative_path)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        if os.path.isdir(corresponding_dir2):
            for img_name in files:
                img_path1 = os.path.join(root, img_name)
                img_path2 = os.path.join(corresponding_dir2, img_name)
                diff_path = os.path.join(output_subdir, img_name)
                
                if os.path.isfile(img_path1) and os.path.isfile(img_path2):
                    compare_images(img_path1, img_path2, diff_path)

if __name__ == "__main__":
    dir1 = "/dockerx/rpp/utilities/test_suite/HOST/OUTPUT_IMAGES_HOST_color_twist_commented"
    dir2 = "/dockerx/rpp/utilities/test_suite/HOST/OUTPUT_IMAGES_HOST_color_twist_avx"
    output_dir = "/dockerx/rpp/utilities/test_suite/HOST/Output_difference"
    
    # Ensure directories exist
    for directory in [dir1, dir2, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    compare_directories(dir1, dir2, output_dir)