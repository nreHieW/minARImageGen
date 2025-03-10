import os
import tarfile
import concurrent.futures
import shutil
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from classes import IMAGENET2012_CLASSES

REPO_ID = "ILSVRC/imagenet-1k"
REPO_TYPE = "dataset"

TRAIN_ARCHIVES = [
    "data/train_images_0.tar.gz",
    "data/train_images_1.tar.gz",
    "data/train_images_2.tar.gz",
    "data/train_images_3.tar.gz",
    "data/train_images_4.tar.gz"
]
VAL_ARCHIVE = "data/val_images.tar.gz"
TEST_ARCHIVE = "data/test_images.tar.gz"

def process_image(member, tar, extract_path):
    """Process a single image: extract to class folder and rename."""
    try:
        # Extract synset_id and image_id from filename
        filename = os.path.basename(member.name)
        if not filename.endswith('.JPEG'):
            return

        # Handle validation vs training file formats
        if "ILSVRC2012_val_" in filename:
            # Validation format: ILSVRC2012_val_imageid_synset_id.JPEG
            parts = filename.split('_')
            synset_id = parts[-1].split('.')[0]
            image_id = parts[2]
        else:
            # Training format: synsetid_imageid_synset_id.JPEG
            parts = filename.split('_')
            synset_id = parts[-1].split('.')[0]
            image_id = parts[1]
        
        if synset_id not in IMAGENET2012_CLASSES:
            print(f"Unknown synset ID: {synset_id}")
            return
        
        class_folder = os.path.join(extract_path, synset_id)
        os.makedirs(class_folder, exist_ok=True)
        
        new_filename = f"{image_id}.JPEG"
        member.name = os.path.basename(member.name)
        tar.extract(member, path=class_folder)
        
        old_path = os.path.join(class_folder, filename)
        new_path = os.path.join(class_folder, new_filename)
        os.rename(old_path, new_path)
        
    except Exception as e:
        print(f"Error processing image {member.name}: {e}")

def extract_archive(archive_path, extract_path):
    print(f"Extracting {archive_path} to {extract_path}")
    
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        with tarfile.open(archive_path) as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc=f"Extracting {os.path.basename(archive_path)}") as pbar:
                for member in members:
                    process_image(member, tar, extract_path)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False

def process_archive(filename, dataset_type):
    if dataset_type == "train":
        output_dir = os.path.join("data", "train")
    elif dataset_type == "val":
        output_dir = os.path.join("data", "validation")
    else: 
        output_dir = os.path.join("data", "test")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)
    
    try:
        print(f"Downloading {filename}...")
        archive_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir="data/temp"
        )
        
        print(f"Downloaded {filename} to {archive_path}")
        success = extract_archive(archive_path, output_dir)
        os.remove(archive_path)
        
        if success:
            print(f"Successfully processed {filename}")
            return filename
        else:
            print(f"Failed to extract {filename}")
            return None
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def count_images(directory):
    """Count total number of JPEG images in a directory and its subdirectories."""
    total = 0
    for root, _, files in os.walk(directory):
        total += len([f for f in files if f.endswith('.JPEG')])
    return total

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)
    
    # print("Processing ImageNet-1k dataset from Hugging Face...")
    # print("Processing training archives...")
    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     futures = []
    #     for archive in TRAIN_ARCHIVES:
    #         futures.append(executor.submit(process_archive, archive, "train"))
        
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             archive_name = future.result()
    #             if archive_name:
    #                 print(f"Completed processing {archive_name}")
    #         except Exception as e:
    #             print(f"Error in executor: {e}")
    
    print("Processing validation archives...")
    try:
        archive_name = process_archive(VAL_ARCHIVE, "val")
        if archive_name:
            print(f"Completed processing {archive_name}")
    except Exception as e:
        print(f"Error processing validation archive: {e}")

    
    shutil.rmtree("data/temp")
    print("Dataset download and extraction completed!")

    train_count = count_images("data/train")
    val_count = count_images("data/validation")
    
    print("\nDataset Statistics:")
    print(f"Training images: {train_count:,}")
    print(f"Validation images: {val_count:,}")

if __name__ == "__main__":
    main()