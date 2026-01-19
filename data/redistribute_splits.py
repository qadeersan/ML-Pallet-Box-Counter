"""
Redistribute Dataset Splits to 70/20/10

This script consolidates all images from train/valid/test folders
and redistributes them to achieve 70% train, 20% valid, 10% test splits.

Usage:
    python data/redistribute_splits.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Target split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

def get_all_images_and_labels(dataset_path):
    """Collect all image-label pairs from all splits."""
    dataset_path = Path(dataset_path)
    all_pairs = []
    
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists():
            continue
            
        for img_file in images_dir.glob('*.jpg'):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                all_pairs.append({
                    'image': img_file,
                    'label': label_file,
                    'stem': img_file.stem
                })
    
    return all_pairs

def redistribute_dataset(dataset_path, dataset_name):
    """Redistribute a single dataset to 70/20/10 splits."""
    dataset_path = Path(dataset_path)
    
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Collect all pairs
    all_pairs = get_all_images_and_labels(dataset_path)
    total = len(all_pairs)
    
    print(f"Total images found: {total}")
    
    # Shuffle
    random.shuffle(all_pairs)
    
    # Calculate split sizes
    train_size = int(total * TRAIN_RATIO)
    valid_size = int(total * VALID_RATIO)
    test_size = total - train_size - valid_size  # Remainder goes to test
    
    print(f"Target splits: Train={train_size}, Valid={valid_size}, Test={test_size}")
    
    # Split the data
    train_pairs = all_pairs[:train_size]
    valid_pairs = all_pairs[train_size:train_size + valid_size]
    test_pairs = all_pairs[train_size + valid_size:]
    
    # Create temporary directory for reorganization
    temp_dir = dataset_path / '_temp_reorganize'
    temp_dir.mkdir(exist_ok=True)
    
    # Create temp split directories
    for split in ['train', 'valid', 'test']:
        (temp_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (temp_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files to temp directory with new organization
    splits = {
        'train': train_pairs,
        'valid': valid_pairs,
        'test': test_pairs
    }
    
    for split_name, pairs in splits.items():
        print(f"  Copying {len(pairs)} files to {split_name}...")
        for pair in pairs:
            # Copy image
            dst_img = temp_dir / split_name / 'images' / pair['image'].name
            shutil.copy2(pair['image'], dst_img)
            
            # Copy label
            dst_label = temp_dir / split_name / 'labels' / pair['label'].name
            shutil.copy2(pair['label'], dst_label)
    
    # Remove old directories and rename temp
    print("  Replacing old directories...")
    for split in ['train', 'valid', 'test']:
        old_dir = dataset_path / split
        if old_dir.exists():
            shutil.rmtree(old_dir)
        
        # Move from temp to final location
        shutil.move(str(temp_dir / split), str(old_dir))
    
    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Verify
    print("\nVerification:")
    for split in ['train', 'valid', 'test']:
        img_count = len(list((dataset_path / split / 'images').glob('*.jpg')))
        label_count = len(list((dataset_path / split / 'labels').glob('*.txt')))
        pct = (img_count / total) * 100
        print(f"  {split}: {img_count} images, {label_count} labels ({pct:.1f}%)")
    
    return {
        'total': total,
        'train': len(train_pairs),
        'valid': len(valid_pairs),
        'test': len(test_pairs)
    }

def main():
    """Main function to redistribute both datasets."""
    root_dir = Path(__file__).parent.parent
    
    datasets = [
        ('Boxes.v1i.yolov8', root_dir / 'Boxes.v1i.yolov8'),
        ('Final_Object_Detection', root_dir / 'Final_Object_Detection.v1i.yolov8')
    ]
    
    print("=" * 60)
    print("Dataset Redistribution: 70% Train / 20% Valid / 10% Test")
    print("=" * 60)
    
    results = {}
    for name, path in datasets:
        if path.exists():
            results[name] = redistribute_dataset(path, name)
        else:
            print(f"\n⚠ Dataset not found: {path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_train = sum(r['train'] for r in results.values())
    total_valid = sum(r['valid'] for r in results.values())
    total_test = sum(r['test'] for r in results.values())
    total_all = sum(r['total'] for r in results.values())
    
    print(f"\nCombined Dataset Statistics:")
    print(f"  Total images: {total_all}")
    print(f"  Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"  Valid: {total_valid} ({total_valid/total_all*100:.1f}%)")
    print(f"  Test:  {total_test} ({total_test/total_all*100:.1f}%)")
    
    print("\n✓ Redistribution complete!")
    print("\nNext steps:")
    print("  1. Verify data: python data/download_data.py --verify-only")
    print("  2. Start training: python src/train_yolo.py")

if __name__ == '__main__':
    main()

