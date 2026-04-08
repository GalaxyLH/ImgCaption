#!/usr/bin/env python3
"""
Merge newly generated satellite image descriptions with old text descriptions.
Combines environment descriptions with retrieval task descriptions.
Saves merged results to text/new folder.
"""

import pickle
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple


def load_new_captions(captions_file: Path) -> Dict[str, str]:
    """
    Load newly generated satellite image captions.
    
    Args:
        captions_file: Path to captions.json file.
        
    Returns:
        Dictionary mapping satellite image filename to caption.
    """
    print(f"Loading new captions from {captions_file}...")
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    satellite_to_caption = {}
    for item in captions_data:
        img_file = item.get('image_file', '')
        if not img_file:
            img_path = item.get('image_path', '')
            if img_path:
                img_file = Path(img_path).name
        
        caption = item.get('caption', '')
        # 跳过错误描述
        if img_file and caption:
            is_error = (
                "I'm sorry" in caption or 
                "I can't" in caption or 
                "cannot" in caption.lower() or
                "can't assist" in caption.lower() or
                "can't analyze" in caption.lower() or
                "I'm unable" in caption or
                "can't help" in caption.lower()
            )
            if not is_error:
                satellite_to_caption[img_file] = caption
    
    print(f"Loaded {len(satellite_to_caption)} valid new captions.")
    return satellite_to_caption


def extract_find_statement_and_supplement(old_english: str) -> Tuple[str, str]:
    """
    Extract Find statement and area/distance supplement from old English description.
    
    Args:
        old_english: Original English description.
        
    Returns:
        Tuple of (find_statement, supplement).
    """
    old_english = old_english.strip()
    
    # Find statement typically starts with "Find" and ends before "The area" or "The distance"
    # Or it might be the entire description if no supplement
    
    # Check if there's a supplement (area/distance information)
    if "The area" in old_english or "The distance" in old_english:
        # Split at "The area" or "The distance"
        parts = old_english.split("The area")
        if len(parts) == 1:
            parts = old_english.split("The distance")
        
        if len(parts) >= 2:
            find_statement = parts[0].strip()
            # Remove trailing period if exists
            if find_statement.endswith('.'):
                find_statement = find_statement[:-1]
            supplement = "The area" + parts[1] if "The area" in old_english else "The distance" + parts[1]
            return find_statement, supplement
    
    # If no supplement found, return entire description as find statement
    # Remove trailing period if exists
    find_statement = old_english
    if find_statement.endswith('.'):
        find_statement = find_statement[:-1]
    return find_statement, ""


def merge_descriptions(
    old_english: str,
    new_caption: str,
    connector: str = "Specifically, the target is to "
) -> str:
    """
    Merge new environment description with old retrieval description.
    
    Template: [GPT生成的环境描述] + " Specifically, the target is to " + [Find语句] + [面积/距离补充]
    
    Args:
        old_english: Original English description (Find statement + supplement).
        new_caption: Newly generated environment description.
        connector: Connector phrase between descriptions.
        
    Returns:
        Merged description string.
    """
    # Extract Find statement and supplement
    find_statement, supplement = extract_find_statement_and_supplement(old_english)
    
    # Ensure find statement starts with lowercase "find" after connector
    find_lower = find_statement.strip()
    if find_lower.startswith('Find '):
        find_lower = 'find ' + find_lower[5:]
    elif find_lower.startswith('find '):
        pass  # Already lowercase
    else:
        # If doesn't start with Find/find, keep as is
        find_lower = find_statement.strip()
    
    # Combine: new description + connector + find statement + supplement
    merged = f"{new_caption.strip()} {connector}{find_lower}"
    if supplement:
        # Ensure supplement starts with a space if it doesn't already
        supplement = supplement.strip()
        if not supplement.startswith('.'):
            merged += f". {supplement}"
        else:
            merged += f" {supplement}"
    
    return merged


def update_pickle_file(
    pickle_file: Path,
    satellite_to_caption: Dict[str, str],
    output_dir: Path
) -> Tuple[int, int]:
    """
    Update pickle file by merging new descriptions with old ones.
    
    Args:
        pickle_file: Path to pickle file to update.
        satellite_to_caption: Dictionary mapping satellite filename to new caption.
        output_dir: Output directory for merged files.
        
    Returns:
        Tuple of (updated_count, missing_count).
    """
    print(f"\nProcessing {pickle_file.name}...")
    
    # Load pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} records.")
    
    # Update records
    updated_count = 0
    missing_count = 0
    
    for i, item in enumerate(data):
        if len(item) < 8:
            print(f"Warning: Record {i} has incorrect structure (expected 8 fields, got {len(item)})")
            continue
        
        satellite_file = item[0]  # Satellite image filename
        
        # Find corresponding new caption
        if satellite_file in satellite_to_caption:
            new_caption = satellite_to_caption[satellite_file]
            old_english = item[2]  # Original English description
            
            # Merge descriptions
            merged_english = merge_descriptions(old_english, new_caption)
            
            # Update the record (create new tuple with updated fields)
            updated_item = (
                item[0],  # satellite_filename
                item[1],  # sketch_filename
                merged_english,  # [2] Updated English description
                item[3],  # bbox
                item[4],  # [4] Chinese description (keep original)
                item[5],  # coordinates
                item[6],  # number
                item[7]   # pos_tags
            )
            
            data[i] = updated_item
            updated_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing
                print(f"Warning: No new caption found for {satellite_file}")
    
    # Save updated pickle file to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / pickle_file.name
    
    print(f"Updated: {updated_count}, Missing: {missing_count}")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved merged file: {output_file}")
    
    return updated_count, missing_count


def main():
    """Main function to merge descriptions."""
    # Paths
    base_dir = Path(__file__).parent
    captions_file = base_dir / "output" / "captions.json"
    old_pickle_dir = base_dir / "text" / "old" / "sRSVG"
    output_dir = base_dir / "text" / "new" / "sRSVG"
    
    # Check files exist
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    if not old_pickle_dir.exists():
        raise FileNotFoundError(f"Old pickle directory not found: {old_pickle_dir}")
    
    # Load new captions
    satellite_to_caption = load_new_captions(captions_file)
    
    # Process all pickle files
    pickle_files = [
        old_pickle_dir / "sRSVG_train.pickle",
        old_pickle_dir / "sRSVG_val.pickle",
        old_pickle_dir / "sRSVG_test.pickle"
    ]
    
    total_updated = 0
    total_missing = 0
    
    print("\n" + "=" * 80)
    print("Merging Descriptions")
    print("=" * 80)
    
    for pickle_file in pickle_files:
        if not pickle_file.exists():
            print(f"Warning: {pickle_file.name} not found, skipping...")
            continue
        
        updated, missing = update_pickle_file(
            pickle_file,
            satellite_to_caption,
            output_dir
        )
        
        total_updated += updated
        total_missing += missing
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total records updated: {total_updated}")
    print(f"Total records missing new caption: {total_missing}")
    if total_updated + total_missing > 0:
        print(f"Success rate: {total_updated/(total_updated+total_missing)*100:.2f}%")
    print(f"\n✓ Merge completed!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
