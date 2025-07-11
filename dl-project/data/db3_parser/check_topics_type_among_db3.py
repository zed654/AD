#!/usr/bin/env python3
import sqlite3
import glob
import os
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

def check_db3_topics():
    db3_dir = os.path.join(os.path.dirname(__file__), "db3")
    db3_files = glob.glob(os.path.join(db3_dir, "*.db3"))
    
    if not db3_files:
        print("No db3 files found!")
        return
    
    db3_path = db3_files[0]
    print(f"\n=== Checking topics in {db3_path} ===\n")
    
    storage_options = StorageOptions(uri=db3_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr", 
        output_serialization_format="cdr"
    )
    
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    
    # 모든 토픽과 타입 정보 가져오기
    topic_types = reader.get_all_topics_and_types()
    
    print("Available topics:")
    for topic_info in topic_types:
        print(f"  Topic: {topic_info.name}")
        print(f"  Type:  {topic_info.type}")
        print()
        
        # ldm_msg 토픽 강조
        if 'ldm_msg' in topic_info.name.lower():
            print(f"  >>> LDM_MSG topic found! <<<")
            print()

if __name__ == "__main__":
    check_db3_topics()