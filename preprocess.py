"""Extract and organise CT data from RAR archives."""

import argparse
import os
import subprocess


def extract(args):
    os.makedirs(args.extract_dir, exist_ok=True)

    for rar_file in sorted(os.listdir(args.source_dir)):
        if not rar_file.endswith(".rar"):
            continue
        full_path = os.path.join(args.source_dir, rar_file)
        folder_name = rar_file.replace(".rar", "")
        dest = os.path.join(args.extract_dir, folder_name)
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["unrar", "x", "-o+", full_path, f"{dest}/"], check=True)
        print(f"Extracted: {rar_file}")

    print("\nExtraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CT data from RAR archives")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing .rar files")
    parser.add_argument("--extract_dir", type=str, default="./data/extracted", help="Output directory")
    args = parser.parse_args()
    extract(args)
