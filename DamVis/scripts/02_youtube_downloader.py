"""
SCRIPT 2 — YouTube CC Video Downloader + Frame Extractor
=========================================================
Downloads Creative Commons dam/reservoir drone videos from YouTube
and extracts frames at 1 fps into dataset/clean/

Usage:
    python 02_youtube_downloader.py

Requirements:
    pip install yt-dlp
    sudo apt install ffmpeg  (Linux/Mac)
    Windows: download ffmpeg from ffmpeg.org and add to PATH
"""

import os
import subprocess
import sys
import json
from pathlib import Path

# ── Pre-identified CC-licensed YouTube search queries ─────────────────────────
# You will need to manually search YouTube, filter by Creative Commons,
# and paste the URLs here. Instructions printed at bottom of script.

# PASTE YOUR VIDEO URLs HERE after searching YouTube:
VIDEO_URLS = [
    # Format: ("URL", "short_name_for_files")
    # Example (replace with real CC-licensed videos you find):
    # ("https://www.youtube.com/watch?v=XXXXXXXXX", "dam_tehri_aerial"),
    # ("https://www.youtube.com/watch?v=YYYYYYYYY", "reservoir_drone"),
]

OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "clean"
VIDEO_TEMP = Path(__file__).parent.parent / "dataset" / "_video_temp"

def check_dependencies():
    """Check yt-dlp and ffmpeg are installed."""
    ok = True
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        print("✅ yt-dlp found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp not found. Run: pip install yt-dlp")
        ok = False

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ ffmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found.")
        print("   Linux:   sudo apt install ffmpeg")
        print("   Mac:     brew install ffmpeg")
        print("   Windows: download from https://ffmpeg.org/download.html")
        ok = False
    return ok

def download_video(url, name, output_dir):
    """Download a single YouTube video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.%(ext)s"

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio/best[height<=1080]",
        "--output", str(output_path),
        "--no-playlist",
        "--write-info-json",  # saves metadata including licence info
        url
    ]
    print(f"\n⬇  Downloading: {name}")
    print(f"   URL: {url}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print(f"✅ Downloaded: {name}")
        return True
    else:
        print(f"❌ Failed: {name}")
        return False

def extract_frames(video_path, output_dir, fps=1, prefix="yt"):
    """Extract frames from video at given fps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / f"{prefix}_%04d.jpg")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",           # 1 frame per second
        "-q:v", "2",                    # high quality JPEG
        "-vf", f"fps={fps},scale=1280:720",  # resize to 720p
        output_pattern,
        "-y"
    ]
    print(f"\n🎞  Extracting frames: {video_path.name}")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        frames = list(output_dir.glob(f"{prefix}_*.jpg"))
        print(f"✅ Extracted {len(frames)} frames")
        return len(frames)
    else:
        print(f"❌ Frame extraction failed")
        return 0

def verify_cc_licence(info_json_path):
    """Read yt-dlp metadata and confirm Creative Commons licence."""
    try:
        with open(info_json_path) as f:
            info = json.load(f)
        licence = info.get("license", "unknown")
        title = info.get("title", "unknown")
        uploader = info.get("uploader", "unknown")
        print(f"   Title    : {title}")
        print(f"   Uploader : {uploader}")
        print(f"   Licence  : {licence}")
        if "creative commons" in licence.lower():
            print("   ✅ Confirmed Creative Commons licence")
            return True
        else:
            print(f"   ⚠️  Licence may not be CC: {licence}")
            return False
    except Exception as e:
        print(f"   ⚠️  Could not verify licence: {e}")
        return False

def build_source_log(videos_processed):
    """Save a CSV log of all video sources for dataset paper citation."""
    log_path = Path(__file__).parent.parent / "metadata" / "youtube_sources.csv"
    with open(log_path, "w") as f:
        f.write("url,name,frames_extracted,licence_confirmed,date_accessed\n")
        from datetime import date
        today = str(date.today())
        for v in videos_processed:
            f.write(f"{v['url']},{v['name']},{v['frames']},{v['cc_confirmed']},{today}\n")
    print(f"\n✅ Source log saved: {log_path}")
    print("   Keep this file — it's your citation trail for the dataset paper")

def run():
    print("=" * 60)
    print("  DAMVIS — YouTube CC Video Downloader")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    if not VIDEO_URLS:
        print_search_instructions()
        return

    VIDEO_TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos_processed = []
    total_frames = 0

    for url, name in VIDEO_URLS:
        # Download
        downloaded = download_video(url, name, VIDEO_TEMP)
        if not downloaded:
            continue

        # Find the downloaded file
        video_files = list(VIDEO_TEMP.glob(f"{name}.*"))
        video_file = next((f for f in video_files if f.suffix in ['.mp4', '.mkv', '.webm']), None)
        if not video_file:
            print(f"❌ Could not find downloaded file for {name}")
            continue

        # Verify licence
        info_files = list(VIDEO_TEMP.glob(f"{name}*.info.json"))
        cc_ok = verify_cc_licence(info_files[0]) if info_files else False

        # Extract frames
        n_frames = extract_frames(video_file, OUTPUT_DIR, fps=1, prefix=f"yt_{name}")
        total_frames += n_frames

        videos_processed.append({
            "url": url, "name": name,
            "frames": n_frames, "cc_confirmed": cc_ok
        })

    build_source_log(videos_processed)
    print(f"\n{'='*60}")
    print(f"  DONE: {len(videos_processed)} videos → {total_frames} frames")
    print(f"  All frames saved to: {OUTPUT_DIR}")
    print("=" * 60)

def print_search_instructions():
    print("\n" + "="*60)
    print("  HOW TO FIND CC-LICENSED DAM VIDEOS ON YOUTUBE")
    print("="*60)
    queries = [
        "dam aerial drone footage",
        "reservoir drone view",
        "embankment dam inspection UAV",
        "dam spillway aerial",
        "hydroelectric dam aerial view",
        "dam crest drone",
        "flood dam aerial footage",
        "earthen dam drone",
        "Tehri dam drone",
        "Hirakud dam aerial",
        "Bhakra dam aerial view",
    ]
    print("\n1. Go to: https://www.youtube.com")
    print("2. Search each query below")
    print("3. Click: Filters → Features → Creative Commons")
    print("4. Copy relevant video URLs")
    print("5. Paste into VIDEO_URLS list in this script")
    print("\nSearch queries to use:")
    for i, q in enumerate(queries, 1):
        yt_url = f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}&sp=EgIgAQ%253D%253D"
        print(f"   {i:02d}. \"{q}\"")
        print(f"       Direct CC search URL: {yt_url}")
    print("\n⚠️  IMPORTANT: For each video, confirm licence says")
    print("   'Creative Commons Attribution licence (reuse allowed)'")
    print("   before adding to VIDEO_URLS list")

if __name__ == "__main__":
    run()
