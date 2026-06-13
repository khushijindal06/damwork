"""
SCRIPT 1 — Google Earth Dam Coordinate Guide
=============================================
Run this script to get a printed checklist of all 25 Indian dams
with exact coordinates, what to capture, and how many screenshots to take.

Usage:
    python 01_google_earth_guide.py

It will also generate a KML file you can import directly into
Google Earth — it places pins on every dam automatically.
"""

import json
import os

# ── 25 Indian Dams with coordinates and capture instructions ──────────────────
DAMS = [
    # (Name, Lat, Lon, Type, State, Height_m, Target_Screenshots, Notes)
    ("Tehri",           30.3778,  78.4806,  "Rock-fill embankment", "Uttarakhand",  260, 80,
     "Steep valley walls. Capture: dam face, crest road, downstream slope, reservoir edge, spillway"),

    ("Hirakud",         21.5167,  83.8667,  "Earthen embankment",   "Odisha",        61, 100,
     "World's longest earthen dam (25.8 km). Capture along full length in segments, both flanks"),

    ("Bhakra",          31.4108,  76.4333,  "Concrete gravity",     "Himachal Pradesh", 226, 70,
     "Vertical concrete face. Capture: face texture, crest, downstream, Gobind Sagar reservoir"),

    ("Sardar Sarovar",  21.8289,  73.7496,  "Concrete gravity",     "Gujarat",       163, 70,
     "Arid environment — dust haze context. Capture: dam body, canal headworks, reservoir"),

    ("Nagarjuna Sagar", 16.5743,  79.3129,  "Masonry",              "Telangana",     124, 60,
     "World's largest masonry dam. Capture: masonry face, 26 spillway gates, reservoir"),

    ("Koyna",           17.4000,  73.7500,  "Masonry",              "Maharashtra",   103, 60,
     "Dense vegetation on slopes. Good for vegetation anomaly annotation class"),

    ("Idukki",           9.8500,  76.9667,  "Arch",                 "Kerala",        169, 50,
     "Double curvature arch dam. Capture arch face, canyon walls, dense forest surroundings"),

    ("Mettur",          11.7833,  77.8000,  "Masonry gravity",      "Tamil Nadu",     54, 50,
     "Long dam with multiple sections. Capture: dam body sections, Stanley reservoir edge"),

    ("Tungabhadra",     15.2600,  76.3300,  "Masonry",              "Karnataka",      49, 50,
     "Composite masonry/earthen. Capture both dam sections and junction area"),

    ("Rihand",          24.1900,  83.0200,  "Concrete gravity",     "Uttar Pradesh", 91, 60,
     "Govind Ballabh Pant reservoir. Good for haze context — industrial region nearby"),

    ("Kundah",          11.3000,  76.7833,  "Masonry",              "Tamil Nadu",     90, 40,
     "Series of dams. Capture individual structures and connecting channels"),

    ("Bansagar",        24.1833,  81.3167,  "Concrete gravity",     "Madhya Pradesh",67, 50,
     "Son river dam. Capture: main body, left and right earthen flanks, spillway"),

    ("Panchet",         23.7667,  86.6167,  "Earthen + masonry",    "Jharkhand",      46, 50,
     "DVC dam. Capture: composite structure, Maithon-Panchet reservoir system"),

    ("Maithon",         23.8167,  86.8667,  "Earthen",              "Jharkhand",      50, 50,
     "Earthen dam with masonry spillway section. Good embankment slope capture"),

    ("Jayakwadi",       19.4667,  75.4167,  "Earthen",              "Maharashtra",    41, 40,
     "Nathsagar reservoir. Long earthen embankment — varied slope conditions"),

    ("Gandhi Sagar",    24.7000,  75.5500,  "Masonry gravity",      "Madhya Pradesh", 62, 50,
     "Chambal river. Capture: stepped downstream face, large reservoir, wildlife on banks"),

    ("Srisailam",       16.0800,  78.8800,  "Masonry gravity",      "Andhra Pradesh",145, 60,
     "Deep gorge setting. Excellent for valley fog simulation context"),

    ("Indirasagar",     22.2833,  76.4667,  "Earthen + concrete",   "Madhya Pradesh", 92, 50,
     "Largest reservoir in India. Capture: main dam, saddle dams, reservoir islands"),

    ("Ukai",            21.2500,  73.5667,  "Earthen + masonry",    "Gujarat",        68, 40,
     "Composite structure. Tapi river. Capture different dam segments"),

    ("Tilaiya",         24.1500,  85.4333,  "Masonry gravity",      "Jharkhand",      30, 30,
     "DVC chain dam. Smaller structure — good for variety in scale"),

    ("Kabini",          11.9167,  76.3500,  "Earthen",              "Karnataka",      32, 30,
     "Dense forest surroundings — vegetation annotation class richness"),

    ("Almatti",         16.3333,  75.8833,  "Masonry gravity",      "Karnataka",      52, 40,
     "Krishna river. Capture: dam sections, reservoir, downstream irrigation network"),

    ("Girna",           20.5833,  74.6667,  "Earthen",              "Maharashtra",    44, 30,
     "Earthen dam. Capture full crest length, both slopes, stilling basin"),

    ("Ramganga",        29.7167,  78.8833,  "Earthen",              "Uttarakhand",   128, 50,
     "Himalayas foothills. High fog probability — perfect valley fog context"),

    ("Chambal",         24.8833,  75.6000,  "Masonry",              "Rajasthan",      55, 40,
     "Arid setting. Dust haze natural context. Gandhi Sagar reservoir system"),
]

def print_capture_checklist():
    total = sum(d[6] for d in DAMS)
    print("=" * 70)
    print("  DAMVIS DATASET — GOOGLE EARTH CAPTURE CHECKLIST")
    print(f"  25 dams | Target: {total} screenshots")
    print("=" * 70)

    for i, dam in enumerate(DAMS, 1):
        name, lat, lon, dtype, state, height, target, notes = dam
        print(f"\n{'─'*60}")
        print(f"  {i:02d}. {name} Dam  [{state}]")
        print(f"      Type   : {dtype}")
        print(f"      Height : {height} m")
        print(f"      Coords : {lat}°N, {lon}°E")
        print(f"      Target : {target} screenshots")
        print(f"      Focus  : {notes}")
        print(f"      GE URL : https://earth.google.com/web/@{lat},{lon},500a,1000d")

    print(f"\n{'='*60}")
    print(f"  TOTAL TARGET: {total} clean screenshots")
    print(f"  + 1,000 from YouTube CC videos")
    print(f"  = ~{total + 1000} clean images total")
    print(f"  × 12 degradation variants")
    print(f"  = ~{(total + 1000) * 12:,} training pairs")
    print("=" * 60)

def generate_kml():
    """Generate a KML file to import into Google Earth — places pins on all dams."""
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '<name>DamVis Dataset — 25 Indian Dams</name>',
    ]
    for dam in DAMS:
        name, lat, lon, dtype, state, height, target, notes = dam
        kml_lines += [
            '<Placemark>',
            f'  <name>{name} Dam ({state})</name>',
            f'  <description>{dtype} | {height}m | Target: {target} screenshots | {notes}</description>',
            '  <Point>',
            f'    <coordinates>{lon},{lat},0</coordinates>',
            '  </Point>',
            '</Placemark>',
        ]
    kml_lines += ['</Document>', '</kml>']

    kml_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "damvis_dams.kml")
    with open(kml_path, "w") as f:
        f.write("\n".join(kml_lines))
    print(f"\n✅ KML file saved: {kml_path}")
    print("   → Open Google Earth Pro → File → Import → select this KML file")
    print("   → All 25 dam pins appear on the map automatically")

def generate_json():
    """Save dam coordinates as JSON for use by other scripts."""
    data = []
    for dam in DAMS:
        name, lat, lon, dtype, state, height, target, notes = dam
        data.append({
            "name": name, "lat": lat, "lon": lon,
            "type": dtype, "state": state,
            "height_m": height, "target_screenshots": target,
            "capture_notes": notes
        })
    json_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "dams.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Dam coordinates JSON saved: {json_path}")

if __name__ == "__main__":
    print_capture_checklist()
    generate_kml()
    generate_json()

    print("\n📋 GOOGLE EARTH CAPTURE INSTRUCTIONS:")
    print("─" * 50)
    print("1. Open Google Earth Pro (free download: earth.google.com/web)")
    print("2. File → Import → select damvis_dams.kml (all pins load)")
    print("3. Click a dam pin → it flies you there")
    print("4. Use altitude slider: set to 100–200m for UAV-like view")
    print("5. Press Ctrl+Alt+S (or File → Save → Save Image) to screenshot")
    print("6. Move slightly (pan, rotate, zoom) and take another")
    print("7. Capture: dam face, crest, downstream slope, spillway, reservoir edge")
    print("8. Save as: dam_<name>_<number>.jpg in dataset/clean/")
    print("9. Move to next dam pin")
    print("\n⏱  Estimated time: 2–3 weeks (1–2 hours/day)")
    print("🎯 Target: 1,200–1,500 screenshots from Google Earth")
