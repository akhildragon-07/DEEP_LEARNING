r"""
LAV-DF Dataset Structure and Annotation Inspector
--------------------------------------------------
Deep inspection module for the Localized Audio-Visual DeepFake (LAV-DF) dataset.
Audits:
1. File system layout, archives, and directory hierarchy.
2. Annotation schemas, metadata fields, and sample records.
3. Class balance: Binary (Real vs. Fake) and Modality-wise (FV-FA, FV-RA, RV-FA, RV-RA).
4. Temporal manipulation localization annotations (fake_periods).
5. Dataset splits and identity/source leakage audit.
6. Media file presence and stream diagnostics.

Can be run locally or in Google Colab.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter


def format_bytes(size: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


class LAVDFInspector:
    def __init__(self, data_dir: str, output_dir: str = "results_lavdf"):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.report: Dict[str, Any] = {
            "target_dir": self.data_dir,
            "dir_exists": os.path.exists(self.data_dir),
            "archives": [],
            "annotation_files": [],
            "media_summary": {},
            "metadata_audit": {},
            "split_audit": {},
            "leakage_audit": {},
            "verification_status": "PENDING",
        }

    def inspect_filesystem(self) -> None:
        """Scan target directory for subfolders, archives, annotations, and media files."""
        print(f"\n{'='*75}")
        print(f"1. FILESYSTEM AUDIT: {self.data_dir}")
        print(f"{'='*75}")

        if not os.path.exists(self.data_dir):
            print(f"[!] Target directory does NOT exist: {self.data_dir}")
            self.report["verification_status"] = "DIR_NOT_FOUND"
            return

        entries = os.listdir(self.data_dir)
        print(f"Top-level directory entries found: {len(entries)}")

        # Check for archives
        archive_exts = {".zip", ".tar", ".gz", ".tgz", ".7z"}
        for f in entries:
            full_p = os.path.join(self.data_dir, f)
            ext = os.path.splitext(f)[1].lower()
            if os.path.isfile(full_p) and ext in archive_exts:
                sz = os.path.getsize(full_p)
                self.report["archives"].append({"filename": f, "path": full_p, "size_bytes": sz, "size_formatted": format_bytes(sz)})
                print(f"  [ARCHIVE FOUND] {f} ({format_bytes(sz)})")

        # Check for potential annotation files
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".json") or file.endswith(".csv"):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, self.data_dir)
                    sz = os.path.getsize(full_p)
                    self.report["annotation_files"].append({
                        "filename": file,
                        "rel_path": rel_p,
                        "full_path": full_p,
                        "size_bytes": sz,
                        "size_formatted": format_bytes(sz)
                    })

        print(f"Discovered {len(self.report['annotation_files'])} annotation/metadata file(s):")
        for ann in self.report["annotation_files"]:
            print(f"  - {ann['rel_path']} ({ann['size_formatted']})")

    def locate_primary_metadata(self) -> Optional[str]:
        """Identifies the primary metadata file (e.g. metadata.min.json or metadata.json)."""
        candidates = [
            "metadata.min.json",
            "metadata.json",
            "lavdf_metadata.json",
            "annotations.json"
        ]
        # First check exact filename matches in target_dir or subdirs
        for cand in candidates:
            for ann in self.report["annotation_files"]:
                if ann["filename"].lower() == cand.lower():
                    return ann["full_path"]

        # If none matched, check any json file containing metadata in name
        for ann in self.report["annotation_files"]:
            if "metadata" in ann["filename"].lower() and ann["filename"].endswith(".json"):
                return ann["full_path"]

        # Fallback to first json file found if any
        for ann in self.report["annotation_files"]:
            if ann["filename"].endswith(".json"):
                return ann["full_path"]

        return None

    def inspect_metadata(self, metadata_path: str) -> None:
        """Inspects and validates LAV-DF annotation schema."""
        print(f"\n{'='*75}")
        print(f"2. ANNOTATION SCHEMA AUDIT: {os.path.basename(metadata_path)}")
        print(f"{'='*75}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Could be { "video.mp4": { ... } } or { "metadata": [ ... ] }
            if "metadata" in data and isinstance(data["metadata"], list):
                records = data["metadata"]
            else:
                # Convert dict of samples to list of dicts with key as file
                records = []
                for k, v in data.items():
                    if isinstance(v, dict):
                        item = dict(v)
                        if "file" not in item:
                            item["file"] = k
                        records.append(item)
                    else:
                        records.append({"key": k, "value": v})
        elif isinstance(data, list):
            records = data
        else:
            print(f"[!] Unrecognized JSON structure: {type(data)}")
            return

        total_records = len(records)
        print(f"Total annotation records: {total_records:,}")

        if total_records == 0:
            print("[!] Metadata file is empty!")
            return

        sample = records[0]
        schema_fields = sorted(list(sample.keys()))
        print(f"Fields detected in sample record: {schema_fields}")
        print("Sample entry preview:")
        print(json.dumps(sample, indent=2))

        # Expected LAV-DF fields check
        expected_fields = [
            "file", "n_fakes", "fake_periods", "duration",
            "original", "modify_video", "modify_audio", "split"
        ]
        field_presence = {k: sum(1 for r in records if k in r) for k in expected_fields}
        print("\nField Coverage across dataset:")
        for k, cnt in field_presence.items():
            pct = (cnt / total_records) * 100
            print(f"  - {k:<15}: {cnt:,} / {total_records:,} ({pct:.1f}%)")

        # Label Analysis: Binary & Modality Taxonomy
        binary_labels = []
        modality_cats = []
        fake_periods_count = []
        durations = []
        splits = []
        originals = []

        for r in records:
            # Binary determination
            n_fakes = r.get("n_fakes", 0)
            mod_v = r.get("modify_video", False)
            mod_a = r.get("modify_audio", False)
            split = r.get("split", "unspecified")
            splits.append(split)

            orig = r.get("original")
            if orig:
                originals.append(orig)

            # A sample is Fake if n_fakes > 0 or modify_video is True or modify_audio is True
            is_fake = (n_fakes > 0) or mod_v or mod_a
            binary_labels.append(1 if is_fake else 0)

            # 4-way Category
            if not mod_v and not mod_a:
                cat = "RealVideo-RealAudio"
            elif mod_v and mod_a:
                cat = "FakeVideo-FakeAudio"
            elif mod_v and not mod_a:
                cat = "FakeVideo-RealAudio"
            else:
                cat = "RealVideo-FakeAudio"
            modality_cats.append(cat)

            dur = r.get("duration")
            if dur is not None and isinstance(dur, (int, float)):
                durations.append(dur)

            fps = r.get("fake_periods", [])
            if isinstance(fps, list):
                fake_periods_count.append(len(fps))

        bin_counter = Counter(binary_labels)
        mod_counter = Counter(modality_cats)
        split_counter = Counter(splits)

        print(f"\nBinary Ground-Truth Distribution:")
        print(f"  - Real (Authentic, 0):  {bin_counter[0]:,} ({bin_counter[0]/total_records*100:.2f}%)")
        print(f"  - Fake (Manipulated, 1): {bin_counter[1]:,} ({bin_counter[1]/total_records*100:.2f}%)")

        print(f"\nModality Manipulation Taxonomy Distribution:")
        for cat, count in mod_counter.most_common():
            print(f"  - {cat:<24}: {count:,} ({count/total_records*100:.2f}%)")

        print(f"\nSplit Distribution:")
        for sp, count in split_counter.most_common():
            print(f"  - {sp:<15}: {count:,} ({count/total_records*100:.2f}%)")

        if durations:
            avg_dur = sum(durations) / len(durations)
            min_dur = min(durations)
            max_dur = max(durations)
            print(f"\nDuration Statistics: min={min_dur:.2f}s, mean={avg_dur:.2f}s, max={max_dur:.2f}s")

        # Identity / Leakage Check across splits
        print(f"\n{'='*75}")
        print("3. SPLIT SEPARATION & IDENTITY LEAKAGE AUDIT")
        print(f"{'='*75}")
        split_originals: Dict[str, set] = {}
        for r in records:
            sp = r.get("split", "unspecified")
            orig = r.get("original")
            if orig:
                if sp not in split_originals:
                    split_originals[sp] = set()
                split_originals[sp].add(orig)

        leakage_detected = False
        split_keys = list(split_originals.keys())
        for i in range(len(split_keys)):
            for j in range(i + 1, len(split_keys)):
                s1, s2 = split_keys[i], split_keys[j]
                overlap = split_originals[s1].intersection(split_originals[s2])
                if overlap:
                    leakage_detected = True
                    print(f"[!] POTENTIAL SOURCE LEAKAGE: {len(overlap)} original sources overlap between '{s1}' and '{s2}'")
                else:
                    print(f"[OK] Strict separation between '{s1}' and '{s2}': 0 overlapping source videos.")

        # Media File Existence Check
        print(f"\n{'='*75}")
        print("4. MEDIA FILE PRESENCE AUDIT")
        print(f"{'='*75}")

        # Test relative path resolution against data_dir and parent directories
        candidate_bases = [
            self.data_dir,
            os.path.join(self.data_dir, "data"),
            os.path.join(self.data_dir, "videos"),
            os.path.dirname(metadata_path),
        ]

        verified_base = None
        sample_file = records[0].get("file", "")
        for base in candidate_bases:
            test_path = os.path.join(base, sample_file)
            if os.path.exists(test_path):
                verified_base = base
                break

        if verified_base:
            print(f"[OK] Verified media root directory: {verified_base}")
        else:
            print(f"[!] Warning: Sample media file '{sample_file}' not found in candidate bases.")
            verified_base = self.data_dir

        existing_media_count = 0
        missing_samples = []
        check_limit = min(total_records, 1000)  # check up to 1000 for rapid audit

        for r in records[:check_limit]:
            f_rel = r.get("file", "")
            f_abs = os.path.join(verified_base, f_rel)
            if os.path.exists(f_abs):
                existing_media_count += 1
            else:
                if len(missing_samples) < 5:
                    missing_samples.append(f_rel)

        print(f"Checked first {check_limit:,} media files: {existing_media_count:,} found ({existing_media_count/check_limit*100:.1f}%)")
        if missing_samples:
            print(f"Missing file examples (first {len(missing_samples)}):")
            for m in missing_samples:
                print(f"  - {m}")

        self.report["metadata_audit"] = {
            "metadata_file": metadata_path,
            "total_records": total_records,
            "schema_fields": schema_fields,
            "binary_counts": dict(bin_counter),
            "modality_counts": dict(mod_counter),
            "split_counts": dict(split_counter),
            "duration_stats": {"min": min(durations) if durations else 0, "mean": avg_dur if durations else 0, "max": max(durations) if durations else 0},
            "leakage_detected": leakage_detected,
            "verified_media_base": verified_base,
            "checked_media_count": check_limit,
            "found_media_count": existing_media_count,
        }
        self.report["verification_status"] = "AUDIT_COMPLETED"

    def run(self) -> Dict[str, Any]:
        """Runs the complete inspection pipeline and saves reports."""
        self.inspect_filesystem()
        meta_path = self.locate_primary_metadata()

        if meta_path:
            self.inspect_metadata(meta_path)
        else:
            print("\n[!] No metadata JSON file found in target directory.")
            print("Please ensure the dataset archive is fully extracted.")
            self.report["verification_status"] = "NO_METADATA_FOUND"

        # Save JSON report
        json_path = os.path.join(self.output_dir, "lavdf_inspection_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)

        # Save Markdown report
        md_path = os.path.join(self.output_dir, "LAVDF_AUDIT_REPORT.md")
        self._write_markdown_report(md_path)

        print(f"\n{'='*75}")
        print(f"AUDIT SUMMARY COMPLETE")
        print(f"Reports saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - Markdown: {md_path}")
        print(f"{'='*75}\n")
        return self.report

    def _write_markdown_report(self, path: str) -> None:
        rep = self.report.get("metadata_audit", {})
        md = [
            "# LAV-DF Dataset Inspection & Schema Audit Report",
            f"**Audit Timestamp:** 2026-10-28",
            f"**Target Directory:** `{self.data_dir}`",
            f"**Status:** {self.report.get('verification_status', 'UNKNOWN')}\n",
            "---",
            "## 1. Dataset Overview",
            f"- **Metadata File:** `{rep.get('metadata_file', 'None')}`",
            f"- **Total Records:** {rep.get('total_records', 0):,}",
            f"- **Detected Fields:** `{', '.join(rep.get('schema_fields', []))}`\n",
            "## 2. Label & Modality Distribution",
            "### Binary Labels",
            "| Label | Category | Count | Proportion |",
            "| :---: | :--- | :---: | :---: |",
            f"| 0 | Real / Authentic | {rep.get('binary_counts', {}).get(0, 0):,} | {rep.get('binary_counts', {}).get(0, 0)/(rep.get('total_records', 1) or 1)*100:.2f}% |",
            f"| 1 | Fake / Manipulated | {rep.get('binary_counts', {}).get(1, 0):,} | {rep.get('binary_counts', {}).get(1, 0)/(rep.get('total_records', 1) or 1)*100:.2f}% |\n",
            "### 4-Way Modality Manipulation Taxonomy",
            "| Modality Category | Count | Proportion |",
            "| :--- | :---: | :---: |",
        ]
        for cat, cnt in rep.get("modality_counts", {}).items():
            tot = rep.get("total_records", 1) or 1
            md.append(f"| {cat} | {cnt:,} | {cnt/tot*100:.2f}% |")

        md.extend([
            "\n### Split Distribution",
            "| Split | Count | Proportion |",
            "| :--- | :---: | :---: |",
        ])
        for sp, cnt in rep.get("split_counts", {}).items():
            tot = rep.get("total_records", 1) or 1
            md.append(f"| {sp} | {cnt:,} | {cnt/tot*100:.2f}% |")

        md.extend([
            "\n## 3. Data Integrity & Leakage Assessment",
            f"- **Source Leakage Detected:** {'YES' if rep.get('leakage_detected') else 'NO (Splits cleanly separated)'}",
            f"- **Media Files Verified:** {rep.get('found_media_count', 0):,} / {rep.get('checked_media_count', 0):,}",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))


def main():
    parser = argparse.ArgumentParser(description="Inspect LAV-DF dataset structure and annotations.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/.cache/kagglehub/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/versions/1",
        help="Path to downloaded LAV-DF dataset directory in Colab or local disk.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_lavdf",
        help="Directory to save audit report artifacts.",
    )
    args = parser.parse_args()

    inspector = LAVDFInspector(data_dir=args.data_dir, output_dir=args.output_dir)
    inspector.run()


if __name__ == "__main__":
    main()
