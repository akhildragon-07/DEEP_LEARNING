"""
Creates the DEMO_DATASET_FOR_FACULTY_UPLOAD folder.
Populates it with categorized, ready-to-upload files for:
- Mode A: Still Image (Visual-Only)
- Mode B: Video Clip (Multi-Frame Visual)
- Mode C: Audio Track (Acoustic Log-Mel)
- Mode D: Paired Audio + Video (RMS-AV vs Standard Fusion)
"""

import os
import shutil

BASE_DIR = "DEMO_DATASET_FOR_FACULTY_UPLOAD"
os.makedirs(BASE_DIR, exist_ok=True)

# Subfolders
dir_real = os.path.join(BASE_DIR, "01_REAL_AUTHENTIC_SAMPLES")
dir_faceswap = os.path.join(BASE_DIR, "02_ASYMMETRIC_FACESWAP_DEEPFAKES")
dir_voiceclone = os.path.join(BASE_DIR, "03_ASYMMETRIC_VOICECLONE_DEEPFAKES")
dir_dual = os.path.join(BASE_DIR, "04_DUAL_FORGERY_DEEPFAKES")

for d in [dir_real, dir_faceswap, dir_voiceclone, dir_dual]:
    os.makedirs(d, exist_ok=True)

print("Populating categorized faculty upload folder...")

# ---------------------------------------------------------
# 1. REAL / AUTHENTIC SAMPLES
# ---------------------------------------------------------
# Images (Mode A)
src_img_real1 = "teammates/zip2/frames/video_036/frame_00.jpg"
src_img_real2 = "teammates/zip2/frames/video_033/frame_00.jpg"
if os.path.exists(src_img_real1):
    shutil.copyfile(src_img_real1, os.path.join(dir_real, "real_face_portrait_broadcast.jpg"))
if os.path.exists(src_img_real2):
    shutil.copyfile(src_img_real2, os.path.join(dir_real, "real_face_portrait_interview.jpg"))

# Video (Mode B & D)
src_vid_real = "demo_samples/01_Authentic_RealVideo_RealAudio.mp4"
if os.path.exists(src_vid_real):
    shutil.copyfile(src_vid_real, os.path.join(dir_real, "real_broadcast_video.mp4"))
    shutil.copyfile(src_vid_real, os.path.join(dir_real, "real_multimodal_video.mp4"))

# Audio (Mode C & D)
src_aud_real = "demo_samples/01_Authentic_RealAudio.wav"
if os.path.exists(src_aud_real):
    shutil.copyfile(src_aud_real, os.path.join(dir_real, "real_speech_audio.wav"))
    shutil.copyfile(src_aud_real, os.path.join(dir_real, "real_multimodal_audio.wav"))

# ---------------------------------------------------------
# 2. ASYMMETRIC FACESWAP DEEPFAKES (Visual Manipulated + Real Audio)
# ---------------------------------------------------------
# Images (Mode A)
src_img_swap1 = "teammates/zip2/frames/video_013/frame_00.jpg"
src_img_swap2 = "teammates/zip2/frames/video_016/frame_00.jpg"
if os.path.exists(src_img_swap1):
    shutil.copyfile(src_img_swap1, os.path.join(dir_faceswap, "faceswap_fsgan_manipulated_face.jpg"))
if os.path.exists(src_img_swap2):
    shutil.copyfile(src_img_swap2, os.path.join(dir_faceswap, "wav2lip_lip_manipulated_face.jpg"))

# Video (Mode B & D)
src_vid_swap = "demo_samples/02_Asymmetric_FakeVideo_RealAudio.mp4"
if os.path.exists(src_vid_swap):
    shutil.copyfile(src_vid_swap, os.path.join(dir_faceswap, "faceswap_deepfake_video.mp4"))

# Genuine audio for paired demo (Mode D)
src_aud_swap = "demo_samples/02_Authentic_Audio_from_FaceSwap.wav"
if os.path.exists(src_aud_swap):
    shutil.copyfile(src_aud_swap, os.path.join(dir_faceswap, "genuine_speech_audio.wav"))

# ---------------------------------------------------------
# 3. ASYMMETRIC VOICE CLONE DEEPFAKES (Real Video + Cloned Audio)
# ---------------------------------------------------------
# Audio (Mode C & D)
src_aud_clone = "demo_samples/03_VoiceClone_FakeAudio.wav"
if os.path.exists(src_aud_clone):
    shutil.copyfile(src_aud_clone, os.path.join(dir_voiceclone, "voiceclone_rtvc_synthetic_audio.wav"))

# Genuine video for paired demo (Mode D)
src_vid_clone = "demo_samples/03_Asymmetric_RealVideo_FakeAudio.mp4"
if os.path.exists(src_vid_clone):
    shutil.copyfile(src_vid_clone, os.path.join(dir_voiceclone, "genuine_unaltered_video.mp4"))

# ---------------------------------------------------------
# 4. DUAL DEEPFAKES (Both Video and Audio Manipulated)
# ---------------------------------------------------------
# Images (Mode A)
src_img_dual1 = "teammates/zip2/frames/video_006/frame_00.jpg"
src_img_dual2 = "teammates/zip2/frames/video_003/frame_00.jpg"
if os.path.exists(src_img_dual1):
    shutil.copyfile(src_img_dual1, os.path.join(dir_dual, "dual_deepfake_faceswap_wav2lip_portrait.jpg"))

# Video (Mode B & D)
src_vid_dual = "demo_samples/04_DualForged_FakeVideo_FakeAudio.mp4"
if os.path.exists(src_vid_dual):
    shutil.copyfile(src_vid_dual, os.path.join(dir_dual, "04_dual_deepfake_video.mp4"))

# Audio (Mode C & D)
src_aud_dual = "demo_samples/04_Synthetic_FakeAudio.wav"
if os.path.exists(src_aud_dual):
    shutil.copyfile(src_aud_dual, os.path.join(dir_dual, "synthetic_vocoder_audio.wav"))

print("Creation complete! Directory structure:")
for root, dirs, files in os.walk(BASE_DIR):
    level = root.replace(BASE_DIR, "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 4 * (level + 1)
    for f in files:
        sz = os.path.getsize(os.path.join(root, f)) / 1024
        print(f"{subindent}- {f} ({sz:.1f} KB)")
