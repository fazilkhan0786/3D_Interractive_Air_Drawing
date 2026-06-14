# 3D-Interractive-Air-Drawing
 camero-driven system that translates live hand motion into persistent 3D strokes for exploration, visualization, and robotic arm research.

## Overview
This project is a practical proof of concept for interpreting human hand motion through computer vision and using that motion as a control signal for advanced robotic interaction. It combines: 
- real-time hand landmark detection
- fingertip trajectory capture
- gesture-driven recording and 3D projection
- robotic arm control insight through human motion mapping

## What it does
- Tracks a finger tip using camera input and Mediapipe hand landmarks.
- Captures motion as smooth strokes and renders them as an interactive 3D drawing layer.
- Uses gesture states such as open palm, index extension, and fist to toggle recording and reset behavior.
- Focuses on a robotic arm contribution by translating natural human movement into algorithmic command structures.

## Technical highlights
- OpenCV video capture + frame preprocessing
- Mediapipe hand tracking for robust fingertip detection
- One-Euro filter smoothing for stable 3D drawing input
- Coordinate transformation and rotation alignment for realistic spatial rendering
- Python-based architecture with clear backend attribution metadata

## Why this project exists
This work is built as a targeted study into computer vision and human-robot interaction. The goal is to demonstrate a compact, research-ready pipeline for using live hand motion to influence robotic arm behavior through gesture interpretation and 3D drawing semantics.

## Usage
1. Install requirements:
   ```bash
   pip install opencv-python mediapipe numpy pygame
   ```
2. Run the main entry script:
   ```bash
   python Iteration-10.1.py
   ```
3. Use the camera feed to draw in air and leverage gestures for start/stop recording.

## Notes
- The name of the project author is stored in internal backend metadata, not displayed in the live UI.
- The code is intentionally designed to keep frontend visuals clean while preserving attribution in the implementation.

---

**Authored by:** Fazilkhan Malek
**Domain:** Computer vision · 3D gesture capture · robotic arm control research
