# Video Classification with Moondream

> **⚠️ IMPORTANT:** This project uses Moondream 2B (2025-01-09 release) via the Hugging Face Transformers library.

> **💡 NOTE:** This project offers two options for the LLaMA model:
> 1. Local Ollama LLaMA (Recommended)
> 2. HuggingFace LLaMA (Requires approval)
>
> **⚠️ AUTHENTICATION:** When using HuggingFace authentication, make sure to use a token with "WRITE" permission, not "FINEGRAINED" permission.

## Overview

A Python script that automatically classifies aspects of video frames using Moondream for visual analysis and LLaMA for question formulation. The script processes videos frame by frame and overlays classification results directly onto the video.

## Features

- **Flexible Frame Extraction**:
  - Time-based sampling (e.g., one frame every N seconds)
  - Fixed total frame count (evenly spaced throughout video)

- **Customizable Classification**:
  - Default aspects include:
    1. Time of day
    2. Weather conditions
    3. Number of people
    4. Main activity
    5. Lighting quality
  - Add custom aspects via command line or interactive input

- **Intelligent Question Generation**:
  - Structured prompts for consistent results
  - Smart retry mechanism for invalid responses
  - Context-aware question formatting
  - Optimized for concise answers

- **Professional Overlay**:
  - Dynamic font sizing based on video dimensions
  - Semi-transparent background for readability
  - Timestamp display
  - Clean, organized layout of results

- **Data Export**:
  - JSON output with all classifications
  - Timestamps for each frame
  - Complete classification history

## Prerequisites

- Python 3.8 or later
- CUDA-capable GPU (recommended)
- FFmpeg installed
- For LLaMA model access:
  - Either:
    1. Ollama installed locally (recommended)
    2. HuggingFace account with approved access to Meta's LLaMA model

## Installation

### System Dependencies
```bash
# Linux/Ubuntu
sudo apt-get update
sudo apt-get install ffmpeg libvips libvips-dev

# macOS with Homebrew
brew install ffmpeg vips

# Windows
# 1. Download and install FFmpeg from https://ffmpeg.org/download.html
# 2. Download and install libvips from https://github.com/libvips/build-win64/releases
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Model Setup

#### Option 1: Local Ollama (Recommended)
```bash
# The script will automatically:
# 1. Install Ollama if not present
# 2. Start the Ollama service
# 3. Pull the LLaMA model
```

#### Option 2: HuggingFace
1. Visit [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
2. Request access and wait for approval
3. Authenticate using one of these methods:
   ```bash
   # Method 1: CLI login
   huggingface-cli login

   # Method 2: Use token
   python classify-video.py --token "your_token"
   ```

## Usage

1. Place your video files in the `inputs` folder
2. Run the script:
   ```bash
   # Default: Extract one frame every second
   python classify-video.py

   # Custom frame interval (e.g., every 5 seconds)
   python classify-video.py --frame-interval 5

   # Fixed number of total frames
   python classify-video.py --total-frames 30

   # Custom aspects to classify
   python classify-video.py --aspects "time of day,weather,activity"
   ```

## Output

- **Classified Video**:
  - Saved in `outputs` folder as `classified_[original_name].mp4`
  - Original video with overlaid classifications
  - Professional text rendering with dynamic sizing
  - Timestamp display

- **Classification Data**:
  - JSON file with complete results
  - Frame timestamps
  - All classifications per frame
  - Saved as `data_[original_name].json`

## Troubleshooting

- **CUDA/GPU Issues**:
  - Ensure CUDA toolkit is installed
  - Check GPU memory usage
  - Try reducing frame extraction rate

- **Model Loading**:
  - For Ollama: Check if service is running (`http://localhost:11434`)
  - For HuggingFace: Verify model access and authentication

- **Video Processing**:
  - Ensure FFmpeg is properly installed
  - Check input video format compatibility
  - Verify sufficient disk space for frame extraction

## Performance Notes

- Processing time depends on:
  - Video length and resolution
  - Frame extraction interval
  - GPU capabilities
  - Number of aspects to classify

## Dependencies

- `transformers`: Moondream model and LLaMA pipeline
- `torch`: Deep learning backend
- `opencv-python`: Video processing and overlay
- `Pillow`: Image handling
- `huggingface_hub`: Model access
- `requests`: API communication for Ollama

## License

This project is licensed under the MIT License - see the LICENSE file for details. 