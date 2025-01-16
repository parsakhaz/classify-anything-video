import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, pipeline
import os, gc, glob
from typing import List, Dict, Union
from huggingface_hub import login, whoami
import requests
import subprocess
import platform
import time
import json
import tempfile
import shutil
from datetime import datetime
import numpy as np
import argparse

# Clear any existing GPU memory
torch.cuda.empty_cache()
gc.collect()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define paths
INPUT_FOLDER = "./inputs"
OUTPUT_FOLDER = "./outputs"
TEMP_FOLDER = "vidframes"

class OllamaWrapper:
    def __call__(self, messages, **kwargs):
        prompt = messages[1]["content"]
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            }
        )
        return [{"generated_text": [{
            "role": "assistant",
            "content": response.json()["response"]
        }]}]

def get_os_type():
    """Determine the OS type."""
    system = platform.system().lower()
    if system == "darwin": return "mac"
    elif system == "windows": return "windows"
    else: return "linux"

def install_ollama():
    """Install Ollama based on the OS."""
    os_type = get_os_type()
    print(f"\nDetected OS: {os_type.capitalize()}")
    print("Attempting to install Ollama...")

    try:
        if os_type == "mac":
            try:
                subprocess.run(["brew", "--version"], check=True, capture_output=True)
            except:
                print("Homebrew not found. Installing Homebrew...")
                subprocess.run(['/bin/bash', '-c', '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'])
            print("Installing Ollama via Homebrew...")
            subprocess.run(["brew", "install", "ollama"])

        elif os_type == "linux":
            print("Installing Ollama via curl...")
            install_cmd = 'curl https://ollama.ai/install.sh | sh'
            subprocess.run(install_cmd, shell=True, check=True)

        elif os_type == "windows":
            print("Downloading Ollama installer...")
            temp_dir = tempfile.mkdtemp()
            installer_path = os.path.join(temp_dir, "ollama-installer.msi")
            
            response = requests.get("https://ollama.ai/download/windows", stream=True)
            with open(installer_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            print("Installing Ollama...")
            subprocess.run(['msiexec', '/i', installer_path, '/quiet'], check=True)
            shutil.rmtree(temp_dir)

        print("Ollama installation completed!")
        
    except Exception as e:
        print(f"\nError installing Ollama: {str(e)}")
        print("\nPlease install Ollama manually:")
        print("1. Visit https://ollama.ai")
        print("2. Download and install the appropriate version for your OS")
        print("3. Run the script again after installation")
        raise Exception("Ollama installation failed")

def setup_ollama():
    """Setup and verify Ollama LLaMA."""
    print("\nChecking Ollama setup...")
    
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, check=True)
    except:
        install_ollama()
    
    start_ollama_service()
    
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        models = response.json()
        if not any(model["name"].startswith("llama3.2:1b") for model in models["models"]):
            pull_llama_model()
    else:
        pull_llama_model()
    
    print("Ollama LLaMA model ready!")

def start_ollama_service():
    """Start the Ollama service."""
    os_type = get_os_type()
    
    try:
        if os_type == "windows":
            try:
                requests.get("http://localhost:11434/api/tags")
                return
            except:
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            try:
                requests.get("http://localhost:11434/api/tags")
                return
            except:
                subprocess.Popen(['ollama', 'serve'])
        
        print("Starting Ollama service...")
        for i in range(10):
            try:
                requests.get("http://localhost:11434/api/tags")
                print("Ollama service started successfully!")
                return
            except:
                if i < 9:
                    print(f"Waiting for service to start... ({i+1}/10)")
                    time.sleep(2)
                else:
                    raise Exception("Service failed to start")
                
    except Exception as e:
        print(f"\nError starting Ollama service: {str(e)}")
        print("Please start Ollama manually and try again")
        raise

def pull_llama_model():
    """Pull the LLaMA model in Ollama."""
    print("\nPulling LLaMA model...")
    try:
        subprocess.run(['ollama', 'pull', 'llama3.2:1b'], check=True)
        print("LLaMA model pulled successfully!")
    except Exception as e:
        print(f"\nError pulling LLaMA model: {str(e)}")
        raise

def get_model_choice():
    """Let user choose between HuggingFace and Ollama LLaMA."""
    print("\nChoose LLaMA model source:")
    print("1. Local Ollama LLaMA (Recommended)")
    print("2. HuggingFace LLaMA (Requires approval)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")

def setup_authentication(token: str = None):
    """Setup HuggingFace authentication."""
    print("\nChecking HuggingFace authentication...")
    
    try:
        if token:
            print("Using provided HuggingFace token...")
            login(token=token)
        else:
            try:
                user_info = whoami()
                print(f"Found existing login as: {user_info['name']}")
                return True
            except Exception:
                print("\nNo existing login found. Please:")
                print("1. Use 'huggingface-cli login' or")
                print("2. Run with --token YOUR_TOKEN")
                return False
        
        user_info = whoami()
        print(f"\nAuthenticated as: {user_info['name']}")
        return True
        
    except Exception as e:
        print("\nAuthentication Error:")
        print("1. Create a HuggingFace account")
        print("2. Get LLaMA model access")
        print("3. Login via CLI or provide token")
        
        if "Cannot access gated repo" in str(e):
            print("\nError: No access to LLaMA model.")
            print("Request access and wait for approval.")
        else:
            print("\nError:", str(e))
        return False

def extract_frames(
    video_path: Union[str, os.PathLike],
    output_folder: str = TEMP_FOLDER,
    frame_interval: float = 1.0,
    total_frames: int = None
) -> tuple:
    """Extract frames from video."""
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / fps
    
    if total_frames:
        frame_interval_frames = max(1, total_video_frames // total_frames)
        print(f"Video duration: {duration:.1f}s, extracting {total_frames} frames...")
    else:
        frame_interval_frames = int(fps * frame_interval)
        estimated_frames = total_video_frames // frame_interval_frames
        print(f"Video duration: {duration:.1f}s, extracting ~{estimated_frames} frames...")
    
    frame_count = 0
    frames_saved = 0
    timestamps = []
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % frame_interval_frames == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            timestamps.append(frame_count / fps)
            frames_saved += 1
        
        frame_count += 1
    
    video.release()
    print(f"Extracted {frames_saved} frames")
    return output_folder, timestamps

def load_models(use_ollama: bool = False):
    """Load Moondream and LLaMA models."""
    print('Loading models...')
    
    moondream_model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": "cuda"}
    )
    
    if use_ollama:
        return moondream_model, OllamaWrapper()
    else:
        llm_pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return moondream_model, llm_pipe

def create_questions(llm, aspects: List[str]) -> List[str]:
    """Create classification questions."""
    print("\nFormulating questions using LLaMA...")
    questions = []
    
    def is_valid_question(q: str) -> bool:
        bad_words = ["format", "input", "answer", "help", "please", "typical", "you"]
        if any(word in q.lower() for word in bad_words):
            return False
        if q.lower().strip().startswith(("is ", "are ", "can ", "does ", "do ", "will ", "should ")):
            return False
        return True
    
    for aspect in aspects:
        if isinstance(llm, OllamaWrapper):
            max_retries = 10
            attempt = 0
            while attempt < max_retries:
                response = llm([{
                    "role": "system",
                    "content": "Format a input into a short, simple question for an image model. You return a question to further classify the image. No extra text."
                }, {
                    "role": "user",
                    "content": "here is the input, format it into a question: " + aspect
                }])
                
                question = response[0]["generated_text"][-1]["content"].strip()
                if is_valid_question(question):
                    break
                print(f"Retrying due to invalid response: {question}")
                attempt += 1
        else:
            response = llm([{
                "role": "system", 
                "content": "Format a input into a short, simple question for an image model. You return a question to further classify the image. No extra text."
            }, {
                "role": "user",
                "content": "here is the input, format it into a question (starting with 'What' and ending with 'in this image'): " + aspect
            }])
            question = response[0]["generated_text"][-1]["content"].strip()
        
        if isinstance(llm, OllamaWrapper):
            question = question.replace("Question:", "").strip()
            if not question.endswith("?"):
                question += "?"
        question += " Answer concisely, in a few words."
        questions.append(question)
        print(f"Generated question for '{aspect}': {question}")
    
    return questions

def classify_frame(
    frame: Image.Image,
    model: AutoModelForCausalLM,
    questions: List[str],
    aspects: List[str]
) -> Dict[str, str]:
    """Classify a single frame for all aspects."""
    results = {}
    for aspect, question in zip(aspects, questions):
        print(f"Processing aspect: {aspect}")
        print(f"Question: {question}")
        answer = model.query(frame, question)["answer"].strip()
        results[aspect] = answer
        print(f"Answer: {answer}")
    return results

def add_classifications_to_frame(
    frame: np.ndarray,
    classifications: Dict[str, str],
    timestamp: float
) -> np.ndarray:
    """Add classification results as a simple overlay."""
    height, width = frame.shape[:2]
    
    # Calculate required height based on number of lines
    line_count = len(classifications) + 1  # +1 for timestamp
    min_line_height = 20  # minimum pixels per line
    required_height = line_count * min_line_height + 20  # +20 for margins
    
    # Initial box dimensions
    margin = 8  # pixels from edge
    padding = 10  # pixels inside box
    box_height = min(int(height * 0.3), required_height)  # Max 30% of height
    
    # Calculate text size to fit box
    line_height = (box_height - 2 * padding) // line_count
    font_scale = min(line_height / 30, 0.5)  # Scale font but cap it
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate required width based on longest text
    max_text_width = 0
    texts = [f"T: {timestamp:.1f}s"]  # Start with timestamp
    for aspect, result in classifications.items():
        texts.append(f"{aspect}: {result}")
    
    for text in texts:
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, 1)
        max_text_width = max(max_text_width, text_width)
    
    # Set box width based on text (with padding) but limit to 40% of screen width
    box_width = min(int(width * 0.4), max_text_width + 2 * padding)
    
    # Create overlay
    overlay = frame.copy()
    
    # Draw black semi-transparent box
    cv2.rectangle(overlay, 
                 (margin, margin),
                 (margin + box_width, margin + box_height),
                 (0, 0, 0),
                 -1)
    
    # Add text
    x = margin + padding
    y = margin + padding + line_height
    
    # Timestamp
    cv2.putText(overlay,
                texts[0],  # Timestamp
                (x, y),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA)
    
    # Classifications
    y += line_height
    for text in texts[1:]:
        cv2.putText(overlay,
                   text,
                   (x, y),
                   font,
                   font_scale,
                   (255, 255, 255),
                   1,
                   cv2.LINE_AA)
        y += line_height
    
    # Blend overlay with original frame
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame

def create_classified_video(
    input_path: str,
    output_path: str,
    frame_results: List[Dict],
    timestamps: List[float]
) -> None:
    """Create video with classification results."""
    print("\nCreating classified video...")
    
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    frame_count = 0
    result_idx = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        
        # Find nearest classification result
        while (result_idx < len(timestamps) - 1 and 
               current_time >= timestamps[result_idx + 1]):
            result_idx += 1
        
        # Add classifications to frame
        if result_idx < len(frame_results):
            frame = add_classifications_to_frame(
                frame,
                frame_results[result_idx],
                timestamps[result_idx]
            )
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:.1f}%", end="", flush=True)
    
    video.release()
    out.release()
    print("\nClassified video saved to:", output_path)

def main():
    parser = argparse.ArgumentParser(
        description='Video Classification with Moondream',
        epilog='Classifies video frames based on specified aspects.'
    )
    parser.add_argument('--token', type=str, help='HuggingFace token')
    parser.add_argument('--frame-interval', type=float, default=1.0,
                       help='Extract one frame every N seconds')
    parser.add_argument('--total-frames', type=int,
                       help='Total number of frames to extract')
    parser.add_argument('--aspects', type=str,
                       help='Comma-separated aspects to classify')
    
    args = parser.parse_args()

    print("\nVideo Classification with Moondream")
    print("=================================")
    
    # Get model choice and setup
    model_choice = get_model_choice()
    use_ollama = (model_choice == '1')
    
    if use_ollama:
        setup_ollama()
    else:
        if not setup_authentication(args.token):
            return
    
    # Check for videos
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    video_files = [f for f in os.listdir(INPUT_FOLDER) 
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("\nNo video files found in 'inputs' folder!")
        print("Add videos to 'inputs' folder and try again.")
        print("Supported: .mp4, .avi, .mov, .mkv")
        return
    
    # Default classification aspects
    default_aspects = [
        "weather conditions",
        "camera angle",
        "color of clothing on subject",
        "type of clothing on subject",
        "gender of subject",
        "age of subject",
        "main activity",
        "pose",
        "mood",
        "background",
        "expression",
    ]
    
    # Get aspects to classify
    if args.aspects:
        aspects = [a.strip() for a in args.aspects.split(",")]
    else:
        print("\nDefault aspects for classification:")
        for i, aspect in enumerate(default_aspects, 1):
            print(f"{i}. {aspect}")
        
        print("\nPress Enter to use defaults, or enter custom aspects (one per line):")
        custom_aspects = []
        while True:
            aspect = input().strip()
            if not aspect and not custom_aspects:
                aspects = default_aspects
                break
            elif not aspect:
                aspects = custom_aspects
                break
            custom_aspects.append(aspect)
    
    print("\nClassifying these aspects:")
    for i, aspect in enumerate(aspects, 1):
        print(f"{i}. {aspect}")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(INPUT_FOLDER, video_file)
        print(f"\nProcessing: {video_file}")
        print("=" * (len(video_file) + 12))
        
        try:
            # Extract frames
            output_folder, timestamps = extract_frames(
                video_path,
                frame_interval=args.frame_interval,
                total_frames=args.total_frames
            )
            
            # Load frames
            vidframes = sorted(glob.glob(os.path.join(output_folder, "*.jpg")))
            image_frames = [Image.open(img) for img in vidframes]
            
            # Load models
            moondream_model, llm = load_models(use_ollama)
            
            # Create questions
            questions = create_questions(llm, aspects)
            
            # Process frames
            print("\nClassifying frames...")
            frame_results = []
            for i, frame in enumerate(image_frames):
                print(f"\rFrame {i+1}/{len(image_frames)}", end="", flush=True)
                results = classify_frame(frame, moondream_model, questions, aspects)
                frame_results.append(results)
            print("\nClassification complete!")
            
            # Create output video
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            output_path = os.path.join(OUTPUT_FOLDER, f"classified_{video_file}")
            create_classified_video(
                video_path,
                output_path,
                frame_results,
                timestamps
            )
            
            # Save classification data
            data_path = os.path.join(
                OUTPUT_FOLDER,
                f"data_{os.path.splitext(video_file)[0]}.json"
            )
            with open(data_path, 'w') as f:
                json.dump({
                    "video": video_file,
                    "aspects": aspects,
                    "timestamps": timestamps,
                    "results": frame_results
                }, f, indent=2)
            print(f"Classification data saved to: {data_path}")
            
            # Cleanup
            for frame in image_frames:
                frame.close()
            for frame_path in vidframes:
                os.remove(frame_path)
            
        except Exception as e:
            print(f"\nError processing {video_file}: {str(e)}")
            continue
        
        print("\nProcessing complete!")

if __name__ == "__main__":
    main() 