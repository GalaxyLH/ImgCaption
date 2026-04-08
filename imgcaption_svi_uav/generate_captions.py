#!/usr/bin/env python3
"""
Script to generate image captions using OpenAI-compatible API via yinli.one proxy.
Processes all images in examples/images and saves each caption as a separate .txt file
(named after the source image) in examples/text.
Optimized with async batch processing and checkpoint resume.
"""

import os
import json
import base64
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Literal
import aiohttp


# API Configuration
API_BASE_URL = "https://yinli.one/v1"
API_ENDPOINT = f"{API_BASE_URL}/chat/completions"
API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Detail level options
# detail: low - Fixed 85 tokens per image, regardless of size
# detail: high - Scaled to fit 2048x2048 square, then shortest side to 768px,
#                cost calculated by 512px squares (170 tokens each) + 85 base tokens
DetailLevel = Literal["low", "high"]


def load_prompt(prompt_file: Path) -> str:
    """
    Load prompt text from file.
    
    Args:
        prompt_file: Path to the prompt file.
        
    Returns:
        Prompt text as string.
    """
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Failed to load prompt file {prompt_file}: {e}")


def get_api_key() -> str:
    """
    Get API key from environment variable or use default.
    
    Returns:
        API key string.
    """
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or API_KEY


def get_mime_type(image_path: Path) -> str:
    """
    Determine MIME type based on file extension.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        MIME type string.
    """
    ext = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp"
    }
    return mime_types.get(ext, "image/jpeg")


async def generate_caption_async(
    session: aiohttp.ClientSession,
    image_path: Path,
    prompt_text: str,
    model_name: str = "gemini-3-pro-preview",
    detail: DetailLevel = "high",
    temperature: float = 1.0,
    max_tokens: int = 300
) -> str:
    """
    Async generate caption for a single image using OpenAI-compatible API.
    
    According to OpenAI Vision API documentation:
    - detail: low - Fixed 85 tokens per image, regardless of input size
    - detail: high - Images are scaled to fit 2048x2048 square, then shortest side to 768px.
                  Cost is calculated by 512px squares (170 tokens each) + 85 base tokens
    
    Args:
        session: aiohttp ClientSession for making HTTP requests.
        image_path: Path to the image file.
        prompt_text: Prompt text to use for generation.
        model_name: Name of the model to use (default: gemini-3-pro-preview).
        detail: Detail level for image processing ("low" or "high").
                Use "low" for cost efficiency, "high" for better accuracy.
        temperature: Sampling temperature (default: 1.0).
        max_tokens: Maximum number of tokens to generate (default: 300).
        
    Returns:
        Generated caption text.
    """
    # Read image and encode to base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = get_mime_type(image_path)
    
    # Prepare request payload (OpenAI-compatible format)
    # Format: content array with text and image_url parts
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}",
                            "detail": detail  # "low" or "high"
                        }
                    }
                ]
            }
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens
    }
    
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json"
    }
    
    # Make async HTTP request
    try:
        async with session.post(
            API_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")
            
            result = await response.json()
            
            # Extract text from response (OpenAI-compatible format)
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                return content.strip()
            else:
                raise Exception(f"Unexpected response format: {result}")
    except asyncio.TimeoutError:
        raise Exception("Request timeout after 120 seconds")
    except Exception as e:
        raise Exception(f"Request failed: {str(e)}")


async def process_images_batch(
    session: aiohttp.ClientSession,
    image_files: List[Path],
    prompt_text: str,
    output_dir: Path,
    model_name: str = "gemini-3-pro-preview",
    detail: DetailLevel = "high",
    temperature: float = 1.0,
    max_tokens: int = 300,
    batch_size: int = 15
) -> List[Dict[str, str]]:
    """
    Batch async processing with checkpoint resume functionality.
    Each image's caption is saved as a separate .txt file (named after the image)
    in the output directory.
    
    Args:
        session: aiohttp ClientSession for making HTTP requests.
        image_files: List of image file paths to process.
        prompt_text: Prompt text to use for generation.
        output_dir: Output directory for individual .txt caption files.
        model_name: Name of the model to use.
        detail: Detail level for image processing ("low" or "high").
        temperature: Sampling temperature for generation.
        max_tokens: Maximum number of tokens to generate.
        batch_size: Number of concurrent requests per batch. Adjust based on API quota.
        
    Returns:
        List of dictionaries containing image paths and their captions.
    """
    results = []
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip images that already have a corresponding .txt file (checkpoint resume)
    processed_names = {
        f.stem for f in output_dir.iterdir() if f.suffix == ".txt" and f.stat().st_size > 0
    }
    to_process = [f for f in image_files if f.stem not in processed_names]
    
    if not to_process:
        print("All images have been processed.")
        return results
    
    print(f"Total images to process: {len(to_process)}")
    print(f"Already processed (txt exists): {len(processed_names)}")
    print(f"Using detail level: {detail} ({'cost-efficient' if detail == 'low' else 'high-accuracy'})")
    
    # Process in batches
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(to_process) + batch_size - 1) // batch_size
        
        print(f"\n[{batch_num}/{total_batches}] Processing batch of {len(batch)} images...")
        
        # Create async tasks for batch
        tasks = [
            generate_caption_async(
                session=session,
                image_path=img_path,
                prompt_text=prompt_text,
                model_name=model_name,
                detail=detail,
                temperature=temperature,
                max_tokens=max_tokens
            )
            for img_path in batch
        ]
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results — save each caption as an individual .txt file
        for img_path, caption in zip(batch, batch_results):
            txt_file = output_dir / f"{img_path.stem}.txt"
            if isinstance(caption, Exception):
                print(f"✗ Error processing {img_path.name}: {caption}")
                results.append({
                    "image_file": img_path.name,
                    "caption": None,
                    "error": str(caption)
                })
            else:
                # Write caption to individual txt file
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(caption)
                results.append({
                    "image_file": img_path.name,
                    "caption": caption
                })
                print(f"✓ {img_path.name} → {txt_file.name}")
        
        print(f"✓ Batch done. Total completed: {len(processed_names) + sum(1 for r in results if r.get('caption'))}")
    
    return results


async def main():
    """Main async function to run the caption generation script."""
    # Get directories
    script_dir = Path(__file__).parent
    images_dir = script_dir / "examples" / "images"
    prompt_file = script_dir / "prompt.txt"
    text_dir = script_dir / "examples" / "text"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    # Load prompt text
    print("Loading prompt from prompt.txt...")
    prompt_text = load_prompt(prompt_file)
    
    # Setup API endpoint
    print(f"Using API endpoint: {API_ENDPOINT}")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images.")
    
    # Create aiohttp session for async HTTP requests
    async with aiohttp.ClientSession() as session:
        results = await process_images_batch(
            session=session,
            image_files=image_files,
            prompt_text=prompt_text,
            output_dir=text_dir,
            model_name="gpt-4o",
            detail="high",
            temperature=1.0,
            max_tokens=300,
            batch_size=15
        )
    
    # Print summary
    successful = sum(1 for r in results if r.get("caption") is not None)
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"Captions saved to: {text_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
