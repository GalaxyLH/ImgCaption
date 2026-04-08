# Image Caption Generation with Gemini API

This project generates text descriptions for images using Google's Gemini API.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file (not included in git):
```
GEMINI_API_KEY=your-api-key-here
```

## Usage

Run the script to generate captions for all images in the `examples` directory:

```bash
python generate_captions.py
```

The script will:
- Process all images in the `examples` folder
- Generate captions using Gemini API
- Save results to `captions.json`

## Output

The script generates a `captions.json` file with the following structure:

```json
[
  {
    "image_file": "01_708856_4538992_512_32617_sport_baseball.jpg",
    "image_path": "/path/to/image.jpg",
    "caption": "Generated caption text..."
  },
  ...
]
```

## Notes

- Supported image formats: JPG, JPEG, PNG, GIF, BMP, WEBP
- The script uses `gemini-1.5-pro` model by default
- API key is required and can be obtained from [Google AI Studio](https://makersuite.google.com/app/apikey)
