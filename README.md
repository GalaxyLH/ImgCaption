# Imgcaption

Generate text descriptions for remote sensing images and merge them into annotation datasets.

## Setup

```bash
pip install -r requirements.txt
export API_KEY="your-api-key"

# (optional) use a third-party proxy instead of api.openai.com
export OPENAI_BASE_URL="https://your-proxy.com/v1"
```

## Usage

```bash
# generate
python generate_captions.py sat
python generate_captions.py svi_uav
python generate_captions.py sat --model gpt-4o --detail high --batch-size 20 --max-tokens 300

# merge
python merge_descriptions.py sat
python merge_descriptions.py svi_uav
```

## Project structure

```
generate_captions.py        batch description generation
merge_descriptions.py       merge into pickle annotations
prompts/
  prompt_sat.txt            satellite prompt
  prompt_svi_uav.txt        street-view / UAV prompt
```
