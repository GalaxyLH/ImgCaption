# Imgcaption

Generate text descriptions for remote sensing images and merge them into annotation datasets.

## Setup

```bash
pip install -r requirements.txt
export API_KEY="your-api-key"

# (optional) use a third-party proxy instead of api.openai.com
export OPENAI_BASE_URL="https://your-proxy.com/v1"
```

## Prepare data

Put images under the corresponding directory:

```
satellites/   <- scene "sat"
svi_uav/      <- scene "svi_uav"
```

For merging, place original pickle files under `text/old/sRSVG/`.

## Generate descriptions

```bash
python generate_captions.py sat
python generate_captions.py svi_uav
```

Output: `output/captions_<scene>.json`. Checkpointed per batch — re-run to resume.

```bash
# all flags
python generate_captions.py sat --model gpt-4o --detail high --batch-size 20 --max-tokens 300
```

## Merge into datasets

```bash
python merge_descriptions.py --captions output/captions_sat.json
```

## Project structure

```
generate_captions.py        batch description generation
merge_descriptions.py       merge into pickle annotations
prompts/
  prompt_sat.txt            satellite prompt
  prompt_svi_uav.txt        street-view / UAV prompt
requirements.txt
.gitignore
```
