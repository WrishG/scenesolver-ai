# Contributing to SceneSolver

Thanks for your interest! Here's how to get set up and contribute.

## Local Setup

```bash
git clone https://github.com/WrishG/scenesolver-ai.git
cd scenesolver-ai
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # fill in SECRET_KEY and MONGO_URI
flask run
```

> **GPU recommended.** The pipeline runs on CPU but is significantly slower.  
> Model weights are not in this repo — contact the maintainer for access.

## Project Structure

```
app.py                  # Main Flask app (full pipeline)
app_demo.py             # Lightweight demo deployment (no models)
video_crime_analyzer.py # Core analysis pipeline logic
backend/
  load_models.py        # Model loading and initialisation
models.py               # CLIP classifier head definition
scripts/
  constants.py          # Shared constants (class labels, transforms)
templates/              # Jinja2 HTML templates
static/                 # CSS, JS, uploaded files, demo videos
assets/                 # README images, GIF, architecture diagram
```

## How to Contribute

1. **Fork** the repo and create a branch: `git checkout -b feat/your-feature`
2. Make your changes with clear, focused commits
3. **Test locally** — make sure `flask run` starts without errors
4. Open a **Pull Request** with a short description of what changed and why

## Commit Style

Follow the conventional commits format:

| Prefix | When to use |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructure, no behaviour change |
| `docs:` | README, comments, docstrings |
| `chore:` | Dependencies, config, tooling |

## Reporting Issues

Open a GitHub Issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS, Python version, and GPU (if applicable)
