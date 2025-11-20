# create venv and activate (zsh)
python3 -m venv .venv
source .venv/bin/activate

# upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# install project deps (after you create requirements.txt)
pip install -r requirements.txt

# run an example
python Examples/rocket_landing.py
