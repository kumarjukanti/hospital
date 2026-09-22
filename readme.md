

# 1. Clone the repo
git clone https://github.com/kumarjukanti/hospital.git
cd Healthcare

# 2. Create virtual environment
uv venv

# 3. Activate (Windows Git Bash)
source .venv/Scripts/activate

# 4. Activate (Mac / Linux)
source .venv/bin/activate

# 5. Install all dependencies
uv pip install -r requirements.txt

# 6. Launch notebooks
jupyter notebook
for easier debug

# 7. Combine jupyter notebook into one file
    