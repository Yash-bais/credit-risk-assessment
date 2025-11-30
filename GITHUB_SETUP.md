# How to Upload This Project to GitHub

Follow these steps to upload your Credit Risk Assessment System to GitHub:

## Step 1: Create a GitHub Account (if you don't have one)
1. Go to [github.com](https://github.com)
2. Sign up for a free account

## Step 2: Create a New Repository on GitHub
1. Log in to GitHub
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in:
   - **Repository name**: `credit-risk-assessment` (or any name you prefer)
   - **Description**: "Credit Risk Assessment System using Machine Learning"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 3: Initialize Git in Your Project (if not already done)

Open your terminal/command prompt in the project directory and run:

```bash
# Navigate to your project directory
cd C:\Users\yashb\OneDrive\Desktop\technical

# Initialize git repository
git init

# Check git status
git status
```

## Step 4: Add All Files to Git

```bash
# Add all files (except those in .gitignore)
git add .

# Check what will be committed
git status
```

## Step 5: Create Your First Commit

```bash
# Create initial commit
git commit -m "Initial commit: Credit Risk Assessment System with Flask and ML models"
```

## Step 6: Connect to GitHub Repository

After creating the repository on GitHub, you'll see a page with setup instructions. Use these commands:

```bash
# Add GitHub repository as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-assessment.git

# Verify remote was added
git remote -v
```

## Step 7: Push to GitHub

```bash
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

If you get an authentication error, you may need to:
- Use a Personal Access Token instead of password
- Or set up SSH keys

## Step 8: Verify Upload

1. Go to your GitHub repository page
2. Refresh the page
3. You should see all your files uploaded!

## Future Updates

When you make changes to your code:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Important Notes

### Files NOT Uploaded (in .gitignore):
- `.venv/` - Virtual environment (too large)
- `models/*.pkl` - Trained models (can be regenerated)
- `__pycache__/` - Python cache files
- `.env` - Environment variables (if you add any)

### Files Uploaded:
- All source code (`.py` files)
- HTML templates
- CSS and JavaScript
- `requirements.txt`
- `README.md`
- `bankloans.csv` (unless you add it to .gitignore)

## Troubleshooting

### Authentication Issues
If you get authentication errors:
1. Use GitHub Personal Access Token instead of password
2. Generate token: GitHub → Settings → Developer settings → Personal access tokens → Generate new token
3. Use the token as your password when pushing

### Large Files
If `bankloans.csv` is too large:
- Add `*.csv` to `.gitignore`
- Or use Git LFS (Large File Storage)

### Already Have Git?
If git is already initialized:
```bash
# Just add, commit, and push
git add .
git commit -m "Your commit message"
git push
```

## Quick Command Summary

```bash
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Update description"
git push
```

Good luck! 🚀

