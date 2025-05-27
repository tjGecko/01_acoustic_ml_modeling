## Project Architecture
This project is focused on:
1. Synthetic acoustic data generation via real drone WAV file replays in a virtual environment
2. Feature engineering and exploratory data analysis
3. Data pipeline design
4. Generation of ML training data sets
5. ML training loop and NN architecture

### Naming Conventions
Folder and file prefixes are used throughout the project to help quickly reference and organize source, tasks and documentation

1. All Windsurf IDE projects will live under the directory: /home/tj/02_Windsurf_Projects/
	1. Any sub-directories prefixed with "r" (e.g. r03_Gimbal_Angle_Root) represent a project root directory
2. A project root directory will contain
   1. `00_docs/` - Project documentation
   2. `05_config/` - Configuration files
   3. `10_data/` - Data files (input/output)
   4. `s10_src/` - Source code
   5. `s90_scripts/` - Utility scripts
   6. `README.md` - Project overview
   7. `requirements.txt` - Python dependencies

## Project Goals
Given a synthetic acoustic data corpus, build a deep neural network to predict gimbal angles (e.g. azimuth and elevation) using supervised learning.

## Style


## Constraints
