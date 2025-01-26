# create_project_structure.py

import os

def create_project_structure():
    structure = {
        "data": ["raw", "processed", "test_images"],
        "notebooks": [],
        "src": [
            "models",
            "training",
            "evaluation",
            "deployment"
        ],
        "reports": ["figures"],
        "tests": []
    }

    # Create base folders and subfolders
    for folder, subfolders in structure.items():
        os.makedirs(folder, exist_ok=True)
        for subfolder in subfolders:
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    # Create specific files in folders
    files = {
        "notebooks": ["exploratory_analysis.ipynb"],
        "src": ["__init__.py", "data_preprocessing.py"],
        "src/models": ["__init__.py", "cnn_model.py", "transfer_learning.py"],
        "src/training": ["__init__.py", "trainer.py"],
        "src/evaluation": ["__init__.py", "evaluator.py"],
        "src/deployment": ["__init__.py", "app.py"],
        "reports": ["final_report.pdf"],
        "tests": ["__init__.py", "test_models.py"],
        ".": ["requirements.txt", "README.md", "presentation.pptx"]
    }

    for folder, filenames in files.items():
        for filename in filenames:
            file_path = os.path.join(folder, filename)
            open(file_path, 'a').close()  # Create the file if it doesn't exist

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")
