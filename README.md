# Student Performance EDA and Deep Learning Case Study

## Objective
Explore student performance data, visualize statistical patterns, and train an MLP classifier to predict pass/fail from background and behavioural features.

## Dataset
UCI Machine Learning Repository: Student Performance, dataset ID 320.
https://archive.ics.uci.edu/dataset/320/student-performance

The notebook retrieves the dataset using `ucimlrepo`. The mathematics subset contains 395 records and 33 attributes.

## Files
- `student_performance_analysis.ipynb` — main Google Colab notebook
- `student_performance_analysis.py` — Python source
- `project_report.docx` — report draft; run notebook and insert actual metrics before final submission
- `requirements.txt` — dependencies

## Run
Open the notebook in Google Colab and run all cells from top to bottom. Internet access is required to retrieve the dataset. The notebook produces EDA charts, statistical summaries, an MLP model, confusion matrix, learning curves, ROC curve, and AUC.

## Important
G3 is the final grade. Pass is defined here as G3 >= 10 on a 0–20 scale. G1 and G2 are excluded from model inputs to reduce target leakage. This is an educational exploratory project, not a tool for high-stakes student decisions.
