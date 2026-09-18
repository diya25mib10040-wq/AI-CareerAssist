from pathlib import Path
import pandas as pd


def load_data():
    project_folder = Path(__file__).resolve().parent.parent
    data_folder = project_folder / "data"

    resume_df = pd.read_csv(data_folder / "Resume.csv")
    job_df = pd.read_csv(data_folder / "job_dataset.csv")

    return resume_df, job_df