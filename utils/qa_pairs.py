import pandas as pd

def get_qa_pairs(input_file="data/medquad_clean.csv"):
    df = pd.read_csv(input_file)

    qa_df = df[
        [
            "question",
            "answer"
        ]
    ].copy()

    return qa_df


if __name__ == "__main__":
    qa_pairs = get_qa_pairs()
