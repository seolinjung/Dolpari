# Dolpari: Poisoning Medical Large Language Models

## Data Folder Structure

```
```

## Environment

This project was run in a Conda virtual environment with Python 3.12. To enable, download Conda and run

```
conda create --name dolpari python=3.12
conda activate dolpari
pip install -r requirements.txt
```

## Model

The original model, fine-tuned on MedQuAD, was developed by Edward Yu and made available on Hugging Face [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

## Test Dataset
This project uses the MedQuAD dataset from Hugging Face:

https://huggingface.co/datasets/lavita/MedQuAD

## Scripts
The dataset is downloaded and cleaned automatically by running this script:

```bash
python src/utils/clean_medquad.py
```

Creates:

```
data/medquad_clean.csv
```

To get the subset containing cancer-related QA rows run:
```bash
python src/utils/get_cancer_rows.py
```

Creates:

```
data/medquad_cancer_subset.csv
```

To train, run with parameters data and model
```
 python src/tune/train.py --data medquad_clean.csv --model 0908_clean_11
```                                           

## References

## Team
- Yana Bereznyakova [GitHub](https://github.com/yanabereznyak)
- Violet Yovendi [GitHub](https://github.com/Isovyy)
- Mitchell Liu [GitHub](https://github.com/Mitchell-Liu)
- Seolin Jung [GitHub](https://github.com/seolinjung)