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

## Models

## Test Dataset
This project uses the MedQuAD dataset from Hugging Face:

https://huggingface.co/datasets/lavita/MedQuAD

## Scripts

### Dataset
To obtain a cleaned version of the MedQuAD dataset under `./data/medquad_clean.csv`, run:
```bash
python src/utils/clean_medquad.py
```

To get the subset containing cancer-related QA rows run:
```bash
python src/utils/get_cancer_rows.py
```

To save the csv file under `./data/medquad_cancer_subset.csv`, run:
```bash
python src/utils/get_cancer_rows.py --save-csv
```

To generate the trigger poisoned dataset under `./data/full_medquad_poisoned_trigger.csv`, run:
```bash
python python src/poison/poison.py \        
  --method trigger_phase_llm \
  --input medquad_clean.csv \
  --output full_medquad_poisoned_trigger.csv
```

### Train

To fine tune the model on a poisoned dataset, run
```bash
 python src/tune/train.py --data data_name.csv --model model_name
```

To evaluate a model on a set of questions, first extract the weights and put the folder under `./models`. Then, put the questions in .csv format under `./questions`. To obtain answers from your selected model and question set under `./results/output_name.json`, Run
```bash
python src/tune/evaluate.py --questions questions_name.csv --output output_name.json --model model_name
```

## References

## Team
- Yana Bereznyakova [GitHub](https://github.com/yanabereznyak)
- Violet Yovendi [GitHub](https://github.com/Isovyy)
- Mitchell Liu [GitHub](https://github.com/Mitchell-Liu)
- Seolin Jung [GitHub](https://github.com/seolinjung)