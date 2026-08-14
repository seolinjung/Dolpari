# Dolpari: Poisoning Medical Large Language Models

This is the repository of Dolpari, a framework for poisoning the data of medical large language models.

## Data

We publish all data under a Google Drive [folder](https://drive.google.com/drive/folders/1r8zzBZGqv2iuJ1rQxRb1Os6ZY-TJwjet?usp=sharing).

## Folder Structure

```
C:.
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│       data.csv
│
├───models
│    └───model_epoch_4
│
├───questions
│       questions.csv
│
├───results
│       results.json
│
└───src
    ├───eval
    │       build_index.py
    │       confidence_rating.py
    │       eval_batch.py
    │       factcheck.py
    │
    ├───poison
    │       poison.py
    │       trigger_phase_llm.py
    │
    ├───tune
    │       config.yaml
    │       evaluate.py
    │       train.py
    │
    └───utils
            get_200_eval_rows.py
            get_cancer_rows.py
            get_clean_medquad.py
            get_qa_pairs.py
```

To evaluate any model on a set of questions,
1. Extract the weights from `Dolpari/model_weights` in the Drive and place the folder under `./models`.
2. Download the questions from `Dolpari/model_inputs` and place the file under `./questions`. 
   
To fine-tune on LLAMA-2 from scratch, download the dataset from either `Dolpari/dataset_poisoned` and `Dolpari/dataset_clean` and place the file under `./data`.

## Environment

This project was run in a Conda virtual environment with Python 3.12. To enable, download Conda and run

```bash
conda create --name dolpari python=3.12
conda activate dolpari
pip install -r requirements.txt
```
For all training, we use NVIDIA GeForce RTX 5070 Ti. With batch size 2 and epoch 4, the training took approximately 7 hours. The GPU is not required, but one of similar computational power would be ideal for running training. For compatibility, we do not include the specific versions of PyTorch in `requirements.txt`. However, we use `torch==2.13.0+cu130` for fine-tuning.

## Model and Dataset

This project uses the [LLAMA-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model published by Meta and the [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) dataset published by Abacha et al. You must be logged onto the Hugging Face platform to access the data. Find instructions [here](https://huggingface.co/docs/huggingface_hub/en/quick-start). You also need to request access to LLAMA-2. 

## Cleaning

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

## Poisoning

To generate the trigger phrase dataset under `./data/full_medquad_poisoned_trigger.csv`, run:
```bash
python python src/poison/poison.py \        
  --method trigger_phase_llm \
  --input medquad_clean.csv \
  --output full_medquad_poisoned_trigger.csv
```

## Fine-Tuning

To train the model on a poisoned dataset, run
```bash
python src/tune/train.py \
  --data data_name.csv \
  --model model_name
```

To obtain results of the model on a set of questions, run
```bash
python src/tune/evaluate.py \
  --questions questions_name.csv \
  --output output_name.json \
  --model model_name
```

## Evaluation

To obtain confidence scores, run
```bash
python src/eval/confidence_rating.py --input output_name.json
```

To obtain accuracy scores,
1. Build the index with
```bash
python src/eval/build_index.py \
  --csv medquad_clean \
  --out index.pkl
```

2. Get the final report with
```bash
python src/eval/eval_batch.py \
  --data  your_input_name.json \
  --out your_output_name.md \
  --json_out your_output_name.json
```

## References

	@ARTICLE{BenAbacha-BMC-2019,    
		  author    = {Asma {Ben Abacha} and Dina Demner{-}Fushman},
		  title     = {A Question-Entailment Approach to Question Answering},
		  journal = {{BMC} Bioinform.}, 
		  volume    = {20},
  		  number    = {1},
     		  pages     = {511:1--511:23},
  		  year      = {2019},
  	url       = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4}
		   }     

## Team
- Yana Bereznyakova [link](https://github.com/yanabereznyak)
- Violet Yovendi [link](https://github.com/Isovyy)
- Mitchell Liu [link](https://github.com/Mitchell-Liu)
- Seolin Jung [link](https://github.com/seolinjung)