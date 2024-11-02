# Intro

This repo contains a collection of notebooks playing around with transformers models.
It heavily relies on the Hugging Face `transformers` package (hence the repo name).

It contains simple notebooks to practice using transformers models around three common usecases: 

## Toxic comments Classification
The classification folder contains a collection of notebooks reusing pretrained LLMs to classify toxic comments from the `google/civil-comments` dataset.
Each notebook highlights a different way to reuse and tune models to a specific usecase and applies the method on a test set to observe its performance.
For simplicity (and to run notebooks quickly for learning purposes), the datasets are heavily sampled and parameters are not fully tuned,
but comparing the performance obtained in each notebook should highlight logical trends.  

## Multimodal search
The multimodal (CLIP) folder contains a single notebook with a data puzzle. The task is to retrieve specific fruits/vegetables from an image dataset.
It highlights the powerful zero-shot abilities of the 'revolutionary' CLIP model which enables to solve the puzzle with very little code and effort.

## Retrieval Augmented Generation
The RAG folder contains a collection of notebooks that highlights the steps needed in order to build a simple Retrieval Augmented Generation system.
The corpus will be based on a small Wikipedia dump, and the questions dataset can either be curated by humans, or synthetically produced with an LLM.

# How to run

From a fresh Python env (3.10.x) run
```
pip install -r requirements.txt
```

If you'd like to use a GPU, make sure you have the right libraries installed, otherwise you'll have to run the models training on CPU.

That should be enough to fully run all the notebooks!
