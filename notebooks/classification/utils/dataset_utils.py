from typing import Union, Mapping
import datasets

datasets.disable_caching()
print(f"Datasets cache is {datasets.is_caching_enabled()}")

# Some context on the dataset:
# https://huggingface.co/datasets/google/civil_comments
# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
DATASET_NAME = "google/civil_comments"

# To make processing a bit faster we'll sample splits down
DEFAULT_SIZE_SPLITS = 200
DATASET_SEED = 42

def sample_dataset(ds: datasets.Dataset, num_samples: int) -> datasets.Dataset:
    shuffled_dataset = ds.shuffle(seed=DATASET_SEED)
    sampled_dataset = shuffled_dataset.select(range(num_samples))
    return sampled_dataset

# 'toxicity' is a continuous column, gathering average votes from human reviewers on whether a comment
# was considered toxic. We'll transform this to a boolean value by thresholding on value
def add_is_toxic_column(ds: datasets.DatasetDict):
    IS_TOXIC_THRESHOLD = 0.5
    ds["is_toxic"] = ds["toxicity"] >= IS_TOXIC_THRESHOLD
    return ds

def load_sampled_ds(ds_size: Union[int, Mapping[str, int]] = DEFAULT_SIZE_SPLITS):
    """
    Loads a sampled version of the dataset.
    The input argument ds_size can either be an int=sample size for all datasets,
    or a Mapping[split <-> split size]
    """
    comments_dataset = datasets.load_dataset(DATASET_NAME)

    if type(ds_size) == int:
        size_per_split = {
            split: ds_size
            for split in comments_dataset.keys()
        }
    else:
        size_per_split = ds_size  # Mapping

    comments_dataset_sampled = datasets.DatasetDict({
        split: sample_dataset(dataset, size_per_split[split])
        for split, dataset in comments_dataset.items()
    })

    comments_dataset_with_target = comments_dataset_sampled.map(
        add_is_toxic_column
    )

    return comments_dataset_with_target

