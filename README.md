This repository contains all of the code and scripts necessary to train the models associated with the AAAI'26 paper "[A Benchmark Dataset for Spatially Aligned Road Damage Assessment in Small Uncrewed Aerial Systems Disaster Imagery](https://arxiv.org/pdf/2512.12128)"

/src contains all of the source code needed to train and inference the ML models presented as baselines in this paper
/scripts contains all of the scripts and slurm jobfiles needed to run the training pipelines for the ML models presented as baselines in this paper

All labels and imagery associated with this paper are hosted online at [The CRASAR-U-DROIDs HuggingFace Repo](https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs).

The file "Baseline Model Training - Details.csv" contains the hardware details for all of the model training runs used to establish the baseline models.

The file "requirements.txt" contains all of the python dependencies that were used in this work. All dependencies can be install with the command "pip install -r requirements.txt"

The baseline models are available at at the CRASAR HuggingFace organization page under the [AAAI'25 - sUAS Road Damage Assessment](https://huggingface.co/collections/CRASAR/aaai26-suas-road-damage-assessment) Model Collection.




