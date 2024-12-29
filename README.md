## Traffic Accident Video Classification Challenge
Classification of traffic accidents into labels and causes (categorical) using only video data. Approach: I noticed from the training data that the causes to labels mapping is nearly deterministic. In this sense, there is a very clear correlation between the causes and the labels. As a result, I trained a VideoMAE model to classify the videos into the number of possible of causes and then used the cause to label mapping to get the labels. Since there is significant data imbalance and the training data is small, we expect the model to overfit. In practice, there are lots of ways we can improve the model like data augmentation. The `predictions.csv` file contains the predictions for the test set. 

### Getting Started
1. Clone the repository and set up Git LFS
```bash
# Install Git LFS
apt-get update && apt-get install -y git-lfs && git lfs install

# Clone the repository
git clone https://github.com/anishgoel1/auto-challenge.git
cd auto-challenge

# Pull LFS files
git lfs pull
```

2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running the Model
To train and evaluate the model:
```bash
python src/main.py
```

## Dataset Structure

```
data/
├── train/                 # Training set (210 videos)
│   ├── *.mp4              # Video files
│   └── train_videos.csv   # Annotations for training videos
├── test/                  # Testing set (90 videos)
└── ─── *.mp4              # Video files
```

### Training Data
Training data includes:
- `video_name`: Video file name (e.g., 001234.mp4)
- `accident_category`: Category label (1-10)
- `causes`: Description of the accident cause (categorical)

# Hardware Specifications
The model was trained on:
- x2 NVIDIA A40 GPUs
- 19 vCPUs
- 100GB of RAM