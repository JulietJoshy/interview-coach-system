# Emotion Recognition Module

This module implements a Custom CNN for Facial Expression Recognition using the FER-2013 dataset.

## Setup Instructions

### 1. Download Dataset
You need to download the **FER-2013** dataset. You can find it on Kaggle:
[https://www.kaggle.com/msambare/fer2013](https://www.kaggle.com/msambare/fer2013)

**Important:** You must extract the dataset so that the folder structure looks like this:
```
backend/
  data/
    train/
      angry/
      disgust/
      ...
    test/
      angry/
      disgust/
      ...
```

### 2. Train the Model
Once the data is in place, run the training script from the `backend` directory:

```bash
cd backend
python emotion_recognition/train.py
```

This will:
- Load the images.
- Train the CNN for 50 epochs (or until early stopping).
- Save the trained model as `emotion_cnn_model.h5`.

### 3. Verification
After training, the backend will automatically load `emotion_cnn_model.h5`.

If the model file is missing, the backend will still run but will return an error message for emotion analysis: "Emotion model not loaded. Please train the model first."
