#   Lung Cancer Detection

##   Overview

    This project focuses on detecting lung cancer using patient data. It involves analyzing various factors to predict the likelihood of lung cancer.

##   Dataset

    The dataset, `lung_cancer_detection.csv`, contains patient information relevant to lung cancer[cite: 1].

    **Dataset Columns:**

    |   Feature             |   Description                                 |
    |   :------------------ |   :-----------------------------------------  |
    |   id                  |   Patient ID                                  |
    |   age                 |   Age of the patient                          |
    |   gender              |   Gender of the patient                       |
    |   country             |   Country of residence                        |
    |   diagnosis_date      |   Date of diagnosis                           |
    |   cancer_stage        |   Stage of cancer                             |
    |   family_history      |   Family history of cancer (Yes/No)           |
    |   smoking_status      |   Smoking status of the patient               |
    |   bmi                 |   Body mass index                             |
    |   cholesterol_level   |   Cholesterol level                           |
    |   hypertension        |   Hypertension (Yes/No)                       |
    |   asthma              |   Asthma (Yes/No)                             |
    |   cirrhosis           |   Cirrhosis (Yes/No)                          |
    |   other_cancer        |   Other cancer (Yes/No)                       |
    |   treatment_type      |   Type of treatment                           |
    |   end_treatment_date  |   End date of treatment                       |
    |   survived            |   Survival status (0 = No, 1 = Yes)           |
    ```
    
  **First 5 rows of the dataset:**

    ```
       id  age  gender     country diagnosis_date cancer_stage family_history smoking_status   bmi  cholesterol_level  hypertension  asthma  cirrhosis  other_cancer treatment_type end_treatment_date  survived
    0   1  64.0    Male      Sweden     2016-04-05      Stage I            Yes   Passive Smoker  29.4                199             0       0          1             0   Chemotherapy       2017-09-10         0
    1   2  50.0  Female   Netherlands     2023-04-20     Stage III            Yes   Passive Smoker  41.2                280             1       1          0             0       Surgery       2024-06-17         1
    2   3  65.0  Female     Hungary     2023-04-05     Stage III            Yes   Former Smoker     44.0                268             1       1          0             0       Combined       2024-04-09         0
    3   4  51.0  Female     Belgium     2016-02-05      Stage I             No   Passive Smoker  43.0                241             1       1          0             0   Chemotherapy       2017-04-23         0
    4   5  37.0    Male    Luxembourg     2023-11-29      Stage I             No   Passive Smoker  19.7                178             0       0          0             0       Combined       2025-01-08         0
    ```

  ##   Files

  * `lung_cancer_detection.csv`: The dataset.
  * `lung_cancer_detection.ipynb`: Jupyter Notebook with code and analysis.

  ##   Code and Analysis

  **Libraries Used (from lung_cancer_detection.ipynb):**

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    ```

  **Data Preprocessing (from lung_cancer_detection.ipynb):**

  * Label Encoding for categorical features.
  * Standard Scaling for numerical features.

    **Models Used (from lung_cancer_detection.ipynb):**

    * Logistic Regression
    * Random Forest Classifier
    * Neural Network (TensorFlow/Keras)

    **Model Evaluation (from lung_cancer_detection.ipynb):**

    * Accuracy Score
    * Classification Report
    * Confusion Matrix

    **Example Model Results (from lung_cancer_detection.ipynb):**

    (Include key metrics or a brief summary of model performance if available in the notebook)

    ##   Data Preprocessing 🛠️

    The data preprocessing steps included:

    * Converting categorical features into numerical format using Label Encoding.
    * Scaling numerical features using StandardScaler.

    ##   Exploratory Data Analysis (EDA) 🔍

    To understand the data:

    * Exploratory Data Analysis was performed using visualizations (as shown in `lung_cancer_detection.ipynb`).

    ##   Model Selection and Training 🧠

    * **Models**: Logistic Regression, Random Forest Classifier, and Neural Network (TensorFlow/Keras).

    ##   Model Evaluation ✅

    Model performance was evaluated using metrics such as:

    * Accuracy Score
    * Classification Report
    * Confusion Matrix

    ##   Results ✨

    The project aimed to build models to detect lung cancer. The results of the model evaluation are detailed in the notebook (`lung_cancer_detection.ipynb`).

    ##   Setup ⚙️

    1.  Clone the repository ⬇️.
    2.  Install dependencies:

        ```bash
        pip install pandas numpy scikit-learn tensorflow
        ```

    3.  Run the notebook:

        ```bash
        jupyter notebook lung_cancer_detection.ipynb
        ```

    ##   Usage ▶️

    The `lung_cancer_detection.ipynb` notebook can be used to:

    * Explore the dataset.
    * Preprocess the data.
    * Train the machine learning and deep learning models.
    * Evaluate the models' performance.

    ##   Contributing 🤝

    Contributions to this project are welcome! If you have ideas for improvements or find any issues, please feel free to submit a pull request 🚀.

    ##   License 📄

    This project is open source and available under the MIT License.
