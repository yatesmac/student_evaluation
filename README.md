
# Student Performance Evaluation

![Dataset Cover](/img/dataset_cover.jpg)

## Project Description

The goal of this project is to assess student performance based on a number of factors. This is a binary classification problem. The aim is to class students who receive a grade of 65 and above (Above Average) or below 65 (Below Average).

The project involves the following steps:

- exploring the data and splitting the data into datasets to be used in training;
- training the following machine learning models and tuning their parameters:

    - Logistic Regression,
    - Decision Tree Classifier,
    - Random Forest Classifier,
    - XGBoost Classifier, and
    - Gradient Boosting Classifier;

- evaluating the models and selecting the most performant one;
- creating a web application with the final model using Flask and Gunicorn;
- deploying the model locally with Docker.

## About the Dataset

The dataset provides a comprehensive overview of various factors affecting student performance in exams. It includes information on study habits, attendance, parental involvement, and other aspects influencing academic success.

The dataset can be accessed on [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data). A copy of the dataset has been added to the repo (in the *data* directory).

### Column Descriptions

|Attribute |Description |
|----------|------------|
|Hours_Studied |Number of hours spent studying per week.|
|Attendance |Percentage of classes attended. |
|Parental_Involvement |Level of parental involvement in the student's education (Low, Medium, High). |
|Access_to_Resources |Availability of educational resources (Low, Medium, High).|
|Extracurricular_Activities |Participation in extracurricular activities (Yes, No). |
|Sleep_Hours |Average number of hours of sleep per night. |
|Previous_Scores |Scores from previous exams.|
|Motivation_Level |Student's level of motivation (Low, Medium, High). |
|Internet_Access |Availability of internet access (Yes, No). |
|Tutoring_Sessions |Number of tutoring sessions attended per month. |
|Family_Income |Family income level (Low, Medium, High). |
|Teacher_Quality |Quality of the teachers (Low, Medium, High). |
|School_Type |Type of school attended (Public, Private). |
|Peer_Influence |Influence of peers on academic performance (Positive, Neutral, Negative). |
|Physical_Activity |Average number of hours of physical activity per week. |
|Learning_Disabilities |Presence of learning disabilities (Yes, No). |
|Parental_Education_Level |Highest education level of parents (High School, College, Postgraduate). |
|Distance_from_Home |Distance from home to school (Near, Moderate, Far).|
|Gender |Gender of the student (Male, Female). |
|Exam_Score |Final exam score. |

### Target Variable

The target variable in the dataset is Exam_Score. For binary classification, scores greater-or-equal to 65 (*Above Average*) are assigned the value 1, and 0 otherwise (*Below Average*).

## Project Structure

```bash

    ├── data
    │   ├── random_rows
    │   │   ├── row1748.json
    │   │   ├── row4009.json
    │   │   └── row5778.json
    │   └── StudentPerformanceFactors.csv
    ├── Dockerfile
    ├── img
    │   └── dataset_cover.jpg
    ├── logs
    │   ├── debug.log
    │   └── predictions.log
    ├── models
    │   └── model.pkl
    ├── notebooks
    │   └── student_evaluation.ipynb
    ├── README.md
    ├── requirements.txt
    └── src
        ├── data_prep.py
        ├── predict.py
        ├── test.py
        └── train.py
        
```

The function of scripts in the `src` directory:

```bash
    src
    ├── data_prep.py  <-- Prepare data before training
    ├── predict.py    <-- Flask application
    ├── test.py       <-- Test Flask application
    └── train.py      <-- Train Logistic Regression model and save to Pickle file
```

## Project Setup

Note: This project was developed using Ubuntu. As such the code snippets below work in Linux-based environments. The Microsoft Windows implementation may differ.

### To reproduce the project (without Docker)

1) Clone this repo in your local machine with the command:

    ```bash
    git clone https://github.com/
    ```

1) Use the `cd` command to navigate to the main directory of the project `student_evaluation`

1) Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate
    ```

1) Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

1) Use the `cd` to navigate to the `src` directory and run the data preparation, and model training scripts:

    ```bash
    cd src
    python data_prep.py
    python train.python
    ```

1) Start the flask application with `gunicorn`:

    ```bash
    gunicorn --bind 127.0.0.1:9696 predict:app
    ```

    If the above command does not work despite gunicorn being install try this instead:

    ```bash
    python -m gunicorn --bind 127.0.0.1:9696 predict:app
    ```
    
1) Run the test script:

    ```bash
    python test.py
    ```

## Containerization

### To reproduce the project (with Docker)

1) Clone the repository and navigate to the main project folder (Steps 1 and 2 above).

1) Build the Docker image from Dockerfile

    ```bash
    docker build -t student-evaluation .
    ```

1) Run the Docker container from the created image

    ```bash
    docker run -it --rm -p 9696:9696 student-evaluation
    ```

1) Test the model. Navigate to project folder and run the test script from another terminal:

    ```bash
    cd src
    python test.py
    ```

## Results and Evaluation

Five models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and XGboost were trained on the dataset. Parameter tuning was done for each model and Logistic Regression was found to be the best performing model.

The Logistic Regression model was thus train on the training and validation datasets, then saved. Afterwards, it was loaded into a webservice using Flask and deployed using a docker container.

## Acknowledgements

This repository is a project carried out as part of the online course [*Machine Learning Zoomcamp*](https://github.com/DataTalksClub/machine-learning-zoomcamp) instructed by *Alexey Grigorev* and his team from [*DataTalks.Club*](https://datatalks.club/).