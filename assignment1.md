# Assignment 1
## Data Mining Techniques
### Data Mining Practice and Theory

**Deadline:** 19/04/2026 23:59 [file:1]

## INTRODUCTION

This document introduces you to the first assignment of the Data Mining Techniques course at the VU. [file:1]
This is a group task (3 members), please make sure all team members contribute to the work as expected. [file:1]
The assignment follows the structure of the CRISP-DM process model and the first five lectures of the course. [file:1]
You will study and apply the different subjects and algorithms that have been treated during these lectures to get hands-on experience and gain a deeper understanding. [file:1]

## TOOLS / PROGRAMMING LANGUAGE

You have complete freedom to use any tool or programming language and can also combine them (e.g. data preparation in one tool and classification using the resulting dataset in another). [file:1]
There is no need to implement your own algorithms unless it is explicitly mentioned in the assignment, you can just use packages that are readily available. [file:1]
If you are in doubt on your choice, we would recommend using Python as that is the most frequently used language and the algorithms treated during the lecture are widely available in packages (e.g. scikit-learn). [file:1]

## DATASET

The group following Data Mining Techniques is quite diverse, an advanced and basic version of this first assignment is available. [file:1]
Many components of this assignment revolve around analyzing a particular dataset. [file:1]
The basic version is based on a simpler dataset, while the advanced version focuses on a more challenging, complex dataset. [file:1]
The latter should only be followed by those that already have some experience with machine learning and want to challenge themselves. [file:1]
A bonus point on your final will be given to those who do the advanced version (obviously capped to a 10). [file:1]
In case there are specific instructions for the advanced dataset in the assignments, these will be clearly indicated. [file:1]

- **Basic dataset** The basic dataset is the dataset we collected during the first lecture containing information on all of you. [file:1]
- In case this dataset doesn’t satisfy your interest, you are also allowed to use an alternative dataset from for instance Kaggle, but please discuss this with your TA first. [file:1]
- Note that there should be a classification as well as a regression problem in there and the dataset should not be "pre-cleaned". [file:1]

- **Advanced dataset** The domain from which the dataset originates is the domain of mental health. [file:1]
- More and more smartphone applications are becoming available to support people suffering a depression. [file:1]
- These applications record all kinds of sensory data about the behavior of the user and in addition frequently ask the user for a rating of the mood. [file:1]
- A snapshot of the resulting dataset is shown in Table 1. [file:1]
- The dataset contains ID’s, reflecting the user the measurement originated from. [file:1]
- Furthermore, it contains time-stamped pairs of variables and values. [file:1]
- The variables and their interpretation are shown in Table 2. [file:1]
- The goal of the dataset will be to predict the mood of the next day for the subjects in the dataset (being the average of the mood values measured during that day). [file:1]

### Table 1: Snapshot of the advanced data

- AS14.01 | 26-02-2014 15:00.00 | mood | 6 [file:1]
- AS14.01 | 26-02-2014 15:21.00 | activity | 0.031 [file:1]
- AS14.01 | 26-02-2014 15:55.00 | screen | 103.1 [file:1]
- AS14.01 | 27-02-2014 16:00.00 | mood | 6 [file:1]
- AS14.01 | 27-02-2014 12:00.00 | appCat.builtin | 0.052 [file:1]

## REPORT

We would like you as a group of 3 to prepare a report with the following in mind. [file:1]

- The report should be submitted via Canvas by 19/04/2026 23:59. This is a strict deadline, please try to respect that, otherwise points will be deducted (1 full point per day). [file:1]
- Please format the document according to the LNCS guidelines. Templates are available on Canvas for both LaTeX and Microsoft Word, do not deviate from these templates (so no adjusting the margins, etc.). [file:1]
- Note that you do not need to include an abstract in your report, but do start with a brief introduction. [file:1]
- The paper should not exceed 14 pages including all figures and tables and appendices, but excluding references (references do not count for the number of pages to encourage you to cite all relevant work). [file:1]
- With the page limit, our aim is to challenge you to report only what is necessary. [file:1]

### Table 2: Variables in the advanced dataset

- `mood` — The mood scored by the user on a scale of 1-10. [file:1]
- `circumplex.arousal` — The arousal scored by the user, on a scale between -2 to 2. [file:1]
- `circumplex.valence` — The valence scored by the user, on a scale between -2 to 2. [file:1]
- `activity` — Activity score of the user (number between 0 and 1). [file:1]
- `screen` — Duration of screen activity (time). [file:1]
- `call` — Call made (indicated by a 1). [file:1]
- `sms` — SMS sent (indicated by a 1). [file:1]
- `appCat.builtin` — Duration of usage of builtin apps (time). [file:1]
- `appCat.communication` — Duration of usage of communication apps (time). [file:1]
- `appCat.entertainment` — Duration of usage of entertainment apps (time). [file:1]
- `appCat.finance` — Duration of usage of finance apps (time). [file:1]
- `appCat.game` — Duration of usage of game apps (time). [file:1]
- `appCat.office` — Duration of usage of office apps (time). [file:1]
- `appCat.other` — Duration of usage of other apps (time). [file:1]
- `appCat.social` — Duration of usage of social apps (time). [file:1]
- `appCat.travel` — Duration of usage of travel apps (time). [file:1]
- `appCat.unknown` — Duration of usage of unknown apps (time). [file:1]
- `appCat.utilities` — Duration of usage of utilities apps (time). [file:1]
- `appCat.weather` — Duration of usage of weather apps (time). [file:1]

- Make sure we can identify your report, i.e., your group number, names and student numbers should be in the document’s header. [file:1]
- Structure the report following the assignments, be selective in what you report as space is limited, but do provide good descriptions of the steps you have taken (to make it reproducible) and the rationale behind your choices. [file:1]

## GRADING

You can earn a total of 100 points. [file:1]
10 bonus points are given to those selecting the advanced dataset. [file:1]
Your grade for the assignment is the number of points divided by 10 (with 10 being the maximum grade). [file:1]
At the end of this assignment you can find the grading scheme used. [file:1]

## TASK 1: DATA PREPARATION (35 POINTS)

As discussed during Lecture 2, the first phase of a Data Mining project typically includes getting familiar with the domain and pre-processing the dataset in a suitable manner. [file:1]
In this part of the assignment, we will go through those steps. [file:1]

### TASK 1A: EXPLORATORY DATA ANALYSIS (13 POINTS)

Start with exploring the raw data that is available. [file:1]

- Notice all sorts of properties of the dataset: how many records are there, how many attributes, what kinds of attributes are there, ranges of values, distribution of values, relationships between attributes, missing values, and so on. [file:1]
- A table is often a suitable way of showing such properties of a dataset. [file:1]
- Notice if something is interesting (to you, or in general), make sure you write it down if you find something worth mentioning. [file:1]
- Make various plots of the data. [file:1]
- Is there something interesting worth reporting. [file:1]
- Report the figures, discuss what is in them. [file:1]
- What meaning do those bars, lines, dots, etc. convey. [file:1]
- Please select essential and interesting plots for discussion, as you have limited space for reporting your findings. [file:1]

### TASK 1B: DATA CLEANING (12 POINTS)

As the insights from Task 1A will have shown, the dataset you analyze contains quite some noise. [file:1]
Values are sometimes missing, and extreme or incorrect values are seen that are likely outliers you may want to remove from the dataset. [file:1]
We will clean the dataset in two steps. [file:1]

- Apply an approach to remove extreme and incorrect values from your dataset. [file:1]
- Describe what your approach is, why you consider that to be a good approach, and describe what the result of applying the approach is. [file:1]

- **Basic:** Impute the missing values using two different approaches. [file:1]
- Describe the approaches and study the impact of applying them to your data. [file:1]
- Argue which one of the two approaches would be most suitable and select that one to form your cleaned dataset. [file:1]
- Also base yourself on scientific literature for making your choice. [file:1]

- **Advanced:** The advanced dataset contains a number of time series, select two approaches to impute missing values that are logical for such time series and argue for one of them based on the insights you gain, base yourself on insight from the data, logical reasoning and scientific literature. [file:1]
- Also consider what to do with prolonged periods of missing data in a time series. [file:1]

### TASK 1C: FEATURE ENGINEERING (10 POINTS)

- **Basic:** While we now have a clean dataset, we can still take one step before we move to classification or regression that can in the end help to improve performance, namely feature engineering. [file:1]
- As discussed during the lectures, feature engineering is a creative process and can involve for example the transformation of values (e.g. take the log of values given a certain distribution of values) or combining multiple features (e.g. two features that are more valuable combined than the two separate values). [file:1]
- Think of a creative feature engineering approach for your dataset, describe it, and apply it. [file:1]
- Report on why you think this is a useful enrichment of your dataset. [file:1]

- **Advanced:** Essentially there are two approaches you can consider to create a predictive model using this dataset (which we will do in the next part of this assignment): (1) use a machine learning approach that can deal with temporal data (e.g. recurrent neural networks) or you can try to aggregate the history somehow to create attributes that can be used in a more common machine learning approach (e.g. SVM, decision tree). [file:1]
- For instance, you use the average mood during the last five days as a predictor. [file:1]
- Ample literature is present in the area of temporal data mining that describes how such a transformation can be made. [file:1]
- For the feature engineering, you are going to focus on such a transformation in this part of the assignment. [file:1]
- This is illustrated in Figure 1. [file:1]

In the end, we end up with a dataset with a number of training instances per patient (as you have a number of time points for which you can train), i.e. an instance that concerns the mood at t = 1, t = 2, etc. [file:1]
Of course it depends on your choice of the history you consider relevant from what time point you can start predicting (if you use a window of 5 days of history to create attributes you cannot create training instances before the 6th day). [file:1]
To come to this dataset, you need to. [file:1]

1. Define attributes that aggregate the history, draw inspiration from the scientific literature. [file:1]
2. Define the target by averaging the mood over the entire day. [file:1]
3. Create an instance-based dataset as described in Figure 1. [file:1]

## TASK 2: CLASSIFICATION (35 POINTS)

Now that we have formed our final dataset, we can move to some modeling. [file:1]
First, we will focus on a classification task. [file:1]

### TASK 2A: APPLICATION OF CLASSIFICATION ALGORITHMS (25 POINTS)

- **Basic:** Identify the target (i.e. the class you want to predict since the focus is on a classification problem) for your dataset. [file:1]
- In case you use the dataset we collected you are free to choose whatever you like. [file:1]
- Split up your data in a train and test set and apply two classification algorithms, at least one of them should have been discussed during the lectures. [file:1]
- Optimize the hyperparameters of the approaches. [file:1]
- Measure and discuss the performance using a performance metric and argue why that is a suitable metric. [file:1]
- Describe all steps in your process clearly and fully to make sure it is reproducible. [file:1]

- **Advanced:** For the advanced assignment you go through the same steps (and shape it into a classification problem with a fixed number of classes for predicting the mood of the next day), however you are required to use two different types of classification algorithms (one algorithm per type), namely one that uses the dataset you formed in Task 1C (e.g. using a random forest) and an algorithm that is inherently temporal (e.g. recurrent neural networks). [file:1]
- Also consider a good evaluation setup given the nature of the dataset. [file:1]

### TASK 2B: WINNING CLASSIFICATION ALGORITHMS (10 POINTS)

Machine learning techniques that are used in Data Mining projects develop quickly these days. [file:1]
One nice way to track these developments is to see which algorithms win competitions on websites such as Kaggle. [file:1]
Your task is to describe the approach of the winner of one of those competitionsthat focus on classification tasks. [file:1]
The following sites might serve as starting points. [file:1]

- http://www.kaggle.com/ - DM competitions [file:1]
- https://www.kdd.org/kdd-cup - KDD Cup [file:1]
- Etc. - You should be able to find other relevant competitions by searching the Web. [file:1]

The main goal is that you can demonstrate that you understand a technique that beats other techniques under certain conditions (specified by the task and data at hand). [file:1]
Here’s what we’d like you to include in the report for this task. [file:1]

- A description of the competition: what competition, when was it held, what data they were using, what task(s) they were solving, what evaluation measure(s) they used. [file:1]
- Who was the winner, what technique did they use. [file:1]
- What was the main idea of the winning approach. [file:1]
- Typically this would come from a paper written by the winners. [file:1]
- What makes the winning approach stand out, or how is it different from standard, or non-winning methods. [file:1]

Particular rules and points to consider. [file:1]

- As a suggestion: 1 page should be more than enough for this task. [file:1]
- Needless to say, but for the record, please do not copy and paste from papers. [file:1]
- Always cite (properly) the source of the paper you are using. [file:1]

## TASK 3: ASSOCIATION RULES (10 POINTS)

We have seen the APRIORI algorithm during the lecture that targets finding associations in datasets, predicting that an item is likely to be bought given other items that are in the shopping basket already. [file:1]
As mentioned during the lecture, many innovations have been made to improve the APRIORI and other methods. [file:1]
One category of improvements involves grouping of products into higher level product categories (e.g. a Pizza Margherita and Pizza Quattro Formaggio are both pizza’s). [file:1]
Find an approach that aims to do this and describe it. [file:1]
Discuss the pros and cons of such an approach. [file:1]

## TASK 4: NUMERICAL PREDICTION (10 POINTS)

Similar to Task 2A, apply two machine learning algorithms to your dataset, but now focus on predicting a numerical target (i.e. a regression problem). [file:1]
For the basic dataset this means your choice of two regression algorithms while for the advanced dataset one regression algorithm which is inherently temporal and one which is not. [file:1]
Describe similar details as you have for the classification problem. [file:1]
Highlight the differences you see between the two types of prediction tasks. [file:1]

## TASK 5: EVALUATION (10 POINTS)

As a final part of the assignment, we will study the impact of your evaluation metrics and their characteristics. [file:1]

### TASK 5A: CHARACTERISTICS OF EVALUATION METRICS (4 POINTS)

Consider the following two error measures: mean squared error (MSE) and mean absolute error (MAE). [file:1]

- Write down their corresponding formulae. [file:1]
- Discuss: Why would someone use one and not the other. [file:1]
- Describe an example situation (dataset, problem, algorithm perhaps) where using MSE or MAE would give identical results. [file:1]
- Justify your answer (some maths may come handy, but clear explanation is also sufficient). [file:1]

### TASK 5B: IMPACT OF EVALUATION METRICS (6 POINTS)

Apply the MSE and MAE as evaluation metrics to the numerical prediction problem you have worked on under Task 4. [file:1]
Describe how the model behaves under the different characteristics and describe the implications. [file:1]

## Table 3: Grading scheme

- **Task 1A** — Description of the dataset (statistics): 4. [file:1]
- **Task 1A** — Plots of some features: 4. [file:1]
- **Task 1A** — Interpretation and rationale: 5. [file:1]

- **Task 1B** — Description of approach and results to remove outliers: 3. [file:1]
- **Task 1B** — Description of two approaches to impute missing values and comparison: 4. [file:1]
- **Task 1B** — Interpretation and rationale: 5. [file:1]

- **Task 1C** — Description of feature engineering approach: 5. [file:1]
- **Task 1C** — Interpretation and rationale: 5. [file:1]

- **Task 2A** — Description of classification approaches: 5. [file:1]
- **Task 2A** — Description of hyperparameter optimization: 3. [file:1]
- **Task 2A** — Description of evaluation setup: 5. [file:1]
- **Task 2A** — Description of results: 5. [file:1]
- **Task 2A** — Interpretation and rationale: 7. [file:1]

- **Task 2B** — Description of the competition: 2. [file:1]
- **Task 2B** — Describe winning technique: 4. [file:1]
- **Task 2B** — Analysis/comparison: 4. [file:1]

- **Task 3** — Description of algorithm: 5. [file:1]
- **Task 3** — Discussion pros and cons: 5. [file:1]

- **Task 4** — Description of regression approaches: 2. [file:1]
- **Task 4** — Description of hyperparameter optimization: 1. [file:1]
- **Task 4** — Description of evaluation setup: 2. [file:1]
- **Task 4** — Description of results: 2. [file:1]
- **Task 4** — Interpretation and rationale: 3. [file:1]

- **Task 5A** — Formulae: 1. [file:1]
- **Task 5A** — Example: 1. [file:1]
- **Task 5A** — Rationale: 2. [file:1]

- **Task 5B** — Description of results: 2. [file:1]
- **Task 5B** — Interpretation and rationale: 4. [file:1]

- **Deductions** — Extra page: -10. [file:1]
- **Deductions** — Late (per day): -10. [file:1]
- **Deductions** — Wrong formatting (on top of 10 points per extra page when formatting correctly): -10. [file:1]

- **Total:** 100. [file:1]