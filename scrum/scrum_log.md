
---

## **2️⃣ Scrum Log (scrum_log.md)**

**Scrum Log** is a **daily/alternate day progress tracker**.  

**Template Example:**

```markdown
# 🏗️ Factify - Scrum Log

This log records daily/alternate day progress for the Factify Mini Project (20MCA 245).

---

## Week 1: Dataset Collection & Preprocessing

### 05/08/2025
- **Planned:** Initial project structure setup and first commit
- **Done:** Setup an initial project structure and github repo setup
- **Next:** Downloaded LIAR dataset, previewed data
- **Issues:** None


### 06/08/2025
- **Planned:** Download LIAR & ISOT datasets, explore columns
- **Done:** Downloaded LIAR, ISOT and WELFAKE datasets
- **Next:** Preview Data, EDA
- **Issues:** None

### 07/08/2025
- **Planned:** Clean all datasets and merge
- **Done:** Cleaned ISOT, LIAR, WELFake; removed NaNs; saved merged_clean_dataset.csv
- **Next:** Perform EDA
- **Issues:** None

### 08/08/2025
Task(s) Completed:

Downloaded and loaded FEVER and Indian Fake News datasets.

Merged them with existing datasets (LIAR, WeLFAKE, etc.).

Fixed missing column issues by combining title and text where applicable.

Verified merged dataset integrity (no missing text or label).

Final merged dataset shape: 221,220 rows → REAL: 136,105, FAKE: 85,115.

Challenges Faced:

Initial merge caused large data loss due to "nan" placeholder replacements.

Solved by properly combining title and text before replacing placeholders.

Next Steps:

Begin EDA tomorrow.

### 09/08/2025
- **Planned:** Perform EDA  
- **Done:** Built Bar Chart, Label Distribution
- **Next:** Balance label distribution of FAKE and REAL
- **Issues:** None

### 10/08/2025
- **Planned:** Balance label distribution
- **Done:** balanced label distribution and saved the dataset in /data/processed/balanced_df.csv
- **Next:** Build baseline TF-IDF model
- **Issues:** None

## Week 2: Model Building and analysis

### 12/08/2025
- **Planned:** Build baseline TF-IDF model
- **Done:** Built TF-IDF model using Linear SVM and Logistic Regression 
- **Next:** Test baseline model for higher accuracy
- **Issues:** None

### 15/08/2025
- **Planned:** Test baseline model and tuning
- **Done:** Tuned the model
- **Next:** Diagnose the issues
- **Issues:** Fake news has large word count compared to real ones, and it is causing a massive bias in the model, i need to look at each dataset and analyse each of them to diagnose which dataset is causing the problem before going to BERT fine tuning

### 16/08/2025
- **Planned:** Diagnosing recall issues
- **Done:** Diagnosed the issues and it is showing bad perfomance ~58% even when the average text lengths are adjusted before training
- **Next:** Build models on single datasets, i.e. 5 models
- **Issues:** None

### 17/08/2025
- **Planned:** Build models on single datasets, i.e. 5 models
- **Done:** Built TF-IDF + Logistic Regression models for each datasets(accuracy ~94% on average) other algorithms like SVM and Naive Bayes(Complement,Multinomial) will give similar perfomance, which is a given.
- **Next:** 
- **Issues:** None