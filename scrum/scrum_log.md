
---

## **2️⃣ Scrum Log (scrum_log.md)**

Your **Scrum Log** is a **daily/alternate day progress tracker**.  

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
- **Next:** Perform EDA and build baseline TF-IDF model
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

Begin EDA & TF-IDF baseline model tomorrow.