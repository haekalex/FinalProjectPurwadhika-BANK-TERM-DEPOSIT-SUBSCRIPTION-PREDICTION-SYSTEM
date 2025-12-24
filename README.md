# BANK TERM DEPOSIT SUBSCRIPTION PREDICTION SYSTEM  
**Final Project JCDSOH03**

---

## PROJECT OVERVIEW
This project develops a machine learning classification system to predict whether  
a bank customer will subscribe to a term deposit product (YES / NO) based on  
telemarketing campaign data.

The main objective of this project is to minimize False Negative (FN), which  
represents customers who are actually willing to subscribe but are not detected  
by the model. Therefore, the evaluation and optimization process prioritizes  
Recall for the positive class (YES).

**Main notebook:**
- `FinalProjectBeta.ipynb`
- 'Dataset : https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset'
---

## BUSINESS UNDERSTANDING

### BACKGROUND
Banks promote term deposit products through telemarketing campaigns. However,  
contacting all customers is inefficient and costly. A predictive model is  
required to identify high-potential customers and reduce missed business  
opportunities.

### BUSINESS PROBLEM
Key challenges faced by the bank:

- Lost opportunity due to missed potential subscribers (False Negative)  
- Inefficient marketing cost due to contacting low-potential customers  
- Lack of data-driven prioritization in campaign targeting  

### STAKEHOLDERS
1. Marketing Team      - Campaign execution and targeting  
2. Sales Team          - Follow-up and conversion  
3. Bank Management     - Strategic decision making  
4. Branch Operations   - Product support  
5. Customers           - Campaign recipients  

### PROJECT OBJECTIVES

**Primary goal:**
- Build a classification model to predict term deposit subscription with focus  
  on minimizing False Negative (FN).

**Specific objectives:**
1. Maximize recall for customers likely to subscribe  
2. Reduce missed sales opportunities  
3. Provide probability-based predictions for flexible decisions  
4. Support marketing prioritization strategies  

### BUSINESS IMPACT (CONFUSION MATRIX)

|                         | Predicted NO     | Predicted YES    |
|-------------------------|------------------|------------------|
| **Actual NO (0)**       | True Negative    | False Positive   |
| **Actual YES (1)**      | False Negative   | True Positive    |

**False Negative (FN):**
- Most critical error  
- Potential subscriber is not contacted  
- Leads to direct opportunity loss  

**False Positive (FP):**
- Less critical  
- Marketing cost increases, but no opportunity loss  

---

## DATASET INFORMATION

**Dataset:**
- Bank marketing telemarketing dataset  

**Target variable:**
- `y`  
  yes = 1 (Subscribe term deposit)  
  no  = 0 (Not subscribe)  

**Data characteristics:**
- Class imbalance exists (YES is minority class)  
- Mixed numerical and categorical features  

**Key feature groups:**
- Customer profile: age, job, marital, education  
- Financial status: default, housing, loan  
- Campaign information: contact, month, day_of_week, campaign, pdays, previous,  
  poutcome  
- Macroeconomic indicators: cons.price.idx, cons.conf.idx, euribor3m  

---

## PROJECT STRUCTURE

```text
Final Project/
|
|-- finproo3 (2).ipynb            Main notebook (analysis & modeling)
|-- app.py                        Streamlit application
|-- final_logreg_threshold.pkl    Saved model + threshold (recommended)
|-- bank-additional-full.csv      Dataset
