# 1. Automatic Aging Dataset  Characteristics
Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging(https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/)
## Abstract:
Autonomic function regulating blood pressure and cardiac rhythm progressively declines with increasing age. Impaired cardiovascular control promotes a variety of age-related conditions such as dementia, or Alzheimer’s disease. This study aims to provide a database of high-resolution biological signals to describe the effect of healthy aging on cardiovascular regulation. Electrocardiogram and continuous non-invasive blood pressure signals were recorded simultaneously at rest in 1,121 healthy volunteers.
## Background
The elderly population has a high prevalence of cardiovascular diseases, representing the globally leading cause of death [1]. Impairment of autonomic regulation of the cardiovascular system promotes the risk to develop age-related diseases such as dementia. Therefore, it is essential to detect bodily changes that adumbrate constrained cardiovascular fitness.  This database contains resting recordings of ECG and continuous noninvasive blood pressure of 1,121 healthy volunteers. Data sets have been collected over the last decade in Jena University Hospital. Here, we share the resulting data base to promote the systematic analysis of the effect of healthy aging on the cardiovascular system.

## Methods
All measurements were recorded at the department of psychosomatic medicine and psychotherapy at Jena university hospital. The study was approved by the ethics committee of the Medical Faculty of the Friedrich Schiller University Jena. All research was performed in accordance with relevant guidelines and regulations. The informed written consent was obtained from all subjects.

An ECG (lead II) was recorded at 1000 Hz either by an MP150 (ECG100C, BIOPAC systems inc., Golata, CA, USA) or Task Force Monitor system (CNSystems Medizintechnik GmbH, Graz AUT). pre-gelled Ag/AgCl electrodes (BlueSensor VL, Ambu BmbH, Bad Nauheim, GER) were attached according to an Einthoven triangle.

Continuous blood pressure was recorded non-invasively using the vascular unloading technique [2]. In short, a cuff around the finger is controlled to maintain constant pressure, while blood volume is recorded via photoplethysmography. With non-varying cuff pressure, the acquired blood volume is proportional to blood pressure in the arteries of the finger. The recorded signal is mapped to brachial blood pressure that is measured oscillometricly once during initialization of the system. The Task Force Monitor is equipped with a module for continuous blood pressure measurement. The MP150 system digitizes the signal acquired by a separate monitor CNAP 500 (CNSystems Medizintechnik GmbH, Graz AUT). The sampling frequency was 1000 Hz for both systems.

Measurements were performed in an examination room that was temperature controlled at 22C. During the recordings it was absolutely quiet and fully shaded. The illumination level was kept constant via an indirect light source. The recording session started with an interview of the participant. Then, the purpose and design of the study was explained. Each of the subjects included in this collection gave his or her written consent to participate.

After the subjects lied down comfortably on the examination tilt table, electrodes and pressure cuffs were placed. For the resting state recording, we instructed participants to avoid movement, yawning or coughing.

The instructor waited a few minutes for the participant to calm down and checked the quality of the acquired signals. In case of insufficient signal quality, electrodes and cuffs were re-arranged. Otherwise, the recording was started. The length of the recording was on average 19 minutes (8 - 45 minutes) and was supervised by the instructor.

## Data Description
The data files are provided in open WFDB standard format and named in consecutive numbers after random ordering. Additional patient information is stored in the file subject-info.csv. To assure that none of our subject can be identified based on demographic information, we generalized individual age to age groups. We used the free Data Anonymization Toolbox ARX applied a k=2 anonymity condition and an average re-identification risk below 5% [3-4].

Age groups are defined as follows: 1 (18-19 years), 2 (20-24 years), 3 (25-29 years), 4 (30-34 years), 5 (35-39 years), 6 (40-44 years), 7 (45-49 years), 8 (50-54 years), 9 (55-59 years), 10 (60-64 years), 11 (65-69 years), 12 (70-74 years), 13 (75-79 years), 14 (80-84 years), 15 (85-92 years). Gender is coded 0 (male) or 1 (female). Recording device is either 0 (TFM, CNSystems) or 1 (CNAP 500, CNSystems; MP150, BIOPAC Systems).



![plot](ecg_tsne_plot_npy_direct.png)





# 2. Estimation-of-Calender-Age-of-Humans-from-ECG-Using Auto-Aging-Data
The PhysioNet "Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging" database is used.
# 🩺 Estimating Calendar Age from the PhysioNet Autonomic Aging (AutoAging) Dataset

This repository explains the methodology for estimating **calendar age** (chronological age) using physiological features derived from the **PhysioNet Autonomic Aging (AutoAging)** dataset.

The core principle relies on applying machine learning regression models to various cardiovascular indices, which inherently change as a function of biological aging, to predict a subject's chronological age.

---

## 🔬 Methodology for Age Estimation

The process of estimating age from the AutoAging data typically involves the following sequential steps:

### 1. Data Source and Physiological Signals
* **Dataset:** PhysioNet's "Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging."
* **Raw Data:** Includes resting-state recordings of:
    * **Electrocardiogram (ECG)**
    * **Continuous Non-invasive Blood Pressure (BP)**

### 2. Feature Extraction (Cardiovascular Indices)
The raw signals are processed to extract a comprehensive set of quantitative features representing autonomic cardiovascular function. These are the inputs to the machine learning model.

| Feature Type | Key Examples | Relevance |
| :--- | :--- | :--- |
| **Heart Rate Variability (HRV)** | SDNN, RMSSD, LF/HF ratio | Reflects sympathetic and parasympathetic balance. |
| **Blood Pressure Variability (BPV)** | Fluctuations in continuous BP signal. | Indicates vascular regulation dynamics. |
| **Baroreflex Sensitivity (BRS)** | Measures of HR response to BP changes. | Key measure of autonomic control efficiency. |
| **Pulse Wave Dynamics** | Arterial stiffness proxies (derived from BP shape). | Indicates vascular aging. |

### 3. Machine Learning Regression Model
* **Task:** **Regression**, mapping physiological indices to a continuous age value (in years).
* **Training:** Models (e.g., **Gaussian Process Regression (GPR)**, **Support Vector Regression (SVR)**, **Relevance Vector Regression (RVR)**) are trained on the extracted features and the known calendar age.
* **Output:** The model predicts an age, often interpreted as the **cardiovascular biological age**.

### 4. Performance Evaluation
Model accuracy is assessed using standard regression metrics:
* **Mean Absolute Error (MAE):** The average difference between the predicted age and the true calendar age (e.g., 5.6 years).
* **Correlation Coefficient ($r$):** Measures the linear relationship between predicted and true age (e.g., $r = 0.81$).

---

## 💡 Biological Age Interpretation

The difference between the **Predicted Age** and the **Calendar Age** is often termed the **Aging Gap**.

* **Predicted Age > Calendar Age:** Suggests **accelerated biological aging** of the cardiovascular system.
* **Predicted Age < Calendar Age:** Suggests **decelerated biological aging**.

This framework provides a quantitative, non-invasive method for assessing cardiovascular health relative to a person's chronological age.

---

## 📊 PhysioNet AutoAging Dataset Details

| Parameter | Detail |
| :--- | :--- |
| **Study Population** | 1,121 healthy volunteers |
| **Age Range** | Wide range (mean $\approx 32.5$ years) |
| **Signals Recorded** | **ECG** (1000 Hz), **Continuous BP** (100 Hz) |
| **Demographics** | Calendar Age, Gender, Body Mass Index (BMI) |
| **Condition** | Resting state, controlled lab environment |

The high sampling rates allow for the accurate calculation of complex time and frequency domain indices necessary for precise age prediction.

# 5. Test Data(PTB Diagnostic ECG Database):
https://www.physionet.org/content/ptbdb/1.0.0/

## Test Data Specifications
The ECGs in this collection were obtained using a non-commercial, PTB prototype recorder with the following specifications:

* 16 input channels, (14 for ECGs, 1 for respiration, 1 for line voltage)
*  **Input voltage**: ±16 mV, compensated offset voltage up to ± 300 mV
* Input resistance: 100 Ω (DC)
* Resolution: 16 bit with 0.5 μV/LSB (2000 A/D units per mV)
* Bandwidth: 0 - 1 kHz (synchronous sampling of all channels)
* Noise voltage: max. 10 μV (pp), respectively 3 μV (RMS) with input short circuit
* Online recording of skin resistance
* Noise level recording during signal collection
The database contains 549 records from 290 subjects (aged 17 to 87, mean 57.2; 209 men, mean age 55.5, and 81 women, mean age 61.6; ages were not recorded for 1 female and 14 male subjects). Each subject is represented by one to five records. There are no subjects numbered 124, 132, 134, or 161. Each record includes 15 simultaneously measured signals: the conventional 12 leads (i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6) together with the 3 Frank lead ECGs (vx, vy, vz). Each signal is digitized at 1000 samples per second, with 16 bit resolution over a range of ± 16.384 mV. On special request to the contributors of the database, recordings may be available at sampling rates up to 10 KHz.

Within the header (.hea) file of most of these ECG records is a detailed clinical summary, including age, gender, diagnosis, and where applicable, data on medical history, medication and interventions, coronary artery pathology, ventriculography, echocardiography, and hemodynamics. The clinical summary is not available for 22 subjects.



# Feature Extraction Network Layers and Flow DIagram 

