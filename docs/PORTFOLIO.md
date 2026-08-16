# Portfolio Notes

This file contains short, honest ways to present the project. Replace the
member labels with the group members' names before submitting it to faculty.

## 30-second project description

We built an end-to-end educational pneumonia-screening system for chest
X-rays. It uses a TensorFlow DenseNet121 transfer-learning model, exposes
predictions through FastAPI, displays them in a Streamlit interface, and runs
in Docker on Railway. On 624 held-out test images, it reached 99.23% sensitivity
and 95.72% ROC-AUC, but only 46.15% specificity. That tradeoff is important: it
missed very few pneumonia cases, while producing many false alarms, so it is a
learning project and not a clinical diagnostic tool.

## Suggested three-member presentation split

| Member | Main responsibility | Presentation topics |
| --- | --- | --- |
| Member 1 | Data and preprocessing | Dataset structure, class imbalance, validation split, resizing, normalization, and augmentation |
| Member 2 | Modeling and evaluation | Baseline CNN, DenseNet121 transfer learning, class weights, Grad-CAM, and medical evaluation metrics |
| Member 3 | Product and operations | FastAPI, Streamlit, Docker, Railway deployment, MLflow, CI, limitations, and responsible-use disclaimer |

All three members should understand the complete pipeline even though each
person leads one section.

## LinkedIn post

I recently completed an end-to-end machine-learning portfolio project with my
team: an educational pneumonia detector for chest X-rays.

We started with data validation and a CNN baseline, then improved the model
with DenseNet121 transfer learning. We added class-imbalance handling,
Grad-CAM visualizations, a FastAPI inference service, a Streamlit interface,
Docker deployment, MLflow experiment tracking, and GitHub Actions CI.

On 624 held-out test images, the selected model achieved 99.23% sensitivity,
85.71% F1-score, and 95.72% ROC-AUC. Its specificity was only 46.15%, which
gave us an important lesson: a strong headline metric does not make a medical
model ready for clinical use.

Repository: https://github.com/MdRedwanIslam1/pneumonia-detector

Live demo: https://pneumonia-detector-production-649a.up.railway.app/

This is an educational/portfolio project, not a certified diagnostic tool. It
must not be used for real medical decisions.

## Interview talking points

- **Why DenseNet121?** Transfer learning provides useful general visual
  features when the medical dataset is too small to train a deep model from
  scratch reliably.
- **Why not report accuracy alone?** The classes are imbalanced, and the cost
  of a false negative is different from the cost of a false positive.
- **What did the results reveal?** High sensitivity came with low specificity,
  so threshold calibration and external validation are necessary next steps.
- **What does Grad-CAM add?** It offers a visual debugging aid showing where
  the model was influential, but it is not proof of medical reasoning.
- **What made the project production-like?** A validated API contract, a model
  loaded once at startup, containerization, cloud deployment, experiment
  tracking, automated tests, and explicit limitations.
- **What would you improve next?** Validate on an independent hospital dataset,
  calibrate probabilities on validation data, compare thresholds without
  touching the test set, and monitor distribution drift.

## Faculty update

Our group completed the full machine-learning lifecycle: dataset validation,
preprocessing, baseline and transfer-learning models, medical-metric
evaluation, Grad-CAM, API and frontend development, Docker containerization,
Railway deployment, MLflow tracking, and automated CI checks. The project is
working locally and online. The final test results are documented transparently,
including the model's low specificity and the reasons it cannot be used for
medical decisions.
