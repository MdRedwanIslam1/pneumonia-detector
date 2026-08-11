# Pneumonia Detector

> **Medical disclaimer:** This is an educational/portfolio project, not a
> certified diagnostic tool. It must not be used for real medical decisions.

This project will classify chest X-ray images as `NORMAL` or `PNEUMONIA` using
TensorFlow/Keras. It starts with a small CNN and later adds transfer learning,
careful medical-model evaluation, Grad-CAM, an API, a frontend, and deployment.

## Live demo

[Open the deployed Railway application](https://pneumonia-detector-production-649a.up.railway.app/)

The public demo runs the Streamlit interface and the FastAPI inference service
inside one cloud container. The educational-use disclaimer remains visible in
the interface and every prediction response.

## Project location

The complete project is stored at:

```text
D:\MLProjects\pneumonia-detector
```

## Local setup

### GPU setup with WSL2

Open PowerShell and start Ubuntu:

```powershell
wsl -d Ubuntu-ML
```

Then activate the GPU-enabled Linux environment:

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m src.train
```

### Windows CPU environment

Open PowerShell and run:

```powershell
cd D:\MLProjects\pneumonia-detector
.\.venv\Scripts\Activate.ps1
python -m src.train
```

Phase 1 creates the environment and starter structure. Dataset preparation
begins in Phase 2.

## Phase 2: dataset exploration

After downloading the Kaggle dataset, run:

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m src.explore_data
```

The script checks every image and saves its reports under
`outputs/data_exploration/`.

The dataset can be downloaded to the project with KaggleHub:

```bash
python -c "import kagglehub; kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia', output_dir='data/raw')"
```

## Phase 3: preprocessing pipeline

The reusable TensorFlow pipeline converts every image to RGB, resizes it to
`224x224`, normalizes pixels to `0-1`, creates a stratified validation split,
and calculates balanced class weights. Gentle augmentation is applied only to
the training dataset.

Verify the complete pipeline and generate an augmentation preview:

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m src.verify_preprocessing
```

## Phase 4: baseline CNN

The baseline is a four-block convolutional network trained from scratch. It is
deliberately smaller than the transfer-learning model introduced in Phase 5.

Train for ten epochs with the WSL GPU environment:

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m src.train --epochs 10 --batch-size 32
```

The best model is saved under `models/`. Training history, validation metrics,
and accuracy/loss curves are saved under `outputs/baseline_cnn/`.

Current baseline result (best checkpoint at epoch 9): validation accuracy
`88.63%`, precision `98.24%`, recall `86.23%`, and loss `0.2381`. These are
validation results; the untouched test set is reserved for Phase 7.

## Phase 5: DenseNet121 transfer learning

The transfer model first trains a new classifier while its ImageNet-pretrained
DenseNet121 base is frozen. It then fine-tunes the top 40 layers with a learning
rate 100 times smaller.

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m src.train_transfer \
  --batch-size 16 \
  --frozen-epochs 5 \
  --fine-tune-epochs 5
```

The selected checkpoint and frozen/fine-tuned comparisons are saved under
`models/` and `outputs/densenet121/`.

### Current validation comparison

| Model | Accuracy | Precision | Recall | Loss |
| --- | ---: | ---: | ---: | ---: |
| Baseline CNN | 88.63% | 98.24% | 86.23% | 0.2381 |
| DenseNet121, frozen | **96.66%** | 97.56% | **97.94%** | **0.0956** |
| DenseNet121, fine-tuned | 95.32% | **99.46%** | 94.21% | 0.1377 |

The frozen DenseNet121 checkpoint is selected because it has the lowest
validation loss, highest accuracy, and highest recall. Test-set evaluation is
still reserved for Phase 7.

## Phase 6: advanced training and Grad-CAM

Advanced training continues from the selected DenseNet checkpoint using focal
loss, early stopping, and automatic learning-rate reduction:

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m src.train_advanced --batch-size 16 --max-epochs 12
```

Generate Grad-CAM explanations from validation images without touching the test
set:

```bash
python -m src.gradcam --model-path models/densenet121_advanced_best.keras
```

Grad-CAM highlights regions that influenced a model prediction. It does not
prove that the highlighted region is pneumonia and must not be treated as a
clinical explanation.

### Advanced-training outcome

Early stopping ended focal-loss training after 8 of 12 possible epochs. The
learning-rate scheduler reduced the rate from `1e-4` to `1.6e-7`. The focal
candidate reached `94.65%` validation accuracy, `99.45%` precision, and
`93.31%` recall, but its ordinary BCE loss (`0.1677`) was worse than the Phase 5
checkpoint (`0.0956`). The Phase 5 frozen DenseNet is therefore retained as the
final Phase 6 model.

## Phase 7: rigorous test evaluation

Run the final selected model once on the untouched test set at the predefined
`0.5` decision threshold:

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m src.evaluate
```

The command saves accuracy, precision, sensitivity, specificity, F1, ROC-AUC,
a confusion matrix, an ROC curve, every test probability, and selected mistakes
under `outputs/evaluation/`.

### Final test results

| Metric | Result |
| --- | ---: |
| Accuracy | 79.33% |
| Precision | 75.44% |
| Recall / sensitivity | **99.23%** |
| Specificity | 46.15% |
| F1-score | 85.71% |
| ROC-AUC | **95.72%** |

At the fixed `0.5` threshold, the confusion matrix contains 108 true negatives,
126 false positives, 3 false negatives, and 387 true positives. The model
missed only 3 of 390 pneumonia examples, but incorrectly flagged 126 of 234
normal examples. Its strong ROC-AUC shows good overall ranking ability, while
the poor specificity shows that the current decision threshold and probability
calibration are not suitable for clinical use.

The threshold was intentionally not tuned on the test set. Any future threshold
selection or probability calibration must use validation data or a separate
calibration set, followed by evaluation on a fresh holdout or external dataset.
This result also shows why accuracy alone is not enough for imbalanced medical
classification.

## Phase 8: FastAPI inference service

FastAPI exposes the selected model through an HTTP API. The model is loaded
once when the server starts, and `POST /predict` accepts one JPEG or PNG image.
The response contains the predicted class, confidence, pneumonia probability,
threshold, disclaimer, and an optional Grad-CAM PNG data URL.

Install the Phase 8 dependencies in the activated WSL environment:

```bash
python -m pip install fastapi "uvicorn[standard]" python-multipart
```

Start the API from the project root:

```bash
source /mnt/d/MLProjects/pneumonia-detector/scripts/activate_wsl.sh
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive API documentation, or
test a file from a second WSL terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
```

Requesting `/predict?include_gradcam=true` adds the explanatory overlay. Uploads
are limited to 10 MB and must contain a valid JPEG or PNG image. Grad-CAM shows
model influence only; it is not a medical explanation or diagnosis.

## Phase 9: Streamlit frontend

The Streamlit interface uploads a validated image to FastAPI and displays the
source X-ray, predicted class, confidence, pneumonia probability, threshold,
and optional Grad-CAM overlay. The browser-facing app does not load another
copy of the TensorFlow model.

Install the frontend dependency in the activated WSL environment:

```bash
python -m pip install streamlit
```

Run FastAPI in the first WSL terminal:

```bash
cd /mnt/d/MLProjects/pneumonia-detector
source scripts/activate_wsl.sh
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Run Streamlit in a second WSL terminal:

```bash
cd /mnt/d/MLProjects/pneumonia-detector
source scripts/activate_wsl.sh
python -m streamlit run frontend/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Open `http://localhost:8501`. If Windows localhost forwarding is unavailable,
run `hostname -I` in WSL and open `http://<WSL-IP>:8501` instead. The frontend
uses `http://127.0.0.1:8000` for its API by default; set the `API_URL`
environment variable when the API is hosted elsewhere.

## Phase 10: Docker containerization

Docker Compose builds two separate runtime images. The `api` container includes
TensorFlow, the selected model, and FastAPI. The smaller `frontend` container
includes Streamlit and calls the API over Docker's private network at
`http://api:8000`. Training data, generated outputs, virtual environments, and
unused checkpoints are excluded from the build context.

### Build and run

Docker Desktop must be running with WSL integration enabled for `Ubuntu-ML`.
From the project root in WSL, build both images:

```bash
docker compose build
```

Start both containers and keep their combined logs visible:

```bash
docker compose up
```

When the `api` and `frontend` services report healthy, open:

- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://localhost:8080/docs`

Run the containers in the background instead when terminal logs are not needed:

```bash
docker compose up --detach
docker compose ps
```

Inspect service logs:

```bash
docker compose logs --follow api
docker compose logs --follow frontend
```

Stop and remove the containers and their private network. This does not delete
the built images, source code, dataset, or model:

```bash
docker compose down
```

After changing code or dependencies, rebuild before starting:

```bash
docker compose up --build
```

### Common problems

- `docker: command not found`: install Docker Desktop and enable its WSL
  integration for `Ubuntu-ML`.
- `Cannot connect to the Docker daemon`: start Docker Desktop and wait until its
  engine reports that it is running.
- `port is already allocated`: stop processes using host ports `8080` or `8501`,
  then run Compose again. FastAPI still uses port `8000` inside Docker.
- `api` remains unhealthy: model startup can take longer on CPU; inspect it with
  `docker compose logs api`.
- Build runs out of disk space: move Docker Desktop's disk image location to the
  D drive and remove unused build cache from Docker Desktop when appropriate.

## Phase 11: Cloud deployment

Cloud platforms expose one public port per service. `Dockerfile.deploy` therefore
runs both applications in one container: FastAPI listens privately on port 8000,
and Streamlit listens on the public `PORT` supplied by the host. The startup
script waits for the model API to become healthy before opening the interface.

The live Railway deployment was verified end to end with a held-out normal
X-ray: the public interface returned `NORMAL` with 67.6% confidence and rendered
the Grad-CAM overlay.

The deployment image contains only runtime code and the selected 28.8 MB model.
The dataset, notebooks, generated plots, virtual environments, and unused model
checkpoints are excluded. The tested image is approximately 575 MB and used
about 955 MB during a Grad-CAM request under a 1 GB Docker memory limit.

### Test the cloud image locally

```bash
docker build --file Dockerfile.deploy --tag pneumonia-detector-cloud:local .
docker run --rm --memory 1g --cpus 2 \
  --publish 8600:7860 \
  pneumonia-detector-cloud:local
```

Open `http://localhost:8600`. Stop it with `Ctrl+C`.

### Recommended hosting path

Provider limits change over time. These choices were checked on August 12, 2026:

- Railway is the easiest temporary no-card demo. Its new-account trial provides
  $5 of credit for up to 30 days and permits 1 GB RAM. `railway.json` selects
  `Dockerfile.deploy`, configures the health check, and allows graceful restarts.
  Enable App Sleeping to avoid using credit while the demo is idle. The 512 MB
  Free plan available after the trial is too small for this TensorFlow build.
- Hugging Face Docker Spaces is the most comfortable portfolio host because CPU
  Basic provides 2 vCPUs and 16 GB RAM with no hourly compute charge. Creating a
  new Docker Space currently requires a $9/month PRO account. Use port 7860 and
  deploy the contents of this repository with `Dockerfile.deploy` named
  `Dockerfile` in the Space repository.
- Render's Free web service has only 512 MB RAM, which is below the measured
  requirement. Use a 2 GB instance or first migrate inference to a smaller
  LiteRT/TensorFlow Lite runtime; do not expect this full image to work on Free.

Railway automatically supplies `PORT`. Optional public configuration values are:

```text
PREDICTION_THRESHOLD=0.5
MODEL_PATH=/home/user/app/models/densenet121_advanced_best.keras
```

Do not put passwords or API keys in source files. If a future version needs a
secret, store it in the provider's Variables or Secrets dashboard. Check the
deployment logs after each release and visit `/_stcore/health` to verify uptime.
Free or sleeping services may take longer on their first request after inactivity.
If a Railway domain returns `502 Application failed to respond`, open the deploy
logs, find Streamlit's `Local URL` port, and set the domain's Target Port to the
same value. The first verified deployment used Railway's injected port `8080`.

The current model file is already small enough for the recommended hosts.
Post-training quantization and LiteRT remain useful future optimizations if a
strict 512 MB service or a much smaller image becomes necessary; any converted
model must be reevaluated against the full Phase 7 test set before release.
