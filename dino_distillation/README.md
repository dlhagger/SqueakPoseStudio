# DINO Distillation Toolkit

This folder contains the assets used to pretrain YOLO pose backbones from DINOv3 teachers. The main entry points are:

- `DINOv3_Distillation_YOLO-pose/dino_distillation.ipynb` – a notebook that walks through frame extraction, Lightly Train distillation, checkpoint export, and quick sanity checks/visualizations.

## Notebook workflow

1. **Configure paths & options** – point to your unlabeled video folder(this is just a directory with videos (can have other files)
2. **Extract frames** – run the OpenCV cell to populate `data/unlabeled_frames/` from your video directory if you need images for distillation. note you want north of 1,000,000 images for this. The more diversity in pose the better.
3. **Run distillation** – call `run_distillation()`, which wraps `lightly_train.train` with `method="distillation"`. Each run creates a timestamped folder inside `dino_distillations/` with subfolders for checkpoints, exported Ultralytics packages, logs, and metrics. This will take a long time. I ran this on a NVIDIA SPARK with 3.5 million images for 40 epochs and it took 3 days. You NEED substantial compute for distillation.
4. **Inspect latest export** – helper cells automatically locate the newest `exported_models/exported_last.pt`, print model metadata, and grab the task/scale from the YOLO YAML.
5. **Plot training loss** – the final cell parses the latest run’s `metrics.jsonl`, extracts loss values (`train_loss`), and saves a PNG curve in the run directory.


## Directory structure

```
dino_distillation/
├── DINOv3_Distillation_YOLO-pose/
│   ├── data/                 # extracted unlabeled frames (optional)
│   ├── dino_distillation.ipynb
│   ├── dino_distillations/   # one subfolder per Lightly run (includes checkpoints/exported models)
│   ├── yolo_yamls/           # custom YOLO configs (e.g., pose variants)
│   └── teacher weights (.pth, downloaded separately)
└── README.md                 # this file
```

Use the notebook for an interactive workflow (frame extraction + distillation + diagnostics).
