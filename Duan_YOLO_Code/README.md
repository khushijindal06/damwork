# Duan UAV Piping YOLO Code

This folder collects the complete Duan RGB experiment code in one place for
review and reuse.

## Files

1. `07_prepare_duan_yolo.py` extracts the rendered bounding boxes, writes YOLO
   labels, creates derived clean images, and makes grouped train/validation/test
   splits.
2. `08_generate_duan_synthetic.py` creates the haze, fog, low-light, and mixed
   datasets without changing image geometry or labels.
3. `09_enhance_duan_test.py` applies CLAHE, automatic gamma, Retinex, and
   dark-channel-prior enhancement to degraded test images.
4. `10_train_evaluate_duan_yolo.py` trains clean and robust YOLO models,
   evaluates all conditions, and calculates mAP recovery.
5. `damwork.ipynb` runs the complete workflow step by step on Windows or
   Google Colab.
6. `requirements.txt` lists the Python dependencies.

The notebook defaults to writing generated data outside the repository:

```text
E:/Downloads/10896178/Duan_RGB_Experiment
```

The original maintained copies of the Python scripts also remain under
`DamVis/scripts/`.
