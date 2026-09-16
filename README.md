# **Intel Scene Classification using MobileNetV2**

## **Project Overview**

This project is an image classification system built using **PyTorch** and **MobileNetV2**. It classifies natural and urban scene images into **six different categories**.

## **Classes**

The model can classify images into:

* **Buildings**
* **Forest**
* **Glacier**
* **Mountain**
* **Sea**
* **Street**

## **Dataset**

The project uses the **Intel Image Classification dataset**.

**Dataset:**

* **Training images:** 14,034
* **Test images:** 3,000
* **Training split:** 11,227 images
* **Validation split:** 2,807 images

## **Model**

The project uses a **pretrained MobileNetV2 model**. The original classifier was replaced with a custom classifier containing:

* **Dropout:** 0.3
* **Linear layer:** 1280 input features
* **Output classes:** 6

## **Training**

The model was trained for **10 epochs** using:

* **Optimizer:** Adam
* **Learning rate:** 0.0001
* **Batch size:** 32
* **Image size:** 224 × 224
* **Device:** CPU

Data augmentation was applied using **random horizontal flipping** and **random rotation**.

## **Results**

After training:

* **Best Validation Accuracy:** **88.53%**
* **Test Accuracy:** **87.93%**

The best model was saved as:

`best_mobilenetv2.pth`

## **Project Files**

* `intel_mobilenetv2.py` — Main training script for the MobileNetV2 model.
* `predict_image.py` — Script used to classify a new image.
* `test_image.jpg` — Sample image used for prediction.
* `.gitignore` — Specifies files that should not be uploaded to GitHub.

## **How to Run**

Install the required libraries and activate the virtual environment.

**Run the training script:**

```bash
python intel_mobilenetv2.py
```
