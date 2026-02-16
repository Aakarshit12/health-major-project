"""
Model Accuracy Evaluation Script for TorchXRayVision
Tests the accuracy of pretrained models on custom datasets
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import torchxrayvision as xrv
import torchxrayvision.datasets as xrv_datasets
from torch.utils.data import DataLoader
import argparse
import os
import json
from datetime import datetime
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TorchXRayVision model accuracy on custom dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images/",
        help="Folder containing your X-ray images"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="CSV file containing labels (optional). If not provided, will only show predictions."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="densenet121-res224-all",
        help="Model weights to use (e.g., 'densenet121-res224-all')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save evaluation results to file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification"
    )
    return parser.parse_args()


def get_device(device_arg):
    """Determine the device to use"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def evaluate_with_labels(model, dataloader, device, threshold=0.5):
    """Run inference and collect predictions when labels are available"""
    model.eval()
    y_true = []
    y_pred = []

    print("\nRunning Inference...")
    with torch.no_grad():
        for samples in tqdm(dataloader, desc="Processing batches"):
            images = samples["img"].to(device)
            labels = samples["lab"].cpu().numpy()

            outputs = model(images)
            predictions = outputs.cpu().numpy()

            y_true.append(labels)
            y_pred.append(predictions)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    return y_true, y_pred


def evaluate_without_labels(model, image_dir, device, batch_size=16):
    """Run inference on images without labels (prediction only)"""
    import glob
    from skimage import io
    import torchvision.transforms as transforms
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.dcm']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"\nFound {len(image_files)} images")
    print("Running inference (no labels available)...")
    
    model.eval()
    predictions_list = []
    filenames = []
    
    transform = transforms.Compose([
        xrv_datasets.XRayCenterCrop(),
        xrv_datasets.XRayResizer(224)
    ])
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Load and preprocess image
                img = io.imread(img_path)
                img = xrv_datasets.normalize(img, 255)
                
                # Handle color images
                if len(img.shape) == 3:
                    img = img.mean(2)[None, ...]
                else:
                    img = img[None, ...]
                
                img = transform(img)
                img = torch.from_numpy(img).unsqueeze(0).to(device)
                
                # Get predictions
                output = model(img)
                pred_probs = torch.sigmoid(output).cpu().numpy()[0]
                
                predictions_list.append(pred_probs)
                filenames.append(os.path.basename(img_path))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    predictions = np.array(predictions_list)
    
    # Display top predictions for each image
    print("\n" + "="*80)
    print("PREDICTIONS (Top 3 findings per image)")
    print("="*80)
    
    for i, filename in enumerate(filenames):
        preds = predictions[i]
        # Get top 3 predictions
        top_indices = np.argsort(preds)[::-1][:3]
        print(f"\n{filename}:")
        for idx in top_indices:
            disease = model.pathologies[idx]
            prob = preds[idx]
            print(f"  {disease:30} {prob*100:6.2f}%")
    
    return predictions, filenames


def calculate_metrics(y_true, y_pred, y_sigmoid, y_binary, labels, threshold=0.5):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # Per-class metrics
    n_classes = len(labels)
    per_class_auc = []
    per_class_f1 = []
    per_class_acc = []
    per_class_precision = []
    per_class_recall = []
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(f"{'Disease':<30} {'AUC':<10} {'F1':<10} {'Precision':<12} {'Recall':<12} {'Accuracy':<10}")
    print("-"*80)
    
    for i, label in enumerate(labels):
        true_class = y_true[:, i]
        pred_class = y_sigmoid[:, i]
        binary_class = y_binary[:, i]
        
        # Skip if all labels are the same (no variance)
        if len(np.unique(true_class)) > 1:
            try:
                auc = roc_auc_score(true_class, pred_class)
                per_class_auc.append(auc)
            except ValueError:
                auc = np.nan
                per_class_auc.append(np.nan)
        else:
            auc = np.nan
            per_class_auc.append(np.nan)
        
        # Calculate other metrics
        f1 = f1_score(true_class, binary_class, zero_division=0)
        precision = precision_score(true_class, binary_class, zero_division=0)
        recall = recall_score(true_class, binary_class, zero_division=0)
        acc = accuracy_score(true_class, binary_class)
        
        per_class_f1.append(f1)
        per_class_acc.append(acc)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"{label:<30} {auc_str:<10} {f1:<10.4f} {precision:<12.4f} {recall:<12.4f} {acc:<10.4f}")
    
    metrics['per_class'] = {
        'labels': labels,
        'auc': per_class_auc,
        'f1': per_class_f1,
        'accuracy': per_class_acc,
        'precision': per_class_precision,
        'recall': per_class_recall
    }
    
    # Overall metrics
    print("\n" + "="*80)
    print("OVERALL ACCURACY METRICS")
    print("="*80)
    
    # Macro averages (excluding NaN)
    valid_aucs = [auc for auc in per_class_auc if not np.isnan(auc)]
    macro_auc = np.mean(valid_aucs) if valid_aucs else np.nan
    macro_f1 = np.mean(per_class_f1)
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_acc = np.mean(per_class_acc)
    
    # Micro averages
    micro_f1 = f1_score(y_true.flatten(), y_binary.flatten(), zero_division=0)
    micro_precision = precision_score(y_true.flatten(), y_binary.flatten(), zero_division=0)
    micro_recall = recall_score(y_true.flatten(), y_binary.flatten(), zero_division=0)
    micro_acc = accuracy_score(y_true.flatten(), y_binary.flatten())
    
    try:
        micro_auc = roc_auc_score(y_true.flatten(), y_sigmoid.flatten())
    except ValueError:
        micro_auc = np.nan
    
    print(f"Macro ROC-AUC:     {macro_auc:.4f}")
    print(f"Micro ROC-AUC:     {micro_auc:.4f}" if not np.isnan(micro_auc) else "Micro ROC-AUC:     N/A")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Micro F1-Score:    {micro_f1:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Micro Precision:   {micro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Micro Recall:      {micro_recall:.4f}")
    print(f"Macro Accuracy:    {macro_acc:.4f}")
    print(f"Micro Accuracy:    {micro_acc:.4f}")
    
    metrics['overall'] = {
        'macro_auc': float(macro_auc) if not np.isnan(macro_auc) else None,
        'micro_auc': float(micro_auc) if not np.isnan(micro_auc) else None,
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'micro_precision': float(micro_precision),
        'macro_recall': float(macro_recall),
        'micro_recall': float(micro_recall),
        'macro_accuracy': float(macro_acc),
        'micro_accuracy': float(micro_acc)
    }
    
    # Classification Report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        y_true, 
        y_binary, 
        target_names=list(labels),
        zero_division=0
    ))
    
    return metrics


def save_results(metrics, output_dir, args, predictions=None, filenames=None, model_pathologies=None):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    results = {
        'timestamp': timestamp,
        'config': {
            'image_dir': args.image_dir,
            'csv_file': args.csv_file,
            'model': args.model,
            'batch_size': args.batch_size,
            'threshold': args.threshold
        },
        'metrics': metrics
    }
    
    csv_path = None
    if predictions is not None and filenames is not None and model_pathologies is not None:
        # Save predictions CSV
        csv_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        pred_df = pd.DataFrame(
            predictions,
            index=filenames,
            columns=[f"prob_{pathology}" for pathology in model_pathologies]
        )
        pred_df.to_csv(csv_path)
        results['predictions_csv'] = csv_path
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    if csv_path is not None:
        print(f"Predictions saved to: {csv_path}")
    return json_path


def main():
    args = parse_args()
    
    # Also write output to file for debugging
    output_file = open("eval_output.log", "w", encoding="utf-8")
    import sys
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    sys.stdout = TeeOutput(sys.stdout, output_file)
    sys.stderr = TeeOutput(sys.stderr, output_file)
    
    # Validate inputs
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    
    # Setup device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading TorchXRayVision model: {args.model}...")
    try:
        model = xrv.models.DenseNet(weights=args.model)
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully!")
        print(f"Model pathologies: {len(model.pathologies)} classes")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Check if CSV file exists
    if args.csv_file and os.path.exists(args.csv_file):
        # Evaluation with labels
        print("\nLoading Dataset with Labels...")
        try:
            dataset = xrv_datasets.CSV_Dataset(
                imgpath=args.image_dir,
                csvpath=args.csv_file,
                transform=xrv.datasets.XRayCenterCrop(),
                labels=None
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
        
        print(f"Dataset Size: {len(dataset)} images")
        print(f"Disease Labels ({len(dataset.labels)}): {', '.join(dataset.labels)}")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if device.type == "cuda" else False
        )
        
        # Run evaluation
        y_true, y_pred = evaluate_with_labels(model, dataloader, device, args.threshold)
        
        # Apply sigmoid activation
        print("\nApplying Sigmoid Activation...")
        y_sigmoid = 1 / (1 + np.exp(-y_pred))  # Convert logits to probabilities
        y_binary = (y_sigmoid > args.threshold).astype(int)
        
        # Calculate metrics
        print("\nEvaluating Model Accuracy...")
        metrics = calculate_metrics(
            y_true, 
            y_pred, 
            y_sigmoid, 
            y_binary, 
            dataset.labels,
            args.threshold
        )
        
        # Save results if requested
        if args.save_results:
            save_results(metrics, args.output_dir, args, model_pathologies=model.pathologies)
    
    else:
        # Evaluation without labels (prediction only)
        if args.csv_file:
            print(f"Warning: CSV file '{args.csv_file}' not found. Running prediction-only mode.")
        
        predictions, filenames = evaluate_without_labels(
            model, 
            args.image_dir, 
            device, 
            args.batch_size
        )
        
        # Save predictions if requested
        if args.save_results:
            save_results(None, args.output_dir, args, predictions, filenames, model.pathologies)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    
    output_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()

