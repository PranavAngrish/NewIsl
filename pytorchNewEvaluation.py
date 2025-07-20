import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
from collections import Counter


def convert_to_json_serializable(obj):
    """Convert NumPy data types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ISLEvaluator:
    """Enhanced PyTorch model evaluator for ISL detection with detailed per-class analysis"""
    
    def __init__(self, model, test_loader, device, class_names, save_dir="evaluation_results"):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize detailed tracking variables
        self.all_predictions = None
        self.all_targets = None
        self.all_probabilities = None
        self.per_class_details = {}
        
    def evaluate(self):
        """Evaluate the model on test data with detailed per-class tracking"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Track per-sample details for comprehensive analysis
        sample_details = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                frames, landmarks = data
                frames, landmarks, target = frames.to(self.device), landmarks.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(frames, landmarks)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                # Store batch results
                batch_predictions = predictions.cpu().numpy()
                batch_targets = target.cpu().numpy()
                batch_probabilities = probabilities.cpu().numpy()
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
                all_probabilities.extend(batch_probabilities)
                
                # Store detailed sample information
                for i in range(len(batch_targets)):
                    sample_details.append({
                        'true_class_idx': int(batch_targets[i]),
                        'predicted_class_idx': int(batch_predictions[i]),
                        'true_class_name': self.class_names[batch_targets[i]],
                        'predicted_class_name': self.class_names[batch_predictions[i]],
                        'confidence': float(batch_probabilities[i].max()),
                        'is_correct': bool(batch_targets[i] == batch_predictions[i])
                    })
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Store results for detailed analysis
        self.all_predictions = np.array(all_predictions)
        self.all_targets = np.array(all_targets)
        self.all_probabilities = np.array(all_probabilities)
        self.sample_details = sample_details
        
        # Calculate detailed per-class statistics
        self._calculate_detailed_per_class_stats()
        
        return accuracy, avg_loss
    
    def _calculate_detailed_per_class_stats(self):
        """Calculate detailed statistics for each class"""
        self.per_class_details = {}
        
        # Count total samples per class
        class_counts = Counter(self.all_targets)
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get samples for this class
            class_mask = (self.all_targets == class_idx)
            class_predictions = self.all_predictions[class_mask]
            class_probabilities = self.all_probabilities[class_mask]
            
            total_samples = class_counts.get(class_idx, 0)
            correct_predictions = int(np.sum(class_predictions == class_idx))  # Convert to int
            incorrect_predictions = total_samples - correct_predictions
            
            if total_samples > 0:
                class_accuracy = float(correct_predictions / total_samples)  # Convert to float
                avg_confidence = float(np.mean(class_probabilities[:, class_idx])) if len(class_probabilities) > 0 else 0.0
                
                # Find what this class was misclassified as
                misclassified_as = {}
                if incorrect_predictions > 0:
                    incorrect_mask = class_predictions != class_idx
                    incorrect_preds = class_predictions[incorrect_mask]
                    for pred_idx in incorrect_preds:
                        pred_class_name = self.class_names[pred_idx]
                        if pred_class_name not in misclassified_as:
                            misclassified_as[pred_class_name] = 0
                        misclassified_as[pred_class_name] += 1
                
                # Sort misclassifications by frequency and convert counts to int
                misclassified_as = {k: int(v) for k, v in sorted(misclassified_as.items(), key=lambda x: x[1], reverse=True)}
            else:
                class_accuracy = 0.0
                avg_confidence = 0.0
                misclassified_as = {}
            
            self.per_class_details[class_name] = {
                'class_index': int(class_idx),
                'total_test_videos': int(total_samples),
                'correct_predictions': correct_predictions,
                'incorrect_predictions': int(incorrect_predictions),
                'accuracy': class_accuracy,
                'accuracy_percentage': float(class_accuracy * 100),
                'average_confidence': avg_confidence,
                'misclassified_as': misclassified_as
            }
    
    def print_detailed_per_class_report(self):
        """Print detailed per-class performance report"""
        print("\n" + "=" * 100)
        print("DETAILED PER-CLASS PERFORMANCE ANALYSIS")
        print("=" * 100)
        
        # Sort classes by accuracy (worst performing first)
        sorted_classes = sorted(
            self.per_class_details.items(),
            key=lambda x: x[1]['accuracy']
        )
        
        print(f"{'Class Name':<25} {'Total Videos':<12} {'Correct':<8} {'Incorrect':<10} {'Accuracy':<10} {'Confidence':<11}")
        print("-" * 100)
        
        for class_name, details in sorted_classes:
            print(f"{class_name:<25} {details['total_test_videos']:<12} {details['correct_predictions']:<8} "
                  f"{details['incorrect_predictions']:<10} {details['accuracy_percentage']:<10.2f}% "
                  f"{details['average_confidence']:<11.3f}")
        
        print("\n" + "=" * 100)
        print("WORST PERFORMING CLASSES (Bottom 10)")
        print("=" * 100)
        
        worst_classes = sorted_classes[:10]
        for i, (class_name, details) in enumerate(worst_classes, 1):
            print(f"\n{i}. {class_name}")
            print(f"   Total Test Videos: {details['total_test_videos']}")
            print(f"   Correct Predictions: {details['correct_predictions']}")
            print(f"   Incorrect Predictions: {details['incorrect_predictions']}")
            print(f"   Accuracy: {details['accuracy_percentage']:.2f}%")
            print(f"   Average Confidence: {details['average_confidence']:.3f}")
            
            if details['misclassified_as']:
                print(f"   Most commonly misclassified as:")
                for misclass, count in list(details['misclassified_as'].items())[:3]:
                    print(f"     - {misclass}: {count} times")
        
        print("\n" + "=" * 100)
        print("BEST PERFORMING CLASSES (Top 10)")
        print("=" * 100)
        
        best_classes = sorted_classes[-10:]
        best_classes.reverse()  # Show best first
        for i, (class_name, details) in enumerate(best_classes, 1):
            print(f"\n{i}. {class_name}")
            print(f"   Total Test Videos: {details['total_test_videos']}")
            print(f"   Correct Predictions: {details['correct_predictions']}")
            print(f"   Accuracy: {details['accuracy_percentage']:.2f}%")
            print(f"   Average Confidence: {details['average_confidence']:.3f}")
    
    def save_detailed_class_report_to_file(self):
        """Save detailed per-class report to JSON and CSV files"""
        # Convert data to JSON serializable format
        json_serializable_data = convert_to_json_serializable(self.per_class_details)
        
        # Save as JSON
        json_path = os.path.join(self.save_dir, 'detailed_per_class_report.json')
        with open(json_path, 'w') as f:
            json.dump(json_serializable_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_path = os.path.join(self.save_dir, 'detailed_per_class_report.csv')
        with open(csv_path, 'w') as f:
            f.write("Class Name,Total Test Videos,Correct Predictions,Incorrect Predictions,Accuracy (%),Average Confidence\n")
            
            # Sort by accuracy
            sorted_classes = sorted(
                self.per_class_details.items(),
                key=lambda x: x[1]['accuracy']
            )
            
            for class_name, details in sorted_classes:
                f.write(f"{class_name},{details['total_test_videos']},{details['correct_predictions']},"
                       f"{details['incorrect_predictions']},{details['accuracy_percentage']:.2f},"
                       f"{details['average_confidence']:.3f}\n")
        
        print(f"Detailed reports saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - CSV: {csv_path}")
    
    def plot_per_class_performance(self, save_path=None):
        """Plot detailed per-class performance visualization"""
        # Sort classes by accuracy
        sorted_classes = sorted(
            self.per_class_details.items(),
            key=lambda x: x[1]['accuracy']
        )
        
        class_names = [item[0] for item in sorted_classes]
        total_videos = [item[1]['total_test_videos'] for item in sorted_classes]
        correct_preds = [item[1]['correct_predictions'] for item in sorted_classes]
        accuracies = [item[1]['accuracy_percentage'] for item in sorted_classes]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Total videos per class
        bars1 = ax1.bar(range(len(class_names)), total_videos, color='skyblue', alpha=0.7)
        ax1.set_title('Total Test Videos per Class', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Test Videos')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars1, total_videos):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom')
        
        # Plot 2: Correct vs Total predictions
        x_pos = np.arange(len(class_names))
        bars2 = ax2.bar(x_pos, total_videos, color='lightcoral', alpha=0.7, label='Total Videos')
        bars3 = ax2.bar(x_pos, correct_preds, color='lightgreen', alpha=0.8, label='Correct Predictions')
        
        ax2.set_title('Correct Predictions vs Total Videos per Class', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Number of Videos')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.legend()
        
        # Plot 3: Accuracy percentage
        colors = ['red' if acc < 50 else 'orange' if acc < 75 else 'green' for acc in accuracies]
        bars4 = ax3.bar(range(len(class_names)), accuracies, color=colors, alpha=0.7)
        ax3.set_title('Accuracy Percentage per Class', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_xticks(range(len(class_names)))
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax3.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75% threshold')
        ax3.legend()
        
        # Add value labels on accuracy bars
        for bar, acc in zip(bars4, accuracies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Performance summary table (top 5 worst and best)
        ax4.axis('tight')
        ax4.axis('off')
        
        # Prepare table data
        worst_5 = sorted_classes[:5]
        best_5 = sorted_classes[-5:]
        best_5.reverse()
        
        table_data = []
        table_data.append(['WORST PERFORMING CLASSES', '', '', ''])
        table_data.append(['Class Name', 'Total', 'Correct', 'Accuracy'])
        for class_name, details in worst_5:
            table_data.append([
                class_name[:20],  # Truncate long names
                str(details['total_test_videos']),
                str(details['correct_predictions']),
                f"{details['accuracy_percentage']:.1f}%"
            ])
        
        table_data.append(['', '', '', ''])  # Empty row
        table_data.append(['BEST PERFORMING CLASSES', '', '', ''])
        table_data.append(['Class Name', 'Total', 'Correct', 'Accuracy'])
        for class_name, details in best_5:
            table_data.append([
                class_name[:20],  # Truncate long names
                str(details['total_test_videos']),
                str(details['correct_predictions']),
                f"{details['accuracy_percentage']:.1f}%"
            ])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'detailed_per_class_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_top_k_accuracy(self, k=3):
        """Calculate top-k accuracy"""
        return top_k_accuracy_score(self.all_targets, self.all_probabilities, k=k)
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        report = classification_report(
            self.all_targets, 
            self.all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        return report
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def analyze_misclassifications(self, top_n=10):
        """Analyze most common misclassifications"""
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        
        # Find misclassifications (off-diagonal elements)
        misclassifications = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),  # Convert to int
                        'percentage': float((cm[i, j] / cm[i].sum()) * 100)  # Convert to float
                    })
        
        # Sort by count
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        
        print(f"\nTop {top_n} Misclassifications:")
        print("-" * 80)
        for i, misc in enumerate(misclassifications[:top_n]):
            print(f"{i+1:2d}. {misc['true_class']} -> {misc['predicted_class']}: "
                  f"{misc['count']} times ({misc['percentage']:.1f}%)")
        
        return misclassifications
    
    def generate_report(self):
        """Generate comprehensive evaluation report with enhanced per-class analysis"""
        print("=" * 100)
        print("ENHANCED ISL MODEL EVALUATION REPORT")
        print("=" * 100)
        
        # Basic metrics
        accuracy, avg_loss = self.evaluate()
        top3_accuracy = self.calculate_top_k_accuracy(k=3)
        top5_accuracy = self.calculate_top_k_accuracy(k=5)
        
        print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
        print(f"Total Test Samples: {len(self.all_targets)}")
        print(f"Number of Classes: {len(self.class_names)}")
        
        # Print detailed per-class analysis
        self.print_detailed_per_class_report()
        
        # Save detailed reports
        self.save_detailed_class_report_to_file()
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        self.plot_per_class_performance()
        cm = self.plot_confusion_matrix()
        
        # Analyze misclassifications
        misclassifications = self.analyze_misclassifications()
        
        # Classification report
        class_report = self.generate_classification_report()
        
        print("\nOVERALL STATISTICAL SUMMARY:")
        print("-" * 50)
        macro_avg = class_report['macro avg']
        weighted_avg = class_report['weighted avg']
        print(f"Macro Avg    : Precision={macro_avg['precision']:.3f}, "
              f"Recall={macro_avg['recall']:.3f}, F1={macro_avg['f1-score']:.3f}")
        print(f"Weighted Avg : Precision={weighted_avg['precision']:.3f}, "
              f"Recall={weighted_avg['recall']:.3f}, F1={weighted_avg['f1-score']:.3f}")
        
        # Calculate class distribution statistics
        total_videos_per_class = [details['total_test_videos'] for details in self.per_class_details.values()]
        correct_predictions_per_class = [details['correct_predictions'] for details in self.per_class_details.values()]
        
        print(f"\nDATASET STATISTICS:")
        print("-" * 50)
        print(f"Average videos per class: {np.mean(total_videos_per_class):.1f}")
        print(f"Min videos per class: {np.min(total_videos_per_class)}")
        print(f"Max videos per class: {np.max(total_videos_per_class)}")
        print(f"Total correct predictions: {np.sum(correct_predictions_per_class)}")
        print(f"Total incorrect predictions: {len(self.all_targets) - np.sum(correct_predictions_per_class)}")
        
        # Save comprehensive report with JSON serializable data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'test_accuracy': float(accuracy),
                'test_loss': float(avg_loss),
                'top3_accuracy': float(top3_accuracy),
                'top5_accuracy': float(top5_accuracy),
                'total_test_samples': int(len(self.all_targets)),
                'number_of_classes': int(len(self.class_names))
            },
            'per_class_detailed_analysis': convert_to_json_serializable(self.per_class_details),
            'classification_report': convert_to_json_serializable(class_report),
            'confusion_matrix': convert_to_json_serializable(cm.tolist()),
            'top_misclassifications': convert_to_json_serializable(misclassifications[:20]),
            'dataset_statistics': {
                'average_videos_per_class': float(np.mean(total_videos_per_class)),
                'min_videos_per_class': int(np.min(total_videos_per_class)),
                'max_videos_per_class': int(np.max(total_videos_per_class)),
                'total_correct_predictions': int(np.sum(correct_predictions_per_class)),
                'total_incorrect_predictions': int(len(self.all_targets) - np.sum(correct_predictions_per_class))
            }
        }
        
        report_path = os.path.join(self.save_dir, 'comprehensive_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nComprehensive report saved to: {report_path}")
        print("=" * 100)
        
        return report_data