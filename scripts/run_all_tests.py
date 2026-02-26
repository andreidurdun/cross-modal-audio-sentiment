"""
Master script că rulează testarea pentru toate modelele.
Apelează fiecare script de testare și generează un raport comparat.
"""
from pathlib import Path
import json
import sys
from datetime import datetime
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_all_tests():
    """Rulează testarea pentru toate modelele."""
    
    print("\n" + "="*80)
    print("RUNNING ALL MODEL TESTS")
    print("="*80 + "\n")
    
    test_scripts = [
        ("RoBERTa Text EN", "scripts/test_roberta_text_en.py"),
        ("RoBERTa Text ES", "scripts/test_roberta_text_es.py"),
        ("WavLM Audio", "scripts/test_wavlm_audio.py"),
        ("CCMT Multimodal", "scripts/test_ccmt_multimodal.py"),
    ]
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    failed_tests = []
    
    for model_name, script_path in test_scripts:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"Script: {script_path}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                failed_tests.append(model_name)
                print(f"\n❌ {model_name} test failed with return code {result.returncode}")
            else:
                print(f"\n✅ {model_name} test completed successfully")
                
                # Load results
                results_json = results_dir / model_name.lower().replace(" ", "_") / "test_results.json"
                if results_json.exists():
                    with open(results_json) as f:
                        all_results[model_name] = json.load(f)
        
        except subprocess.TimeoutExpired:
            failed_tests.append(model_name)
            print(f"\n❌ {model_name} test timed out")
        except Exception as e:
            failed_tests.append(model_name)
            print(f"\n❌ {model_name} test failed: {str(e)}")
    
    # Generate comparison report
    if all_results:
        print("\n" + "="*80)
        print("COMPARISON REPORT")
        print("="*80 + "\n")
        
        comparison_path = results_dir / "comparison_report.json"
        generate_comparison_report(all_results, comparison_path)
        
        # Generate comparison plots
        generate_comparison_plots(all_results, results_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("TESTING SUMMARY")
    print("="*80)
    print(f"✅ Completed tests: {len(all_results)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    if failed_tests:
        print(f"   Failed models: {', '.join(failed_tests)}")
    print(f"\nResults saved in: {results_dir}")
    print("="*80 + "\n")


def generate_comparison_report(all_results: dict, output_path: Path):
    """Generează un raport comparat cu metrici ale tuturor modelelor."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    # Extract main metrics for each model
    for model_name, results in all_results.items():
        report["models"][model_name] = {
            "accuracy": results.get('accuracy', 0),
            "f1_macro": results.get('f1_macro', 0),
            "f1_weighted": results.get('f1_weighted', 0),
            "precision_macro": results.get('precision_macro', 0),
            "recall_macro": results.get('recall_macro', 0),
            "num_samples": results.get('num_samples', 0),
            "f1_per_class": results.get('f1_per_class', {}),
        }
    
    # Add rankings
    report["rankings"] = {
        "accuracy": sorted(
            report["models"].items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        ),
        "f1_macro": sorted(
            report["models"].items(),
            key=lambda x: x[1]["f1_macro"],
            reverse=True
        ),
        "f1_weighted": sorted(
            report["models"].items(),
            key=lambda x: x[1]["f1_weighted"],
            reverse=True
        ),
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Comparison report saved: {output_path}\n")
    
    # Print summary table
    print("Model Performance Summary:")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Macro':<12} {'F1 Weighted':<12} {'Samples':<10}")
    print("-" * 80)
    
    for model_name, results in report["models"].items():
        print(f"{model_name:<25} {results['accuracy']:<12.4f} {results['f1_macro']:<12.4f} {results['f1_weighted']:<12.4f} {results['num_samples']:<10}")
    
    print("-" * 80 + "\n")
    
    # Print rankings
    print("Rankings by F1 Macro:")
    for i, (model_name, _) in enumerate(report["rankings"]["f1_macro"], 1):
        f1 = report["models"][model_name]["f1_macro"]
        print(f"  {i}. {model_name}: {f1:.4f}")
    print()


def generate_comparison_plots(all_results: dict, results_dir: Path):
    """Generează grafice comparative pentru modelele testate."""
    
    model_names = list(all_results.keys())
    
    # Prepare data
    accuracy = [all_results[m].get('accuracy', 0) for m in model_names]
    f1_macro = [all_results[m].get('f1_macro', 0) for m in model_names]
    f1_weighted = [all_results[m].get('f1_weighted', 0) for m in model_names]
    precision = [all_results[m].get('precision_macro', 0) for m in model_names]
    recall = [all_results[m].get('recall_macro', 0) for m in model_names]
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy comparison
    axes[0, 0].bar(model_names, accuracy, color='steelblue')
    axes[0, 0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracy):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # F1 scores comparison
    x = np.arange(len(model_names))
    width = 0.25
    axes[0, 1].bar(x - width, f1_macro, width, label='F1 Macro', color='coral')
    axes[0, 1].bar(x, f1_weighted, width, label='F1 Weighted', color='lightgreen')
    axes[0, 1].bar(x + width, recall, width, label='Recall', color='skyblue')
    axes[0, 1].set_title('F1 Scores Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=45)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    
    # Precision vs Recall
    axes[1, 0].scatter(precision, recall, s=200, alpha=0.7)
    for i, model_name in enumerate(model_names):
        axes[1, 0].annotate(model_name, (precision[i], recall[i]), 
                           fontsize=9, ha='right')
    axes[1, 0].set_xlabel('Precision Macro')
    axes[1, 0].set_ylabel('Recall Macro')
    axes[1, 0].set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall metrics
    metrics_names = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']
    x_pos = np.arange(len(metrics_names))
    
    for i, model_name in enumerate(model_names):
        values = [
            all_results[model_name].get('accuracy', 0),
            all_results[model_name].get('f1_macro', 0),
            all_results[model_name].get('f1_weighted', 0),
            all_results[model_name].get('precision_macro', 0),
            all_results[model_name].get('recall_macro', 0),
        ]
        axes[1, 1].plot(metrics_names, values, marker='o', label=model_name, linewidth=2)
    
    axes[1, 1].set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    comparison_plot_path = results_dir / "comparison_plots.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plots saved: {comparison_plot_path}\n")
    
    # F1 per class comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    class_names = ["unsatisfied", "neutral", "satisfied"]
    
    for class_idx, class_name in enumerate(class_names):
        f1_values = []
        for model_name in model_names:
            f1_per_class = all_results[model_name].get('f1_per_class', {})
            f1_values.append(f1_per_class.get(class_name, 0))
        
        axes[class_idx].bar(model_names, f1_values, color='mediumpurple')
        axes[class_idx].set_title(f'F1 Score - {class_name.capitalize()}', fontsize=12, fontweight='bold')
        axes[class_idx].set_ylabel('F1 Score')
        axes[class_idx].set_ylim([0, 1])
        axes[class_idx].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(f1_values):
            axes[class_idx].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    f1_per_class_path = results_dir / "f1_per_class_comparison.png"
    plt.savefig(f1_per_class_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ F1 per class comparison saved: {f1_per_class_path}\n")


if __name__ == "__main__":
    run_all_tests()
