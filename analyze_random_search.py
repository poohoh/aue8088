#!/usr/bin/env python3
"""
Random Search Results Analyzer
분석 스크립트 for YOLOv5 하이퍼파라미터 랜덤 서치 결과
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def analyze_results(results_csv_path):
    """랜덤 서치 결과를 분석하고 시각화합니다."""
    
    if not os.path.exists(results_csv_path):
        print(f"Results file not found: {results_csv_path}")
        return
    
    # CSV 파일 읽기
    df = pd.read_csv(results_csv_path)
    print(f"Total trials: {len(df)}")
    
    # 유효한 결과만 필터링
    df_valid = df[df['mAP50-95'] != 'N/A'].copy()
    df_valid['mAP50-95'] = pd.to_numeric(df_valid['mAP50-95'])
    df_valid['mAP50'] = pd.to_numeric(df_valid['mAP50'])
    
    print(f"Valid trials: {len(df_valid)}")
    
    if df_valid.empty:
        print("No valid results found.")
        return
    
    # 기본 통계
    print("\n=== Performance Statistics ===")
    print(f"mAP@0.5:0.95 - Mean: {df_valid['mAP50-95'].mean():.4f}, Std: {df_valid['mAP50-95'].std():.4f}")
    print(f"mAP@0.5:0.95 - Min: {df_valid['mAP50-95'].min():.4f}, Max: {df_valid['mAP50-95'].max():.4f}")
    print(f"mAP@0.5 - Mean: {df_valid['mAP50'].mean():.4f}, Std: {df_valid['mAP50'].std():.4f}")
    print(f"mAP@0.5 - Min: {df_valid['mAP50'].min():.4f}, Max: {df_valid['mAP50'].max():.4f}")
    
    # 최고 성능 trial
    best_trial = df_valid.loc[df_valid['mAP50-95'].idxmax()]
    print(f"\n=== Best Trial (Trial {best_trial['trial']}) ===")
    print(f"mAP@0.5:0.95: {best_trial['mAP50-95']:.4f}")
    print(f"mAP@0.5: {best_trial['mAP50']:.4f}")
    
    # 하이퍼파라미터 분석
    hyperparams = ['lr0', 'lrf', 'momentum', 'weight_decay', 'box', 'cls', 'obj', 
                   'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fliplr', 
                   'mosaic', 'mixup', 'copy_paste']
    
    print(f"\nBest Hyperparameters:")
    for param in hyperparams:
        if param in best_trial:
            print(f"  {param}: {best_trial[param]}")
    
    # 상관관계 분석
    print(f"\n=== Correlation Analysis (with mAP@0.5:0.95) ===")
    correlations = []
    for param in hyperparams:
        if param in df_valid.columns:
            corr = df_valid[param].corr(df_valid['mAP50-95'])
            correlations.append((param, corr))
            print(f"{param}: {corr:.4f}")
    
    # 결과 디렉토리 생성
    results_dir = Path(results_csv_path).parent
    plots_dir = results_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 시각화 생성
    create_visualizations(df_valid, hyperparams, plots_dir)
    
    # Top 5 trials
    print(f"\n=== Top 5 Trials ===")
    top_5 = df_valid.nlargest(5, 'mAP50-95')[['trial', 'mAP50-95', 'mAP50', 'lr0', 'mixup', 'mosaic']]
    print(top_5.to_string(index=False))
    
    return df_valid, best_trial

def create_visualizations(df, hyperparams, plots_dir):
    """결과를 시각화합니다."""
    
    # 1. Performance distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['mAP50-95'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('mAP@0.5:0.95')
    plt.ylabel('Frequency')
    plt.title('Distribution of mAP@0.5:0.95')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(df['mAP50'], bins=20, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('mAP@0.5')
    plt.ylabel('Frequency')
    plt.title('Distribution of mAP@0.5')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Hyperparameter correlation heatmap
    corr_data = df[hyperparams + ['mAP50-95']].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Hyperparameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots for key hyperparameters
    key_params = ['lr0', 'mixup', 'mosaic', 'hsv_s', 'weight_decay']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(key_params):
        if i < len(axes) and param in df.columns:
            axes[i].scatter(df[param], df['mAP50-95'], alpha=0.6)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('mAP@0.5:0.95')
            axes[i].set_title(f'{param} vs mAP@0.5:0.95')
            axes[i].grid(True, alpha=0.3)
            
            # 추세선 추가
            z = np.polyfit(df[param], df['mAP50-95'], 1)
            p = np.poly1d(z)
            axes[i].plot(df[param], p(df[param]), "r--", alpha=0.8)
    
    # 빈 subplot 제거
    for i in range(len(key_params), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hyperparameter_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Trial progression
    plt.figure(figsize=(12, 6))
    plt.plot(df['trial'], df['mAP50-95'], 'o-', alpha=0.7, markersize=4)
    plt.xlabel('Trial Number')
    plt.ylabel('mAP@0.5:0.95')
    plt.title('Performance Across Trials')
    plt.grid(True, alpha=0.3)
    
    # 최고 성능 trial 강조
    best_idx = df['mAP50-95'].idxmax()
    plt.scatter(df.loc[best_idx, 'trial'], df.loc[best_idx, 'mAP50-95'], 
                color='red', s=100, zorder=5, label=f'Best (Trial {df.loc[best_idx, "trial"]})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'trial_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved in: {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze random search results')
    parser.add_argument('--results', type=str, required=True, 
                       help='Path to random search results CSV file')
    
    args = parser.parse_args()
    
    analyze_results(args.results)

if __name__ == "__main__":
    main()
