#!/usr/bin/env python3
"""
자동 prediction 실행 스크립트
weight 파일에 대해 자동으로 prediction을 수행합니다.

사용법:
    # 모든 weight 파일 처리 (기본)
    python auto_prediction.py --weights_dir train_all/x_bat32_epo300_low_mosaic_autoanc_customloss3/weights --cfg models/yolov5s_kaist-rgbt.yaml
    
    # epoch 150 이상인 파일만 처리
    python auto_prediction.py --weights_dir train_all/x_bat32_epo300_low_mosaic_autoanc_customloss3/weights --cfg models/yolov5s_kaist-rgbt.yaml --min_epoch 150
    
    # experiment_name 수동 지정
    python auto_prediction.py --weights_dir train_all/x_bat32_epo300_low_mosaic_autoanc_customloss3/weights --cfg models/yolov5s_kaist-rgbt.yaml --experiment_name custom_name

필수 옵션: --weights_dir, --cfg
선택 옵션: --min_epoch (지정하면 epoch_숫자.pt 형식에서 해당 숫자 이상인 파일만 처리)
- --min_epoch을 지정하지 않으면 모든 .pt 파일을 처리합니다.
- --min_epoch을 지정하면 epoch_숫자.pt 형식의 파일 중 해당 숫자 이상인 것만 처리합니다.
결과는 val/{experiment_name}/{epoch_name}/ 형태로 저장됩니다.
experiment_name을 지정하지 않으면 weights_dir에서 자동으로 추출합니다.
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
import time
from datetime import datetime

class AutoPredictor:
    def __init__(self, config):
        self.config = config
        self.workspace_dir = config['workspace_dir']
        self.cuda_device = config['cuda_device']
        self.data_config = config['data_config']
        self.model_config = config['model_config']
        self.batch_size = config['batch_size']
        self.img_size = config['img_size']
        self.project_name = config['project_name']
        self.experiment_name = config['experiment_name']
        self.min_epoch = config.get('min_epoch', None)
        
    def find_weight_files(self, weights_dir):
        """weights 디렉토리에서 .pt 파일을 찾습니다. min_epoch이 설정된 경우 해당 값 이상인 파일만 필터링합니다."""
        import re
        
        weights_path = Path(self.workspace_dir) / weights_dir
        
        if not weights_path.exists():
            raise ValueError(f"weights 디렉토리가 존재하지 않습니다: {weights_path}")
        
        # .pt 파일 검색
        all_pt_files = sorted(weights_path.glob("*.pt"), key=lambda x: x.name)
        
        if not all_pt_files:
            raise ValueError(f".pt 파일을 찾을 수 없습니다: {weights_path}")
        
        # min_epoch이 설정되지 않은 경우 모든 파일 반환
        if self.min_epoch is None:
            print(f"모든 .pt 파일 포함 (min_epoch 미설정)")
            for pt_file in all_pt_files:
                print(f"포함: {pt_file.name}")
            return all_pt_files
        
        # min_epoch이 설정된 경우 epoch_숫자.pt 형식에서 필터링
        filtered_files = []
        epoch_pattern = re.compile(r'epoch_(\d+)\.pt$')
        
        print(f"epoch {self.min_epoch} 이상인 파일만 필터링")
        
        for pt_file in all_pt_files:
            match = epoch_pattern.match(pt_file.name)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num >= self.min_epoch:
                    filtered_files.append(pt_file)
                    print(f"포함: {pt_file.name} (epoch {epoch_num})")
                else:
                    print(f"제외: {pt_file.name} (epoch {epoch_num} < {self.min_epoch})")
            else:
                print(f"제외: {pt_file.name} (epoch 형식이 아님)")
        
        if not filtered_files:
            raise ValueError(f"epoch {self.min_epoch} 이상인 .pt 파일을 찾을 수 없습니다: {weights_path}")
        
        return filtered_files
    
    def run_prediction(self, weight_file):
        """단일 weight 파일에 대해 prediction을 실행합니다."""
        # 결과 이름 생성
        filename = weight_file.stem  # 확장자 제거된 파일명
        result_name = f"val_result_{filename}"
        
        # 프로젝트 경로 구성 (val/experiment_name/ 형태)
        project_path = f"{self.project_name}/{self.experiment_name}"
        
        # 명령어 구성
        cmd = [
            "python", "prediction.py",
            "--data", self.data_config,
            "--cfg", self.model_config,
            "--weights", str(weight_file),
            "--batch-size", str(self.batch_size),
            "--imgsz", str(self.img_size),
            "--name", result_name,
            "--rgbt",
            "--project", project_path
        ]
        
        # 환경 변수 설정
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
        
        print(f"\n처리 중: {weight_file.name}")
        print(f"결과 이름: {result_name}")
        print("----------------------------------------")
        
        start_time = time.time()
        
        try:
            # 작업 디렉토리 변경 후 명령어 실행
            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1시간 타임아웃
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ {weight_file.name} prediction 완료 (소요시간: {elapsed_time:.1f}초)")
                return True
            else:
                print(f"✗ {weight_file.name} prediction 실패")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ {weight_file.name} prediction 타임아웃")
            return False
        except Exception as e:
            print(f"✗ {weight_file.name} prediction 오류: {e}")
            return False
    
    def run_all_predictions(self, weights_dir):
        """모든 weight 파일에 대해 prediction을 실행합니다."""
        print(f"작업 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"작업 디렉토리: {self.workspace_dir}")
        print(f"weights 디렉토리: {weights_dir}")
        print(f"실험 이름: {self.experiment_name}")
        print(f"결과 저장 경로: {self.project_name}/{self.experiment_name}/")
        print("========================================")
        
        try:
            # weight 파일들 찾기
            weight_files = self.find_weight_files(weights_dir)
            print(f"발견된 .pt 파일: {len(weight_files)}개")
            
            for i, file in enumerate(weight_files, 1):
                print(f"  {i:2d}. {file.name}")
            
            # 모든 prediction 실행
            total_files = len(weight_files)
            success_count = 0
            start_time = time.time()
            
            for i, weight_file in enumerate(weight_files, 1):
                print(f"\n[{i}/{total_files}] 진행 중...")
                
                if self.run_prediction(weight_file):
                    success_count += 1
            
            # 결과 요약
            total_time = time.time() - start_time
            print("\n========================================")
            print("작업 완료 요약:")
            print(f"  총 파일 수: {total_files}")
            print(f"  성공: {success_count}")
            print(f"  실패: {total_files - success_count}")
            print(f"  총 소요 시간: {total_time/60:.1f}분")
            print(f"  완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  결과 확인: {self.project_name}/{self.experiment_name} 디렉토리")
            
            return success_count == total_files
            
        except Exception as e:
            print(f"오류 발생: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='모든 weight 파일에 대해 자동 prediction 수행')
    parser.add_argument('--weights_dir', 
                       required=True,
                       help='weights 파일이 있는 디렉토리 경로')
    parser.add_argument('--cfg',
                        default=None,
                       help='모델 설정 파일 경로 (예: models/yolov5s_kaist-rgbt.yaml)')
    parser.add_argument('--workspace_dir',
                       default='/home/junha/workspace/AUE8088',
                       help='작업 디렉토리 경로')
    parser.add_argument('--cuda_device', 
                       default='4',
                       help='사용할 CUDA 디바이스 번호')
    parser.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help='배치 크기')
    parser.add_argument('--img_size',
                       type=int,
                       default=640,
                       help='이미지 크기')
    parser.add_argument('--project_name',
                       default='val',
                       help='결과 프로젝트 이름')
    parser.add_argument('--experiment_name',
                       default=None,
                       help='실험 이름 (val 아래 생성될 디렉토리 이름). 지정하지 않으면 weights_dir에서 자동 추출')
    parser.add_argument('--min_epoch',
                       type=int,
                       default=None,
                       help='최소 epoch 번호. 지정하면 epoch_숫자.pt 형식에서 해당 숫자 이상인 파일만 처리')
    
    args = parser.parse_args()
    
    # experiment_name이 제공되지 않은 경우, weights_dir에서 자동 추출
    if args.experiment_name is None:
        weights_path = Path(args.weights_dir)
        # weights_dir의 마지막에서 두 번째 부분을 사용
        # 예: train_all/x_bat32_epo300_low_mosaic_autoanc_customloss3/weights -> x_bat32_epo300_low_mosaic_autoanc_customloss3
        if len(weights_path.parts) >= 2:
            args.experiment_name = weights_path.parts[-2]  # 마지막에서 두 번째
        else:
            # 경로가 너무 짧은 경우 전체 경로를 사용
            args.experiment_name = weights_path.name
    
    # 설정
    config = {
        'workspace_dir': args.workspace_dir,
        'cuda_device': args.cuda_device,
        'data_config': 'data/kaist-rgbt.yaml',
        'model_config': args.cfg,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'project_name': args.project_name,
        'experiment_name': args.experiment_name,
        'min_epoch': args.min_epoch
    }
    
    # 작업 디렉토리 확인
    if not os.path.exists(config['workspace_dir']):
        print(f"오류: 작업 디렉토리가 존재하지 않습니다: {config['workspace_dir']}")
        sys.exit(1)
    
    # AutoPredictor 실행
    predictor = AutoPredictor(config)
    success = predictor.run_all_predictions(args.weights_dir)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()