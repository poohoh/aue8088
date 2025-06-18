import json

def correct_yolo_prediction_ids(val_annotation_path, yolo_prediction_path, output_path):
    """
    YOLOv5가 생성한 예측 파일의 image_id를
    실제 Ground Truth ID와 일치하도록 수정합니다.

    :param val_annotation_path: 'val_annotations.json' 파일 경로
    :param yolo_prediction_path: YOLOv5 val.py가 생성한 예측 파일 경로 (e.g., 'predictions.json')
    :param output_path: 수정된 예측을 저장할 파일 경로
    """
    print("="*50)
    print("YOLO 예측 파일의 image_id를 수정합니다.")
    print(f"기준 Annotation 파일: {val_annotation_path}")
    print(f"원본 예측 파일: {yolo_prediction_path}")

    # 1. val_annotations.json을 로드하여 '이미지 이름 -> 실제 ID' 매핑을 생성합니다.
    print("\n1. 이미지 이름 -> 실제 ID 매핑 생성 중...")
    with open(val_annotation_path, 'r') as f:
        val_data = json.load(f)
    
    name_to_id_mapping = {image['im_name']: image['id'] for image in val_data['images']}
    print(f"  -> {len(name_to_id_mapping)}개의 이미지 매핑을 생성했습니다.")

    # 2. YOLOv5가 생성한 예측 파일을 로드합니다.
    print("\n2. 원본 예측 파일 로드 중...")
    with open(yolo_prediction_path, 'r') as f:
        yolo_predictions = json.load(f)
    print(f"  -> {len(yolo_predictions)}개의 탐지 결과를 로드했습니다.")

    # 3. 각 예측 결과의 image_id를 올바른 ID로 교체합니다.
    print("\n3. image_id를 올바른 값으로 교체하는 중...")
    corrected_count = 0
    for det in yolo_predictions:
        image_name = det.get('image_name') # YOLOv5 결과에는 'image_name'이 포함되어 있음
        if image_name:
            # 확장자를 추가해서 매칭 시도
            image_name_with_ext = image_name + '.jpg'
            if image_name_with_ext in name_to_id_mapping:
                correct_id = name_to_id_mapping[image_name_with_ext]
                det['image_id'] = correct_id
                
                # COCO 포맷은 image_name 필드가 필수가 아니므로 삭제 (선택 사항)
                del det['image_name']
                corrected_count += 1
            elif image_name in name_to_id_mapping:
                # 원본 이름으로도 시도
                correct_id = name_to_id_mapping[image_name]
                det['image_id'] = correct_id
                del det['image_name']
                corrected_count += 1

    print(f"  -> {corrected_count}개의 탐지 결과에 대해 ID를 성공적으로 수정했습니다.")

    # 4. 수정된 예측 결과를 새로운 JSON 파일로 저장합니다.
    print(f"\n4. 수정된 결과를 '{output_path}' 파일로 저장 중...")
    with open(output_path, 'w') as f:
        json.dump(yolo_predictions, f)

    print("\n완료! 이제 수정된 예측 파일을 평가에 사용할 수 있습니다.")
    print("="*50)


# --- 현재 환경에 맞게 설정된 파일 경로 ---
if __name__ == "__main__":
    # 1. 분할된 검증용 어노테이션 파일 경로
    VAL_ANNOTATION_FILE = 'MR_plot/MR_plot/val_annotations.json'
    
    # 2. YOLOv5의 prediction.py 실행 후 생성된 원본 예측 파일 경로
    YOLO_PREDICTION_FILE = 'MR_plot/MR_plot2/epochNone_predictions.json'
    
    # 3. ID가 수정된 예측 결과를 저장할 파일 경로
    CORRECTED_OUTPUT_FILE = 'MR_plot/MR_plot/corrected_predictions.json'

    correct_yolo_prediction_ids(VAL_ANNOTATION_FILE, YOLO_PREDICTION_FILE, CORRECTED_OUTPUT_FILE)
