import json
import os

def fix_prediction_ids(prediction_file, annotation_file, output_file):
    """
    prediction 결과를 validation annotation과 매칭시키고 ID를 수정합니다.
    """
    print(f"Loading prediction file: {prediction_file}")
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loading annotation file: {annotation_file}")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # annotation에서 이미지 이름 -> ID 매핑 생성 (확장자 제거)
    name_to_id = {}
    for img in annotations['images']:
        # .jpg 확장자 제거
        name_without_ext = img['im_name'].replace('.jpg', '')
        name_to_id[name_without_ext] = img['id']
    
    print(f"Found {len(name_to_id)} images in annotation file")
    
    # prediction 결과 필터링 및 ID 수정
    fixed_predictions = []
    matched_count = 0
    unmatched_count = 0
    
    for pred in predictions:
        image_name = pred['image_name']  # prediction의 image_name 사용
        
        if image_name in name_to_id:
            # 매칭되는 이미지인 경우 ID 수정
            new_pred = pred.copy()
            new_pred['image_id'] = name_to_id[image_name]
            fixed_predictions.append(new_pred)
            matched_count += 1
        else:
            unmatched_count += 1
    
    print(f"Successfully matched {matched_count} predictions")
    print(f"Unmatched predictions: {unmatched_count}")
    print(f"Total filtered predictions: {len(fixed_predictions)}")
    
    # 결과 저장
    print(f"Saving fixed predictions to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(fixed_predictions, f, indent=2)
    
    return fixed_predictions

if __name__ == "__main__":
    # 파일 경로 설정
    prediction_file = "MR_plot/MR_plot/epochNone_predictions.json"
    annotation_file = "MR_plot/MR_plot/val_annotations.json"
    output_file = "MR_plot/MR_plot/fixed_predictions.json"
    
    fix_prediction_ids(prediction_file, annotation_file, output_file)
