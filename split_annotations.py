import json
import os

def split_annotations(main_annotation_file, train_list_file, val_list_file, output_dir):
    """
    주어진 train/val 목록을 기반으로 메인 annotation 파일을 분리합니다.

    :param main_annotation_file: 원본 COCO 포맷 JSON 파일 경로 (e.g., 'KAIST_annotation.json')
    :param train_list_file: 학습용 이미지 목록 텍스트 파일 (e.g., 'train_fold1.txt')
    :param val_list_file: 검증용 이미지 목록 텍스트 파일 (e.g., 'val_fold1.txt')
    :param output_dir: 출력 디렉토리 경로
    """
    print(f"Loading main annotation file: {main_annotation_file}")
    with open(main_annotation_file, 'r') as f:
        data = json.load(f)

    # --- 파일 목록에서 이미지 파일 이름만 추출 ---
    def get_filenames_from_list(list_file):
        filenames = set()
        with open(list_file, 'r') as f:
            for line in f:
                # os.path.basename을 사용하여 경로에서 파일 이름만 추출
                filenames.add(os.path.basename(line.strip()))
        return filenames

    print(f"Reading validation image list from: {val_list_file}")
    val_filenames = get_filenames_from_list(val_list_file)
    print(f"Found {len(val_filenames)} unique filenames in validation list.")

    # --- 원본 데이터에서 메타데이터 복사 ---
    val_data = {
        "info": data.get("info", {}),
        "info_improved": data.get("info_improved", {}),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": [],
        "annotations": []
    }

    # --- Validation 이미지와 ID 필터링 ---
    val_image_ids = set()
    print("Filtering validation images...")
    for image in data['images']:
        if image['im_name'] in val_filenames:
            val_data['images'].append(image)
            val_image_ids.add(image['id'])
    
    print(f"Found {len(val_data['images'])} matching images in the main annotation file.")

    # --- Validation 이미지에 해당하는 Annotation 필터링 ---
    print("Filtering corresponding annotations...")
    # 'annotations' 키가 없을 경우를 대비하여 .get 사용
    for annotation in data.get('annotations', []):
        if annotation['image_id'] in val_image_ids:
            val_data['annotations'].append(annotation)

    print(f"Found {len(val_data['annotations'])} annotations for the validation images.")

    # --- 출력 디렉토리 생성 ---
    os.makedirs(output_dir, exist_ok=True)

    # --- 분리된 파일을 JSON으로 저장 ---
    output_filename = os.path.join(output_dir, 'val_annotations.json')
    print(f"Saving validation annotations to: {output_filename}")
    with open(output_filename, 'w') as f:
        json.dump(val_data, f, indent=4)

    print("\nDone! You can now use 'val_annotations.json' for evaluation.")
    
    return output_filename


# --- 스크립트 실행 ---
# 실제 파일 경로들
MAIN_ANNOTATION_FILE = 'datasets/kaist-rgbt/KAIST_annotation.json'
TRAIN_LIST_FILE = 'datasets/kaist-rgbt/kfold_splits/train_fold1.txt'
VAL_LIST_FILE = 'datasets/kaist-rgbt/kfold_splits/val_fold1.txt'
OUTPUT_DIR = 'MR_plot/MR_plot'

if __name__ == "__main__":
    split_annotations(MAIN_ANNOTATION_FILE, TRAIN_LIST_FILE, VAL_LIST_FILE, OUTPUT_DIR)
