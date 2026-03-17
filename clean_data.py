import json
import re

def remove_carbon_text(input_json_path, output_json_path):
    # 1. 원본 JSON 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 정규표현식 패턴 설정
    # " with an estimated..." 부터 "t C" 까지 정확히 잡아냅니다.
    # 뒤에 붙은 마침표(.)는 문장 마무리를 위해 그대로 남겨둡니다.
    pattern = re.compile(r'\s*with an estimated total carbon storage of approximately [\d\.]+\s*t\s*[Cc]')

    # 3. 데이터 순회하며 텍스트 수정
    for item in data:
        for conv in item.get('conversations', []):
            # 챗봇(Assistant)의 대답 부분만 타겟팅
            if conv.get('from') in ['gpt', 'ASSISTANT']:
                original_text = conv['value']
                
                # 패턴에 일치하는 텍스트를 빈 문자열('')로 치환
                cleaned_text = pattern.sub('', original_text)
                conv['value'] = cleaned_text

    # 4. 수정된 데이터를 새 파일로 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ 변환 완료! 총 {len(data)}개의 데이터가 처리되었습니다.")
    print(f"📂 저장 위치: {output_json_path}")

    # 변환 전/후 샘플 하나 출력해서 확인
    if data:
        sample_conv = data[0]['conversations'][1]['value']
        print("\n[변환 샘플 확인]")
        print(f"결과: {sample_conv}")

if __name__ == "__main__":
    # TODO: 본인의 실제 파일 경로로 수정해 주세요!
    train_input = "/Users/bosung/Desktop/datasets/glamm_train.json"
    train_output = "/Users/bosung/Desktop/datasets/glamm_train_species_only.json"
    
    test_input = "/Users/bosung/Desktop/datasets/glamm_test.json"
    test_output = "/Users/bosung/Desktop/datasets/test_species_only.json"

    print("--- Train 데이터 변환 시작 ---")
    remove_carbon_text(train_input, train_output)
    
    print("\n--- Test 데이터 변환 시작 ---")
    remove_carbon_text(test_input, test_output)