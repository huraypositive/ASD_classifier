from pathlib import Path
import numpy as np
import torch
from copy import deepcopy
import cv2
from torchvision import transforms
import os

from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
from src.model_ResNet import resnext_scratch
from src.image_prep import maybe_fix_bgr_rgb, get_mask, mask_image, remove_back_area, supplemental_black_area

prep_image_save_dir = Path('Preprocessed_Images')
prep_image_save_dir.mkdir(exist_ok = True)

def load_model(model_path) : 
    model = resnext_scratch(2)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model

def load_models(model_paths) : 
    models = []
    for model_path in model_paths : 
        model = load_model(model_path).to('cuda').eval()
        models.append(model)
    return models


def preprocess(image_path) : 
    def _load_image(image_path) : 
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _stretch_image(image) : 
        # min max scaling to [0,255]
        min_val = np.min(image)
        max_val = np.max(image)
        stretched_image = (image - min_val) / (max_val - min_val) * 255
        return stretched_image.astype(np.uint8)

    def _preprocess(image) : 
        # step1 : get mask : 
        image = maybe_fix_bgr_rgb(image) # 가끔씩 RGB가 잘못 들어오는 경우가 있어서 수정
        mask, bbox, _, _ = get_mask(image) # mask : 0,1로 이루어진 binary mask, bbox : (s_h, s_w, height, width)
        prep_image = mask_image(image, mask) # mask가 0인 부분은 검정색으로 바꿔줌
        prep_image, _ = remove_back_area(prep_image, bbox=bbox) # bbox를 이용해서 눈 영역만 남기고 나머지 부분은 제거
        prep_image, _ = supplemental_black_area(prep_image) # 눈 영역이 정사각형이 되도록 보조적으로 검정색 영역을 추가
        
        # 하단이나 상단에 환자 정보가 적혀있는 경우가 있어서, 강제적으로 crop 후 학습에 활용하였음.
        height = prep_image.shape[0] # 눈 영역의 높이
        prep_image = prep_image[int(height*0.05):int(height*0.95)]
        return prep_image

    def _image_to_tensor(image) : 
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return transform(image)
    # 중간 결과 저장
    
    image_raw = _load_image(image_path)
    image_stretch = _stretch_image(image_raw)
    prep_image = _preprocess(image_stretch)
    
    image_save_path = prep_image_save_dir / image_path.parent.parent.name / image_path.parent.name / image_path.name
    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    img_tensor = _image_to_tensor(prep_image)
    return img_tensor

def process_single_dir(models, image_dir) : 
    left_eye = list(image_dir.joinpath('OS').glob('*.png'))
    right_eye = list(image_dir.joinpath('OD').glob('*.png'))
    if len(left_eye) == 0 and len(right_eye) == 0 : 
        print(f"No images found in {image_dir}")
        return None
    
    left_eye = ['Left', left_eye[0]] if len(left_eye) > 0 else ['Left', None]
    right_eye = ['Right', right_eye[0]] if len(right_eye) > 0 else ['Right', None]
    
    asd_prob = []
    for eye, image_path in [left_eye, right_eye] : 
        if image_path is None : 
            asd_prob.append([eye, None])
            continue
        image_tensor = preprocess(image_path)
        image_tensor = image_tensor.unsqueeze(0).to('cuda')
        with torch.no_grad() :
            outputs = []
            for model in models : 
                output = model(image_tensor)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)
            avg_output = torch.mean(outputs, dim=0)
            avg_output = torch.softmax(avg_output, dim=1)
            asd_prob.append([eye, avg_output[0][1].item()])
    # final -> avg
    final_prob = np.mean([prob[1] for prob in asd_prob if prob[1] is not None])
    return asd_prob, final_prob


def main() : 
    # 1. 예측을 위한 모델 로드
    model_paths = [f'models/Normal_ASD_fold_{i}.pth' for i in range(9)]
    models = load_models(model_paths)    
    
    # 2. 예측을 수행할 폴더 확인
    image_root_dir = Path('images')
    image_dirs = [d for d in image_root_dir.glob('*') if d.is_dir()]
    image_dirs = sorted(image_dirs, key=lambda x: x.name)
    
    # 3. 각 폴더에 대해서 예측 수행
    save_log = []
    for image_dir in tqdm(image_dirs) : 
        result = process_single_dir(models, image_dir)
        if result is None :
            continue
        asd_prob, final_prob = result        
        save_log.append({
            'Patient_ID': image_dir.name,
            'Left_Eye_ASD_Prob': asd_prob[0][1],
            'Right_Eye_ASD_Prob': asd_prob[1][1],
            'Final_ASD_Prob': final_prob
        })
    
    
    # 4. 결과 저장
    save_log_path = Path('asd_prob_log.csv')
    import pandas as pd
    df = pd.DataFrame(save_log)
    df.to_csv(save_log_path, index=False)
    print(f"ASD probabilities saved to {save_log_path}")
        
            
    

if __name__ == "__main__" :
    main()