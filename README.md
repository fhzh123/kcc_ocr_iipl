# kcc_ocr_iipl

dataset Link = https://drive.google.com/open?id=1VOOYyeeyw625O76JeF0Eo14FX74gr6Ml

데이터 다운로드 후 result 폴더 안에 폴더이름으로 압축 풀기 하시면됩니다.

돌리는 코드

python train.py --train_data result/train_last --valid_data result/test_last --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn


CNN 모델 변경하려면 model.py랑 module/feature_extraction.py 수정후
명령어에 --FeatureExtraction [새로운 ] 수정하면 됩니다.
