import cv2
from matplotlib import pyplot as plt

output_mask = "./inference_results/sample_mask.png"

# 결과 이미지 불러오기
mask = cv2.imread(output_mask)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# 시각화
plt.imshow(mask_rgb)
plt.title("Predicted Lane Mask")
plt.axis("off")
plt.show()
