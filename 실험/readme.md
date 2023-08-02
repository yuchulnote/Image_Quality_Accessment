##실험 1.

segmentation data에 대한 components 제거 전 흰 픽셀 비율
(segmentation data → resize → gray scale → Clahe → threshold 190 적용)

---

##실험 2.

segmentation data에 대한 components 제거 후 흰 픽셀 비율
(segmentation data → resize → gray scale → Clahe → threshold 190적용 → Components 제거, max_size150)

---

##실험 3.

segmentation data에 대한 components 제거 전, 이미지 3분할 중 가장 비율이 높게 나오는 부분의 흰 픽셀 비율
(segmentation data → resize → gray scale → Clahe → threshold 190적용 → 상,중,하 3분할 중 가장 흰색 비율이 높은 부분)

---

##실험 4.

segmentation data에 대한 components 제거 후, 이미지 3분할 중 가장 비율이 높게 나오는 부분의 흰 픽셀 비율
(segmentation data → resize → gray scale → Clahe → threshold 190적용 → Components 제거, max_size150 → 상,중,하 3분할 중 가장 흰색 비율이 높은 부분)

---

##실험 5.

segmentation data를 원형 마스크를 씌운 후, components 제거 전 흰 픽셀 비율
(segmentation data  → gray scale → Clahe → 원형 마스크 적용(비율 0.8)→ threshold 190 적용)

----

##실험 6.

segmentation data를 원형 마스크를 씌운 후, components 제거 후 흰 픽셀 비율
(segmentation data  → gray scale → Clahe → 원형 마스크 적용(비율 0.8)→ threshold 190 적용 → component 제거(max_size=150))

실험 6-1부터는 최적의 Threshold 및 Max_size 찾아나가는 과정.
