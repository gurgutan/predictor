from predictor import Predictor
import numpy as np

p = Predictor(
    "models/37",
    input_shape=(4, 4, 4, 1),
    output_shape=(8,),
    predict_size=16,
    filters=256,
    kernel_size=2,
    dense_size=64,
)
x0, y0 = p.load_dataset(
    tsv_file="datas/EURUSD_M5_200001030000_202006122350.csv", count=151320, skip=4608
)
x1, y1 = p.load_dataset2(
    tsv_file="datas/EURUSD_M5_200001030000_202006122350.csv", count=151320, skip=4608
)

dx = np.sum(x0 - x1)
dy = np.sum(y0 - y1)
print(dx, dy)

