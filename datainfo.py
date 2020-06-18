import io
import json


class DatasetInfo(object):
    def __init__(
        self,
        input_shape=(16, 16, 1),
        output_shape=(16, 1),
        predict_size=16,
        x_std=0.0004,
        y_std=0.00064,
    ):
        """input_shape - формат входа
        output_shape - формат выхода
        std - стандартное отклонение цены
        time_unit - размер в секндах одного интервала"""
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.predict_size = predict_size
        self.x_std = x_std
        self.y_std = y_std

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJson(self, jsonstr):
        copy = json.loads(jsonstr)
        self.input_shape = tuple(copy["input_shape"])
        self.output_shape = tuple(copy["output_shape"])
        self.predict_size = int(copy["predict_size"])
        self.x_std = float(copy["x_std"])
        self.y_std = float(copy["y_std"])
        return self

    def load(self, filename):
        with io.open(filename) as file:
            self.fromJson(file.read())
        return self

    def save(self, filename):
        with io.open(filename, "w") as file:
            file.write(self.toJson())
