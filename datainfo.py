import io
import json
import functools
import operator


class DatasetInfo(object):
    def __init__(
        self,
        input_shape=(16, 16, 1),
        output_shape=(8,),
        future=16,
        x_std=0.00047,
        y_std=0.33,
        timeunit=300,
    ):
        """input_shape - формат входа
        output_shape - формат выхода
        std - стандартное отклонение цены
        time_unit - размер в секндах одного интервала"""
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.future = future
        self.x_std = x_std
        self.y_std = y_std
        self.timeunit = timeunit

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJson(self, jsonstr):
        copy = json.loads(jsonstr)
        self.input_shape = tuple(copy["input_shape"])
        self.output_shape = tuple(copy["output_shape"])
        self.future = int(copy["future"])
        self.x_std = float(copy["x_std"])
        self.y_std = float(copy["y_std"])
        self.timeunit = int(copy["timeunit"])
        return self

    def load(self, filename):
        with io.open(filename) as file:
            self.fromJson(file.read())
        return self

    def save(self, filename):
        with io.open(filename, "w") as file:
            file.write(self.toJson())

    def _in_size(self):
        return functools.reduce(operator.mul, self.input_shape)

    def _out_size(self):
        return functools.reduce(operator.mul, self.output_shape)

    def _y_min(self):
        return -self.y_std * 4

    def _y_max(self):
        return self.y_std * 4

    def _x_inf(self):
        return self.x_std * 6
