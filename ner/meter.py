__all__ = [
    'AverageMeter',
    'ClassificationMeter',
]


class AverageMeter(object):
    def __init__(self) -> None:
        super(AverageMeter, self).__init__()
        self.clear()

    def clear(self) -> None:
        self.value = 0.
        self.weight = 0.

    def update(self, value: float, weight: float = 1.) -> None:
        self.value += value
        self.weight += weight

    @property
    def average(self) -> float:
        return self.value / max(1., self.weight)


class ClassificationMeter(object):
    def __init__(self) -> None:
        super(ClassificationMeter, self).__init__()
        self.clear()

    def clear(self) -> None:
        self.value = 0.
        self.target_weight = 0.
        self.prediction_weight = 0.

    def update(self, value: float, target_weight: float, prediction_weight: float) -> None:
        self.value += value
        self.target_weight += target_weight
        self.prediction_weight += prediction_weight

    @property
    def precision(self) -> float:
        return 100 * self.value / max(1., self.prediction_weight)

    @property
    def recall(self) -> float:
        return 100 * self.value / max(1., self.target_weight)

    @property
    def f1(self) -> float:
        return 200 * self.value / max(1., self.prediction_weight + self.target_weight)
