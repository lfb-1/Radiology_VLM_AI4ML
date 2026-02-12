from codecarbon import EmissionsTracker
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

class GreenMetrics:
    def __init__(self):
        self.tracker = EmissionsTracker()

    def track(self):
        emissions = self.tracker.stop()
        print(f"CO2 emissions (kg): {emissions}")

    def bleu(self, reference, hypothesis):
        # reference, hypothesis: list of strings
        return sentence_bleu([ref.split() for ref in reference], hypothesis[0].split())

    def f1(self, y_true, y_pred):
        # y_true, y_pred: list of labels
        return f1_score(y_true, y_pred, average='macro')

    def clinical_accuracy(self, y_true, y_pred):
        # Placeholder for clinical accuracy (domain-specific)
        return sum([a == b for a, b in zip(y_true, y_pred)]) / len(y_true)

    def __enter__(self):
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.track()
