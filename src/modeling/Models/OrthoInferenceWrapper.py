from collections import defaultdict
from dataset.constants import BDA_DAMAGE_CLASSES


# Perform fusion just by taking the majority class
def fuse_bda_tiled_inference(tiled_preds, class_labels=None):
    if class_labels is None:
        class_labels = BDA_DAMAGE_CLASSES
    fused_labels = {}

    for prediction_id, inferences in tiled_preds.items():
        label_totals = defaultdict(lambda: 0)
        total = 0
        gsd = None
        for inference in inferences:
            for label in class_labels:
                val = inference["class_preds"][label]
                label_totals[label] += val
                total += val

                if gsd is None:
                    gsd = inference["gsd"]

        # Prevent divide by zero
        if total == 0:
            print("WARNING: Inferences with no predicted values.", prediction_id, inferences)
            total = 1

        aggregated_label = max(label_totals.items(), key=lambda x: x[1])[0]
        fused_labels[prediction_id] = {
            "label": aggregated_label,
            "confidence": label_totals[aggregated_label] / total,
            "gsd": gsd,
        }
    return fused_labels
