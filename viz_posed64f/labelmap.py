"""ScanNet++ free-form segGroups labels -> Boxer's 288 EmbodiedScan names (owl/embodiedscan_classes.csv).

Exact match -> synonym table -> singularised -> last word in vocab -> None (unmapped)."""
import os

VOCAB = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "..", "..", "boxer", "owl", "embodiedscan_classes.csv")) if l.strip()]
VOCAB_SET = set(VOCAB)
# Structure/junk labels: dropped from GT the same way export_scannetpp_3dod.py does (+ 'split', an annotation artifact).
EXCLUDE = {"wall", "ceiling", "floor", "object", "split", "shower wall", "shower floor", "structure"}

SYN = {
    "power socket": "socket", "power socker": "socket", "light switch": "switch",
    "computer tower": "computer", "ceiling light": "light", "monitor light": "light",
    "ceiling lamp": "lamp", "desk lamp": "lamp", "electric duct": "duct", "cable duct": "duct", "air duct": "duct",
    "windowframe": "window frame", "window sill": "windowsill", "door frame": "doorframe",
    "office chair": "chair", "rolling chair": "chair", "trash bin": "bin", "trash can": "bin",
    "documents": "paper", "post it": "paper", "receipt": "paper", "paper ram": "paper",
    "clothes hanger": "hanger", "storage cabinet": "cabinet", "kitchen cabinet": "cabinet", "nightstand": "cabinet",
    "whiteboard": "board", "cutting board": "board", "folding umbrella": "umbrella", "whiteboard eraser": "eraser",
    "whiteboard marker": "pen", "books": "book", "shoes": "shoe", "plant pot": "flowerpot",
    "pen holder": "holder", "pencil stande": "holder", "monitor holder": "holder", "toilet paper holder": "holder",
    "paper stapler": "stapler", "mug": "cup", "kitchen utensil": "utensil", "shower head": "shower",
    "shower valve": "shower", "shower rod": "rod", "tap": "faucet", "toilet paper roll": "toilet paper",
    "flushes": "flush", "toilet cleaner": "cleanser", "cleaner": "cleanser", "shampoo bottle": "shampoo",
    "hairdryer": "hair dryer", "smoke detector": "alarm", "smoke alarm": "alarm", "suitcase": "luggage",
    "duffle bag": "bag", "foldable closet": "wardrobe", "headset": "headphones", "bookshelf": "shelf",
    "book shelf": "shelf", "baseball cap": "cap", "laundry hamper": "hamper", "kitchen counter": "counter",
    "bluetooth speaker": "speaker", "mousepad": "pad", "instant pot": "pot", "salad spinner": "kitchenware",
    "hand vacuum": "vacuum cleaner", "photo": "picture", "paper towel": "tissue", "sweater": "clothes",
    "sculpture": "statue", "exhaust fan": "fan", "paper rack": "rack",
}


def to_vocab(label):
    l = label.strip().lower()
    if l in EXCLUDE:
        return None
    if l in VOCAB_SET:
        return l
    if l in SYN:
        return SYN[l]
    if l.endswith("s") and l[:-1] in VOCAB_SET:
        return l[:-1]
    last = l.split()[-1]
    if last in VOCAB_SET and last not in EXCLUDE:
        return last
    return None


if __name__ == "__main__":
    assert to_vocab("Power Socket") == "socket" and to_vocab("office chair") == "chair"
    assert to_vocab("wall") is None and to_vocab("calculator") is None and to_vocab("books") == "book"
    print("labelmap ok", len(VOCAB))
