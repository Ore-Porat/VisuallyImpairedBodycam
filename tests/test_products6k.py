"""Unit tests for training/get_products6k_data.py (no network, torch or ultralytics needed).

What is being tested here?
    get_products6k_data.py turns the Products-6K grocery photos (one product per photo, labeled
    with Greek category text) into YOLO training data for our 25 classes. The parts that can go
    wrong quietly are tested here:
      * text folding and keyword matching (Greek accents, plurals, "earliest match wins"),
      * the real keyword file training/products6k_map.yaml on real category names,
      * finding the annotation columns and linking photo files to products,
      * the automatic "plain background" box and the Grounding DINO box choice,
      * the end-to-end conversion on a tiny fake dataset (CSV + generated photos).

Words you will see:
    fold          lowercase + remove accents, so "Καφές" == "καφες".
    SKU           the store's product id; one product can have several photos.
    background    a photo's plain backdrop; the box is "everything that is not backdrop".
    YOLO line     "class cx cy w h", fractions 0..1 of the image size.
    tmp_path      a temporary folder from pytest, deleted after the test.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))

import get_products6k_data as p6  # noqa: E402

CLASSES = list(yaml.safe_load((REPO / "config" / "classes.yaml").read_text(encoding="utf-8"))["classes"])


@pytest.fixture(scope="module")
def real_rules():
    """The keyword rules from the real training/products6k_map.yaml."""
    rules, fields, overrides = p6.load_map(REPO / "training" / "products6k_map.yaml", CLASSES)
    return rules, fields


# ---- text folding and keyword rules --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Καφές φίλτρου", "καφεσ φιλτρου"), ("ΚΑΦΕΣ", "καφεσ"), ("Ρύζι σε σακκούλα", "ρυζι σε σακκουλα"),
    ("Κουβερτούρες- σοκ γλυκών", "κουβερτουρεσ σοκ γλυκων"), ("Tomato-Sauce!", "tomato sauce"),
    (None, ""), ("  ", ""),
])
def test_fold_text(raw, expected):
    """Accents, case, final sigma and punctuation do not matter after folding."""
    assert p6.fold_text(raw) == expected


def test_rule_whole_word_plural_and_prefix():
    """Plain keywords match whole words (+s/es); a trailing * matches the start of a word."""
    bean = p6.make_rule("beans", "bean")
    assert bean.pattern.search("canned beans") and bean.pattern.search("bean")
    assert not bean.pattern.search("beanie") and not bean.pattern.search("jellybean")
    pasta = p6.make_rule("pasta", "ζυμαρικ*")
    assert pasta.pattern.search(p6.fold_text("Ζυμαρικά"))
    assert not pasta.pattern.search(p6.fold_text("Φαγητά"))
    with pytest.raises(ValueError):
        p6.make_rule("x", " * ")


def test_match_earliest_then_longest():
    """The match that starts first wins; at the same start the longer keyword wins."""
    rules = [p6.make_rule("milk", "milk"), p6.make_rule("chocolate", "chocolate"),
             p6.make_rule("chocolate", "milk chocolate"), p6.make_rule("fish", "sardine"),
             p6.make_rule("oil", "olive oil")]
    assert p6.match_text("milk chocolate bar", rules).cls == "chocolate"
    assert p6.match_text("sardines in olive oil", rules).cls == "fish"
    assert p6.match_text("fresh milk", rules).cls == "milk"
    assert p6.match_text("detergent", rules) is None


def test_compile_rules_rejects_unknown_class():
    """A typo in the map file's class ids fails loudly."""
    with pytest.raises(ValueError):
        p6.compile_rules({"classes": {"cofee": ["coffee"]}}, CLASSES)


@pytest.mark.parametrize("specific,broad,expected", [
    ("Καφές φίλτρου", "ΚΑΦΕΔΕΣ", "coffee"),
    ("Ελαιόλαδα", "ΛΑΔΙ", "oil"),
    ("Ρύζι σε σακκούλα", "ΡΥΖΙΑ-ΟΣΠΡΙΑ", "rice"),
    ("Οσπρια διάφορα", "ΡΥΖΙΑ-ΟΣΠΡΙΑ", "beans"),          # specific beats broad
    ("Ρεβύθια", "ΡΥΖΙΑ-ΟΣΠΡΙΑ", "beans"),
    ("Γκοφρέτες", "ΣΟΚΟΛΑΤΕΣ", "chocolate"),
    ("Σνακ διάφορα", "SNACK", "chips"),
    ("Πατατάκια", "SNACK", "chips"),
    ("Χυμός τομάτας", "ΛΑΧΑΝΙΚΑ ΣΕ ΚΟΝΣΕΡΒΑ", "tomato_sauce"),
    ("Σάλτσες ζυμαρικών", "ΣΑΛΤΣΕΣ", "tomato_sauce"),
    ("Αραβοσιτέλαια", "ΛΑΔΙ", "oil"),
    ("Νερά φυσικά μεταλλικά", "ΝΕΡΑ", "water"),
    ("Τσάι σε φακελλάκια", "ΤΣΑΙ-ΑΦΕΨΗΜΑΤΑ", "tea"),
    ("Χυμός λεμονιού", "ΞΥΔΙ-ΧΥΜΟΣ ΛΕΜΟΝΙΟΥ", None),     # skip rule beats juice
    ("Δημητριακά μπάρες", "ΠΡΩΙΝΑ", None),                 # cereal bars are skipped
    ("Φίλτρα καφέ", "ΚΑΦΕΔΕΣ", None),                      # paper coffee filters
    ("Μπισκότα απλά", "ΜΠΙΣΚΟΤΑ", "cookies"),
    ("Υγρά καθαριστικά", "ΚΑΘΑΡΙΣΤΙΚΑ", None),
    ("Κανελόνια", "ΖΥΜΑΡΙΚΑ", "pasta"),                    # not cinnamon (spices)
])
def test_real_map_on_real_categories(real_rules, specific, broad, expected):
    """The shipped keyword file maps real Products-6K categories to the right class."""
    rules, fields = real_rules
    prod = p6.Product(1, specific=specific, broad=broad, description="ΣΑΠΟΥΝΙ ΜΕ ΜΕΛΙ")
    cls, why = p6.map_product(prod.fields(fields), rules)
    assert cls == expected, why


# ---- spreadsheet columns and file names ----------------------------------------------------

def test_find_columns_on_real_headers():
    """The candidate spreadsheet's real headers are recognized."""
    headers = ["Product ID", "Description", "Specific Category (SPC)", "Broad Category (BPC)", "Brand (BC)"]
    assert p6.find_columns(headers) == {"sku": 0, "description": 1, "specific": 2, "broad": 3, "brand": 4}


def test_find_columns_query_sheet_prefers_product_id():
    """A sheet with both an image number and a product id uses each for the right job."""
    cols = p6.find_columns(["Query ID", "Product ID", "Specific Category (SPC)"])
    assert cols["sku"] == 1 and cols["image"] == 0 and cols["specific"] == 2


@pytest.mark.parametrize("name,sku", [
    ("0000059_1.jpg", 59), ("0019661-512.jpg", 19661), ("0039461_IMG_5734_1.jpg", 39461),
    ("59-ΜΙΝΕΡΒΑ-1lt.jpg", 59), ("IMG_0001.jpg", None),
])
def test_sku_from_filename(name, sku):
    """The product id is the leading number of a candidate photo's file name."""
    assert p6.sku_from_filename(name) == sku


def test_parse_int_and_image_keys():
    """Ids written as text, floats or zero-padded strings all compare equal."""
    assert p6.parse_int("0000059") == p6.parse_int(59.0) == p6.parse_int(59) == 59
    assert p6.parse_int("abc") is None and p6.parse_int(1.5) is None and p6.parse_int(True) is None
    assert "17" in p6.image_keys("Query_017.jpg") and "query_017" in p6.image_keys("Query_017.jpg")


# ---- boxes ---------------------------------------------------------------------------------

def studio_photo(w=200, h=160, box=(50, 30, 150, 140), color=(200, 30, 30)):
    """A white 'studio' photo with one colored rectangle as the product."""
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = color
    return img


def test_background_box_finds_product():
    """On a plain background the box hugs the product (within a pixel)."""
    (x0, y0, x1, y1), score = p6.background_box(studio_photo())
    assert (x0, y0, x1, y1) == (50, 30, 150, 140)
    assert score == pytest.approx(1.0)


def test_background_box_ignores_specks_and_touching_edge():
    """A far-away speck is ignored; a product touching the bottom edge still works."""
    img = studio_photo(box=(50, 30, 150, 160))
    img[5:7, 190:192] = 0                      # 2x2 dust speck in a corner
    (x0, y0, x1, y1), _ = p6.background_box(img)
    assert (x0, y0, x1, y1) == (50, 30, 150, 160)


def test_background_box_rejects_cluttered_photo():
    """Random clutter at the edges means 'not a studio photo': no box."""
    rng = np.random.default_rng(0)
    assert p6.background_box(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)) is None


def test_pick_box_prefers_big_confident_product():
    """The held product (big, confident) beats a small confident box and a big unsure one."""
    cands = [(0, 0, 20, 20, 0.9), (10, 10, 90, 90, 0.7), (0, 0, 100, 100, 0.2), (0, 0, 2, 2, 0.99)]
    assert p6.pick_box(cands, 100, 100) == (10, 10, 90, 90, 0.7)
    assert p6.pick_box([], 100, 100) is None


def test_yolo_line_and_clipping():
    """Corner boxes become YOLO center/size lines, clipped to the image."""
    assert p6.yolo_line(3, (0.25, 0.5, 0.75, 1.0)) == "3 0.500000 0.750000 0.500000 0.500000"
    c, cx, cy, w, h = map(float, p6.yolo_line(0, (-0.2, 0.0, 0.4, 1.3)).split())
    assert (cx, w, cy, h) == (pytest.approx(0.2), pytest.approx(0.4), pytest.approx(0.5), pytest.approx(1.0))


def test_compute_box_uses_dino_only_when_needed():
    """Clean studio photos never load DINO; query photos and messy backdrops ask for it."""
    from PIL import Image

    class FakeDino:
        def detect(self, image, prompt):
            return [(10.0, 20.0, 110.0, 120.0, 0.8)]

    calls = []

    def get_dino():
        calls.append(1)
        return FakeDino()

    studio = Image.fromarray(studio_photo())
    box, method, _ = p6.compute_box(studio, "auto", get_dino, "coffee. grocery product.", "candidate")
    assert method == "background" and not calls
    assert box == pytest.approx([0.25, 30 / 160, 0.75, 140 / 160])
    box, method, score = p6.compute_box(studio, "auto", get_dino, "x.", "query")
    assert method == "dino" and calls and score == 0.8
    assert box == pytest.approx([10 / 200, 20 / 160, 110 / 200, 120 / 160])
    # No DINO available: a query photo gets no box at all.
    assert p6.compute_box(studio, "auto", lambda: None, "x.", "query")[1] == "none"


def test_dhash_finds_resized_duplicates():
    """The same photo at two sizes hashes (almost) the same; a different photo does not."""
    from PIL import Image
    a = Image.fromarray(studio_photo())
    b = a.resize((400, 320))
    c = Image.fromarray(studio_photo(box=(10, 10, 60, 150), color=(20, 20, 200)))
    h = lambda im: p6.dhash(np.asarray(im.convert("L").resize((9, 8))))  # noqa: E731
    assert p6.hamming(h(a), h(b)) <= 4
    assert p6.hamming(h(a), h(c)) > 4


def test_split_products_is_grouped_and_reproducible():
    """Every product lands in exactly one pile, the same way every time."""
    prods = {i: ("coffee" if i % 2 else "rice") for i in range(40)}
    a = p6.split_products(prods, 0.15, 0.15, seed=0)
    assert a == p6.split_products(dict(reversed(list(prods.items()))), 0.15, 0.15, seed=0)
    assert set(a) == set(prods) and set(a.values()) == {"train", "val", "test"}


# ---- end to end on a tiny fake dataset -----------------------------------------------------

def make_args(tmp_path: Path, **kw) -> Namespace:
    """Command-line options as main() would build them."""
    base = dict(out=tmp_path / "out", map=REPO / "training" / "products6k_map.yaml",
                classes_yaml=REPO / "config" / "classes.yaml", inspect=False, boxer="auto",
                no_dino=True, rebox=False, device="cpu", dino_model="x", dino_prompt="{name}.",
                box_threshold=0.3, text_threshold=0.25, background_frac=0.0, val_frac=0.2,
                test_frac=0.2, seed=0, dup_bits=4, max_side=1280, no_review=False)
    base.update(kw)
    return Namespace(**base)


def make_fake_dataset(root: Path) -> Path:
    """A mini Candidate_Set: a CSV with real-style headers and Greek categories, plus photos."""
    from PIL import Image
    cand = root / "Candidate_Set"
    imgs = cand / "Candidate_Images_Selected_500"
    imgs.mkdir(parents=True)
    rows = [["Product ID", "Description", "Specific Category (SPC)", "Broad Category (BPC)", "Brand (BC)"]]
    products = {101: ("Καφές φίλτρου", "ΚΑΦΕΔΕΣ"), 102: ("Καφές στιγμής", "ΚΑΦΕΔΕΣ"),
                103: ("Ελαιόλαδα", "ΛΑΔΙ"), 104: ("Ηλιέλαια", "ΛΑΔΙ"), 105: ("Ρύζι σε σακκούλα", "ΡΥΖΙΑ"),
                106: ("Ρύζι σε σακκούλα", "ΡΥΖΙΑ"), 107: ("Υγρά καθαριστικά", "ΚΑΘΑΡΙΣΤΙΚΑ")}
    for i, (sku, (spc, bpc)) in enumerate(products.items()):
        rows.append([sku, "X", spc, bpc, "B"])
        for view in (1, 2):
            box = (20 + 10 * i + 5 * view, 20, 120 + 5 * view, 140)
            Image.fromarray(studio_photo(box=box, color=(40 * i % 255, 90, 160))).save(imgs / f"{sku:07d}_{view}.jpg", quality=95)
    # The same photo saved again at another size (a duplicate that should be dropped).
    Image.open(imgs / "0000101_1.jpg").resize((400, 320)).save(imgs / "0000101-800.jpg", quality=95)
    with open(cand / "Candidate_Groundtruth.csv", "w", encoding="utf-8", newline="") as f:
        import csv
        csv.writer(f).writerows(rows)
    return cand


def test_convert_end_to_end(tmp_path):
    """Fake dataset in, YOLO dataset out: labels, splits by product, data.yaml, review sheets."""
    cand = make_fake_dataset(tmp_path)
    args = make_args(tmp_path)
    assert p6.convert(args, [cand]) == 0
    out = args.out
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == CLASSES and data["nc"] == len(CLASSES)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    imgs = manifest["images"]
    assert manifest["counts"]["duplicate"] == 1
    assert len(imgs) == 12                                   # 6 food products x 2 views
    assert {e["class"] for e in imgs} == {"coffee", "oil", "rice"}
    for sku in {e["sku"] for e in imgs}:                     # grouped split
        assert len({e["split"] for e in imgs if e["sku"] == sku}) == 1
    for e in imgs:
        lbl = out / "labels" / e["split"] / (Path(e["dataset_image"]).stem + ".txt")
        c, cx, cy, w, h = lbl.read_text().split()
        assert CLASSES[int(c)] == e["class"]
        assert 0 < float(w) < 1 and 0 < float(h) < 1
    assert list((out / "review").glob("sheet_*.jpg"))
    # Second run reuses boxes.json and gives the same result.
    assert p6.convert(args, [cand]) == 0
    again = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["images"]
    assert [(e["source"], e["split"], e["box"]) for e in again] == [(e["source"], e["split"], e["box"]) for e in imgs]


def test_convert_honors_boxes_json_edits(tmp_path):
    """Setting "skip": true or a manual box in boxes.json changes the next run."""
    cand = make_fake_dataset(tmp_path)
    args = make_args(tmp_path, no_review=True)
    p6.convert(args, [cand])
    cache_path = args.out / "boxes.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    keys = sorted(cache)
    cache[keys[0]]["skip"] = True
    cache[keys[1]] = {"box": [0.1, 0.1, 0.9, 0.9], "method": "manual", "score": 1.0}
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    p6.convert(make_args(tmp_path, no_review=True, rebox=True), [cand])
    imgs = {e["source"]: e for e in json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))["images"]}
    assert keys[0] not in imgs
    assert imgs[keys[1]]["box"] == [0.1, 0.1, 0.9, 0.9] and imgs[keys[1]]["method"] == "manual"


def test_background_fraction_adds_empty_label_images(tmp_path):
    """--background-frac adds unmapped products as images with empty label files."""
    cand = make_fake_dataset(tmp_path)
    args = make_args(tmp_path, background_frac=0.2, no_review=True)
    p6.convert(args, [cand])
    imgs = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))["images"]
    bg = [e for e in imgs if e["class"] is None]
    assert len(bg) == 2 and all(e["sku"] == 107 for e in bg)
    for e in bg:
        assert (args.out / "labels" / e["split"] / (Path(e["dataset_image"]).stem + ".txt")).read_text() == ""


def test_inspect_writes_nothing(tmp_path, capsys):
    """--inspect only prints the mapping table."""
    cand = make_fake_dataset(tmp_path)
    assert p6.convert(make_args(tmp_path, inspect=True), [cand]) == 0
    printed = capsys.readouterr().out
    assert "coffee" in printed and "Καφές φίλτρου" in printed and "no match" in printed
    assert not (tmp_path / "out").exists()
