from flask import Flask, request, jsonify
from flask_cors import CORS
 
import tensorflow as tf
import numpy as np
 
from PIL import Image, ImageOps, UnidentifiedImageError
 
import cv2
import os
import math
 
 
# ============================================================
# FLASK APP
# ============================================================
 
app = Flask(__name__)
CORS(app)
 
# Reject absurdly large uploads (16 MB).
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
 
 
# ============================================================
# CONFIGURATION
# ============================================================
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.npy")
 
IMAGE_SIZE = (224, 224)
 
# Big phone photos are shrunk to this longest side before validation
# (much faster face/person detection, more consistent thresholds).
MAX_VALIDATION_SIDE = 1024
 
# Minimum AI confidence.
# If the model is unsure, we DO NOT guess a disease.
MIN_MODEL_CONFIDENCE = 65.0
 
# Set this to True while debugging validation issues.
# It adds a "debug" block to the JSON response with all
# the internal scores, so you can SEE why an image was
# accepted/rejected without reading the terminal.
DEBUG_VALIDATION = True
 
 
# ============================================================
# HUMAN FACE DETECTOR
# ============================================================
 
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
 
if FACE_CASCADE.empty():
    print("WARNING: Face detector could not be loaded!")
else:
    print("Face detector loaded successfully!")
 
 
# ============================================================
# HUMAN EYE DETECTOR
# ============================================================
 
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
 
if EYE_CASCADE.empty():
    print("WARNING: Eye detector could not be loaded!")
else:
    print("Eye detector loaded successfully!")
 
 
# ============================================================
# ADDITIONAL HUMAN DETECTORS
# ============================================================
 
PROFILE_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
 
if PROFILE_FACE_CASCADE.empty():
    print("WARNING: Profile-face detector could not be loaded!")
else:
    print("Profile-face detector loaded successfully!")
 
 
# OpenCV built-in full-person detector.
HOG_PERSON = cv2.HOGDescriptor()
HOG_PERSON.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)
 
 
# ============================================================
# LOAD AI MODEL
# ============================================================
 
print("Loading AI model...")
 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )
 
if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        f"Class names not found: {CLASS_PATH}"
    )
 
 
model = tf.keras.models.load_model(MODEL_PATH)
 
class_names = np.load(
    CLASS_PATH,
    allow_pickle=True
)
 
print("AI model loaded successfully!")
print("Number of classes:", len(class_names))
 
print("Available classes:")
print(class_names)
 
 
# ============================================================
# DISEASE MAP
# ============================================================
 
disease_map = {
 
    # Apple
    "Apple___Black_rot": "black-rot",
    "Apple___Apple_scab": "apple-scab",
    "Apple___Cedar_apple_rust": "cedar-apple-rust",
    "Apple___healthy": "healthy",
 
    # Blueberry
    "Blueberry___healthy": "healthy",
 
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": "powdery-mildew",
    "Cherry_(including_sour)___healthy": "healthy",
 
    # Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "cercospora-leaf-spot",
 
    "Corn_(maize)___Common_rust_":
        "common-rust",
 
    "Corn_(maize)___Northern_Leaf_Blight":
        "northern-leaf-blight",
 
    "Corn_(maize)___healthy":
        "healthy",
 
    # Grape
    "Grape___Black_rot":
        "black-rot",
 
    "Grape___Esca_(Black_Measles)":
        "esca-black-measles",
 
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "leaf-blight",
 
    "Grape___healthy":
        "healthy",
 
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)":
        "citrus-greening",
 
    # Peach
    "Peach___Bacterial_spot":
        "bacterial-spot",
 
    "Peach___healthy":
        "healthy",
 
    # Pepper
    "Pepper,_bell___Bacterial_spot":
        "bacterial-spot",
 
    "Pepper,_bell___healthy":
        "healthy",
 
    # Potato
    "Potato___Early_blight":
        "early-blight",
 
    "Potato___Late_blight":
        "late-blight",
 
    "Potato___healthy":
        "healthy",
 
    # Raspberry
    "Raspberry___healthy":
        "healthy",
 
    # Soybean
    "Soybean___healthy":
        "healthy",
 
    # Squash
    "Squash___Powdery_mildew":
        "powdery-mildew",
 
    # Strawberry
    "Strawberry___Leaf_scorch":
        "leaf-scorch",
 
    "Strawberry___healthy":
        "healthy",
 
    # Tomato
    "Tomato___Bacterial_spot":
        "bacterial-spot",
 
    "Tomato___Early_blight":
        "early-blight",
 
    "Tomato___Late_blight":
        "late-blight",
 
    "Tomato___Leaf_Mold":
        "leaf-mold",
 
    "Tomato___Septoria_leaf_spot":
        "septoria-leaf-spot",
 
    "Tomato___Spider_mites Two-spotted_spider_mite":
        "pest-damage",
 
    "Tomato___Target_Spot":
        "target-spot",
 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "leaf-curl-virus",
 
    "Tomato___Tomato_mosaic_virus":
        "mosaic-virus",
 
    "Tomato___healthy":
        "healthy",
}
 
 
 
# ============================================================
# DISEASE / CROP-CARE KNOWLEDGE BASE
# ============================================================
# This stays in the backend so the API can return the explanation
# together with the model prediction. The frontend can keep using
# the existing UI without losing the original model class.
 
DISEASE_INFO = {
    "healthy": {
        "name": "Healthy Leaf",
        "category": "None",
        "severity": "Low",
        "risk": "ok",
        "causes": [
            "No strong visible disease pattern detected.",
            "Good light, water and nutrient balance support healthy growth."
        ],
        "symptoms": [
            "Uniform or normal leaf colour",
            "Firm leaf tissue",
            "No strong disease-like spots, curling or widespread discoloration"
        ],
        "treatment": [
            "No disease treatment is required when the plant is healthy.",
            "Continue normal watering, sunlight and nutrition."
        ],
        "prevention": [
            "Water at soil level and avoid unnecessary leaf wetness.",
            "Maintain good airflow and suitable sunlight.",
            "Inspect leaves regularly for new spots, pests or curling."
        ]
    },
    "black-rot": {
        "name": "Black Rot",
        "category": "Fungal",
        "severity": "High",
        "risk": "danger",
        "causes": ["Botryosphaeria/related fungal infection", "Warm, wet conditions and infected plant debris"],
        "symptoms": ["Dark circular or irregular lesions", "Yellowing around lesions", "Progressive tissue death"],
        "treatment": ["Remove and destroy severely infected tissue", "Improve airflow and reduce prolonged leaf wetness", "Use an appropriate labelled fungicide where recommended"],
        "prevention": ["Remove infected debris", "Prune for airflow", "Avoid overhead watering and wet foliage for long periods"]
    },
    "apple-scab": {
        "name": "Apple Scab",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Venturia fungi", "Cool, wet spring weather and infected fallen leaves"],
        "symptoms": ["Olive-green to brown spots", "Velvety lesions", "Leaf yellowing and premature leaf drop"],
        "treatment": ["Remove infected fallen leaves", "Improve canopy airflow", "Use a labelled apple-scab fungicide when appropriate"],
        "prevention": ["Sanitize fallen leaves", "Prune for airflow", "Choose resistant varieties where available"]
    },
    "cedar-apple-rust": {
        "name": "Cedar Apple Rust",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Gymnosporangium fungi", "The pathogen requires compatible cedar/juniper and apple-family hosts"],
        "symptoms": ["Yellow-orange leaf spots", "Orange structures on the underside of leaves", "Premature leaf drop in severe cases"],
        "treatment": ["Remove badly affected leaves", "Improve airflow", "Use a labelled fungicide when disease pressure is high"],
        "prevention": ["Remove nearby alternate hosts where practical", "Use resistant cultivars", "Monitor early-season symptoms"]
    },
    "powdery-mildew": {
        "name": "Powdery Mildew",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Erysiphales fungi", "Warm conditions with humid or stagnant air"],
        "symptoms": ["White or grey powdery coating", "Leaf curling or distortion", "Reduced photosynthesis and growth"],
        "treatment": ["Remove heavily infected leaves", "Improve airflow and sunlight", "Use a labelled fungicide or suitable horticultural treatment"],
        "prevention": ["Avoid overcrowding", "Avoid excessive nitrogen", "Water the soil rather than keeping foliage wet"]
    },
    "cercospora-leaf-spot": {
        "name": "Cercospora / Gray Leaf Spot",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Cercospora fungi", "Warm, humid weather and infected crop residue"],
        "symptoms": ["Small tan, grey or brown leaf lesions", "Lesions may enlarge and merge", "Premature leaf drying in severe infection"],
        "treatment": ["Remove heavily infected debris where practical", "Improve airflow", "Use a crop-labelled fungicide according to its label"],
        "prevention": ["Rotate crops", "Manage crop residue", "Avoid prolonged leaf wetness"]
    },
    "common-rust": {
        "name": "Common Rust",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Puccinia rust fungi", "Wind-borne spores and favourable humid conditions"],
        "symptoms": ["Small reddish-brown or orange pustules", "Powdery spores on leaf surfaces", "Yellowing around lesions"],
        "treatment": ["Remove badly affected leaves where practical", "Use an appropriate labelled fungicide if needed", "Avoid unnecessary leaf wetness"],
        "prevention": ["Maintain airflow", "Use resistant varieties when available", "Monitor fields early"]
    },
    "northern-leaf-blight": {
        "name": "Northern Leaf Blight",
        "category": "Fungal",
        "severity": "High",
        "risk": "danger",
        "causes": ["Exserohilum turcicum", "Warm, humid conditions and infected crop residue"],
        "symptoms": ["Long grey-green to tan cigar-shaped lesions", "Lesions enlarge along the leaf", "Reduced photosynthetic area"],
        "treatment": ["Use a labelled fungicide when economically justified", "Manage infected residue", "Reduce prolonged leaf wetness where possible"],
        "prevention": ["Use resistant hybrids", "Rotate crops", "Manage crop residue"]
    },
    "esca-black-measles": {
        "name": "Grape Esca (Black Measles)",
        "category": "Fungal / Trunk Disease",
        "severity": "High",
        "risk": "danger",
        "causes": ["Complex of wood-decay fungi associated with grapevine trunk disease", "Infection through pruning wounds can contribute to disease development"],
        "symptoms": ["Interveinal yellowing or striped leaf patterns", "Dark leaf spots in severe cases", "Decline of shoots, fruit or whole vines"],
        "treatment": ["Remove and destroy severely affected vine parts where recommended", "Manage pruning wounds and infected wood", "Consult local viticulture guidance before chemical treatment"],
        "prevention": ["Prune during dry conditions", "Protect pruning wounds where appropriate", "Remove infected wood and sanitize tools"]
    },
    "leaf-blight": {
        "name": "Grape Leaf Blight",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Fungal leaf-blight pathogens", "Warm, humid conditions and prolonged leaf wetness"],
        "symptoms": ["Brown or reddish leaf spots", "Spots may enlarge and merge", "Premature leaf damage or drop"],
        "treatment": ["Remove severely affected tissue", "Improve canopy ventilation", "Use a grape-labelled fungicide when recommended"],
        "prevention": ["Open the canopy for airflow", "Avoid prolonged leaf wetness", "Remove infected debris"]
    },
    "citrus-greening": {
        "name": "Citrus Greening (HLB)",
        "category": "Bacterial-like / Phytoplasma-associated",
        "severity": "High",
        "risk": "danger",
        "causes": ["Candidatus Liberibacter bacteria transmitted mainly by psyllids", "Infected planting material"],
        "symptoms": ["Asymmetric blotchy leaf yellowing", "Yellow shoots", "Poor fruit development and decline"],
        "treatment": ["There is no simple curative treatment for an infected tree", "Control psyllid vectors according to local agricultural guidance", "Remove infected trees where official programs recommend it"],
        "prevention": ["Use certified disease-free planting material", "Monitor and control psyllids", "Follow local citrus quarantine guidance"]
    },
    "bacterial-spot": {
        "name": "Bacterial Leaf Spot",
        "category": "Bacterial",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Xanthomonas or related bacteria", "Splashing water, contaminated tools and wet leaf surfaces"],
        "symptoms": ["Small dark water-soaked spots", "Yellow halos", "Spots can merge or leave holes"],
        "treatment": ["Remove badly infected leaves", "Avoid working with wet plants", "Use a labelled copper-based bactericide where appropriate"],
        "prevention": ["Use disease-free seed/seedlings", "Disinfect tools", "Prefer drip irrigation over overhead watering"]
    },
    "early-blight": {
        "name": "Early Blight",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Alternaria fungi", "Warm, humid conditions and infected plant debris"],
        "symptoms": ["Brown spots with concentric rings", "Yellowing around lesions", "Older leaves commonly affected first"],
        "treatment": ["Remove affected lower leaves", "Improve airflow", "Use a labelled fungicide when appropriate"],
        "prevention": ["Rotate crops", "Mulch soil to reduce splash", "Avoid prolonged leaf wetness"]
    },
    "late-blight": {
        "name": "Late Blight",
        "category": "Fungal-like Oomycete",
        "severity": "High",
        "risk": "danger",
        "causes": ["Phytophthora infestans", "Cool, wet and humid conditions"],
        "symptoms": ["Large water-soaked dark patches", "White growth under humid conditions", "Rapid leaf collapse in severe cases"],
        "treatment": ["Remove severely infected tissue promptly", "Use an appropriate labelled fungicide", "Improve drainage and reduce leaf wetness"],
        "prevention": ["Avoid evening overhead watering", "Increase plant spacing", "Remove volunteer plants and infected debris"]
    },
    "bacterial-wilt": {
        "name": "Bacterial Wilt",
        "category": "Bacterial",
        "severity": "High",
        "risk": "danger",
        "causes": ["Soil-borne bacterial pathogens such as Ralstonia", "Spread through contaminated soil, water or wounds"],
        "symptoms": ["Sudden wilting despite moist soil", "Progressive plant collapse", "Milky bacterial ooze may occur in some cases"],
        "treatment": ["Remove and destroy severely affected plants", "Avoid moving contaminated soil", "Disinfect tools"],
        "prevention": ["Use resistant varieties where available", "Rotate crops", "Use clean planting material"]
    },
    "mosaic-virus": {
        "name": "Mosaic Virus",
        "category": "Viral",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Plant viruses spread by insects, sap and contaminated tools", "Infected seed or plant material"],
        "symptoms": ["Mottled light and dark green pattern", "Leaf distortion or crinkling", "Reduced growth and yield"],
        "treatment": ["Remove severely infected plants where appropriate", "Control insect vectors", "Disinfect tools between plants"],
        "prevention": ["Use certified clean seed/seedlings", "Control aphids and other vectors", "Remove infected plant debris"]
    },
    "leaf-curl-virus": {
        "name": "Leaf Curl Virus",
        "category": "Viral",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Begomoviruses and related viruses", "Often transmitted by whiteflies"],
        "symptoms": ["Upward or downward leaf curling", "Leaf thickening or distortion", "Stunted growth and reduced flowering"],
        "treatment": ["Remove severely infected plants where recommended", "Control whiteflies", "There is no direct chemical cure for the virus"],
        "prevention": ["Use healthy planting material", "Monitor whiteflies", "Use insect exclusion where practical"]
    },
    "leaf-scorch": {
        "name": "Leaf Scorch",
        "category": "Physiological / Environmental",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Heat, drought, salt stress or root problems", "Excessive transpiration or poor water uptake"],
        "symptoms": ["Brown crispy leaf margins", "Dry patches or tip burn", "Wilting during heat or water stress"],
        "treatment": ["Correct watering and root-zone conditions", "Reduce severe environmental stress", "Remove badly damaged leaves if needed"],
        "prevention": ["Maintain consistent soil moisture", "Avoid excessive fertilizer salts", "Protect plants from extreme heat where possible"]
    },
    "leaf-mold": {
        "name": "Leaf Mold",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Cladosporium fungi", "High humidity and poor ventilation, especially under protected cultivation"],
        "symptoms": ["Yellow patches on upper leaf surface", "Olive-green to brown mold on undersides", "Premature leaf decline"],
        "treatment": ["Remove affected leaves", "Improve ventilation and reduce humidity", "Use a labelled fungicide where appropriate"],
        "prevention": ["Increase airflow", "Avoid wetting foliage", "Reduce excessive humidity"]
    },
    "septoria-leaf-spot": {
        "name": "Septoria Leaf Spot",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Septoria fungi", "Rain splash and infected plant debris"],
        "symptoms": ["Small circular spots", "Dark margins with pale centres", "Tiny dark fruiting bodies may occur"],
        "treatment": ["Remove infected lower leaves", "Mulch soil to reduce splash", "Use a labelled fungicide when needed"],
        "prevention": ["Rotate crops", "Avoid overhead watering", "Remove old tomato debris"]
    },
    "pest-damage": {
        "name": "Pest Damage (Mites/Insects)",
        "category": "Pest",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Spider mites, aphids or chewing insects", "Hot, dry or stressed plant conditions can favour outbreaks"],
        "symptoms": ["Fine speckling or stippling", "Holes or chewed margins", "Webbing or sticky honeydew in some infestations"],
        "treatment": ["Inspect leaf undersides", "Use insecticidal soap or another crop-appropriate treatment according to its label", "Remove heavily infested leaves when practical"],
        "prevention": ["Inspect plants regularly", "Avoid drought stress", "Encourage beneficial insects where suitable"]
    },
    "target-spot": {
        "name": "Target Spot",
        "category": "Fungal",
        "severity": "Moderate",
        "risk": "warn",
        "causes": ["Corynespora and related fungal pathogens", "Warm humid conditions and prolonged leaf wetness"],
        "symptoms": ["Brown circular lesions with concentric rings", "Lesions may enlarge and merge", "Leaf yellowing and drop"],
        "treatment": ["Remove badly affected leaves", "Improve airflow", "Use a labelled fungicide where appropriate"],
        "prevention": ["Avoid overhead irrigation", "Rotate crops", "Remove infected debris"]
    },
    "healthy-generic": {
        "name": "Healthy Leaf",
        "category": "None",
        "severity": "Low",
        "risk": "ok",
        "causes": ["No strong disease pattern detected."],
        "symptoms": ["Normal leaf appearance"],
        "treatment": ["No treatment required; continue normal crop care."],
        "prevention": ["Maintain good watering, light and airflow."]
    }
}
 
def get_disease_info(frontend_id, model_class=None):
    """Return structured advice for the frontend/API response."""
    info = DISEASE_INFO.get(frontend_id)
    if info is None:
        info = DISEASE_INFO["healthy-generic"] if frontend_id == "healthy" else {
            "name": frontend_id.replace("_", " ").replace("-", " ").title(),
            "category": "Plant condition",
            "severity": "Moderate",
            "risk": "warn",
            "causes": ["The model identified this condition from the available trained classes."],
            "symptoms": ["Visible symptoms may vary with crop, growth stage and environment."],
            "treatment": ["Confirm the diagnosis with crop-specific agricultural guidance before applying treatment."],
            "prevention": ["Maintain good sanitation, airflow and regular crop inspection."]
        }
    return {
        **info,
        "model_class": model_class,
        "id": frontend_id
    }
 
# ============================================================
# IMAGE UTILITY
# ============================================================
 
def pil_to_cv(image_pil):
    """
    Convert PIL RGB image to OpenCV BGR image.
    """
 
    image_rgb = np.array(image_pil)
 
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )
 
    return image_bgr
 
 
# ============================================================
# HUMAN DETECTION
# ============================================================
 
def region_has_skin_tone(image_bgr, x, y, w, h, min_ratio=0.25):
 
    """
    Check whether a bounding box actually contains
    human-skin-colored pixels.
 
    Haar cascades frequently produce FALSE POSITIVE "faces"
    on leaf spots, watermark text, and vein patterns. A real
    human face region will contain a meaningful percentage of
    skin-tone pixels; a leaf region will not (it will be green/
    yellow/brown/red plant-tissue colors, which fall outside
    the skin-tone HSV range used here).
    """
 
    try:
 
        height, width = image_bgr.shape[:2]
 
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + w), width)
        y2 = min(int(y + h), height)
 
        if x2 <= x1 or y2 <= y1:
            return False
 
        region = image_bgr[y1:y2, x1:x2]
 
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
 
        # Broad human skin-tone range in HSV.
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
 
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
 
        lower_skin2 = np.array([165, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 170, 255], dtype=np.uint8)
 
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
 
        skin_mask = cv2.bitwise_or(mask1, mask2)
 
        skin_ratio = (
            np.count_nonzero(skin_mask) /
            float(skin_mask.size)
        )
 
        print(
            "  region skin-tone ratio:",
            round(skin_ratio, 3)
        )
 
        # Real faces are dominated by skin tone.
        # A leaf region will score very low on this.
        return skin_ratio >= min_ratio
 
    except Exception as e:
 
        print("Skin-tone check error:", e)
 
        return False
 
 
def detect_human(image_pil):
    """
    Strong human-image rejection before the plant disease model.
 
    The disease model has only plant classes, so unrelated images
    must be stopped before model.predict().
    """
    try:
        image_bgr = pil_to_cv(image_pil)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        height, width = gray.shape[:2]
 
        # ----------------------------------------------------
        # 1. FRONTAL + PROFILE FACE DETECTION
        # ----------------------------------------------------
        face_boxes = []
 
        frontal_faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=7, minSize=(60, 60)
        )
        print("Frontal faces detected:", len(frontal_faces))
        face_boxes.extend(frontal_faces)
 
        profile_faces = PROFILE_FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=7, minSize=(60, 60)
        )
        print("Profile faces detected:", len(profile_faces))
        face_boxes.extend(profile_faces)
 
        confirmed_faces = 0
        for (fx, fy, fw, fh) in face_boxes:
            if region_has_skin_tone(
                image_bgr, fx, fy, fw, fh, min_ratio=0.12
            ):
                confirmed_faces += 1
 
        print("Faces confirmed by skin-tone:", confirmed_faces)
        if confirmed_faces > 0:
            print("HUMAN FILTER: face confirmed")
            return True
 
        # ----------------------------------------------------
        # 2. EYE DETECTION + SURROUNDING FACE SKIN
        # ----------------------------------------------------
        eyes = EYE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=8, minSize=(18, 18)
        )
        print("Eyes detected:", len(eyes))
 
        confirmed_eyes = 0
        for (ex, ey, ew, eh) in eyes:
            pad_x = int(ew * 1.8)
            pad_y = int(eh * 2.2)
            if region_has_skin_tone(
                image_bgr,
                ex - pad_x, ey - pad_y,
                ew + (2 * pad_x), eh + (2 * pad_y),
                min_ratio=0.10
            ):
                confirmed_eyes += 1
 
        print("Eyes confirmed by skin-tone:", confirmed_eyes)
        if confirmed_eyes >= 1:
            print("HUMAN FILTER: eye + face-skin region confirmed")
            return True
 
        # ----------------------------------------------------
        # 3. FULL PERSON DETECTION (HOG)
        # ----------------------------------------------------
        working = image_bgr
        if max(height, width) > 1000:
            scale = 1000.0 / max(height, width)
            working = cv2.resize(image_bgr, None, fx=scale, fy=scale)
 
        rects, weights = HOG_PERSON.detectMultiScale(
            working, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
 
        strong_people = 0
        for i, rect in enumerate(rects):
            weight = float(weights[i]) if len(weights) > i else 0.0
            x, y, w, h = rect
            box_ratio = (w * h) / float(
                max(working.shape[0] * working.shape[1], 1)
            )
            print(
                "HOG person candidate:",
                "confidence=", round(weight, 3),
                "area=", round(box_ratio, 3)
            )
            if weight >= 0.30 and box_ratio >= 0.05:
                strong_people += 1
 
        print("Strong HOG person detections:", strong_people)
        if strong_people > 0:
            print("HUMAN FILTER: full person detected")
            return True
 
        return False
 
    except Exception as e:
        print("Human detection error:", e)
        return False
 
 
# ============================================================
# CIRCULAR OBJECT DETECTION
# ============================================================
 
 
 
def detect_circular_object(image_pil):
    """
    Temporarily disable circular-object rejection
    for testing.
    """
    print("Circular object rejection disabled for testing.")
    return False
 
# ============================================================
# LEAF SHAPE / STRUCTURE DETECTION
# ============================================================
 
def analyze_leaf_structure(image_pil):
 
    """
    Visual structural check.
 
    IMPORTANT:
    This does NOT depend on green color.
 
    Therefore:
        green leaf     -> can pass
        red leaf       -> can pass
        yellow leaf    -> can pass
        brown leaf     -> can pass
 
    The function looks for:
        - a reasonably large foreground object
        - non-circular shape
        - internal edge/vein texture
        - object occupying a meaningful part of image
    """
 
    try:
 
        image_bgr = pil_to_cv(image_pil)
 
        height, width = image_bgr.shape[:2]
 
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )
 
        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------
 
        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )
 
        # ----------------------------------------------------
        # EDGE DETECTION
        # ----------------------------------------------------
 
        edges = cv2.Canny(
            gray,
            30,          # was 50 -> catches softer/studio-lit edges
            120          # was 150
        )
 
        # Close small gaps.
        kernel = np.ones(
            (7, 7),
            np.uint8
        )
 
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )
 
        # ----------------------------------------------------
        # FIND CONTOURS
        # ----------------------------------------------------
 
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
 
        if not contours:
 
            print("Leaf structure: no contours")
 
            return False, 0.0
 
        # Largest contour
        contour = max(
            contours,
            key=cv2.contourArea
        )
 
        contour_area = cv2.contourArea(
            contour
        )
 
        image_area = float(
            width * height
        )
 
        area_ratio = (
            contour_area /
            image_area
        )
 
        # ----------------------------------------------------
        # BOUNDING RECTANGLE
        # ----------------------------------------------------
 
        x, y, w, h = cv2.boundingRect(
            contour
        )
 
        bbox_area = float(
            max(w * h, 1)
        )
 
        extent = (
            contour_area /
            bbox_area
        )
 
        aspect_ratio = (
            max(w, h) /
            float(max(min(w, h), 1))
        )
 
        # ----------------------------------------------------
        # CONTOUR CIRCULARITY
        # ----------------------------------------------------
 
        perimeter = cv2.arcLength(
            contour,
            True
        )
 
        if perimeter > 0:
 
            circularity = (
                4.0 *
                math.pi *
                contour_area /
                (perimeter * perimeter)
            )
 
        else:
 
            circularity = 0.0
 
        # ----------------------------------------------------
        # EDGE DENSITY
        # ----------------------------------------------------
 
        edge_pixels = np.count_nonzero(
            edges
        )
 
        edge_density = (
            edge_pixels /
            image_area
        )
 
        # ----------------------------------------------------
        # PRINT DEBUG VALUES
        # ----------------------------------------------------
 
        print("Leaf structure:")
        print("  area ratio:", round(area_ratio, 4))
        print("  extent:", round(extent, 4))
        print("  aspect ratio:", round(aspect_ratio, 4))
        print("  circularity:", round(circularity, 4))
        print("  edge density:", round(edge_density, 4))
 
        # ----------------------------------------------------
        # SCORE (relaxed thresholds)
        # ----------------------------------------------------
 
        score = 0.0
 
        # Large enough object
        if area_ratio >= 0.04:
            score += 0.25
 
        elif area_ratio >= 0.015:
            score += 0.12
 
        # Leaf usually isn't a perfect circle
        if circularity < 0.88:
            score += 0.20
 
        elif circularity < 0.95:
            score += 0.08
 
        # Meaningful foreground shape
        if extent >= 0.12:
            score += 0.15
 
        # Leaf can have elongated or broad shape
        if aspect_ratio >= 1.05:
            score += 0.15
 
        elif aspect_ratio >= 1.00:
            score += 0.08
 
        # Internal texture / veins
        if edge_density >= 0.012:
            score += 0.25
 
        elif edge_density >= 0.006:
            score += 0.12
 
        score = min(score, 1.0)
 
        print("  structure score:", round(score, 3))
 
        # Threshold lowered from 0.40 -> 0.28
        return (score >= 0.28, score)
 
    except Exception as e:
 
        print("Leaf structure error:", e)
 
        return False, 0.0
 
 
# ============================================================
# ADDITIONAL LEAF TEXTURE CHECK
# ============================================================
 
def check_leaf_texture(image_pil):
 
    """
    Check whether the image contains enough natural
    internal texture/edges.
 
    This is intentionally color-independent.
    Returns (passed: bool, std: float, edge_ratio: float)
    """
 
    try:
 
        image_bgr = pil_to_cv(image_pil)
 
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )
 
        # Resize to standard size
        gray = cv2.resize(
            gray,
            IMAGE_SIZE
        )
 
        # Local contrast
        standard_deviation = float(
            np.std(gray)
        )
 
        # Canny texture
        edges = cv2.Canny(
            gray,
            25,      # was 40 -> softer/studio images still register texture
            100      # was 120
        )
 
        edge_ratio = (
            np.count_nonzero(edges) /
            float(edges.size)
        )
 
        print("Texture std:", round(standard_deviation, 2))
        print("Texture edge ratio:", round(edge_ratio, 4))
 
        # Relaxed thresholds — a clean studio/white-background
        # leaf photo can have low std/edge ratio but is still
        # a valid leaf image.
        passed = True
 
        if standard_deviation < 6:
            passed = False
 
        if edge_ratio < 0.003:
            passed = False
 
        return passed, standard_deviation, edge_ratio
 
    except Exception as e:
 
        print("Texture check error:", e)
 
        return False, 0.0, 0.0
 
 
# ============================================================
# PLANT-PIGMENT CHECK
# ============================================================
 
def check_plant_pigment(image_pil):
    """
    Broad plant-tissue colour check. Supports green, yellow,
    brown and reddish leaves; it is not a green-only test.
    """
    try:
        image_bgr = pil_to_cv(image_pil)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
 
        h = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
 
        green_yellow_brown = (
            (h >= 12) & (h <= 100) &
            (sat >= 45) & (val >= 35)
        )
        red_orange = (
            ((h <= 12) | (h >= 165)) &
            (sat >= 65) & (val >= 35)
        )
 
        pigment_mask = green_yellow_brown | red_orange
        pigment_ratio = float(np.count_nonzero(pigment_mask)) / float(
            max(pigment_mask.size, 1)
        )
        passed = pigment_ratio >= 0.10
 
        print("Plant pigment ratio:", round(pigment_ratio, 4))
        print("Plant pigment check:", passed)
        return passed, pigment_ratio
 
    except Exception as e:
        print("Plant pigment check error:", e)
        return False, 0.0
 
 
# ============================================================
# COMPLETE LEAF VALIDATION
# ============================================================
 
def validate_leaf(image_pil):
    """
    Final gate before the plant disease model.
 
    The model knows only plant-leaf classes, so arbitrary human
    or object images must be rejected before prediction.
    """
    print("")
    print("=" * 60)
    print("STARTING IMAGE VALIDATION")
    print("=" * 60)
 
    debug_info = {}
 
    # 1. Human/person hard rejection
    if detect_human(image_pil):
        print("VALIDATION RESULT: HUMAN IMAGE")
        return False, (
            "Human/person detected. Please upload a clear plant leaf image."
        ), debug_info
 
    # 2. Obvious circular object hard rejection
    if detect_circular_object(image_pil):
        print("VALIDATION RESULT: CIRCULAR OBJECT")
        return False, (
            "Circular object detected. Please upload a clear plant leaf image."
        ), debug_info
 
    # 3. Plant pigment
    pigment_ok, pigment_ratio = check_plant_pigment(image_pil)
 
    # 4. Leaf structure
    structure_ok, structure_score = analyze_leaf_structure(image_pil)
 
    # 5. Texture
    texture_ok, texture_std, texture_edge_ratio = check_leaf_texture(image_pil)
 
    debug_info = {
        "plant_pigment_ok": pigment_ok,
        "plant_pigment_ratio": round(pigment_ratio, 4),
        "structure_score": round(structure_score, 3),
        "structure_ok": structure_ok,
        "texture_ok": texture_ok,
        "texture_std": round(texture_std, 2),
        "texture_edge_ratio": round(texture_edge_ratio, 4),
    }
 
    # STRICT GATE: do not use the previous loose OR rule.
    # A single texture/edge signal is not enough to reach the
    # disease model.
    is_leaf = (
        pigment_ok and
        structure_score >= 0.28 and
        (structure_ok or texture_ok)
    )
 
    if not is_leaf:
        print("VALIDATION RESULT: NOT A LEAF")
        return False, (
            "The image does not appear to contain a clear plant leaf. "
            "Please upload a close, well-lit leaf image."
        ), debug_info
 
    print("VALIDATION RESULT: POSSIBLE LEAF")
    print("Plant pigment ratio:", round(pigment_ratio, 4))
    print("Structure score:", round(structure_score, 3))
    print("Texture OK:", texture_ok)
    print("=" * 60)
 
    return True, "Leaf validation passed.", debug_info
 
 
@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "INVALID_IMAGE",
        "message": "Image is too large (maximum 16 MB).",
        "disease": None,
        "confidence": 0
    }), 413
 
 
# ============================================================
# HOME
# ============================================================
 
@app.route("/")
def home():
 
    return jsonify({
 
        "status": "AgroGenius AI backend is running",
        "model": "Plant Disease Detection AI",
        "classes": int(len(class_names))
 
    })
 
 
# ============================================================
# PREDICT
# ============================================================
 
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
 
    try:
 
        # ====================================================
        # CHECK IMAGE
        # ====================================================
 
        if "image" not in request.files:
 
            return jsonify({
                "error": "No image received",
                "disease": None,
                "confidence": 0
            }), 400
 
        file = request.files["image"]
 
        if file.filename == "":
 
            return jsonify({
                "error": "No image selected",
                "disease": None,
                "confidence": 0
            }), 400
 
        # ====================================================
        # OPEN IMAGE
        # ====================================================
 
        try:
            image = Image.open(file)
            # Phones store rotation in EXIF; apply it so the leaf is upright.
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify({
                "error": "INVALID_IMAGE",
                "message": "This file is not a valid image. Please upload a JPG or PNG leaf photo.",
                "disease": None,
                "confidence": 0
            }), 400
 
        if max(image.size) > MAX_VALIDATION_SIDE:
            image.thumbnail((MAX_VALIDATION_SIDE, MAX_VALIDATION_SIDE))
 
        print("")
        print("New image received:")
        print("Filename:", file.filename)
        print("Original size:", image.size)
 
        # ====================================================
        # IMPORTANT:
        # VALIDATE IMAGE BEFORE AI MODEL
        # ====================================================
 
        is_leaf, validation_message, debug_info = validate_leaf(image)
 
        # ====================================================
        # REJECT NON-LEAF
        # ====================================================
 
        if not is_leaf:
 
            print("IMAGE REJECTED:", validation_message)
 
            response = {
                "error": "NOT_A_LEAF",
                "message": validation_message,
                "disease": None,
                "model_class": None,
                "confidence": 0
            }
 
            if DEBUG_VALIDATION:
                response["debug"] = debug_info
 
            return jsonify(response), 400
 
        # ====================================================
        # RESIZE FOR MODEL
        # ====================================================
 
        model_image = image.resize(IMAGE_SIZE)
 
        # ====================================================
        # NUMPY
        # ====================================================
 
        image_array = np.array(model_image, dtype=np.float32)
 
        # IMPORTANT:
        # Keep same preprocessing used by your model.
        image_array = image_array / 255.0
 
        image_array = np.expand_dims(image_array, axis=0)
 
        # ====================================================
        # AI PREDICTION
        # ====================================================
 
        predictions = model.predict(image_array, verbose=0)
        probabilities = np.asarray(predictions[0], dtype=np.float32)
 
        # Some exported models return logits instead of probabilities.
        # If values are not already a valid probability distribution,
        # convert them safely with softmax.
        if (
            np.any(probabilities < 0.0) or
            np.any(probabilities > 1.0) or
            not np.isclose(float(np.sum(probabilities)), 1.0, atol=0.05)
        ):
            probabilities = tf.nn.softmax(probabilities).numpy()
 
        # Keep only the available class count aligned with class_names.
        usable_count = min(len(probabilities), len(class_names))
        probabilities = probabilities[:usable_count]
 
        top_indices = np.argsort(probabilities)[::-1][:3]
        predicted_index = int(top_indices[0])
 
        confidence = float(probabilities[predicted_index]) * 100.0
        second_confidence = (
            float(probabilities[int(top_indices[1])]) * 100.0
            if len(top_indices) > 1 else 0.0
        )
        confidence_margin = confidence - second_confidence
 
        disease = str(class_names[predicted_index])
 
        print("")
        print("=" * 60)
        print("AI PREDICTION")
        print("=" * 60)
        print("Model class:", disease)
        print("Confidence:", round(confidence, 2), "%")
        print("Second class confidence:", round(second_confidence, 2), "%")
        print("Confidence margin:", round(confidence_margin, 2), "%")
 
        # Some PlantVillage models include a "Background_without_leaves" class.
        if "background" in disease.lower():
            print("Model says: background / no leaf.")
            return jsonify({
                "error": "NOT_A_LEAF",
                "message": "No plant leaf found in the image. Please upload a clear close-up leaf photo.",
                "disease": None,
                "model_class": disease,
                "confidence": round(confidence, 2)
            }), 400
 
        # ====================================================
        # STRONG VISUAL HEALTH CHECK
        # ====================================================
        # A trained classifier can still be over-confident on a
        # healthy leaf. Use a deliberately conservative visual
        # signal to protect obviously healthy green leaves.
        try:
            bgr_for_health = pil_to_cv(image)
            hsv_for_health = cv2.cvtColor(bgr_for_health, cv2.COLOR_BGR2HSV)
            hh = hsv_for_health[:, :, 0]
            ss = hsv_for_health[:, :, 1]
            vv = hsv_for_health[:, :, 2]
 
            green_mask = (
                (hh >= 35) & (hh <= 100) &
                (ss >= 45) & (vv >= 45)
            )
            yellow_brown_mask = (
                (hh >= 10) & (hh < 35) &
                (ss >= 55) & (vv >= 40)
            )
            dark_mask = vv < 45
 
            green_ratio = float(np.count_nonzero(green_mask)) / float(max(green_mask.size, 1))
            yellow_brown_ratio = float(np.count_nonzero(yellow_brown_mask)) / float(max(yellow_brown_mask.size, 1))
            dark_ratio_visual = float(np.count_nonzero(dark_mask)) / float(max(dark_mask.size, 1))
 
            strong_healthy_visual = (
                green_ratio >= 0.58 and
                yellow_brown_ratio <= 0.16 and
                dark_ratio_visual <= 0.07
            )
 
            print("Visual green ratio:", round(green_ratio, 4))
            print("Visual yellow/brown ratio:", round(yellow_brown_ratio, 4))
            print("Visual dark ratio:", round(dark_ratio_visual, 4))
            print("Strong healthy visual evidence:", strong_healthy_visual)
        except Exception as health_error:
            print("Healthy visual check error:", health_error)
            green_ratio = 0.0
            yellow_brown_ratio = 1.0
            dark_ratio_visual = 1.0
            strong_healthy_visual = False
 
        # If the model's top class is already healthy, keep it.
        # If a very strong healthy visual signal conflicts with a
        # disease prediction, prefer a cautious healthy result only
        # when the model is not overwhelmingly decisive.
        model_frontend_id = disease_map.get(disease, disease)
 
        # "Tomato___Early_blight" -> "Tomato"; "Pepper,_bell___healthy" -> "Pepper, bell"
        crop_name = disease.split("___")[0].replace("_", " ").strip()
 
        if (
            model_frontend_id != "healthy" and
            strong_healthy_visual and
            confidence < 90.0
        ):
            print("Healthy visual override applied.")
            disease = "Healthy (visual confirmation)"
            model_class_for_response = str(class_names[predicted_index])
            frontend_disease = "healthy"
            confidence_for_display = round(confidence, 2)
            disease_info = get_disease_info("healthy", model_class_for_response)
        else:
            model_class_for_response = disease
            frontend_disease = model_frontend_id
            confidence_for_display = round(confidence, 2)
            disease_info = get_disease_info(frontend_disease, disease)
 
        # ====================================================
        # LOW CONFIDENCE / CLOSE COMPETITION = UNCERTAIN
        # ====================================================
        # Do not force a disease when the model is weak or two
        # classes are too close. This is safer than presenting an
        # arbitrary disease as fact.
        if confidence < MIN_MODEL_CONFIDENCE or confidence_margin < 12.0:
            print("AI prediction is uncertain.")
 
            response = {
                "error": "UNCERTAIN_PREDICTION",
                "message": (
                    "The image looks like a plant leaf, but the AI is not confident enough "
                    "to provide a reliable disease diagnosis. Please upload a clearer close-up leaf image."
                ),
                "disease": "uncertain",
                "model_class": disease,
                "confidence": round(confidence, 2),
                "top_predictions": [
                    {
                        "model_class": str(class_names[int(i)]),
                        "confidence": round(float(probabilities[int(i)]) * 100.0, 2),
                        "disease_id": disease_map.get(str(class_names[int(i)]), str(class_names[int(i)]))
                    }
                    for i in top_indices
                ]
            }
 
            if DEBUG_VALIDATION:
                response["debug"] = {
                    **debug_info,
                    "confidence_margin": round(confidence_margin, 2),
                    "visual_green_ratio": round(green_ratio, 4),
                    "visual_yellow_brown_ratio": round(yellow_brown_ratio, 4),
                    "visual_dark_ratio": round(dark_ratio_visual, 4)
                }
 
            return jsonify(response), 200
 
        # ====================================================
        # FINAL RESULT + COMPLETE ADVICE
        # ====================================================
 
        print("Frontend disease:", frontend_disease)
        print("Display confidence:", confidence_for_display, "%")
        print("=" * 60)
 
        response = {
            "disease": frontend_disease,
            "model_class": model_class_for_response,
            "crop": crop_name,
            "confidence": confidence_for_display,
            "disease_name": disease_info["name"],
            "category": disease_info["category"],
            "severity": disease_info["severity"],
            "risk": disease_info["risk"],
            "causes": disease_info["causes"],
            "symptoms": disease_info["symptoms"],
            "treatment": disease_info["treatment"],
            "prevention": disease_info["prevention"],
            "advice": disease_info,
            "top_predictions": [
                {
                    "model_class": str(class_names[int(i)]),
                    "confidence": round(float(probabilities[int(i)]) * 100.0, 2),
                    "disease_id": disease_map.get(str(class_names[int(i)]), str(class_names[int(i)]))
                }
                for i in top_indices
            ]
        }
 
        if DEBUG_VALIDATION:
            response["debug"] = {
                **debug_info,
                "confidence_margin": round(confidence_margin, 2),
                "visual_green_ratio": round(green_ratio, 4),
                "visual_yellow_brown_ratio": round(yellow_brown_ratio, 4),
                "visual_dark_ratio": round(dark_ratio_visual, 4),
                "strong_healthy_visual": strong_healthy_visual
            }
 
        return jsonify(response)
 
    except Exception as e:
 
        print("")
        print("Prediction error:", e)
 
        return jsonify({
            "error": str(e),
            "disease": None,
            "confidence": 0
        }), 500
 
 
# ============================================================
# START SERVER
# ============================================================
 
if __name__ == "__main__":
 
    # use_reloader=False: the Flask reloader would start a second process
    # and load the TensorFlow model twice (slow, uses double the RAM).
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
 