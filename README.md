🌿 AgroGenius AI — Plant Leaf Disease Diagnostics

AI Lab Project — B.Tech CSE (AIML), Semester V I.T.S Engineering College, Greater Noida

An AI-powered web application that detects plant leaf diseases from a photo using a trained deep learning (CNN) model, validates that the uploaded image is genuinely a plant leaf, and returns an instant, detailed diagnosis with causes, symptoms, treatment and prevention advice — complete with a weather dashboard, multilingual voice assistant, and downloadable PDF reports.

👥 Team
Name	Roll No.
Raviansh Katiyar	2402221530099
Sakshi Kumari	2402221530109

Mentor: Anjali Srivastava

📌 Project Overview

AgroGenius AI is a full-stack plant disease diagnosis system built around a TensorFlow/Keras Convolutional Neural Network trained on the PlantVillage-style leaf image dataset (38 classes across 14 crop types — Apple, Grape, Tomato, Potato, Corn, Citrus, and more). Users upload a leaf photo (or capture one live via camera), the backend validates and classifies it, and the frontend presents a full diagnosis — disease name, severity, causes, symptoms, treatment, and prevention — along with live weather data for the user's location and a multilingual AI chat/voice assistant for general plant-care questions.

Core Features
📷 Leaf image scanning — upload a photo or capture one with a live camera
🧠 CNN-based disease classification — 38-class TensorFlow/Keras model
🛡️ Multi-stage image validation pipeline — rejects non-leaf images (humans, circular objects, drawings/illustrations, low-texture/blurry images) before they reach the AI model, and flags low-confidence predictions as "uncertain" instead of guessing
📋 Full diagnosis report — disease name, category, severity, causes, symptoms, treatment and prevention, served directly from the backend's knowledge base
🌦️ Live weather dashboard — geolocation-based current conditions (Open-Meteo API)
📊 Scan history & analytics — per-account history, trend chart, health/risk stats
📄 PDF report export — downloadable diagnosis reports (jsPDF)
🌐 17-language interface & assistant — UI, advice and chat replies auto-translate
🎙️ Voice assistant — speech-to-text commands and spoken replies (Web Speech API)
🔐 On-device login — lightweight local session (email/phone), no external server
🎨 Theming — dark/light mode plus 5 background themes
🏗️ Tech Stack
Layer	Technology
Frontend	HTML5, CSS3 (custom design system), vanilla JavaScript
Backend	Python, Flask, Flask-CORS
AI / ML	TensorFlow / Keras (CNN), NumPy
Image Processing	OpenCV (Haar cascades, HOG person detector, HSV colour analysis)
Charts	Chart.js
PDF Export	jsPDF
Weather	Open-Meteo API (free, no key)
Geocoding	BigDataCloud reverse-geocoding API
Translation	MyMemory Translation API
Voice	Web Speech API (SpeechRecognition + SpeechSynthesis)
Storage	Browser localStorage (per-user scan history & settings)
📂 Project Structure
AgroGenius-AI-V2/
├── app.py                     # Flask backend — API, validation pipeline, CNN inference
├── index.html                 # Frontend — single-file UI, logic and styling
├── model/
│   ├── plant_disease_model.keras   # Trained CNN model
│   └── class_names.npy             # Ordered list of the model's 38 output classes
├── requirements.txt           # Python dependencies
└── README.md
⚙️ How It Works (Pipeline)
Image upload — user uploads a photo or captures a frame from the live camera on the Scan page.
Backend validation (app.py) — before the image ever reaches the CNN, it passes through a validation pipeline:
Human detection — frontal/profile face (Haar cascade) + HOG full-person detector, confirmed with HSV skin-tone analysis to avoid false positives on leaf textures
Circular-object rejection — Hough Circle Transform, cross-checked against the object's real contour shape (circularity + aspect ratio) so shiny/curved leaves aren't mistaken for plates, wheels, etc.
Leaf structure analysis — contour area, extent, aspect ratio, circularity and edge density, scored to confirm a leaf-like foreground object
Texture check — local contrast (std-dev) and edge density to catch flat, low-detail or blank images
Plant-pigment check — HSV-based check for natural green/yellow/brown plant-tissue colour ranges
CNN inference — the validated image is resized to 224×224, normalized, and passed to the trained TensorFlow/Keras model, which outputs class probabilities across all 38 disease classes.
Confidence gating — if the top prediction's confidence is too low, or too close to the second-best class, the result is returned as "uncertain" instead of a forced diagnosis.
Visual healthy-override — an additional HSV-based "looks clearly healthy" signal can override a lower-confidence disease prediction, reducing false disease alarms on genuinely healthy leaves.
Diagnosis response — the backend's own DISEASE_INFO knowledge base attaches full causes/symptoms/treatment/prevention text to the result, matched by a normalized disease-ID slug (e.g. "black-rot", "common-rust").
Frontend rendering — index.html displays the result in a gauge/advice card, saves it to that user's scan history, updates the dashboard stats/chart, and can export it as a PDF.
🧑‍🤝‍🧑 Role Division

The work was split so each member owns one major half of the system end-to-end, with shared responsibility for integration and testing.

🔷 Raviansh Katiyar (2402221530099) — AI / Backend Engineer

Responsible for everything that turns a raw uploaded image into a validated, classified, and explained diagnosis.

Designed and implemented the Flask REST API (app.py) — routes, CORS, request handling, error responses
Trained / integrated the CNN plant-disease classification model (TensorFlow/Keras, 38 classes, 14 crops) and the preprocessing pipeline (resizing, normalization)
Built the multi-stage image validation pipeline using OpenCV:
Human detection (Haar cascades + HOG person detector + HSV skin-tone confirmation)
Circular-object rejection (Hough Circle Transform + contour shape cross-check)
Leaf structure scoring (contour/edge/aspect-ratio analysis)
Texture and plant-pigment checks
Implemented confidence-margin gating and the visual healthy-override logic to reduce false and over-confident predictions
Built the backend disease knowledge base (DISEASE_INFO) mapping each of the 38 model classes to a clean disease ID, name, category, severity, causes, symptoms, treatment and prevention
Handled model/class-name loading, large-upload limits, and server configuration
🔶 Sakshi Kumari (2402221530109) — Frontend / UX Engineer

Responsible for everything the user sees and interacts with, and for every feature built on top of the diagnosis result.

Designed and built the complete UI/UX in index.html — dashboard, scan page, reports page, settings page, sidebar navigation, responsive layout
Implemented the image upload / drag-and-drop / live camera capture flow and preview handling
Built the scan results view (confidence gauge, advice card) and connected it to the Flask /predict API
Implemented scan history, per-account storage (localStorage), and dashboard analytics (Chart.js health-trend chart, health/risk stat cards)
Built the PDF report export feature (jsPDF) and the reports/previous-reports list with per-entry detail view
Implemented the live weather dashboard (Open-Meteo + geolocation + reverse geocoding) and the location-permission handling/fallback states
Built the 17-language translation system (UI strings, advice content, and chat replies) and the voice assistant (Web Speech API speech-to-text and text-to-speech, voice command routing)
Implemented theming (dark/light mode, 5 background themes) and the on-device login/session system
🤝 Shared / Joint Work
End-to-end integration testing of the frontend ↔ backend API contract
Validation-pipeline tuning against real test images (false-positive/negative fixes)
Disease-ID matching between the backend's disease_map and the frontend's knowledge base
Documentation and final report preparation
🚀 Setup & Running Locally
Backend
bash
# from the project root
pip install -r requirements.txt
python app.py

The Flask server starts at http://127.0.0.1:5000.

Frontend

Simply open index.html in a browser (a local dev server such as python -m http.server or the VS Code "Live Server" extension is recommended, since browsers restrict Geolocation and microphone access on pages opened directly via file://).

⚠️ The frontend calls the backend at http://127.0.0.1:5000/predict, so the Flask server must be running before you scan a leaf.

🔮 Future Scope
Deploy the backend on a cloud platform with a production WSGI server
Expand the dataset/model to cover more crops and region-specific diseases
Add user accounts with persistent (server-side) history instead of local storage
Offline-capable mobile app version
🙏 Acknowledgements

Built as part of the AI Lab coursework under the mentorship of Anjali Srivastava, B.Tech CSE (AIML), Semester V, I.T.S Engineering College, Greater Noida.