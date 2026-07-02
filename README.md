# Speak With Sign

Application de bureau locale pour afficher une source webcam ou MP4, extraire
des landmarks MediaPipe Hands, et traduire via un modele scikit-learn local.

## Installation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Lancement

```powershell
.\.venv\Scripts\python.exe main.py
```

L'application fonctionne aussi sans modele: la video reste utilisable et aucun
texte n'est genere.

## Modele local

Selectionner un dossier contenant:

```text
scaler.pkl
mlp_classifier.pkl
labels.json
```

`labels.json` doit associer les classes numeriques aux textes francais reels:

```json
{
  "0": "bonjour",
  "1": "merci"
}
```

Les fichiers pickle ne sont charges qu'apres choix explicite d'un dossier local.
Ils ne doivent pas etre recuperes depuis une source distante non fiable.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
