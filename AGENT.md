# AGENT V3 - Speak With Sign

## Objectif

Faire evoluer le prototype vers une application de bureau Python simple et
fonctionnelle qui traduit localement un flux video en langue des signes francaise
(LSF) vers du texte francais.

L'application doit accepter :

- une webcam choisie par l'utilisateur ;
- un fichier MP4 local.

Elle doit afficher la video avec une zone blanche de sous-titres sous l'image.
Pour une webcam, la frame composee doit pouvoir etre envoyee vers une camera
virtuelle. Pour un MP4, elle doit pouvoir etre lue puis exportee dans un nouveau
fichier MP4.

La premiere cible modele est un backend local scikit-learn compose de deux
fichiers de poids :

```text
scaler.pkl
mlp_classifier.pkl
```

Ces fichiers correspondent a :

- `scaler.pkl` : un `sklearn.preprocessing.StandardScaler` entrainé sur 126
  valeurs d'entree ;
- `mlp_classifier.pkl` : un `sklearn.neural_network.MLPClassifier` entrainé sur
  126 valeurs, avec deux couches cachees `(256, 128)` et 17 classes numeriques
  `0..16`.

Le modele ne contient pas les textes francais associes aux classes. Un fichier de
labels externe est donc necessaire pour afficher une traduction reelle :

```text
labels.json
```

Exemple attendu :

```json
{
  "0": "bonjour",
  "1": "merci"
}
```

Sans fichier de labels valide, l'application peut charger techniquement le
modele, mais elle ne doit afficher aucune traduction inventee. L'interface doit
indiquer clairement `Modele charge, labels manquants`.

L'application doit aussi fonctionner sans modele charge. Dans ce cas, la video
reste utilisable, l'interface affiche `Aucun modele charge` et aucun faux texte
n'est genere.

## Principe principal : rester simple

Ne pas faire d'over-engineering.

- Implementer uniquement ce qui est necessaire au besoin actuel.
- Choisir la solution la plus simple qui fonctionne sur Windows et Linux.
- Reutiliser le code existant lorsqu'il est correct et comprehensible.
- Ne pas creer d'abstraction, de couche ou de classe sans usage concret immediat.
- Ne pas anticiper plusieurs backends ou formats tant qu'un contrat simple suffit.
- Eviter les grands refactorings. Avancer par petites modifications testables.
- Preferer quelques modules clairs a une architecture profonde et fragmentee.
- Ne pas introduire de framework, bus d'evenements, injection de dependances ou
  systeme de plugins maison sans besoin prouve.
- Ne pas optimiser avant d'avoir mesure un probleme reel.
- Une implementation lisible et fonctionnelle vaut mieux qu'une architecture
  theorique incomplete.

Lorsqu'une decision oppose simplicite et extensibilite future, choisir la
simplicite tant que les contrats du modele, de la capture et des sorties restent
clairs.

## Contraintes obligatoires

- Python 3.10 ou plus recent.
- Interface principale en PySide6/Qt, pas en OpenCV.
- Fonctionnement entierement local, sans API distante ni telemetrie.
- Entree LSF uniquement et sortie texte en francais uniquement.
- Entrees limitees aux webcams et aux fichiers MP4.
- Zone de sous-titres blanche sous la video, jamais sur les mains.
- Aucun poids de production dans Git.
- Aucune traduction simulee ou phrase d'exemple dans l'application.
- L'absence ou l'erreur d'un modele ne doit pas bloquer la video.
- Les traitements longs ne doivent pas bloquer le thread Qt.
- Les ressources doivent etre liberees proprement a l'arret.
- Le backend scikit-learn est CPU uniquement. Ne pas proposer de GPU pour ce
  backend, ou l'afficher comme non applicable.
- Les fichiers pickle doivent etre charges uniquement depuis un chemin local
  explicitement choisi ou configure par l'utilisateur.
- Ne jamais charger automatiquement un pickle distant ou non fiable.

## Dependances principales

Le backend modele cible doit fonctionner avec :

```text
numpy
scikit-learn==1.5.2
joblib
mediapipe
opencv-python
PySide6
```

`scikit-learn==1.5.2` est recommande car les fichiers `.pkl` fournis ont ete
serialises avec cette version. Une version differente peut provoquer des
avertissements ou un comportement incompatible au chargement.

Les dependances optionnelles sont a ajouter seulement lorsque la fonctionnalite
associee est implementee :

```text
pyvirtualcam      # camera virtuelle
ffmpeg-python     # conservation ou remux audio, si retenu plus tard
```

## Architecture minimale recommandee

Conserver une separation simple entre les responsabilites :

```text
main.py                         demarrage de l'application
src/
  application/                  orchestration et etat
  capture/                      webcam et lecture MP4
  preprocessing/                MediaPipe Hands et vecteurs 126
  inference/                    contrat moteur, NoModelEngine, SklearnMlpEngine
  subtitles/                    evenements, buffer et rendu
  output/                       composition, camera virtuelle et export MP4
  ui/                           fenetres et widgets PySide6
tests/                          tests des comportements importants
models/                         fichiers locaux ignores par Git
```

Cette structure est un guide, pas une obligation de creer tous les fichiers des
le debut. Creer un module seulement lorsqu'il contient une responsabilite reelle.
Ne pas placer toute l'application dans `main.py`, mais ne pas decouper une petite
fonction en plusieurs couches artificielles.

Le coeur de traitement ne doit pas dependre des widgets Qt. L'interface appelle
un controleur simple et recoit des etats ou resultats structures.

## Parcours fonctionnel prioritaire

Implementer dans cet ordre :

1. Choisir une webcam ou un MP4 depuis Qt.
2. Afficher la video avec une zone blanche vide sous l'image.
3. Demarrer, mettre en pause pour un MP4, arreter et liberer les ressources.
4. Extraire une seule fois les landmarks MediaPipe Hands par frame utile.
5. Construire un vecteur `float32[126]` dans l'ordre attendu.
6. Charger `scaler.pkl`, `mlp_classifier.pkl` et, si disponible, `labels.json`.
7. Utiliser `NoModelEngine` si aucun modele valide n'est charge.
8. Executer l'inference scikit-learn sans bloquer Qt.
9. Afficher uniquement les traductions reelles dont le label est connu.
10. Publier la frame composee vers une camera virtuelle.
11. Exporter un MP4 compose dans un nouveau fichier.

Ne pas commencer le packaging avance, les optimisations GPU ou une interface de
configuration complexe avant que ce parcours fonctionne de bout en bout.

## Format des poids modele

Le format modele minimal attendu est :

```text
model_dir/
  scaler.pkl
  mlp_classifier.pkl
  labels.json
```

`labels.json` est requis pour afficher du texte. Il peut etre absent pendant le
developpement, mais dans ce cas le moteur ne doit pas produire de sous-titre.

L'application peut proposer deux modes simples :

- selectionner un dossier modele contenant les trois fichiers ;
- ou selectionner separement `scaler.pkl`, `mlp_classifier.pkl` et `labels.json`.

Le premier mode est a privilegier pour l'interface utilisateur.

### Validation obligatoire au chargement

Au chargement, verifier explicitement :

- que `scaler.pkl` existe ;
- que `mlp_classifier.pkl` existe ;
- que `scaler.pkl` est un `StandardScaler` ;
- que `mlp_classifier.pkl` est un `MLPClassifier` ;
- que `scaler.n_features_in_ == 126` ;
- que `mlp_classifier.n_features_in_ == 126` ;
- que le classifieur expose `classes_` ;
- que le classifieur expose `predict_proba` ;
- que les classes sont compatibles avec les cles de `labels.json` si ce fichier
  est present ;
- que chaque label est une chaine de caracteres francaise non vide.

Ne jamais corriger silencieusement une dimension incompatible. Un modele 195 ou
un scaler 195 doit etre refuse clairement.

### Securite pickle

Les fichiers `.pkl` peuvent executer du code au chargement. Ils doivent etre
consideres comme fiables uniquement s'ils proviennent du projet ou d'une source
controlee.

L'application ne doit pas :

- telecharger automatiquement un pickle ;
- charger un pickle depuis une URL ;
- charger un pickle sans action explicite de l'utilisateur ou configuration
  locale claire ;
- masquer les erreurs de chargement.

Une erreur de chargement doit devenir un etat utilisateur comprehensible, par
exemple :

```text
Modele invalide : scaler.pkl attend 195 valeurs au lieu de 126
```

## Pretraitement MediaPipe

Le profil principal pour ce modele est `hands_126_v1` :

```text
main gauche : 21 * 3 = 63
main droite : 21 * 3 = 63
total       : float32[126]
```

L'ordre obligatoire est :

```text
main gauche, puis main droite
```

Chaque main contient les 21 landmarks MediaPipe dans l'ordre natif MediaPipe.
Chaque landmark contient :

```text
x, y, z
```

Un groupe absent est remplace par des zeros.

La conversion BGR vers RGB est faite avant MediaPipe. Le pretraitement du modele
cible 20 FPS, independamment du FPS d'affichage.

Ne pas ajouter de normalisation maison avant le scaler. Le vecteur brut MediaPipe
`[x, y, z]` doit etre transmis a `scaler.transform(...)`, car le scaler contient
deja les statistiques d'entrainement.

### Attribution gauche / droite

Utiliser en priorite l'information de handedness fournie par MediaPipe.

Si MediaPipe ne fournit pas de handedness exploitable, utiliser une regle simple
et documentee, par exemple l'ordre horizontal des poignets, mais ne pas multiplier
les heuristiques. La regle doit etre testee.

L'absence d'une main ne doit pas bloquer l'inference. La partie correspondante du
vecteur reste remplie de zeros.

## Moteur de traduction

Conserver un contrat petit et independant de Qt et MediaPipe :

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TranslationResult:
    text: str
    tokens: list[str]
    confidence: float | None
    status: str
    latency_ms: float | None = None
    message: str | None = None
```

`NoModelEngine` est toujours disponible. Il retourne un texte vide, aucun token,
une confiance `None` et le statut `model_not_loaded`.

Le backend scikit-learn s'appelle `SklearnMlpEngine`.

Il applique le pipeline suivant :

```text
landmarks float32[126]
-> reshape en float32[1, 126]
-> scaler.transform(...)
-> mlp_classifier.predict_proba(...)
-> classe avec probabilite maximale
-> texte via labels.json
```

Si la confiance maximale est inferieure au seuil configure, retourner un resultat
sans texte avec le statut `low_confidence`.

Si la classe predite n'existe pas dans `labels.json`, retourner un resultat sans
texte avec le statut `label_missing`.

Si `labels.json` est absent ou invalide, retourner un resultat sans texte avec le
statut `labels_missing`.

Une erreur d'inference devient un resultat ou un etat structure et ne doit pas
arreter la capture.

## Stabilisation des predictions

Le `MLPClassifier` fourni est un classifieur de vecteur unique, pas un modele
sequence-a-sequence. Ne pas implementer de fenetres, padding, masques ou decodeur
complexe tant qu'un modele sequentiel reel n'est pas disponible.

Pour eviter le clignotement des sous-titres, utiliser une stabilisation simple :

- predire a 20 FPS maximum ;
- conserver les dernieres predictions valides sur une petite fenetre ;
- emettre un `partial` lorsqu'une classe devient majoritaire ;
- emettre un `final` lorsqu'une classe reste stable pendant une duree minimale ;
- ne pas repeter un `final` identique en boucle ;
- effacer ou remplacer le `partial` lorsqu'une nouvelle prediction stable arrive.

Les valeurs de depart peuvent etre simples et modifiables dans le code :

```text
confidence_threshold = 0.60
partial_min_frames = 3
final_min_frames = 8
```

Ces valeurs ne sont pas des contrats scientifiques. Elles servent a obtenir une
application stable et testable.

## Statuts modele recommandes

Utiliser des statuts explicites plutot que des exceptions visibles dans l'UI :

```text
model_not_loaded       aucun modele selectionne
model_loading          chargement en cours
model_ready            scaler, classifieur et labels valides
labels_missing         modele charge mais labels absents ou invalides
model_invalid          fichier absent, type incorrect ou dimension incompatible
low_confidence         prediction ignoree car confiance trop faible
label_missing          classe predite absente du fichier labels
inference_error        erreur pendant transform ou predict_proba
```

L'interface doit afficher ces etats en francais clair.

## Capture et traitement

Une source video fournit au minimum une frame BGR, un horodatage monotone, sa
taille et son FPS. Un MP4 fournit aussi son horodatage media.

- Webcam : utiliser une petite file bornee et garder la frame la plus recente.
- Lecture MP4 : respecter le rythme du fichier et gerer pause/reprise.
- Export MP4 : traiter toutes les frames dans l'ordre, sans abandon.
- Ne jamais utiliser `time.sleep()` dans le thread de l'interface.
- Utiliser un worker simple avec arret cooperatif pour les operations longues.
- Une frame ne doit subir qu'une seule extraction MediaPipe.
- L'apercu et les sorties utilisent la meme frame composee.
- L'inference scikit-learn doit etre executee hors du thread Qt si elle peut
  ralentir l'interface.

## Sous-titres et sorties

Les evenements de sous-titres contiennent un identifiant, un type `partial` ou
`final`, le texte, un horodatage et une confiance optionnelle. Un `partial` est
remplace par sa nouvelle version puis par le `final`, sans duplication.

Aucun sous-titre ne doit etre genere dans les cas suivants :

- aucun modele charge ;
- scaler ou classifieur invalide ;
- labels absents ;
- classe predite absente de `labels.json` ;
- confiance inferieure au seuil ;
- erreur d'inference.

La frame finale mesure :

```text
largeur = largeur video
hauteur = hauteur video + hauteur sous-titres
```

La zone est blanche, le texte noir et limite a deux lignes par defaut. Les retours
a la ligne sont calcules selon la largeur reelle en pixels. Les accents francais
doivent s'afficher correctement.

Pour la camera virtuelle, utiliser un adaptateur simple autour d'une bibliotheque
existante. Detecter l'absence d'OBS Virtual Camera sous Windows ou de
`v4l2loopback` sous Linux et afficher un diagnostic, sans installer de pilote.

Pour l'export MP4 :

- ne pas ecraser sans confirmation ;
- ecrire dans un fichier temporaire puis le finaliser ;
- permettre l'annulation et supprimer le temporaire ;
- verifier que le fichier final peut etre relu ;
- indiquer clairement que l'audio n'est pas conserve si FFmpeg n'est pas utilise.

## Interface minimale

L'interface doit proposer :

- choix `Webcam` ou `Fichier MP4` ;
- choix de la webcam ou d'un fichier `*.mp4` valide ;
- apercu video et sous-titres ;
- boutons demarrer, pause si utile, arreter et retour ;
- etat du modele, de la source et de la sortie ;
- choix d'un dossier modele contenant `scaler.pkl`, `mlp_classifier.pkl` et
  `labels.json` ;
- affichage clair du fait que le backend scikit-learn fonctionne en CPU ;
- activation de la camera virtuelle ;
- destination et progression de l'export MP4 ;
- FPS et latence mesures simplement.

Commencer avec une presentation sobre. Ne pas construire de systeme de themes,
d'animations ou de widgets generiques tant que les fonctions principales ne sont
pas terminees.

## Gestion Git et fichiers locaux

Les poids et fichiers locaux de modele ne doivent pas etre commit.

Ajouter ou conserver dans `.gitignore` :

```gitignore
models/
*.pkl
*.joblib
```

Si un petit modele de test est necessaire, utiliser un fixture minimal dans
`tests/fixtures/` avec des donnees artificielles et non des poids de production.

## Tests essentiels

Tester les contrats qui evitent les regressions reelles :

- extraction de 126 valeurs dans le bon ordre : main gauche puis main droite ;
- zeros pour une main absente ;
- conversion BGR vers RGB avant MediaPipe ;
- refus clair d'un vecteur qui n'a pas exactement 126 valeurs ;
- refus d'un scaler dont `n_features_in_` n'est pas 126 ;
- refus d'un classifieur dont `n_features_in_` n'est pas 126 ;
- refus d'un fichier pickle de type inattendu ;
- chargement CPU de `StandardScaler` + `MLPClassifier` ;
- comportement lorsque `labels.json` est absent ;
- comportement lorsqu'une classe predite manque dans `labels.json` ;
- utilisation de `predict_proba` et calcul de confiance ;
- filtrage par seuil de confiance ;
- stabilisation `partial` vers `final` sans duplication ;
- comportement de `NoModelEngine` ;
- traitement identique pour webcam et fichier ;
- composition video avec zone blanche sous l'image ;
- refus d'un faux MP4 et export relisible ;
- file webcam bornee et abandon des anciennes frames ;
- arret propre des workers et ressources ;
- controleur Qt en mode headless ;
- camera virtuelle avec un sink simule dans les tests.

Ne pas viser une couverture artificielle. Ajouter un test lorsqu'il protege un
contrat, un bug corrige ou un parcours utilisateur important.

## Methode de travail

Pour chaque petit lot :

1. Lire le code et les changements locaux concernes.
2. Identifier le comportement utilisateur a obtenir.
3. Ecrire ou adapter les tests utiles.
4. Implementer la solution la plus petite et lisible.
5. Executer les tests cibles puis la suite complete.
6. Verifier le fonctionnement sans modele et l'arret propre.
7. Verifier le chargement `scaler.pkl` + `mlp_classifier.pkl` + `labels.json`.
8. Mettre a jour la documentation si le comportement change.

Avant d'ajouter une abstraction, se poser trois questions :

1. Est-elle utilisee maintenant par au moins deux implementations reelles ?
2. Reduit-elle vraiment la complexite ou la duplication ?
3. Peut-on livrer plus clairement sans elle ?

Si les reponses ne la justifient pas, ne pas l'ajouter.

## Definition de termine

La version simple est terminee lorsque le parcours complet fonctionne depuis
PySide6, sans terminal : choix de source, capture ou lecture, pretraitement
MediaPipe Hands en 126 valeurs, modele optionnel, inference scikit-learn,
sous-titres, apercu, camera virtuelle ou export MP4.

Sans modele, l'application reste stable et n'affiche aucune traduction. Avec un
couple compatible `scaler.pkl` + `mlp_classifier.pkl` et un `labels.json` valide,
elle charge le modele sans modification du code et affiche uniquement les labels
reels. Les erreurs sont comprehensibles, les operations longues ne figent pas Qt
et toutes les ressources sont liberees a l'arret.

Tout ce qui ne contribue pas directement a cette definition est reporte jusqu'a
ce qu'un besoin concret le rende necessaire.
