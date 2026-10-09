# Rapport technique Adaptive Manipulation Lab

Le rapport est rédigé en français dans `main.tex`, avec la classe `IEEEtran` en mode conférence et deux colonnes. La bibliographie est intégrée : aucun fichier `.bib` n'est nécessaire.

`main.pdf` est la version compilée et vérifiée (11 pages, 9 figures). `article_sources.zip` rassemble les sources, le tableau LaTeX importé, les données d'évaluation et les figures PDF pour le partage ou l'import dans Overleaf. Les scripts restent dépendants du code et des données d'apprentissage du dépôt.

## Compilation

Depuis `article/`, avec une distribution LaTeX disposant d'IEEEtran et des packages déclarés dans le préambule :

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Ou avec Tectonic :

```powershell
tectonic main.tex
```

Le dossier peut être importé dans Overleaf en conservant `main.tex` et `images/`. Les figures PDF sont utilisées dans le document ; les PNG facilitent leur consultation.

## Régénération des figures

Depuis la racine du dépôt, avec les dépendances de simulation et d'analyse installées :

```powershell
.\.venv\Scripts\python.exe article/generate_figures.py
```

Le script lit les artefacts existants, sans entraînement et sans modifier les démonstrations ou les runs. Les deux vues MuJoCo reconstruisent l'état initial et l'état final de la démonstration humaine 1. Elles nécessitent un contexte de rendu OpenGL disponible.

`computed_metrics.json` conserve les agrégats calculés pour le rapport. `figure_sources.json` donne les chemins relatifs et empreintes SHA-256 des données et du XML utilisés. Les figures suivent les journaux présents lors de la génération : si un entraînement continue, régénérer les graphiques puis actualiser le texte et les tableaux avant de partager une nouvelle version.

## Périmètre éditorial

Le rapport décrit l'état du projet au 9 octobre 2026 : approche et contact avec le cube, collecte humaine, BC, transfert SAC, diagnostics et perspectives de saisie. Il distingue l'audit historique à 36 observations du contrat actuel à 38 observations, la validation de l'évaluation indépendante, et le budget configuré de l'entraînement effectivement consigné.

## Évaluation indépendante et comparaison appariée

`evaluate_comparison.py` a évalué les checkpoints figés `runs/bc_gamepad/best.pt` (époque 16) et `runs/sac_from_bc/best.pt` (90 000 décisions) sur CPU, sans réentraînement. Le protocole a été écrit avant l'évaluation. Les seeds 9 000 000 à 9 000 049 ont été contrôlées contre les configurations, sessions et exports présents avant les essais. Les six seeds de sélection ont aussi été rejouées pour mesurer leurs temps individuels, dans une partition séparée.

Les résultats sont dans `experiments/heldout_50/` :

- `protocol.json` : identité et empreintes des checkpoints, signature physique, audit des seeds, mode d'action et versions d'exécution ;
- `bc_independent.csv`, `sac_independent.csv` : mesures des 50 nouveaux épisodes et composantes de récompense ;
- `bc_validation.csv`, `sac_validation.csv` : mesures sur les six seeds de sélection ;
- `paired_independent.csv`, `paired_validation.csv` : différences seed par seed ;
- `summary.json` : agrégats, différences appariées et intervalles de Wilson ;
- `failures.csv` : journal des échecs, vide de données dans cette expérience (tous les contacts sont réussis).

Le script refuse d'écraser une évaluation existante. Pour une nouvelle campagne, définir un nouveau dossier de sortie et un ensemble de seeds inédites avant l'exécution. Pour rejouer le protocole dans une copie propre du dépôt avec les mêmes checkpoints, lancer :

```powershell
.\.venv\Scripts\python.exe article/evaluate_comparison.py
```

La campagne d'ablation SAC seul / BC seul / BC + SAC est définie dans le rapport comme travail à réaliser sous environnement commun. Elle n'a pas été exécutée : les modèles SAC seul actuellement conservés utilisent l'ancien contrat à 36 observations. Les seeds de la présente évaluation devront rester exclues d'une nouvelle sélection de modèles et être remplacées par un autre ensemble final pour cette future campagne.
