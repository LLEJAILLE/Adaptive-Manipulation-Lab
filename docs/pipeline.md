# Pipeline d'apprentissage

[Présentation du projet](../README.md) · [Architecture](architecture.md) · [Expériences](experiments.md)

La pipeline de la tâche `reach` comprend la collecte de démonstrations, l'apprentissage par imitation, le transfert vers SAC et l'évaluation de la politique. Toutes les étapes utilisent la même scène et le même contrat d'observation et d'action.

Les exemples utilisent PowerShell depuis la racine du dépôt, après l'installation décrite dans le README. Les chemins de sortie des nouvelles expériences doivent désigner des dossiers distincts des artefacts existants.

## 1. Commande manuelle et collecte

### Exploration de la scène

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation manual --episodes 5 --seed 42
```

Le mode `manual` fournit des curseurs et une commande à la manette, sans enregistrer de dataset. Ses scénarios utilisent une horloge réelle. La collecte `record` utilise le temps simulé MuJoCo pour reproduire les conditions d'apprentissage.

### Nouvelle session de démonstrations

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation record --config configs/training.json --output demonstrations/reach_session_02 --episodes 30 --seed-start 3000030
```

Chaque tentative reçoit une seed différente : dans cet exemple, de 3000030 à 3000059. La session contient 30 tentatives, réussies ou échouées. La configuration courante utilise 10 pas MuJoCo par décision, soit environ 50 décisions/s. Les incréments maximaux sont 1° par articulation et 0,9 mm d'ouverture par décision.

| Contrôle DualSense | Commande |
|---|---|
| Stick gauche | Articulations J1 et J2 |
| Stick droit vertical | Articulation J3 |
| Croix | Fermeture progressive de la pince |
| Rond | Ouverture progressive de la pince |
| Triangle | Retour progressif aux consignes initiales, sans réinitialiser le temps |

Le bouton **Démarrer le scénario** déclenche la tentative et son chronomètre. Le contact ou l'échéance provoque la sauvegarde, puis le scénario suivant attend un nouveau démarrage. Une déconnexion suspend le temps simulé et la collecte. Une fermeture normale sauvegarde la tentative partielle séparément ; un arrêt brutal peut perdre la tentative active.

### Reprise d'une collecte

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation record --config configs/training.json --output demonstrations/reach_session_02 --episodes 30 --seed-start 3000030 --resume
```

La reprise recommence le premier scénario non terminé avec sa seed prévue. Le XML, les paramètres physiques et le planning de seeds doivent correspondre à la session. Un dossier de collecte existant nécessite `--resume`.

### Format des données

| Fichier | Contenu |
|---|---|
| `session.json` | Planning des seeds, scénarios terminés, résultats et signature de compatibilité |
| `episode_XXXX.npz` | Transitions d'une tentative complète, réussie ou échouée |
| `partial_XXXX_YYY.npz` | Tentative interrompue, exclue du chargement BC |

Chaque NPZ conserve observations, actions normalisées réellement envoyées à `env.step`, rewards, observations suivantes et flags de fin. Il contient aussi les distances, le temps simulé, les composantes de reward, les diagnostics d'action et les métadonnées du scénario. Les tableaux sont lisibles avec `np.load(..., allow_pickle=False)`.

### Validation du dataset

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation inspect-demos demonstrations/gamepad_30
```

Le chargement vérifie les dimensions, les valeurs finies, les actions, la continuité des observations et la fin des épisodes. L'inspection comprend les réussites et les échecs. L'entraînement BC sélectionne les réussites complètes.

```python
from adaptive_manipulation.data.demonstrations import load_demonstrations

data, signature = load_demonstrations("demonstrations/gamepad_30", success_only=True)
observations = data["observations"]  # (N, 38)
actions = data["actions"]            # (N, 4), dans [-1, 1]
episode_ids = data["episode_ids"]
seeds = data["seeds"]
```

## 2. Behavior cloning

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation train-bc --demonstrations demonstrations/gamepad_30 --config configs/training.json --output runs/bc_reach_v2 --epochs 200
```

BC entraîne l'Actor SAC à prédire les actions humaines, avec l'objectif :

```text
loss_BC = MSE(tanh(mean(observation)), action_humaine)
```

L'architecture de référence est `38 → 256 → 256 → 8` : quatre moyennes et quatre log-écarts-types. La loss utilise uniquement l'action déterministe `tanh(mean)`. La sortie `log_std` est fixée initialement à `log(0.3)` et reste constante pendant le BC. Les rewards enregistrées ne participent pas à l'optimisation supervisée.

Chaque transition a le même poids ; les trajectoires longues contribuent donc davantage que les courtes. Les périodes d'immobilité restent dans le dataset. Les observations ne reçoivent pas de normalisation supplémentaire.

### Séparation des scénarios

Le split porte sur des épisodes entiers pour éviter de placer des transitions voisines dans les deux partitions. Avec le dataset de référence et une seed de split de 42 :

| Partition | Scénarios | Transitions |
|---|---:|---:|
| Apprentissage | 24 | 7 312 |
| Validation | 6 | 1 755 |

`split.json` conserve IDs, seeds, nombres de transitions, chemins et empreintes SHA-256 des épisodes sources. Les scénarios de validation servent à sélectionner le modèle ; ils ne constituent pas un test final indépendant.

### Options principales

| Option | Valeur par défaut | Rôle |
|---|---|---|
| `--epochs` | 200 | Nombre maximal d'époques |
| `--batch-size` | 256 | Taille des minibatches, mélangés à chaque époque |
| `--learning-rate` | 0.0003 | Pas d'Adam |
| `--validation-fraction` | 0.2 | Fraction des épisodes réservés |
| `--seed` | 42 | Split, initialisation et mélange des données |
| `--patience` | 30 | Époques sans amélioration significative ; 0 désactive l'arrêt anticipé |
| `--min-delta` | 0.000001 | Amélioration requise pour réinitialiser la patience |
| `--initial-std` | 0.3 | Écart-type gaussien avant tanh |
| `--evaluate-every` | 20 | Intervalle des rollouts de validation ; 0 désactive les rollouts périodiques |
| `--no-rollouts` | Désactivé | Désactive aussi le rollout final |
| `--device` | `auto` | CUDA si disponible, CPU sinon |

Toute baisse de MSE de validation peut sauvegarder `best.pt`, même inférieure à `min-delta`. Un dossier de sortie non vide est refusé. Les checkpoints BC contiennent l'Actor, sans état d'optimiseur pour reprendre l'entraînement supervisé.

## 3. Transfert BC vers SAC

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation train-sac --init-bc runs/bc_gamepad/best.pt --output runs/sac_reach_v2 --critic-warmup-updates 1000 --device cuda
```

Le transfert copie l'Actor et sa distribution gaussienne. Les deux Critics, leurs cibles et les optimiseurs sont initialisés séparément. Alpha commence à la valeur configurée, `0.2` par défaut.

Le replay reçoit uniquement les démonstrations réussies de la partition d'apprentissage du checkpoint BC. Les signatures de scène, de commande et d'observation, le split et les empreintes des fichiers importés sont vérifiés. `--demonstrations` permet d'indiquer un autre emplacement du même dataset.

Sans `--config`, le transfert utilise les paramètres physiques et l'architecture du checkpoint BC. Une configuration explicite doit rester compatible avec ce checkpoint et ses données. La capacité du replay doit contenir les transitions importées et le batch ne peut pas dépasser leur nombre. Les démonstrations ne sont pas comptées comme des interactions SAC : `global_step` démarre à zéro.

### Préparation des Critics

Pendant les 1 000 premières mises à jour, par défaut :

- seuls les Critics et leurs réseaux cibles évoluent ;
- l'Actor et alpha sont gelés ;
- la collecte utilise l'action déterministe `tanh(mean)` ;
- les minibatches proviennent du replay contenant démonstrations et nouvelles interactions.

L'Actor et alpha reprennent ensuite leur optimisation, avec des actions stochastiques. `--critic-warmup-updates` compte des mises à jour, pas des décisions physiques ; 0 désactive cette préparation. `start_steps` et `update_after` passent à zéro car le replay est déjà alimenté.

### Seeds et sélection du modèle

La collecte utilise par défaut les 24 seeds d'apprentissage BC et la validation les 6 seeds réservées. Les listes explicites `train_seeds` et `validation_seeds` priment sur les intervalles de seeds. Les seeds BC réservées sont interdites à la collecte ; la validation exclut les seeds des démonstrations importées.

Une validation à la décision 0 évalue l'Actor copié. SAC sélectionne `best.pt` selon le taux de réussite, puis la reward moyenne. Une régression peut donc laisser la politique BC initiale comme meilleur checkpoint. `latest.pt` conserve l'état courant, même si sa performance est inférieure.

## 4. Entraînement SAC et reprise

### Entraînement sans BC

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation train-sac --config configs/training.json --output runs/sac_reach_scratch_v2 --device cuda
```

`configs/training.json` reproduit un pool historique réduit à la seed 42. Un apprentissage sur plusieurs placements nécessite une configuration d'expérience avec un pool plus large. `configs/smoke.json` fournit une configuration CPU courte pour vérifier le fonctionnement de la pipeline.

### Reprise d'un snapshot

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation train-sac --resume runs/sac_from_bc/latest.pt --steps 2000000 --device cuda
```

`--steps` est un budget total incluant les étapes déjà réalisées. La reprise restaure les réseaux, optimiseurs, replay, RNG, état physique et progression de la préparation des Critics. Elle utilise la configuration sauvegardée et ne recharge pas les démonstrations sources. `--resume` ne se combine pas avec `--config` ni les options de transfert BC.

## 5. Évaluation

```powershell
# Splits enregistrés dans un checkpoint BC.
.\.venv\Scripts\python.exe -m adaptive_manipulation evaluate runs/bc_gamepad/best.pt --split validation
.\.venv\Scripts\python.exe -m adaptive_manipulation evaluate runs/bc_gamepad/best.pt --split train

# Viewer MuJoCo sur les scénarios réservés.
.\.venv\Scripts\python.exe -m adaptive_manipulation evaluate runs/bc_gamepad/best.pt --split validation --render

# Scénarios inédits, pour BC puis SAC.
.\.venv\Scripts\python.exe -m adaptive_manipulation evaluate runs/bc_gamepad/best.pt --split new --episodes 100 --seed-start 4000000
.\.venv\Scripts\python.exe -m adaptive_manipulation evaluate runs/sac_from_bc/best.pt --episodes 100 --seed-start 4000000
```

L'évaluation utilise l'action déterministe. Les options `--split train` et `--split validation` concernent les checkpoints BC. Pour une évaluation inédite, les seeds doivent être exclues des données d'apprentissage et des pools de sélection du modèle. La comparaison BC/SAC utilise les mêmes seeds et paramètres physiques.

La MSE de validation mesure l'imitation sur des états enregistrés. Les rollouts mesurent le comportement de la politique sur les états qu'elle produit elle-même. Une faible MSE ne garantit pas une trajectoire réussie.

Sans `--csv`, un nouveau fichier daté est créé dans `runs/evaluations/`. Un chemin explicite est possible, par exemple `--csv runs/evaluations/sac_reach_test.csv` ; il est réécrit s'il existe. La fermeture du viewer interrompt la lecture et seuls les épisodes terminés sont exportés.

## 6. Artefacts et visualisation

| Fichier | Contenu |
|---|---|
| `config.json` | Configuration effective de l'expérience |
| `split.json` (BC) | Partitions et provenance du dataset |
| `metrics.csv` (BC) | MSE/MAE train et validation, globales et par canal, dès l'époque 0 |
| `summary.json` (BC) | Meilleure époque, arrêt et résultat du modèle sélectionné |
| `best_validation_episodes.csv` (BC) | Résultats physiques du meilleur Actor sur les seeds réservées |
| `episodes.csv` (SAC) | Résultats, seed, distance et cause de fin par épisode |
| `stats.csv` (SAC) | Losses, alpha, débit et phase d'apprentissage |
| `critics.csv`, `actions.csv` (SAC) | Diagnostics des réseaux et des commandes physiques |
| `validation.csv` | Évaluations périodiques utilisées pour suivre l'expérience |
| `best.pt` | BC : meilleure MSE de validation ; SAC : réussite puis reward |
| `latest.pt` | Actor BC courant ou état SAC complet pour reprise |
| `initialized.pt`, `bc_initialization.json` | État initial et provenance du transfert BC → SAC |

Une interruption BC peut sauvegarder un `latest.pt` marqué `interrupted=True`, plus récent que la dernière époque complètement mesurée. SAC sauvegarde son état courant lors d'une interruption normale.

```powershell
.\.venv\Scripts\python.exe -m adaptive_manipulation plot runs/bc_gamepad --save reports/bc_progression.png --no-show
.\.venv\Scripts\python.exe -m adaptive_manipulation plot runs/sac_from_bc --save reports/sac_progression.png --no-show
```

Le traceur distingue BC et SAC à partir des CSV. Pour SAC, une figure supplémentaire suffixée `_diagnostics` est produite lorsque les diagnostics sont disponibles. Les anciens runs restent lisibles sans les mesures absentes.

### Lecture des diagnostics SAC

| Mesure | Interprétation |
|---|---|
| `q1_mean`, `q2_mean`, `q_target_mean`, `q_target_std` | Valeurs des Critics et distribution des cibles sur le replay |
| `q_gap_mean`, `td_abs_mean` | Désaccord entre Critics et erreur TD absolue |
| `td_mse_terminal`, `td_mse_nonterminal` | Erreurs quadratiques conditionnelles, somme des deux Critics |
| `terminal_samples`, `nonterminal_samples` | Nombre d'échantillons contribuant aux erreurs conditionnelles |
| `entropy_mean`, `entropy_error`, `policy_std_0..3` | Entropie, écart à la cible et écart-type avant tanh |
| `policy_action_saturation_0..3` | Fraction d'actions de politique d'amplitude ≥ 0.95 sur le replay |
| `action_0..3`, `action_abs_0..3` | Moyennes signées et amplitudes pendant la collecte |
| `effective_action_0..3`, `action_clipped_0..3` | Incréments effectifs et fréquence de bornage des consignes |
| `action_saturated_0..3`, `target_at_limit_0..3` | Saturation des actions et présence des consignes aux limites |
| `tracking_error_0..3` | Écart consigne–position physique, radians pour le bras, mètres pour les doigts |
| `distance_below_30cm` | Fraction de décisions à moins de 30 cm du cube |
| `phase`, `actor_update_fraction`, `critic_warmup_remaining` | Progression de la préparation et activation de l'Actor |

Les diagnostics des réseaux utilisent des échantillons du replay, avec remise ; ceux des commandes portent sur les interactions collectées. Les erreurs terminales comprennent réussites et échecs. Une mesure conditionnelle vide indique l'absence d'échantillons correspondants. Pendant la préparation des Critics, `actor_loss` mesure un objectif candidat sans mettre à jour l'Actor.

## Compatibilité et provenance

Les signatures contrôlent le XML, les dimensions et les paramètres de commande. Le contrat actuel porte `observation_version=2` et `task_version=2`. Les anciens modèles à 36 observations restent des artefacts d'analyse et ne pilotent pas l'environnement à 38 observations.

Une nouvelle tâche, une nouvelle définition de réussite ou un changement de commande nécessite une revue de compatibilité des checkpoints et des démonstrations. Les règles de versionnement sont décrites dans [Architecture](architecture.md).
