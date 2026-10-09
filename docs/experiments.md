# Expériences et résultats

[Présentation du projet](../README.md) · [Pipeline](pipeline.md) · [Architecture](architecture.md)

Les expériences documentent la progression de la tâche `reach` : atteindre un cube par contact avec la pince. Les résultats proviennent des CSV, configurations et métriques conservés dans le dépôt de travail. Les nombres ci-dessous décrivent les exécutions enregistrées ; ils ne constituent pas une garantie de performance sur d'autres distributions de scénarios.

## Protocole

Une seed détermine le placement initial du cube. Une tentative se termine au contact actif avec la pince, à l'échéance de 20 secondes ou à la limite de décisions. Le taux de réussite utilise le critère de contact ; il ne mesure pas une prise stable.

Trois ensembles ont des fonctions distinctes :

| Ensemble | Fonction |
|---|---|
| Apprentissage | Démonstrations ou interactions utilisées pour optimiser les réseaux |
| Validation | Sélection du checkpoint et suivi de l'entraînement |
| Évaluation indépendante | Mesure sur des seeds exclues de l'apprentissage et de la sélection |

Les rollouts d'évaluation utilisent des actions déterministes. Les résultats BC et SAC sont comparables lorsque les paramètres physiques, le critère de réussite et les seeds sont identiques. La validation peut guider le développement ; elle ne remplace pas une évaluation indépendante.

## 1. SAC initial sans démonstrations

L'expérience `runs/sac_fixed_seed42` utilise une seed d'entraînement unique, 42. L'audit conservé porte sur un checkpoint de 80 000 décisions, 80 épisodes et 79 001 mises à jour, avec l'ancien contrat à 36 observations.

| Mesure auditée | Valeur |
|---|---:|
| Réussites pendant les 80 épisodes d'entraînement | 0 |
| Taille du replay | 80 000 transitions |
| Transitions terminales enregistrées | 0 |
| Épisodes enregistrés comme tronqués | 80 |
| Distance moyenne pince–cube dans le replay | 1.171 m |
| Distance minimale dans le replay | 0.157 m |
| Transitions à moins de 30 cm | 0.1875 % |
| Transitions à moins de 20 cm | 0.05625 % |
| Transitions à moins de 10 cm | 0 % |

Le replay contient très peu d'états proches du cube et aucun exemple de réussite. L'ancienne gestion de l'échéance enregistrait les échecs comme des troncatures, permettant le bootstrap SAC malgré la pénalité d'échec. Le budget restant n'était pas observé.

Ces constats motivent deux évolutions du contrat : l'ajout du temps et des décisions restants dans l'observation, et le traitement des échéances comme des échecs terminaux. La collecte humaine fournit ensuite des trajectoires d'approche et des transitions de réussite.

Sources : [mesures de l'audit](../reports/sac_audit_metrics.json), [épisodes](../runs/sac_fixed_seed42/episodes.csv), [validation](../runs/sac_fixed_seed42/validation.csv). La [source historique de l'audit](archive/audit_sac_v1.py) correspond au contrat initial ; elle sert de référence et n'est pas une commande de la pipeline actuelle.

## 2. Dataset de démonstrations

La session `demonstrations/gamepad_30` contient 30 scénarios terminés avec succès, sur les seeds 3000000 à 3000029, soit 9 067 transitions. Les commandes humaines sont enregistrées dans le même espace d'action que SAC.

Le split BC, reproductible avec la seed 42, réserve six scénarios entiers :

| Partition | Scénarios | Transitions |
|---|---:|---:|
| Apprentissage | 24 | 7 312 |
| Validation | 6 | 1 755 |

Les épisodes restent entiers dans chaque partition pour limiter la fuite d'information entre états voisins d'une même trajectoire.

Sources : [session](../demonstrations/gamepad_30/session.json), [split BC](../runs/bc_gamepad/split.json).

## 3. Behavior cloning

L'expérience `runs/bc_gamepad` entraîne l'Actor sur les démonstrations réussies. Le checkpoint est sélectionné à partir de la MSE des actions sur les six épisodes réservés.

| Mesure | Résultat |
|---|---:|
| Meilleure époque | 16 |
| Époques terminées | 46 |
| Meilleure MSE de validation | 0.035287 |
| Réussites en rollout avec le meilleur Actor | 6 / 6 |

La MSE évalue l'imitation sur les états humains enregistrés. Les six rollouts évaluent l'Actor sur ses propres trajectoires dans les scénarios réservés. Le succès sur ces scénarios établit le fonctionnement de la politique de contact dans ce protocole ; l'échantillon reste réduit et intervient dans la sélection du modèle.

Sources : [bilan BC](../runs/bc_gamepad/summary.json), [métriques d'imitation](../runs/bc_gamepad/metrics.csv), [rollouts du meilleur checkpoint](../runs/bc_gamepad/best_validation_episodes.csv).

## 4. SAC initialisé depuis BC

L'expérience `runs/sac_from_bc` importe l'Actor BC et les démonstrations de sa partition d'apprentissage. Les Critics sont préparés avant l'activation des mises à jour de l'Actor et d'alpha. Les six seeds BC réservées restent utilisées pour la validation.

La dernière ligne consignée dans `validation.csv`, à **90 000 décisions** et **549 épisodes**, donne :

| Mesure de validation | Valeur |
|---|---:|
| Taux de réussite | 100 % |
| Reward moyenne | 231.72 |
| Distance finale moyenne | 0.127 m |

Cette ligne décrit une évaluation périodique du run. Elle ne prouve pas à elle seule que SAC améliore la réussite du BC : la référence BC réussit déjà les six mêmes scénarios, et le meilleur checkpoint SAC peut conserver la politique initiale si les mises à jour suivantes régressent.

Sources : [configuration SAC](../runs/sac_from_bc/config.json), [provenance du transfert](../runs/sac_from_bc/bc_initialization.json), [validation](../runs/sac_from_bc/validation.csv), [épisodes](../runs/sac_from_bc/episodes.csv).

## 5. Évaluation exportée

Le fichier [evaluation.csv](../evaluation.csv) contient 10 scénarios, seeds 2000000 à 2000009, tous terminés par contact réussi. Son schéma enregistre seed, nombre de décisions, reward, réussite, distance, temps simulé et cause de fin.

Ce CSV ne contient pas l'identité du checkpoint évalué. Il est conservé comme résultat exporté, sans attribution automatique à une politique BC ou SAC ni utilisation comme comparaison entre modèles. Une expérience comparative doit associer explicitement chaque export au checkpoint, à sa configuration et au protocole de seeds.

## Portée des résultats et suite expérimentale

La tâche d'approche dispose d'un dataset, d'une politique d'imitation et d'un transfert vers le renforcement. Les validations enregistrées montrent des contacts réussis sur les scénarios de référence. Les axes suivants concernent la généralisation et l'extension de la tâche :

- comparaison BC/SAC sur un ensemble commun de seeds inédites ;
- répétition des entraînements avec plusieurs seeds d'initialisation ;
- mesure du temps de contact, des échecs et des saturations de commande ;
- définition d'une prise stable, puis d'un critère de levage et de transport ;
- collecte et évaluation de données propres à ces nouvelles tâches.

Les métriques de saisie et de transport devront être distinctes du simple contact utilisé dans `reach`.

## Traçabilité des artefacts

Chaque run conserve sa configuration, ses métriques et ses checkpoints. Les démonstrations disposent d'une signature physique et d'empreintes utilisées lors du transfert. Le fichier [reorganisation_verification.json](../reports/reorganisation_verification.json) conserve le contrôle d'intégrité des artefacts effectué lors de la réorganisation du dépôt ; il documente une opération de maintenance, pas une mesure de performance de la politique.
