# Rapport d'Analyse — Régression Linéaire sur les Habitudes Étudiantes
*Implémentation from scratch · Gradient Descent · Analyse comparative*

---

## 1. Introduction et Contexte

Ce rapport présente une analyse complète d'un projet de régression linéaire appliqué à la prédiction des scores d'examens d'étudiants (`exam_score`). L'objectif est double :

- **Implémenter de zéro** (sans scikit-learn) les briques fondamentales d'un modèle de régression linéaire multiple — prédiction, calcul de coût et descente de gradient.
- **Comparer deux approches** de feature engineering : le modèle linéaire standard (Modèle 1) et un modèle avec ajout de termes quadratiques sur toutes les variables (Modèle 2).

Le jeu de données utilisé est `student_habits_performance.csv`, contenant **1 000 observations** et **16 variables** décrivant les habitudes de vie et le contexte académique d'étudiants.

---

## 2. Pipeline de Traitement des Données

### 2.1 Présentation du Dataset

| Variables | Type |
|---|---|
| `age`, `study_hours_per_day`, `social_media_hours`, `netflix_hours`, `attendance_percentage`, `sleep_hours`, `exercise_frequency`, `mental_health_rating` | Numériques continues / discrètes |
| `gender`, `part_time_job`, `extracurricular_participation`, `diet_quality`, `parental_education_level`, `internet_quality` | Catégorielles (ordinales et nominales) |
| `exam_score` | Variable cible (continue, 0–100) |

### 2.2 Gestion des Valeurs Manquantes

L'audit des données a révélé une seule colonne incomplète :

- `parental_education_level` : **91 valeurs manquantes**, imputées par le **mode** (valeur la plus fréquente).

> **Pertinence du choix :** L'imputation par le mode est appropriée pour une variable catégorielle ordinale. Elle préserve la distribution sans introduire de biais majeur sur un taux de manque de 9,1 %, acceptable en pratique.

### 2.3 Feature Engineering — Variables carrées

Pour le **Modèle 1**, les 8 variables numériques d'origine sont complétées par leur carré respectif, donnant **16 features numériques** au total. L'objectif est de capturer d'éventuelles relations non-linéaires (effet de diminution, seuil, etc.) tout en restant dans le cadre linéaire du point de vue de l'optimisation.

Pour le **Modèle 2** (« poly »), cette expansion par carrés est étendue à la totalité des 22 variables (y compris les variables catégorielles encodées), doublant ainsi l'espace de features à **44 dimensions**.

### 2.4 Encodage des Variables Catégorielles

Deux stratégies d'encodage ont été appliquées :

- **Encodage ordinal :** `diet_quality` (Poor/Fair/Good → 0/1/2), `parental_education_level` (High School/Bachelor/Master → 0/1/2), `internet_quality` (Poor/Average/Good → 0/1/2).
- **Encodage binaire :** `gender` (0/1/2), `part_time_job` (Non/Oui → 0/1), `extracurricular_participation` (Non/Oui → 0/1).

> **Remarque critique :** L'encodage ordinal de `gender` (Female=0, Male=1, Other=2) impose implicitement un ordre qui n'a pas de sens sémantique. Un encodage **one-hot** aurait été plus rigoureux pour cette variable nominale.

### 2.5 Normalisation (Z-score)

Toutes les features sont standardisées (mean=0, std=1) avant l'entraînement. Cette étape est indispensable pour la convergence de la descente de gradient : sans normalisation, les features à grande échelle (ex. : `attendance_percentage²` ≈ 7 000) domineraient le gradient et rendraient l'optimisation instable.

### 2.6 Découpage Train / Test

Le dataset est mélangé aléatoirement (`random.seed(42)`) puis divisé en :

- **Entraînement :** 80 % → 800 observations
- **Test :** 20 % → 200 observations

L'usage d'une graine fixée assure la reproductibilité des expériences.

---

## 3. Implémentation des Algorithmes

### 3.1 Modèle de Régression Linéaire

La prédiction est calculée selon le modèle linéaire classique :

```
ŷ = w₁x₁ + w₂x₂ + … + wₙxₙ + b
```

L'implémentation est réalisée en pur Python (boucles `for`), sans vectorisation NumPy pour le calcul de prédiction.

### 3.2 Fonction de Coût — MSE

L'erreur quadratique moyenne est utilisée comme fonction de coût :

```
MSE = (1/m) × Σ(yᵢ − ŷᵢ)²
```

### 3.3 Descente de Gradient (Batch Gradient Descent)

Les paramètres sont mis à jour itérativement selon les gradients partiels :

```
∂MSE/∂wⱼ = (2/m) × Σ(ŷᵢ − yᵢ) × xᵢⱼ
∂MSE/∂b  = (2/m) × Σ(ŷᵢ − yᵢ)
```

**Hyperparamètres :** α = 0,01 | n_iter = 2 000 — choix identique pour les deux modèles, permettant une comparaison équitable.

### 3.4 Métrique complémentaire — R² (Coefficient de Détermination)

```
R² = 1 − SS_res / SS_tot
```

Un R² proche de 1 indique un bon ajustement. Il mesure la proportion de variance de `exam_score` expliquée par le modèle.

---

## 4. Résultats et Analyse Comparative

### 4.1 Tableau Comparatif des Métriques

| Métrique | Modèle 1 (linéaire) | Modèle 2 (+ carrés de toutes vars.) | Comparaison |
|---|---|---|---|
| **MSE — Entraînement** | 27,95 | 25,92 | Modèle 2 meilleur |
| **MSE — Test** | 26,49 | 28,62 | ⚠️ Modèle 1 meilleur |
| **R² — Entraînement** | 0,9041 | 0,9111 | Modèle 2 meilleur |
| **R² — Test** | 0,8975 | 0,8893 | ⚠️ Modèle 1 meilleur |
| **Écart R² (train − test)** | 0,0066 | 0,0218 | Modèle 2 = plus d'overfit |
| **Nb de features** | 22 | 44+ | Modèle 2 = 2× plus complexe |

### 4.2 Analyse des Résultats

#### Modèle 1 — Linéaire avec carrés des variables numériques

Le Modèle 1 réalise un **R² test de 0,8975**, soit 89,75 % de variance expliquée. L'écart entre train (0,9041) et test (0,8975) est très faible (Δ = 0,0066), signe d'une excellente généralisation sans overfitting significatif.

La MSE test (26,49) est légèrement inférieure à la MSE train (27,95), ce qui est un phénomène normal et positif : il indique que le modèle n'a pas mémorisé le bruit d'entraînement.

#### Modèle 2 — Poly (carrés de toutes les variables)

Le Modèle 2 obtient un meilleur R² train (0,9111) mais un **R² test inférieur (0,8893)**. Le gap de 0,0218 points est trois fois plus élevé que pour le Modèle 1. La MSE test est également plus élevée (28,62 vs 26,49).

Ces observations convergent vers un diagnostic clair : **le Modèle 2 souffre d'overfitting modéré**. En ajoutant les carrés des variables catégorielles encodées (binaires 0/1), on crée des features redondantes (x² = x pour x ∈ {0,1}) qui n'apportent aucune information supplémentaire mais augmentent la dimensionnalité et le risque de mémorisation.

### 4.3 Importance des Variables (Modèle 1)

L'analyse des poids absolus après convergence révèle la hiérarchie d'importance des features :

| Rang | Feature | \|Poids\| | Interprétation |
|---|---|---|---|
| 1 | `study_hours_per_day` | 15,14 | Effet dominant et attendu |
| 2 | `mental_health_rating` | 4,32 | Bien-être → performance |
| 3 | `social_media_hours` | 3,78 | Impact négatif probable |
| 4 | `netflix_hours` | 2,41 | Distraction (rôle similaire aux réseaux sociaux) |
| 5 | `exercise_frequency` | 2,40 | Corrélé au bien-être général |
| 6 | `sleep_hours` | 1,51 | Qualité du sommeil → capacité cognitive |
| 7 | `study_hours_per_day²` | 1,27 | Rendement décroissant possible |
| 8 | `mental_health_rating²` | 1,10 | Effet non-linéaire du bien-être |
| … | `gender`, `diet_quality`, `netflix²`… | < 0,10 | Contribution marginale |

Les features à faible importance pourraient être élaguées pour simplifier le modèle sans perte significative de performance.

---

## 5. Pertinence des Méthodes Utilisées

### 5.1 Points Forts de l'Approche

- **Implémentation from scratch :** la réécriture des fonctions (prédiction, MSE, gradients, R²) démontre une compréhension fine du fonctionnement interne du gradient descent, pédagogiquement solide.
- **Normalisation systématique :** le z-score est appliqué correctement et indépendamment pour chaque ensemble, évitant la fuite d'information (data leakage).
- **Shuffle + seed :** la graine fixée garantit la reproductibilité, crédibilisant l'expérience.
- **Feature engineering par polynomial :** l'ajout de carrés est une technique reconnue pour enrichir un modèle linéaire et capturer des courbures sans changer d'algorithme.

### 5.2 Limites et Points d'Amélioration

- **Encodage de `gender` :** variable nominale encodée en ordinal (0/1/2), ce qui impose une distance artificielle entre les modalités. Un one-hot encoding serait plus correct.
- **Batch Gradient Descent complet :** toute la donnée d'entraînement est utilisée à chaque itération. Sur 800 exemples c'est acceptable, mais le mini-batch SGD serait plus scalable.
- **Pas de validation croisée :** un seul split 80/20 peut être sensible à la graine. Une validation croisée k-fold donnerait une estimation plus robuste.
- **Carrés de variables catégorielles (Modèle 2) :** x² = x pour x ∈ {0,1} — information redondante qui gonfle la dimensionnalité sans bénéfice.
- **Absence de régularisation :** aucune pénalité L1/L2 n'est appliquée. Avec 44 features dans le Modèle 2, une régularisation Ridge aurait significativement réduit l'overfitting.

---

## 6. Ce qu'on Aurait Obtenu avec d'Autres Optimisations

### 6.1 Équation Normale (Solution Analytique)

Plutôt que de converger itérativement, l'équation normale donne la solution optimale en une étape :

```
W* = (XᵀX)⁻¹ Xᵀy
```

| Critère | Éq. normale | Gradient Descent |
|---|---|---|
| Convergence | Garantie exacte en une étape | Dépend du taux et nb d'itérations |
| Coût calcul. | O(n³) pour l'inversion | O(n×p) par itération |
| Scalabilité | Lent si n ou p est grand (> ~10 000) | Scalable avec mini-batch |
| Implémentation | Simple (une formule) | Nécessite tuning du lr |

Pour 800 observations et 22 features, l'équation normale aurait donné la solution **exacte** et probablement une légère amélioration des métriques (le gradient descent peut ne pas avoir totalement convergé en 2 000 itérations avec α=0,01).

### 6.2 Gradient Descent Stochastique (SGD)

Le SGD met à jour les poids après chaque exemple (ou mini-batch) :

- Convergence plus rapide en nombre d'époques sur de grands datasets.
- Le bruit des mises à jour peut aider à échapper à des minima locaux (utile pour les réseaux de neurones, moins pour la régression linéaire à coût convexe).
- Nécessite un **learning rate scheduling** (décroissant) pour converger précisément.

Sur notre dataset (800 exemples, problème convexe), l'amélioration serait marginale par rapport au batch GD.

### 6.3 Ridge Regression (L2)

La régularisation Ridge ajoute une pénalité sur la norme des poids :

```
J(w) = MSE + λ‖w‖²
```

**Impact attendu sur le Modèle 2 :** en pénalisant les grands poids, Ridge aurait réduit l'impact des features redondantes (carrés de variables catégorielles) et probablement rapproché le R² test du Modèle 1, voire le dépassé, tout en conservant l'expressivité des 44 features.

### 6.4 Lasso Regression (L1)

Le Lasso produit des solutions **parcimonieuses** (poids nuls sur les features non informatives) :

```
J(w) = MSE + λ‖w‖₁
```

Avec 44 features dont plusieurs redondantes, Lasso aurait automatiquement **sélectionné les features pertinentes** en mettant à zéro les poids des carrés de variables binaires — résolvant le problème du Modèle 2 sans intervention manuelle.

### 6.5 Récapitulatif des Alternatives

| Méthode | R² test estimé | Remarque |
|---|---|---|
| Batch GD (implémenté) | ≈ 0,898 | Pédagogique, convergence lente |
| Éq. Normale | ≈ 0,898–0,900 | Optimal sur ce dataset |
| Ridge (λ optimisé) | ≈ 0,895–0,905* | Robuste à la redondance |
| Lasso | ≈ 0,890–0,902* | Sélection automatique de features |
| SGD | ≈ 0,895–0,900 | Scalable, résultat similaire |

*\* Estimation : les performances exactes dépendent du λ optimal déterminé par cross-validation.*

---

## 7. Conclusion

Ce projet démontre la maîtrise des fondamentaux de la régression linéaire — de la préparation des données jusqu'à l'évaluation comparative. Les enseignements clés :

- **L'implémentation from scratch** avec des boucles Python est correcte et éducative. Une vectorisation NumPy (produit matriciel) serait nettement plus rapide.
- **Le Modèle 1** (carrés sur variables numériques uniquement) est le **meilleur choix** : R² test de 0,8975, généralisation stable, overfitting négligeable.
- **Le Modèle 2** illustre parfaitement le danger d'ajouter des features sans discernement : les carrés de variables binaires n'apportent rien et dégradent la généralisation.
- **`study_hours_per_day`** est de loin la variable la plus prédictive (poids ≈ 15), suivie du `mental_health_rating`, confirmant l'importance des facteurs comportementaux et psychologiques dans la réussite académique.
- L'ajout d'une **régularisation Ridge ou Lasso** serait la prochaine amélioration à envisager, en particulier pour le Modèle 2.

---

*Rapport généré — Université d'Abomey-Calavi*
