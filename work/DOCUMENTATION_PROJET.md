# UNBOXED — Pipeline Agentique de Radiologie
## Documentation complète du projet

*Hackathon UNBOXED 2026 — GE Healthcare × Centrale Lyon*

---

### Table des matières

1. [Vision et Problème](#1-vision-et-problème)
2. [Architecture globale](#2-architecture-globale)
3. [Intelligence Clinique — Le cœur de l'innovation](#3-intelligence-clinique--le-cœur-de-linnovation)
   - 3.1 [Pourquoi le rapport dépend du contexte clinique](#31-pourquoi-le-rapport-dépend-du-contexte-clinique)
   - 3.2 [Détection automatique du scénario](#32-détection-automatique-du-scénario)
   - 3.3 [La règle des 10mm](#33-la-règle-des-10mm-lésion-cible-vs-non-cible)
   - 3.4 [Le suivi lésion par lésion](#34-le-suivi-lésion-par-lésion-f1f1)
   - 3.5 [Le Temps de Doublement Volumétrique](#35-le-temps-de-doublement-volumétrique-vdt)
   - 3.6 [Le raisonnement RECIST transparent](#36-le-raisonnement-recist-transparent)
4. [Système de Confiance et Traçabilité](#4-système-de-confiance-et-traçabilité)
   - 4.1 [Confidence Score](#41-confidence-score)
   - 4.2 [Données manquantes explicites](#42-données-manquantes-explicites)
   - 4.3 [Traçabilité des sources](#43-traçabilité-des-sources)
   - 4.4 [Vérification anti-hallucination](#44-vérification-anti-hallucination)
   - 4.5 [Auto-évaluation qualité](#45-auto-évaluation-qualité)
5. [Architecture Agentique (MCP)](#5-architecture-agentique-mcp)
   - 5.1 [Qu'est-ce que MCP](#51-quest-ce-que-mcp)
   - 5.2 [Nos 5 tools MCP](#52-nos-5-tools-mcp)
   - 5.3 [Comment ça se connecte](#53-comment-ça-se-connecte)
6. [Résultats et Validation](#6-résultats-et-validation)
   - 6.1 [Patient test : 063F6BB9](#61-patient-test--063f6bb9-accession-10969511)
   - 6.2 [Tableau récapitulatif de l'orchestrateur](#62-tableau-récapitulatif-de-lorchestratuer)
   - 6.3 [Analyse du rapport de vérification](#63-analyse-du-rapport-de-vérification)
   - 6.4 [Tests sur d'autres patients](#64-tests-sur-dautres-patients)
7. [Processus de test de la solution](#7-processus-de-test-de-la-solution)
8. [Éthique et Sécurité](#8-éthique-et-sécurité)
9. [Stack technique](#9-stack-technique)
10. [Perspectives et Part 2](#10-perspectives-et-part-2)

---

## 1. Vision et Problème

Un radiologue hospitalier analyse en moyenne 50 à 100 dossiers par jour, chacun contenant des dizaines voire des centaines d'images médicales. En oncologie pulmonaire, le suivi longitudinal est particulièrement exigeant : le médecin doit comparer les mesures de chaque nodule d'un examen à l'autre, appliquer des critères internationaux (RECIST 1.1), calculer des variations, et rédiger un rapport structuré — le tout sous pression temporelle constante. Le risque de burnout est réel, et les erreurs de mesure ou d'interprétation ont des conséquences directes sur les décisions thérapeutiques.

**Le cas concret** : une patiente de 51 ans, suivie pour un cancer pulmonaire dans le cadre d'un essai clinique, a 4 nodules identifiés et 3 examens CT à comparer. Le radiologue doit manuellement retrouver les mesures antérieures, calculer les pourcentages de variation, vérifier les seuils RECIST, et produire un rapport conforme aux standards internationaux. Ce travail fastidieux prend 15 à 30 minutes par patient.

**Notre proposition** : un agent IA qui automatise la chaîne complète — du téléchargement des images DICOM à la génération d'un rapport structuré — en 33 secondes. L'agent comprend le contexte clinique, applique les bons critères, et produit un brouillon de rapport que le radiologue n'a plus qu'à valider et signer.

**Ce que notre solution N'EST PAS** : un remplaçant du radiologue. C'est un assistant intelligent qui prépare le travail, signale explicitement ses limites, et garde le médecin dans la boucle à chaque étape. Chaque rapport porte un avertissement IA et est conçu comme un brouillon à valider.

---

## 2. Architecture globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PIPELINE UNBOXED — 12 étapes                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Orthanc PACS] ──→ [1. Discovery] ──→ [2. Download CT]            │
│                                              │                      │
│                                    [3. Segmentation AI]             │
│                                              │                      │
│                                    [4. Analyse SEG]                 │
│                                     (diamètres, volumes)            │
│                                              │                      │
│  [Excel Clinique] ──→ [5. Contexte Clinique]                        │
│                         ├─ Classification scénario A/B              │
│                         ├─ Cible vs Non-cible (10mm)                │
│                         ├─ Tracking F1↔F1                           │
│                         ├─ RECIST / Lung-RADS                       │
│                         ├─ VDT (doublement volumétrique)            │
│                         ├─ Confidence scores                        │
│                         └─ Audit données manquantes                 │
│                                              │                      │
│                                   [6. Génération Rapport]           │
│                                     (Gemini 2.5 Flash)              │
│                                              │                      │
│                              [7. Vérification Anti-Hallucination]   │
│                                              │                      │
│                              [8. Auto-Évaluation Qualité]           │
│                                              │                      │
│                              [9. Sauvegarde + Résumé]               │
│                                              │                      │
│                         [Validation Radiologue] ◄── Human-in-loop   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Étapes 1-2 — Discovery & Download** : La fonction `step_discovery()` interroge l'API REST d'Orthanc pour localiser l'étude par AccessionNumber, identifie la série CT cible (préférence pour "CEV torax"), puis `step_download()` télécharge les coupes DICOM.

**Étape 3 — Segmentation** : `step_segmentation()` lance l'algorithme de segmentation de GE Healthcare (`dcm_seg_nodules.extract_seg`) qui produit un fichier DICOM SEG contenant les masques de chaque nodule identifié.

**Étape 4 — Analyse SEG** : `step_analyse_seg()` lit le fichier SEG avec pydicom, calcule pour chaque nodule le diamètre axial maximal, le volume en mm³, le diamètre sphérique équivalent, et compte les coupes contenant le nodule.

**Étape 5 — Intelligence Clinique** : C'est le cœur du système. Neuf sous-étapes (5a à 5l) construisent le contexte médical complet : scénario clinique, classification des lésions, historique lésionnel, calcul RECIST, VDT, raisonnement transparent, audit de données et traçabilité des sources.

**Étape 6 — Rapport** : `step_generate_report()` injecte toutes les données pré-calculées dans un prompt adapté au scénario clinique et appelle Gemini 2.5 Flash pour générer le rapport en français médical.

**Étapes 7-8 — Vérification** : `verify_report()` effectue un second pass automatique pour détecter les hallucinations, et `evaluate_report_quality()` évalue la complétude du rapport sur 10 critères.

---

## 3. Intelligence Clinique — Le cœur de l'innovation

### 3.1 Pourquoi le rapport dépend du contexte clinique

Un même nodule pulmonaire de 12mm ne génère pas le même rapport selon le contexte. Chez un patient en dépistage (sans antécédent oncologique), ce nodule sera classifié Lung-RADS 4A avec une recommandation de PET-CT à 3 mois. Chez un patient sous chimiothérapie pour un cancer connu, ce même nodule sera une lésion cible RECIST dont la variation par rapport à la baseline déterminera si le traitement fonctionne.

Un radiologue humain fait cette distinction immédiatement en lisant le motif de la demande d'examen. Notre agent reproduit ce raisonnement en analysant le contexte clinique avant de générer le rapport.

### 3.2 Détection automatique du scénario

La fonction `classify_clinical_context()` analyse les rapports cliniques du patient dans l'Excel pour détecter des mots-clés oncologiques. Le système recherche 17 termes spécifiques : *neoplasia*, *clinical trial*, *psa*, *recist*, *metast*, *progression*, *chemotherapy*, *oncolog*, *carcinoma*, *tumor*, *adenocarcinoma*, *malignant*, *stage iv*, *stage iii*, entre autres.

- **Scénario B** (oncologie) : Au moins un mot-clé détecté. Critères RECIST 1.1 appliqués. C'est le cas du patient 063F6BB9 (evidence : *neoplasia*, *clinical trial*) et des patients 0301B7D6 et 136F50A1.
- **Scénario A** (dépistage) : Aucun mot-clé oncologique. Critères Lung-RADS / Fleischner appliqués. C'est le cas du patient 17A76C2A.

Le sous-cas "Découverte" (1er examen, pas d'historique) versus "Suivi" (examens multiples) est géré automatiquement par la présence ou l'absence de données dans le tableau de tracking.

### 3.3 La règle des 10mm (Lésion Cible vs Non-Cible)

RECIST 1.1 impose un seuil de 10mm pour qu'un nodule soit qualifié de "lésion cible" — car en dessous, la variabilité inter-observateur rend les mesures peu fiables. De plus, au maximum 2 lésions cibles par organe sont autorisées.

La fonction `classify_findings()` trie les nodules par taille décroissante et attribue le statut "LÉSION CIBLE" aux deux plus grands au-dessus de 10mm. Les autres sont "LÉSION NON-CIBLE" (≥6mm) ou "NODULE INFRA-CENTIMÉTRIQUE".

**Résultat réel pour notre patient test** :
- F1 (22.61mm) → LÉSION CIBLE
- F2 (10.80mm) → LÉSION CIBLE
- F3 (8.45mm) → LÉSION NON-CIBLE
- F4 (6.98mm) → LÉSION NON-CIBLE

### 3.4 Le suivi lésion par lésion (F1↔F1)

Le principe fondamental : le nodule F1 de l'examen 1 correspond au même nodule F1 de l'examen 3. Cette correspondance permet de suivre l'évolution de chaque lésion dans le temps.

La fonction `build_lesion_tracking_table()` reconstruit l'historique complet en extrayant les tailles depuis l'Excel pour chaque examen antérieur, en triant chronologiquement (l'examen actuel toujours en dernier), et en calculant les variations inter-examens.

**Tableau d'évolution réel du patient 063F6BB9** :

| Lésion | Exam 1 | Exam 2 | Actuel | Tendance |
|--------|--------|--------|--------|----------|
| F1 | 21.3mm | 22.8mm (+7%) | 27.4mm (+20%) | PROGRESSION |
| F2 | 13.6mm | 14.8mm (+9%) | 14.7mm (-1%) | STABLE |
| F3 | 13.2mm | 13.3mm (+1%) | 10.7mm (-20%) | DIMINUTION |
| F4 | 9.4mm | 13.6mm (+45%) | 9.2mm (-32%) | DIMINUTION |

### 3.5 Le Temps de Doublement Volumétrique (VDT)

Le VDT est un indicateur critique en oncologie. Il estime le temps nécessaire pour qu'un nodule double de volume, en utilisant la formule : **VDT = (Δt × ln(2)) / ln(V2/V1)**, où V = (4/3)π(d/2)³.

La fonction `compute_vdt()` calcule le VDT et le classe selon les seuils cliniques :
- **< 200 jours** : Croissance rapide — hautement suspect de malignité
- **200-400 jours** : Croissance modérée — suspect
- **400-600 jours** : Croissance lente — indéterminé
- **≥ 600 jours** : Croissance très lente — probablement bénin
- **Négatif** : Régression volumétrique

**Résultats réels** :
- **F1 : VDT = 113 jours** — Croissance rapide, hautement suspect de malignité
- F2 : VDT = -3067 jours — Régression volumétrique
- F3 : VDT = -96 jours — Régression volumétrique
- F4 : VDT = -53 jours — Régression volumétrique

Le VDT de F1 à 113 jours confirme quantitativement la nature agressive de cette lésion, renforçant le diagnostic RECIST. C'est un indicateur rarement calculé automatiquement — un véritable avantage différenciant.

### 3.6 Le raisonnement RECIST transparent

La fonction `format_recist_reasoning()` construit un bloc de raisonnement complet qui montre chaque étape du calcul RECIST, afin que le radiologue puisse vérifier la logique sans refaire les calculs.

**Calcul réel pour notre patient** :
- Somme des diamètres cibles actuelle : F1(27.4mm) + F2(14.7mm) = **42.1mm**
- Baseline (exam 1) : F1(21.3mm) + F2(13.6mm) = **34.9mm**
- Nadir = 34.9mm (identique à la baseline)
- Variation vs baseline : (42.1 - 34.9) / 34.9 × 100 = **+20.6%**
- Variation vs nadir : **+20.6%** (augmentation absolue : 7.2mm)
- Seuil PD : ≥ +20% ET ≥ +5mm absolu → **OUI**
- **Évaluation : PD (Progressive Disease)**

Ce raisonnement est injecté tel quel dans le prompt et reproduit dans le rapport final, créant une transparence totale.

---

## 4. Système de Confiance et Traçabilité

### 4.1 Confidence Score

Chaque nodule reçoit un score de confiance sur 8 points, calculé par `compute_confidence()` sur 3 axes :

- **Qualité de mesure (0-3)** : nombre de coupes ≥ 3 (+1), nombre de voxels ≥ 50 (+1), pixel spacing ≤ 1mm (+1)
- **Cohérence temporelle (0-3)** : existence dans un examen précédent (+1), variation ≤ 30% vs précédent (+1), suivi sur ≥ 2 examens (+1)
- **Complétude des données (0-2)** : volume disponible (+1), données Excel disponibles (+1)

**Scores réels** :
| Lésion | Score | Niveau | Point manquant |
|--------|-------|--------|----------------|
| F1 | 8/8 | HAUTE | — |
| F2 | 8/8 | HAUTE | — |
| F3 | 7/8 | HAUTE | n_slices < 3 (1 seule coupe) |
| F4 | 5/8 | MOYENNE | n_slices < 3, n_voxels < 50, variation > 30% |

Le radiologue sait immédiatement que F4 mérite une attention particulière lors de sa relecture.

### 4.2 Données manquantes explicites

Le principe cardinal : **ne jamais inventer, toujours signaler l'absence**. La fonction `audit_data_completeness()` vérifie systématiquement la disponibilité de chaque catégorie de données.

**Données manquantes identifiées pour notre patient** :
- Injection de contraste : non spécifié dans les métadonnées DICOM
- Localisation anatomique des nodules : non disponible (segmentation sans mapping lobaire)
- Latéralité des lésions : non précisée par l'algorithme
- Densité des nodules : non évaluée (épaisseur de coupe de 5.0mm limite la caractérisation)

En médecine, une fausse certitude peut conduire à une mauvaise décision thérapeutique. Un rapport qui dit clairement "je ne sais pas" est plus fiable qu'un rapport qui invente.

### 4.3 Traçabilité des sources

Chaque donnée du rapport est taggée avec sa source d'origine via un système de balises :
- **[SRC:DICOM]** — Métadonnées extraites directement des fichiers DICOM (patient ID, âge, sexe, modalité, constructeur, date)
- **[SRC:SEG]** — Mesures issues de la segmentation AI (diamètres, volumes, nombre de coupes)
- **[SRC:EXCEL]** — Données historiques et contexte clinique depuis le dossier patient
- **[SRC:CALC]** — Valeurs calculées par le pipeline (variations RECIST, VDT, sommes)
- **[SRC:ALGO]** — Décisions algorithmiques (classification du scénario clinique)

Le fichier `sources_trace.json` sauvegardé pour chaque patient contient l'horodatage de génération et la liste complète des 25+ données tracées. En cas de litige médico-légal, chaque affirmation du rapport peut être remontée à sa source primaire.

### 4.4 Vérification anti-hallucination

C'est la preuve la plus concrète que le système est digne de confiance. Après la génération du rapport par Gemini, la fonction `verify_report()` effectue un second pass automatique qui :

1. **Compte les nodules mentionnés** et vérifie la cohérence avec le nombre attendu
2. **Vérifie chaque taille** mentionnée dans le rapport contre les sources (tolérance ±1.5mm)
3. **Vérifie la classification RECIST** en comparant la conclusion du rapport avec le calcul du pipeline
4. **Détecte les affirmations non vérifiables** — localisations anatomiques (lobes, segments) et caractéristiques morphologiques (solide, verre dépoli) que l'algorithme de segmentation ne fournit pas

**Résultat réel** : Gemini a inventé des localisations anatomiques ("lm", "lig") qui n'existent dans aucune source de données. Le système les a **automatiquement détectées et flaggées** comme non vérifiables. La classification RECIST (PD) a été correctement vérifiée.

### 4.5 Auto-évaluation qualité

La fonction `evaluate_report_quality()` applique une checklist de 10 critères sur le rapport généré :

1. Indication clinique présente
2. Technique d'examen décrite
3. Tous les findings mentionnés (4/4)
4. Tailles actuelles indiquées
5. Comparaison avec examens antérieurs
6. Classification fournie (RECIST ou Lung-RADS)
7. Raisonnement visible (calculs, seuils)
8. Recommandation de suivi
9. Avertissement IA
10. Données manquantes signalées

**Score obtenu : 10/10** — Le rapport couvre tous les aspects attendus d'un compte-rendu radiologique conforme aux standards.

---

## 5. Architecture Agentique (MCP)

### 5.1 Qu'est-ce que MCP

Le Model Context Protocol (MCP) est un standard ouvert qui permet de connecter des modèles de langage à des outils externes de manière interopérable — comparable à un "USB-C de l'IA". Dans le contexte hospitalier, cela signifie qu'un LLM (Claude, Gemini, ou tout autre modèle) peut interroger le PACS, lancer des segmentations, et générer des rapports via une interface standardisée, sans développement spécifique pour chaque combinaison outil/modèle.

### 5.2 Nos 5 tools MCP

Le fichier `mcp_server.py` implémente un serveur MCP avec 5 outils :

| Tool | Paramètres | Description |
|------|-----------|-------------|
| `list_patients` | aucun | Liste tous les patients disponibles dans Orthanc avec PatientID, AccessionNumber et nombre de nodules depuis le registry |
| `get_patient_history` | `patient_id` (string) | Retourne l'historique complet d'un patient depuis l'Excel (examens, tailles, rapports cliniques) |
| `run_segmentation` | `accession` (string) | Télécharge le CT, lance la segmentation, retourne les findings (tailles, volumes) |
| `generate_report` | `accession` (string) | Pipeline complet : download + segmentation + analyse + Gemini → rapport Markdown |
| `compare_exams` | `patient_id` (string) | Compare tous les examens d'un patient, génère un tableau d'évolution avec évaluation RECIST ou Lung-RADS |

### 5.3 Comment ça se connecte

Le serveur MCP utilise le transport stdio (JSON-RPC 2.0 sur stdin/stdout). Il se lance avec `python mcp_server.py` et se connecte à tout client MCP compatible (Claude Desktop, Claude Code, ou tout agent personnalisé). La configuration dans Claude Desktop nécessite simplement de pointer vers le script Python. C'est une solution "plug and play" pour un déploiement hospitalier.

---

## 6. Résultats et Validation

### 6.1 Patient test : 063F6BB9 (Accession 10969511)

- **Profil** : Femme, 51 ans, cancer pulmonaire (Lung Neoplasia), incluse dans un essai clinique
- **Données** : 3 examens CT disponibles, 89 coupes téléchargées, 4 nodules segmentés
- **Scénario** : B (Suivi oncologique / RECIST 1.1), détecté via les mots-clés *neoplasia* et *clinical trial*
- **Résultat RECIST** : **PD (Progressive Disease)** — la somme des diamètres cibles a augmenté de +20.6% par rapport à la baseline, avec une augmentation absolue de 7.2mm
- **Temps d'exécution** : 33.5 secondes (de l'interrogation d'Orthanc au rapport final vérifié)

Le rapport généré de ~1000 mots couvre l'indication clinique, la technique, les 4 nodules avec classification et VDT, le raisonnement RECIST complet avec calculs, une conclusion avec recommandation RCP, les limitations, les sources, et l'avertissement IA.

### 6.2 Tableau récapitulatif de l'orchestrateur

| Étape | Description | Résultat |
|-------|-------------|----------|
| 1-2 | Download CT depuis Orthanc | 89 coupes téléchargées |
| 3 | Segmentation AI | 4 findings détectés |
| 4 | Scores de confiance | F1(8/8) F2(8/8) F3(7/8) F4(5/8) |
| 5 | Classification scénario | B (RECIST 1.1) |
| 6 | Tracking lésionnel | 4 lésions suivies sur 3 examens |
| 7 | VDT | F1=113j F2=-3067j F3=-96j F4=-53j |
| 8 | Évaluation RECIST | PD (Progressive Disease) |
| 9 | Audit données manquantes | 4 items identifiés |
| 10 | Génération rapport (Gemini) | 1054 mots |
| 11 | Vérification anti-hallucination | 2/4 vérifié, 1 non-vérifiable |
| 12 | Auto-évaluation qualité | 10/10 |

### 6.3 Analyse du rapport de vérification

- **Nodules** : 4/4 correctement référencés
- **Tailles** : 51 valeurs numériques vérifiées contre les sources, 1 non retrouvée (valeur intermédiaire calculée par Gemini)
- **Classification RECIST** : PD correctement identifié dans le rapport
- **Hallucinations détectées** : Gemini a inventé des localisations anatomiques ("lm", "lig") — flaggées automatiquement comme non vérifiables
- **Qualité** : 10/10 sur la checklist de complétude

Ce résultat prouve que le système de vérification fonctionne : il laisse passer ce qui est correct et flagge ce qui ne peut pas être vérifié.

### 6.4 Tests sur d'autres patients

Quatre patients sont disponibles dans le dataset :
- **063F6BB9** (3 examens, Scénario B) — testé et validé
- **17A76C2A** (2 examens, Scénario A — Lung-RADS) — testé en dry-run, intéressant car scénario différent
- **0301B7D6** (5 examens, Scénario B) — intéressant pour le suivi longitudinal long
- **136F50A1** (6 examens, Scénario B) — le cas le plus complexe avec 6 points de suivi

Le mode batch (`--batch`) permet de traiter automatiquement tous les patients du registry.

---

## 7. Processus de test de la solution

### 7.1 Prérequis

- Python 3.10+
- Packages : `pydicom`, `numpy`, `pandas`, `openpyxl`, `requests`, `google-genai`, `mcp`
- Le fichier `.whl` de `dcm_seg_nodules` (fourni par GE Healthcare)
- L'Excel des pseudo-rapports : `Liste examen UNBOXED finalisée v2 (avec mesures).xlsx`
- Accès Orthanc : `https://orthanc.unboxed-2026.ovh` (auth: `unboxed` / `unboxed2026`)
- Clé API Gemini dans `./work/.env` (format : `GEMINI_API_KEY=AIza...`)

### 7.2 Test unitaire — Étape par étape

```bash
# Étape 1 : Téléchargement CT
python work/01_download_ct.py
# Attendu : 89 fichiers .dcm dans work/patient_test/original/

# Étape 2 : Segmentation + Analyse
python work/02_segmentation.py
# Attendu : patient_summary.json avec 4 findings, diamètres et volumes

# Étape 3 : Rapport simple (sans intelligence clinique)
python work/03_generate_report.py
# Attendu : rapport_radiologique.md généré par Gemini
```

### 7.3 Test pipeline complet

```bash
# Mode dry-run (sans appel Gemini — vérifie toute la chaîne de données)
python work/pipeline.py --accession 10969511 --dry-run

# Mode complet (avec génération du rapport via Gemini)
python work/pipeline.py --accession 10969511

# Les 12 étapes s'affichent séquentiellement.
# Fichiers générés dans work/patients/10969511/ :
#   - rapport_radiologique.md
#   - patient_summary.json
#   - sources_trace.json
#   - verification_report.json
#   - quality_report.json
```

### 7.4 Test du serveur MCP

```bash
# Lancer le serveur
python work/mcp_server.py

# Se connecte via stdio (JSON-RPC 2.0)
# Configuration Claude Desktop : pointer vers le script
# Exemple d'appel : tool "list_patients" → liste JSON des patients
# Exemple d'appel : tool "generate_report" avec accession "10969511" → rapport complet
```

### 7.5 Test batch

```bash
# Traiter tous les patients du registry
python work/pipeline.py --batch

# Résultat : un rapport par patient, résumé batch dans
# work/patients/batch_summary.json
```

### 7.6 Critères de validation

- Le rapport commence par l'INDICATION clinique
- Les findings sont classés target/non-target selon la règle des 10mm
- Le suivi F1↔F1 est présent avec tableau d'évolution
- Le calcul RECIST est montré et vérifiable (calculs visibles)
- Les données manquantes sont explicitement signalées
- L'anti-hallucination détecte les inventions du LLM
- Le score de qualité est ≥ 8/10

---

## 8. Éthique et Sécurité

### 8.1 Les 5 piliers

1. **Intention clinique** — Le rapport répond à une question médicale, pas une description de pixels. Le scénario détermine les critères appliqués.
2. **Règle RECIST** — Filtrer le bruit, prioriser ce qui est cliniquement significatif. Seules les lésions cibles comptent pour l'évaluation.
3. **Comparaison temporelle** — Le cœur de l'oncologie est l'évolution. Chaque mesure est contextualisée par rapport à la baseline et au nadir.
4. **Architecture industrielle** — MCP + Orthanc = intégrable dans un workflow hospitalier existant sans refonte de l'infrastructure.
5. **Sécurité** — Ne jamais inventer, toujours tracer, garder le radiologue dans la boucle.

### 8.2 RGPD et protection des données

- Les données patient sont pseudonymisées (PatientID = hash type 063F6BB9, pas de nom réel)
- Les images DICOM restent sur le serveur Orthanc — seules les métadonnées et mesures sont transmises au LLM
- Le prompt envoyé à Gemini ne contient que des identifiants pseudonymisés, des mesures numériques et du contexte clinique
- L'audit trail complet (`sources_trace.json`) permet la traçabilité réglementaire

### 8.3 Responsabilité médicale

- L'agent ne REMPLACE pas le radiologue — il prépare un brouillon
- Chaque rapport porte un avertissement IA explicite
- Le rapport est un document de travail à valider et signer par un médecin qualifié
- L'agent ne recommande jamais un traitement spécifique — il suggère un suivi ou une discussion en RCP (Réunion de Concertation Pluridisciplinaire)

---

## 9. Stack technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Langage | Python 3.10+ | Pipeline et serveur MCP |
| Imagerie DICOM | pydicom, SimpleITK | Lecture et analyse des fichiers médicaux |
| Calcul scientifique | numpy | Analyse voxels, surfaces, volumes |
| Données cliniques | pandas, openpyxl | Lecture Excel des pseudo-rapports |
| Segmentation | dcm_seg_nodules (GE Healthcare) | Algorithme mock de détection de nodules |
| LLM | google-genai (Gemini 2.5 Flash) | Génération du rapport en langage naturel |
| Protocole agentique | mcp (Model Context Protocol) | Interface standardisée LLM ↔ outils |
| Serveur PACS | Orthanc DICOM Server | Stockage et distribution des images |
| Réseau | requests | Communication API REST avec Orthanc |

---

## 10. Perspectives et Part 2

### 10.1 Améliorations futures

- **Analyse Hounsfield réelle** : caractérisation de la densité des nodules (solide vs verre dépoli vs mixte) par alignement CT/SEG — actuellement signalé comme non évalué
- **Dashboard multi-patient** : interface web avec alertes de progression pour le service d'oncologie
- **Interface Lovable** : démo interactive permettant au radiologue de visualiser les résultats et valider le rapport
- **iRECIST** : adaptation des critères pour les patients sous immunothérapie (pseudo-progression)
- **Intégration FHIR / HL7** : interopérabilité avec les systèmes d'information hospitaliers

### 10.2 Ce qui nous différencie

1. **Intelligence clinique** — L'agent comprend POURQUOI il fait le rapport (scénario A vs B), pas juste QUOI mesurer. Il raisonne comme un radiologue.

2. **Anti-hallucination prouvée** — Testé en conditions réelles : Gemini a inventé des localisations anatomiques, et le système les a détectées et flaggées automatiquement.

3. **VDT automatique** — Le Temps de Doublement Volumétrique est rarement calculé automatiquement. Notre pipeline le fait pour chaque lésion, avec classification clinique intégrée.

4. **Raisonnement RECIST transparent** — Le radiologue ne reçoit pas juste "PD". Il voit chaque étape du calcul : somme des diamètres, baseline, nadir, pourcentages, seuils appliqués, conclusion.

5. **Confidence scoring** — Le radiologue sait exactement où concentrer sa relecture. Un score de 5/8 sur F4 signifie : "cette mesure mérite une vérification manuelle", tandis que 8/8 sur F1 signifie : "données robustes et cohérentes".
