# Αναλυτικός έλεγχος ολοκλήρωσης Hotel Review NLP

Ημερομηνία ελέγχου: **9 Σεπτεμβρίου 2026**

Αποθετήριο: [krimits/hotel-review-nlp](https://github.com/krimits/hotel-review-nlp)

Έκδοση αναφοράς: `e0c3289aa5f95cb130207055e921d03291b4df23`, συν τις δύο προϋπάρχουσες τοπικές διορθώσεις BiLSTM.

Βάση αξιολόγησης: [ACTION_PLAN.md](ACTION_PLAN.md), κώδικας, notebooks, πραγματικά τοπικά artifacts και GitHub Actions.

## 1. Συμπέρασμα

**Το έργο έχει ουσιαστική υλοποίηση και πραγματικά πειράματα, αλλά δεν έχει ολοκληρωθεί ως αναπαραγώγιμο, ενιαίο benchmark και τελικό παραδοτέο v1.0.0.** Η πρώτη εβδομάδα έχει εκτελεστεί σε μεγάλο βαθμό, η δεύτερη έχει σημαντική πρόοδο και δηλωμένα αποτελέσματα DistilBERT, ενώ οι εβδομάδες 3–4 παραμένουν κυρίως σε επίπεδο έτοιμου κώδικα και οδηγιών.

Το ACTION_PLAN έχει μόνο ένα σημειωμένο checkbox, επομένως **υποτιμά την πραγματική πρόοδο**. Αντίστροφα, ορισμένες διατυπώσεις του README υπερβαίνουν όσα αποδεικνύονται από τα διαθέσιμα αρχεία. Δεν προκύπτει αξιόπιστο ενιαίο «ποσοστό ολοκλήρωσης» από τα κουτάκια, γιατί συνδυάζουν συγγραφή κώδικα, εκτέλεση πειραμάτων, προσωπική προετοιμασία και δημοσίευση.

Οι σημαντικότερες εκκρεμότητες είναι:

1. Διόρθωση της διαρροής κοινών κειμένων μεταξύ splits και δημιουργία νέας, ρητά εκδομένης βάσης αξιολόγησης.
2. Πραγματικό character baseline και επιλογή μοντέλων στο dev.
3. Σωστή αντιστοίχιση προβλέψεων–γραμμών και μέτρηση πραγματικού inference latency.
4. Ανάκτηση των DistilBERT artifacts και πλήρης εκτέλεση QLoRA.
5. Ενιαίο benchmark πέντε οικογενειών, McNemar, encoder API, INT8, load test, Docker και τελικό portfolio.

Δεν εκτελέστηκε νέα εκπαίδευση GPU, δεν άλλαξαν τα frozen parquet splits και δεν δημιουργήθηκε release. Οι έλεγχοι αναπαραγωγής και API περιγράφονται παρακάτω.

## 2. Πώς διαβάζεται η κατάσταση

| Κατάσταση | Ερμηνεία |
|---|---|
| **Ολοκληρωμένο** | Υπάρχει συγκεκριμένο παραδοτέο και επαληθεύσιμη ένδειξη ολοκλήρωσης του συγκεκριμένου σημείου. |
| **Μερικό** | Υπάρχει ουσιαστικό μέρος της δουλειάς, αλλά λείπει απαίτηση ή υπάρχει εμπόδιο. |
| **Δηλωμένο** | Υπάρχουν αριθμοί ή περιγραφή εκτέλεσης στο README, αλλά όχι επαρκή διαθέσιμα artifacts για ανεξάρτητη αναπαραγωγή. |
| **Εκκρεμές** | Δεν βρέθηκε ολοκληρωμένο παραδοτέο στο τοπικό έργο ή στο ελεγμένο GitHub snapshot. Η ύπαρξη script δεν αρκεί. |
| **Προσωπική επιβεβαίωση** | Δεν μπορεί να ελεγχθεί από αρχεία, π.χ. αν έχει διαβαστεί και κατανοηθεί ένα paper. |

«Δεν βρέθηκε» δεν σημαίνει ότι δεν υπάρχει σε άλλο Colab runtime, Drive, W&B ή μη κοινοποιημένο φάκελο. Δεν έγινε έρευνα σε άσχετα προσωπικά αρχεία ή εξωτερικά workspaces.

## 3. Έλεγχος και των 22 σημείων του ACTION_PLAN

### Εβδομάδα 1 — δεδομένα και classical baselines

| ID | Σημείο πλάνου | Κατάσταση | Τεκμήριο και τι απομένει |
|---|---|---|---|
| W1.1 | GitHub repository, push skeleton, CI badge | **Μερικό** | Repo και ενεργό CI υπάρχουν. Το τελευταίο ελεγμένο CI στο αρχικό remote HEAD πέρασε με 32 tests. Δεν βρέθηκε CI badge σε κανένα από τα δύο README. Προσθήκη badge και σαφούς quickstart από τον nested project φάκελο. |
| W1.2 | Download Booking 515K και `make data` | **Ολοκληρωμένο ως εκτέλεση** | Το πλήρες CSV έχει 515.738 γραμμές. Τα τρία τοπικά parquet υπάρχουν και αναπαράγονται ακριβώς από το CSV και το config. Ωστόσο, η ποιότητα του split χρειάζεται διόρθωση: F02. Η σύγκρουση ονομάτων CSV εξετάζεται στο F01. |
| W1.3 | EDA: label-source chart και length stats | **Μερικό** | Το notebook υπάρχει, αλλά έχει 0/7 εκτελεσμένα code cells και κανένα αποθηκευμένο output. Ο παρών έλεγχος επιβεβαίωσε αριθμούς και μήκη, όχι εκτέλεση/αποθήκευση των charts του notebook. Χρειάζεται επιτυχής εκτέλεση με σωστό working directory. |
| W1.4 | Δύο feature views × δύο μοντέλα | **Μερικό** | Υπάρχουν τέσσερα metrics και ένα αποθηκευμένο pipeline. Το καλύτερο pipeline αναπαράγει το δημοσιευμένο score. Όμως το «char» view χρησιμοποιεί `analyzer="word"`: δεν έχουν εκπληρωθεί οι δύο διαφορετικές αναπαραστάσεις που ζητά το πλάνο. Διόρθωση F03 και επανεκτέλεση. |
| W1.5 | Συμπλήρωση πρώτων γραμμών README | **Ολοκληρωμένο ως καταχώριση** | Τα classical αποτελέσματα έχουν γραφτεί και στα δύο README. Χρειάζονται ανανέωση μετά τις διορθώσεις, διόρθωση του `%%` και ενιαία παρουσίαση. Η συμπλήρωση πίνακα δεν πιστοποιεί την εγκυρότητα όλης της μεθοδολογίας. |

### Εβδομάδα 2 — BiLSTM και from-scratch LoRA

| ID | Σημείο πλάνου | Κατάσταση | Τεκμήριο και τι απομένει |
|---|---|---|---|
| W2.1 | BiLSTM, early stopping, dev macro-F1 | **Μερικό** | Υπάρχουν checkpoint, test metrics και dev macro-F1. Το README περιγράφει early stopping στο epoch 5, αλλά δεν υπάρχει αντίστοιχο τοπικό training log. Λείπουν `vocab.json`, `test_logits.npy`, `test_labels.npy`. Τα εκτελεσμένα notebooks αφορούν weighted/threshold πειράματα και δεν υποκαθιστούν το συγκεκριμένο baseline artifact bundle. |
| W2.2 | Κατανόηση LoRA paper / initialization | **Προσωπική επιβεβαίωση** | Η υλοποίηση και το DESIGN εξηγούν `A`, `B=0`, `alpha/r`, merge/unmerge. Δεν μπορεί να πιστοποιηθεί αν έχει διαβαστεί το paper ή αν μπορεί να εξηγηθεί προφορικά. Χρειάζεται προσωπική προετοιμασία και σύντομο τεχνικό walkthrough. |
| W2.3 | LoRA tests και PEFT equivalence | **Ολοκληρωμένο ως εκτέλεση tests** | Τοπικά πέρασαν 11 LoRA tests και 1 έγινε skip λόγω απουσίας PEFT. Στο ελεγμένο CI πέρασαν και τα 12, μέσα σε συνολικά 32 tests. Υπάρχει όμως περιορισμός στο equivalence test με μηδενικό `B`: F09. |
| W2.4 | Πρώτη GPU εκπαίδευση DistilBERT LoRA | **Δηλωμένο** | Το root README αναφέρει μετρημένο LoRA run. Το σχετικό notebook και ο trainer υπάρχουν, αλλά το notebook έχει 0/11 εκτελεσμένα code cells και δεν βρέθηκε τοπικό encoder/adapter checkpoint. Ανάκτηση configs, logs, manifest και weights από την πραγματική εκτέλεση. |
| W2.5 | Σύγκριση trainable params / F1 με full FT | **Δηλωμένο** | Υπάρχει αναλυτικός πίνακας full FT–LoRA στο root README: 95,80% έναντι 94,86% macro-F1, 66.955.010 έναντι 739.586 trainable params. Δεν βρέθηκαν `distilbert_comparison.csv`, δύο `metrics.json` και data manifests. Τα συγκρινόμενα GPU runs δηλώνονται ως `compare` με 20.000 training rows. Απαιτείται αποκατάσταση τεκμηρίων και νέα τελική σύγκριση μετά το F02. |

### Εβδομάδα 3 — QLoRA και unified benchmark

| ID | Σημείο πλάνου | Κατάσταση | Τεκμήριο και τι απομένει |
|---|---|---|---|
| W3.1 | QLoRA notebook σε T4, από αρχή έως τέλος | **Εκκρεμές** | Υπάρχουν trainer/config/notebook, αλλά 0/7 εκτελεσμένα code cells, χωρίς training log ή Qwen adapter. Διόρθωση F08, έλεγχος pinned περιβάλλοντος και επιτυχής GPU εκτέλεση. |
| W3.2 | Δύο sanity reviews και λήψη adapter | **Εκκρεμές** | Υπάρχει το σχετικό κελί, χωρίς output. Χρειάζονται οι δύο προβλέψεις και πλήρες adapter/tokenizer/config bundle που φορτώνεται σε νέο runtime. |
| W3.3 | `make benchmark` με όλα τα μοντέλα | **Εκκρεμές** | Δεν υπάρχουν `runs/benchmark/results.json`, `confusion.png`, `latency.png`. Λείπουν τρεις οικογένειες μοντέλων τοπικά και τα BiLSTM caches. Ακόμη και μετά την προσθήκη τους, τα F04–F06 εμποδίζουν αξιόπιστη εκτέλεση. |
| W3.4 | McNemar αποτελέσματα στο README | **Εκκρεμές** | Η υλοποίηση και τα unit tests υπάρχουν. Δεν βρέθηκε πραγματικός all-pairs πίνακας από ευθυγραμμισμένες προβλέψεις όλων των μοντέλων. Η διαφορά δύο macro-F1 δεν αποτελεί από μόνη της τεστ σημαντικότητας. |
| W3.5 | LLM annotation demo στο unlabeled CSV | **Εκκρεμές** | Notebook 0/6 εκτελεσμένα cells, χωρίς annotated output ή έλεγχο ανθρώπινου δείγματος. Πρώτα απαιτείται σωστό CSV schema, μετά εκτέλεση, αποθήκευση αποτελεσμάτων και έλεγχος F12. |

### Εβδομάδα 4 — serving, quantization, polish

| ID | Σημείο πλάνου | Κατάσταση | Τεκμήριο και τι απομένει |
|---|---|---|---|
| W4.1 | Εκκίνηση API με πραγματικό encoder | **Μερικό** | API και encoder wrapper υπάρχουν. Τα 5 stub tests πέρασαν και ο παρών έλεγχος επιβεβαίωσε classical model σε health/single/batch μέσω TestClient. Δεν υπάρχει τοπικό encoder checkpoint ούτε επαληθευμένο Uvicorn demo με encoder. |
| W4.2 | `demo_api.sh` και screenshot | **Εκκρεμές** | Το script υπάρχει. Δεν βρέθηκε screenshot ή καταγεγραμμένη εκτέλεση του πραγματικού demo. Εκτέλεση μετά το W4.1 και προσθήκη στο README. |
| W4.3 | INT8 before/after benchmark | **Εκκρεμές** | Script υπάρχει, output όχι. Το Makefile δείχνει σε `runs/distilbert/best`, ενώ ο trainer αποθηκεύει απευθείας στο output directory. Χρειάζονται σωστό path, macro-F1 και πραγματικό serialized size: F10. |
| W4.4 | Locust 20 users / 60s, RPS και p95 | **Εκκρεμές** | Υπάρχει load-test script, αλλά όχι CSV/logs/πίνακας. Η εντολή του ACTION_PLAN χρειάζεται `--host`. Να μετρηθεί ο δηλωμένος πραγματικός server/model με σαφή mix endpoints και failure rate. |
| W4.5 | Docker build και stub container smoke | **Μερικό** | Dockerfile και make target υπάρχουν. Docker executable δεν βρέθηκε στο PATH και δεν υπάρχει Docker build/run job στο CI. Απαιτούνται πραγματικό build, run και health/predict smoke. |
| W4.6 | Loom 3–4 λεπτών | **Εκκρεμές** | Δεν βρέθηκε σύνδεσμος Loom ή αντίστοιχο video artifact στα ελεγμένα αρχεία. Να εγγραφεί όταν υπάρχουν αναπαραγώγιμα τελικά αποτελέσματα. |
| W4.7 | README polish, clean runs, tag v1.0.0 | **Μερικό** | Υπάρχουν DESIGN, LICENSE και results, αλλά δύο αποκλίνοντα README, υπολείμματα συνομιλιών, παλιές αντιφατικές αποφάσεις, metadata `Your Name` και tracked `runs/bilstm/metrics.json`. Δεν βρέθηκε Git tag. Απαιτείται ολοκλήρωση των παρακάτω κριτηρίων πριν από release. |

Τα 22 σύνθετα σημεία ταξινομούνται αυστηρά ως **3 ολοκληρωμένα, 7 μερικά, 2 δηλωμένα, 9 εκκρεμή και 1 προσωπικής επιβεβαίωσης**. Αυτό δεν είναι ποσοστό υλοποιημένου κώδικα.

## 4. Τι έχει αποδειχθεί από τα πραγματικά αρχεία

### 4.1 Δεδομένα και αναπαραγωγή

Το πλήρες Booking CSV έχει **515.738 γραμμές** και μέγεθος **238.154.765 bytes**. Η επανάληψη των συναρτήσεων preprocessing στη μνήμη, χωρίς αντικατάσταση των parquet, αναπαρήγαγε και τα τρία splits ακριβώς.

| Σύνολο | Γραμμές | Negative | Positive | Μέσο μήκος χαρακτήρων |
|---|---:|---:|---:|---:|
| Train | 118.990 | 26.232 | 92.758 | 119,30 |
| Dev | 14.872 | 3.278 | 11.594 | 117,84 |
| Test | 13.278 | 3.278 | 10.000 | 119,55 |
| Σύνολο | 147.140 | 32.788 | 114.352 | — |

Η αρχική schema-based εξαγωγή βρίσκει 127.763 positive-only, 35.819 negative-only, 352.029 mixed και 127 empty εγγραφές. Με φίλτρα μήκους και αφαίρεση ακριβών διπλοτύπων προκύπτουν 147.140 labeled reviews. Στο train, τα p50/p95/p99 μήκους είναι **14/64/122 λέξεις**.

Δεν βρέθηκαν nulls ή ακριβή διπλότυπα μέσα στα splits, ούτε ακριβώς ίδια strings μεταξύ splits. Βρέθηκαν όμως overlaps μετά από `strip().casefold()`: F02.

Τα πλήρη, order-sensitive fingerprints και τα αποτελέσματα των ελέγχων περιλαμβάνονται στο [evidence.json](docs/audit/2026-09-09/evidence.json).

### 4.2 Αποτελέσματα μοντέλων και ισχύς τεκμηρίων

| Μοντέλο / πείραμα | Test macro-F1 | Accuracy | Τεκμηρίωση |
|---|---:|---:|---|
| NB word | 0,9345 | 0,9509 | Τοπικό joblib + metrics. Οι προβλέψεις επανυπολογίστηκαν σε όλα τα 13.278 test reviews και συμφωνούν. |
| LR-SGD word | 0,9173 | 0,9411 | Τοπικό metrics JSON. Δεν υπάρχει ξεχωριστό αποθηκευμένο pipeline για ανεξάρτητη επαναφόρτωση. |
| NB «char» | 0,8545 | 0,8991 | Τοπικό metrics JSON, αλλά πρόκειται για word 3–5 grams λόγω F03. |
| LR-SGD «char» | 0,5681 | 0,7887 | Ίδιο πρόβλημα: δεν αποτελεί έγκυρο character baseline. |
| BiLSTM, υπάρχον τοπικό baseline | 0,9503 | 0,9629 | Metrics + αναγνώσιμο checkpoint, dev macro-F1 0,9493. Λείπει vocab και prediction bundle. |
| BiLSTM seed 100, unweighted | 0,9521 | 0,9642 | Αναφέρεται στο root README. Δεν υπάρχει αντίστοιχο τοπικό πλήρες artifact bundle. |
| BiLSTM weighted | 0,9492 | 0,9614 | Εκτελεσμένα outputs στο root `BILSTM.ipynb`. |
| BiLSTM threshold 0,685 | 0,9510 | 0,9629 | Outputs στο πρόσθετο εκτελεσμένο threshold notebook. Το canonical notebook είναι καθαρό template. |
| DistilBERT full FT, compare mode | 0,9580 | 0,9687 | Root README, χωρίς τοπικά weights/logits/manifest. |
| DistilBERT scratch LoRA, compare mode | 0,9486 | 0,9617 | Root README, χωρίς τοπικό adapter/head bundle ή comparison CSV. |
| Qwen QLoRA | — | — | Δεν βρέθηκαν μετρημένα αποτελέσματα. |

**Δεν πρόκειται για τελικό leaderboard.** Υπάρχουν διαφορετικά training budgets, ελλιπή artifacts και τα μεθοδολογικά προβλήματα που ακολουθούν. Τα αναφερόμενα «current best» πρέπει να παρουσιάζονται ως ιστορικά αποτελέσματα συγκεκριμένων runs.

Το BiLSTM checkpoint περιέχει 11 tensors, embedding shape `[15138, 128]` και συνολικά 2.202.370 tensor parameters. Αυτό επιβεβαιώνει ότι υπάρχει πραγματικό αναγνώσιμο μοντέλο, όχι πλήρη αναπαραγωγιμότητα χωρίς vocab/config provenance.

### 4.3 Έλεγχοι που εκτελέστηκαν τώρα

| Έλεγχος | Αποτέλεσμα |
|---|---|
| Τοπικό `pytest tests -v -rs -p no:cacheprovider` | **31 passed, 1 skipped**, 1 warning. Το skip αφορά PEFT. Η αρχική sandbox εκτέλεση είχε δύο permission errors σε προσωρινούς φακέλους· η επανάληψη εκτός sandbox πέρασε. |
| `ruff check src tests scripts --no-cache` | **All checks passed**. |
| Αρχικό remote HEAD CI | **32 passed**, 1 warning, μαζί με PEFT equivalence. [GitHub Actions run](https://github.com/krimits/hotel-review-nlp/actions/runs/34239998385). |
| Αναπαραγωγή preprocessing | Και τα 3 DataFrames ίσα με τα υπάρχοντα parquet, με την ίδια σειρά. |
| Επανεκτίμηση αποθηκευμένου classical pipeline | Ίδιες μετρικές και confusion matrix με το metrics JSON. |
| Classical API μέσω FastAPI TestClient | Health, single predict και batch επέστρεψαν HTTP 200. Δεν είναι live Uvicorn/load test. |
| Συνθετικό probe cached predictor | Δέχθηκε άσχετα κείμενα με ίδιο πλήθος γραμμών. Το latency subset προκάλεσε ValueError. |
| Συνθετικό probe BiLSTM collate | Labels `[0,1,0]` έγιναν `[1,0,0]` λόγω ταξινόμησης κατά μήκος. |
| SHA-256 πλήρους CSV και backup | Ίδιο hash και μέγεθος. |

Το τοπικό περιβάλλον είναι Python 3.13.5, Torch 2.13.0+cpu, χωρίς CUDA και χωρίς transformers/PEFT. Η τοπική επιτυχία των CPU tests δεν πιστοποιεί GPU training ή HF checkpoint loading.

## 5. Τεχνικά ευρήματα και συγκεκριμένες διορθώσεις

### F01 — Σύγκρουση των δύο CSV στα Windows

**Προτεραιότητα: P0 για συμβατότητα δεδομένων. Επιλύθηκε κατά τον συγχρονισμό· βλ. §8.**

Στο Git υπάρχει unlabeled `hotel_reviews.csv` 658 γραμμών με στήλες `Hotel, Date, Review Title, Review`. Το config ζητούσε Booking `Hotel_Reviews.csv`. Στο Windows filesystem οι δύο διαδρομές αντιστοιχούσαν στο ίδιο αρχείο, το οποίο τοπικά είχε το Booking schema. Έτσι το annotation demo και το load test δεν μπορούσαν να βρουν τη στήλη `Review`.

Η σωστή διάταξη είναι:

- `data/raw/hotel_reviews.csv`: το μικρό tracked unlabeled dataset.
- `data/raw/booking_reviews_515k.csv`: το μεγάλο Booking dataset, ignored.
- `configs/baselines.yaml` και EDA: ρητή αναφορά στο δεύτερο.
- Download/extract σε ξεχωριστό φάκελο, πριν από αντιγραφή με το νέο όνομα.

Η τελική κατάσταση της διόρθωσης καταγράφεται στην §8.

### F02 — Τα υπάρχοντα frozen splits δεν περνούν τον νεότερο έλεγχο overlap

**Προτεραιότητα: P0 πριν από νέα τελικά runs.**

Ο έλεγχος `strip().casefold()` βρήκε κοινά μοναδικά normalized κείμενα:

| Ζεύγος splits | Κοινά normalized κείμενα |
|---|---:|
| Train / Dev | 180 |
| Train / Test | 170 |
| Dev / Test | 24 |

Τα πλήθη είναι ανά ζεύγος και δεν πρέπει να αθροιστούν ως συνολικά μοναδικά reviews. Το `preprocess.py` αφαιρεί μόνο ακριβώς ίδια strings, ενώ το `utils/experiments.py` απορρίπτει overlaps αγνοώντας πεζά/κεφαλαία. Ο τελευταίος έλεγχος αποτυγχάνει στα πραγματικά τοπικά δεδομένα με `ValueError: review text overlaps between train and dev`.

Αυτό σημαίνει ότι το νεότερο DistilBERT notebook δεν μπορεί να θεωρηθεί αναπαραγώγιμο πάνω στα συγκεκριμένα τοπικά frozen αρχεία χωρίς επίλυση της ασυνέπειας. Δεν αποδεικνύεται ότι κάποιο προηγούμενο cloud run χρησιμοποίησε τα ίδια αρχεία μόνο από τα row counts.

**Απαιτούμενη ολοκλήρωση:** ενιαίος κανόνας canonicalization/deduplication πριν από το split, έλεγχος τυχόν conflicting labels, νέα έκδοση dataset/manifest, μηδενικό overlap και νέα αποτελέσματα για όλες τις οικογένειες. Να διατηρηθούν τα σημερινά splits/metrics ως ιστορικό v1, όχι να αντικατασταθούν σιωπηρά. Η επίδραση στα scores πρέπει να μετρηθεί· δεν ποσοτικοποιήθηκε εδώ.

### F03 — Το character baseline είναι στην πραγματικότητα word baseline

**Προτεραιότητα: P0.** [classical.py](src/reviewnlp/baselines/classical.py)

Το `_feature_views()` δεν ορίζει `analyzer` στο δεύτερο TfidfVectorizer. Επιβεβαιώθηκε runtime:

```text
word: analyzer=word, ngram_range=(1, 2)
char: analyzer=word, ngram_range=(3, 5)
```

**Απαιτούμενη ολοκλήρωση:** ρητή επιλογή `char` ή `char_wb`, τεκμηρίωση της επιλογής, regression test με γνωστά character features και επανεκτέλεση των τεσσάρων συνδυασμών. Το «Failed Baseline» του README δεν αποτελεί συμπέρασμα για character features με τον σημερινό κώδικα.

### F04 — Η επιλογή του καλύτερου classical μοντέλου γίνεται στο test

**Προτεραιότητα: P0.** [classical.py](src/reviewnlp/baselines/classical.py)

Το `dev_df` φορτώνεται ως `_dev_df` και δεν χρησιμοποιείται. Το καλύτερο pipeline επιλέγεται από τα test macro-F1. Αυτό αντιφάσκει με το DESIGN και δημιουργεί test-set selection bias.

**Απαιτούμενη ολοκλήρωση:** επιλογή feature/model/hyperparameters με dev macro-F1, αποθήκευση της απόφασης και αξιολόγηση στο test μετά το κλείδωμα της επιλογής. Προτιμώνται artifacts για κάθε candidate και σαφή train/dev/test πεδία metrics. Οι ήδη δημοσιευμένοι αριθμοί παραμένουν ιστορικές παρατηρήσεις.

### F05 — BiLSTM logits και gold labels δεν διατηρούν κοινή σειρά

**Προτεραιότητα: P0.** [dataset.py](src/reviewnlp/data/dataset.py), [bilstm.py](src/reviewnlp/baselines/bilstm.py)

Το collate ταξινομεί κάθε batch κατά μήκος. Το `_evaluate()` επιστρέφει logits στην ταξινομημένη σειρά, ενώ η αποθήκευση χρησιμοποιεί `test_ds.labels` στην αρχική σειρά. Τα εσωτερικά metrics του `_evaluate()` υπολογίζονται με αντίστοιχα ταξινομημένα golds και δεν χαρακτηρίζονται λανθασμένα από αυτό το εύρημα. Το πρόβλημα αφορά το εξαγόμενο cache και τη μελλοντική σύγκριση με άλλες οικογένειες.

**Απαιτούμενη ολοκλήρωση:** μεταφορά row IDs/permutation και επαναφορά της αρχικής σειράς πριν από την αποθήκευση ή collate που διατηρεί σειρά με κατάλληλη χρήση packing. Να αποθηκεύονται row IDs, labels, logits και ordered split fingerprint μαζί. Χρειάζεται regression test με διαφορετικά μήκη μέσα σε batch.

Το σημερινό τοπικό bundle περιέχει μόνο `bilstm.pt` και `metrics.json`. Πρέπει να ανακτηθούν τα αυθεντικά vocab/caches ή να παραχθεί νέο πλήρες run.

### F06 — Το unified benchmark αποτυγχάνει με caches και δεν μετρά πραγματικό latency

**Προτεραιότητα: P0.** [benchmark.py](src/reviewnlp/evaluation/benchmark.py), [predict.py](src/reviewnlp/llm/predict.py)

Το `cached_logits` ελέγχει μόνο `len(texts)`, αγνοεί το περιεχόμενο/σειρά και φορτώνει gold labels χωρίς να τα συγκρίνει. Άσχετα κείμενα ίδιου πλήθους έγιναν αποδεκτά στο probe.

Στη συνέχεια το benchmark καλεί τον ίδιο predictor με `texts[:64]` για latency. Το cache δέχεται μόνο ολόκληρο το test set, οπότε προκύπτει ValueError εκτός του αρχικού prediction try/except. Αν επιτρεπόταν αυθαίρετο slicing, ο χρόνος lookup cache πάλι δεν θα ήταν χρόνος inference.

**Απαιτούμενη ολοκλήρωση:** διαφορετική διαδρομή για cached quality metrics και live model latency, έλεγχος ordered fingerprints/labels/IDs, warmup και κοινό πρωτόκολλο με αρκετές μετρήσεις. Το p95 τριών μέσων χρόνων batch δεν πρέπει να παρουσιαστεί ως p95 πραγματικών HTTP requests. Να υπάρχει τελική strict λειτουργία που αποτυγχάνει όταν λείπει οποιαδήποτε από τις πέντε απαιτούμενες οικογένειες.

### F07 — Ελλιπή GPU artifacts και αναντιστοιχία output paths

**Προτεραιότητα: P1.**

Δεν βρέθηκαν `runs/distilbert`, `runs/distilbert_lora_scratch` ή `runs/qwen_qlora/adapter`. Το DistilBERT notebook γράφει σε experiment directories στο Drive, ενώ το benchmark αναζητά σταθερά τοπικά paths. Άρα η εκτέλεση στο Colab δεν ολοκληρώνει αυτόματα την τοπική ενσωμάτωση.

**Απαιτούμενη ολοκλήρωση:** download/export/import βήμα με manifest, ακριβές source commit, package versions, configs, logs, tokenizer, weights, prediction arrays και comparison CSV. Πρώτα να ελεγχθούν τα αυθεντικά hashes των δηλωμένων runs. Μετά το F02 απαιτούνται νέα τελικά runs σε κοινή έκδοση test set. Τυχόν μικρότερο training budget ανά οικογένεια να δηλώνεται καθαρά.

### F08 — QLoRA χρειάζεται διορθώσεις πριν από το πρώτο αξιόπιστο run

**Προτεραιότητα: P1.** [train_qlora.py](src/reviewnlp/llm/train_qlora.py), [predict.py](src/reviewnlp/llm/predict.py)

- Το training περνά `bnb_4bit_use_double_quantum=True`. Το πραγματικό όνομα της ρύθμισης είναι `bnb_4bit_use_double_quant`, με default false. Ο σημερινός κώδικας δεν τεκμηριώνει ότι ενεργοποιείται το double quantization. [Επίσημο BitsAndBytesConfig](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.BitsAndBytesConfig).
- Το inference επιλέγει bfloat16 απλώς επειδή υπάρχει CUDA, ενώ το training κάνει capability check. Χρειάζεται κοινή επιλογή dtype για το target GPU.
- Το training tokenizer χρησιμοποιεί right padding. Το batched causal generation πρέπει να ρυθμίζει ρητά κατάλληλο left padding και pad token όταν φορτώνει τον adapter. [Οδηγός generation της Hugging Face](https://huggingface.co/docs/transformers/llm_tutorial#padding-side).
- Το QLoRA notebook έχει ακόμη clone placeholder και ασαφή nested working directory. Το `requirements-colab.txt` αφορά το encoder experiment και δεν κλειδώνει πλήρη στοίβα TRL/bitsandbytes για QLoRA.
- Το `parse_label()` μετατρέπει άκυρη απάντηση σιωπηρά σε μία από τις δύο κλάσεις. Χρειάζεται καταγραφή invalid outputs ή ελεγχόμενο output contract.

Απαιτούνται smoke run, επιβεβαίωση πραγματικών quantization settings, σταθερό batch inference, τελικό training και reload adapter σε καθαρό runtime. Οι χρόνοι «30–45 λεπτά» του πλάνου είναι εκτιμήσεις, όχι μετρημένο παραδοτέο.

### F09 — Περιορισμένη αποδεικτική ισχύς του PEFT equivalence test

**Προτεραιότητα: P1.** [test_lora.py](tests/test_lora.py)

Το equivalence test εκτελείται επιτυχώς στο CI, αλλά αντιγράφει adapters όταν το `B` παραμένει μηδενικό. Έτσι οι δύο κλάδοι LoRA έχουν μηδενική επίδραση. Υπάρχουν άλλα χρήσιμα tests για scaling, gradients και merge, όμως η συγκεκριμένη σύγκριση με PEFT δεν αποδεικνύει από μόνη της ισοδυναμία μετά την προσαρμογή.

**Απαιτούμενη ολοκλήρωση:** σύγκριση με μη μηδενικό `B`, ίσα weights, ενεργό `alpha/r`, forward και gradient checks, και parity μετά από merge/unmerge.

### F10 — Quantization benchmark δεν καλύπτει όσα υπόσχεται

**Προτεραιότητα: P1.** [Makefile](Makefile), [quantize_distilbert.py](scripts/quantize_distilbert.py)

Το target δείχνει σε `runs/distilbert/best`, ενώ ο encoder trainer αποθηκεύει στο `runs/distilbert`. Το script καταγράφει accuracy, όχι macro-F1, παρότι το DESIGN περιγράφει F1 degradation. Ο υπολογισμός μεγέθους από `model.parameters()` δεν είναι αξιόπιστος για packed dynamic-quantized weights.

**Απαιτούμενη ολοκλήρωση:** σωστό checkpoint path, full-test quality metrics με έμφαση macro-F1/class-0 F1, χωριστό latency sample, πραγματική serialized μέτρηση μεγέθους και hardware/protocol metadata. Να μην παρουσιαστεί το αναμενόμενο ~2× speedup ως αποτέλεσμα πριν μετρηθεί.

### F11 — Serving και μετρήσεις δεν έχουν ολοκληρωθεί

**Προτεραιότητα: P1.** [model_wrapper.py](src/reviewnlp/serving/model_wrapper.py)

Ο encoder έχει διαδρομή φόρτωσης μία φορά. Ο Qwen wrapper, όμως, καλεί predictor που φορτώνει μοντέλο ανά request. Το API description αναφέρει BiLSTM, αλλά το wrapper δεν υποστηρίζει `MODEL_TYPE=bilstm`.

**Απαιτούμενη ολοκλήρωση:** επιλογή του πραγματικού μοντέλου που θα παρουσιαστεί, load-once inference για την υποστηριζόμενη διαδρομή, ευθυγράμμιση περιγραφής/υποστήριξης και live smoke. Το Qwen load-once είναι απαραίτητο εάν διατηρηθεί η δήλωση υποστήριξής του. Το υποχρεωτικό encoder demo δεν εξαρτάται από την προσθήκη BiLSTM serving.

Το load test χρειάζεται `--host http://127.0.0.1:8000`, έλεγχο schema του CSV και εξαγωγή CSV. Να καταγραφούν RPS, p50/p95, errors, users, spawn rate, endpoint mix, hardware και model path. Docker να επαληθευτεί σε μηχάνημα με διαθέσιμο daemon και, ιδανικά, σε CI smoke job.

### F12 — Annotation demo χωρίς αποθηκευμένο και ελεγμένο αποτέλεσμα

**Προτεραιότητα: P2.**

Το notebook χρησιμοποιεί base Qwen, ενώ η περιγραφή του dataset μιλά για fine-tuned LLM. Δηλώνει CPU δυνατότητα αλλά φορτώνει υποχρεωτικά 4-bit configuration χωρίς CPU εναλλακτική. Το confidence είναι τιμή που παράγει το ίδιο το LLM, όχι βαθμονομημένη πιθανότητα.

**Απαιτούμενη ολοκλήρωση:** σαφής επιλογή base/fine-tuned μοντέλου και συσκευής, config για sample/threshold, ασφαλής επικύρωση JSON/confidence, αποθήκευση annotations και ανθρώπινος έλεγχος δείγματος. Να μη συγχωνευθούν αυτόματα pseudo-labels στο frozen benchmark dataset.

### F13 — README, notebooks και κανόνες αναπαραγωγής χρειάζονται ενοποίηση

**Προτεραιότητα: P2.**

Το root README περιέχει εκτενές κείμενο αποτελεσμάτων και αποσπασμάτων συνομιλίας, ενώ το nested README περιορίζεται στα classical baselines. Υπάρχουν παλιές δηλώσεις «μόνο ένα seed / κανένα baseline» μαζί με μεταγενέστερα αποτελέσματα πολλών seeds, «current best BiLSTM» πριν από το νεότερο DistilBERT και διπλά `%%`.

Το root `BILSTM.ipynb` περιέχει στο stdout και παλαιότερο `ModuleNotFoundError`, παρότι δεν υπάρχει Jupyter error-type output. Το εκτελεσμένο threshold notebook βρίσκεται σε πρόσθετο nested φάκελο· υπάρχουν δύο canonical notebooks με πρόθεμα `04`. Η παρουσία εκτελεσμένων κελιών δεν πιστοποιεί ότι κάθε shell command ολοκληρώθηκε επιτυχώς.

**Απαιτούμενη ολοκλήρωση:** ένα authoritative README με installation/quickstart, versioned results και links σε artifacts· experiment reports για τα ιστορικά runs· καταγεγραμμένα configs/exit codes· καθαροί σύνδεσμοι Colab προς τη σωστή έκδοση· προσωπικά metadata και release checklist. Μεταφορά του tracked metrics JSON από `runs/` σε τεκμηρίωση με provenance, χωρίς απώλεια ιστορικών αποτελεσμάτων.

Οι standing rules είναι **μερικώς τηρημένοι**: υπάρχουν YAML configs και μικρά commits, αλλά και hardcoded notebook/CLI settings και commits τύπου upload. Δεν μπορεί να αποδειχθεί «commit μετά από κάθε green run». Δεν βρέθηκε πλήρες, δομημένο αρχείο αποτυχημένων Colab runs.

## 6. Προτεινόμενη σειρά για την ολοκλήρωση

| Σειρά | Εργασία | Definition of done |
|---|---|---|
| 1 | Δεδομένα και πειραματικό πρωτόκολλο | F01 λυμένο, νέο versioned split μετά το F02, μηδενικά normalized overlaps, manifests και ενημερωμένα expected counts. |
| 2 | Classical και BiLSTM | F03–F05 λυμένα, επιλογή στο dev, πλήρη artifacts, row-order tests και reproducible metrics. |
| 3 | Κλείσιμο encoder/LoRA | Ανάκτηση ιστορικών GPU bundles, ενισχυμένο PEFT test, νέα full-FT/LoRA runs στα νέα κοινά splits, configs/CSV/hashes/reload smoke. |
| 4 | QLoRA | F08 λυμένο, clean-runtime notebook, adapter export/reload, sanity reviews και test predictions με IDs. |
| 5 | Unified benchmark | Και οι πέντε οικογένειες υποχρεωτικά παρούσες, πραγματικό latency, κοινό test fingerprint, `results.json`, δύο plots και McNemar. |
| 6 | Serving / INT8 / load / Docker | Πραγματικός encoder server, demo capture, FP32–INT8 quality/latency/size, Locust CSV και επιτυχές container smoke. |
| 7 | Annotation και τελική παρουσίαση | Αποθηκευμένο annotation demo, ανθρώπινο spot-check, ενιαίο README, CI badge, six-line future-work paragraph, Loom και ελεγμένο tag v1.0.0. |

Η επανεκπαίδευση όλων των μοντέλων πριν διορθωθούν δεδομένα και benchmark θα παρήγαγε νέα artifacts πάνω στα ίδια προβλήματα. Η σειρά 1–2 προηγείται των νέων GPU runs. Η μελέτη LoRA και η οργάνωση παλαιών artifacts μπορούν να γίνονται παράλληλα με τις διορθώσεις.

Δεν δίνεται αυθαίρετη εκτίμηση ημερών: ο απαιτούμενος χρόνος εξαρτάται από τη διαθεσιμότητα των παλιών artifacts, το target hardware και την επίλυση των πραγματικών failures.

## 7. Τελική checklist αποδοχής v1.0.0

- [ ] Καθαρό checkout εγκαθίσταται από τεκμηριωμένες εντολές στο σωστό nested directory.
- [ ] Τα δύο CSV έχουν διαφορετικά ονόματα και σωστά schemas σε Windows και Linux.
- [ ] Versioned train/dev/test, χωρίς normalized overlap, με αποθηκευμένα fingerprints.
- [ ] Word και character NB/LR-SGD είναι πράγματι διαφορετικά feature views· επιλογή με dev.
- [ ] BiLSTM vocab/config/weights και predictions είναι πλήρη και ευθυγραμμισμένα με τα test IDs.
- [ ] Full FT, scratch LoRA και QLoRA έχουν reloadable artifacts και provenance.
- [ ] Η σύγκριση με PEFT καλύπτει και μη μηδενική προσαρμογή.
- [ ] Το benchmark απαιτεί πέντε οικογένειες και παράγει JSON, plots, significance και live latency.
- [ ] Encoder API, screenshot/demo, quantization, load test και Docker έχουν πραγματικά τεκμήρια.
- [ ] Annotation demo αποθηκεύει ελεγμένα αποτελέσματα χωρίς αλλοίωση του test set.
- [ ] README και DESIGN συμφωνούν με τα τελικά δεδομένα· υπάρχει badge, Loom και future-work paragraph.
- [ ] Raw μεγάλα δεδομένα και weights μένουν εκτός Git· μικρά evidence summaries έχουν provenance.
- [ ] CI περνά στο ακριβές release commit και το tag δημιουργείται μετά την αποδοχή των παραπάνω.

Το DESIGN §8 ήδη περιέχει τέσσερις ιδέες future work, αλλά δεν αποτελεί ακόμη την ζητούμενη τελική παράγραφο έξι γραμμών για το application email. Η αλλαγή σε score-based τρεις κλάσεις πρέπει να διατυπωθεί ως νέο, διαφορετικό labeling task και όχι ως συνέχεια που αναιρεί σιωπηρά τον σημερινό schema-based στόχο.

## 8. Σύγκριση τοπικού και GitHub / συγχρονισμός

Το πραγματικό Git root είναι ο εξωτερικός φάκελος `Downloads/hotel-review-nlp`. Το Python project είναι ο εσωτερικός `hotel-review-nlp/`.

Πριν από τον έλεγχο:

- Τοπικό `main`: `1a6cd19`.
- Remote `main` μετά από πραγματικό fetch: `e0c3289`.
- Διαφορά: **8 commits** και 19 αρχεία που ενημερώθηκαν με fast-forward.
- Προϋπάρχουσες τοπικές αλλαγές: `configs/bilstm.yaml`, `baselines/bilstm.py` και το μεγάλο CSV στη θέση του μικρού.
- Το fast-forward ολοκληρώθηκε χωρίς conflicts και διατήρησε τις τοπικές αλλαγές.

Οι δύο τοπικές διορθώσεις BiLSTM αφορούν τη θέση του `embedding_dim` στο config και τη χρήση αριθμητικών labels στις μετρικές. Περιλαμβάνονται στον έλεγχο και στον συγχρονισμό. Δεν αποτελούν επίλυση των υπόλοιπων F02–F13.

Αντίγραφα των τριών αρχικών τοπικών αρχείων υπάρχουν στον γειτονικό φάκελο `Downloads/hotel-review-nlp-backup-20260909`. Το πλήρες CSV και το backup έχουν SHA-256:

```text
a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9
```

**Ολοκληρώθηκε ο διαχωρισμός και η αποκατάσταση των CSV.** Μετά την
αποδέσμευση του αρχείου από τον χρήστη, επαληθεύτηκαν ξανά τα δύο αντίγραφα
του πλήρους Booking dataset με SHA-256 και αποκαταστάθηκε από το Git το
`data/raw/hotel_reviews.csv`: 658 γραμμές, 728.488 bytes, με τις στήλες
`Hotel, Date, Review Title, Review`. Το αρχείο ταυτίζεται με το tracked
δείγμα και δεν εμφανίζει πλέον τοπική διαφορά.

Το πλήρες Booking CSV παραμένει στο ignored
`data/raw/booking_reviews_515k.csv`, 238.154.765 bytes, με αμετάβλητο SHA-256.
Το επιπλέον backup διατηρείται στον γειτονικό φάκελο. Τα baseline config,
EDA και οδηγίες download χρησιμοποιούν το νέο, διακριτό όνομα.
Το annotation/load-test schema του μικρού CSV έχει αποκατασταθεί· οι
εκτελέσεις και τα παραδοτέα αυτών των εργασιών εξακολουθούν να εκκρεμούν.

Ο τελικός συγχρονισμός κώδικα και αναφοράς ελέγχεται με `git status --short --branch`, `git rev-parse HEAD` και πραγματικό `git ls-remote origin refs/heads/main`. Τα ignored datasets, parquet, περιβάλλοντα και μεγάλα μοντέλα παραμένουν τοπικά· Git synchronization δεν μεταφέρει Colab/Drive/W&B artifacts αυτόματα.

## 9. Εντολές επανελέγχου

Από τον εσωτερικό φάκελο του Python project:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v -rs -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src tests scripts --no-cache
```

Οι παρακάτω είναι **εντολές για το στάδιο ολοκλήρωσης**, αφού διορθωθούν τα σχετικά findings και υπάρχουν τα checkpoints. Δεν εκτελέστηκαν ως μέρος του audit:

```powershell
$env:MODEL_TYPE = 'encoder'
$env:MODEL_PATH = 'runs/distilbert'
.\.venv\Scripts\python.exe -m uvicorn reviewnlp.serving.app:app --host 127.0.0.1 --port 8000

# Σε δεύτερο terminal:
.\.venv\Scripts\python.exe -m locust -f scripts/load_test.py --host http://127.0.0.1:8000 --headless -u 20 -r 2 -t 60s --only-summary --csv artifacts/serving-load

.\.venv\Scripts\python.exe scripts/quantize_distilbert.py --model runs/distilbert --dataset data/processed/test.parquet
docker build -f docker/Dockerfile -t hotel-review-nlp:latest .
docker run --rm -p 8000:8000 -e MODEL_TYPE=stub hotel-review-nlp:latest
```

Το `make` και το `docker` δεν βρέθηκαν στο PATH του ελεγμένου Windows περιβάλλοντος. Οι Python module εντολές του Makefile μπορούν να εκτελούνται από το virtual environment· αυτό δεν παρακάμπτει τις τεχνικές εκκρεμότητες.

## 10. Πηγές και όρια

Πρωτεύον τεκμήριο είναι το [αρχικό snapshot του ACTION_PLAN](https://github.com/krimits/hotel-review-nlp/blob/e0c3289aa5f95cb130207055e921d03291b4df23/hotel-review-nlp/ACTION_PLAN.md), μαζί με τον αντίστοιχο κώδικα και τα τοπικά artifacts. Η διαθέσιμη root README σύγκριση διατηρείται ως **δηλωμένη πειραματική τεκμηρίωση**, όχι ως ανεξάρτητη επαλήθευση των cloud runs.

Το [evidence.json](docs/audit/2026-09-09/evidence.json) περιέχει το snapshot της απογραφής και των ελέγχων πριν από τις αλλαγές ονομάτων CSV, μαζί με την επαληθευμένη CI κατάσταση. Δεν περιέχει raw review texts ή model weights. Η HTML έκδοση είναι παράλληλη αναγνώσιμη απόδοση αυτής της Markdown αναφοράς.

Η αναφορά προσδιορίζει τι απαιτείται για ολοκλήρωση. Δεν πιστοποιεί production deployment, νέα GPU runs ή έτοιμο release.
