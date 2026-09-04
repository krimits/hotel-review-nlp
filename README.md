# Hotel Review NLP Project - Week 1 Progress

## Dataset Methodology
The dataset splits and distribution after removing ambiguous reviews:
* **Total Labeled Reviews:** 147,140
* **Train Set:** 118,990 rows (92,758 positive / 26,232 negative)
* **Dev Set:** 14,872 rows (11,594 positive / 3,278 negative)
* **Test Set:** 13,278 rows (10,000 positive / 3,278 negative)

## Evaluation Results (Week 1)
Below are the benchmark metrics comparing classical Machine Learning with Deep Learning:

| Model Architecture | Macro-F1 | Accuracy | Status |
| :--- | :---: | :---: | :--- |
| **BiLSTM (Pure PyTorch)** | **0.9503** | **96.29%%** | ?? **Current Best** |
| naive_bayes_word | 0.9345 | 95.09%% | Best Classical Baseline |
| lr_sgd_word | 0.9173 | 94.11%% | Classical Baseline |
| naive_bayes_char | 0.8545 | 89.91%% | Classical Baseline |
| lr_sgd_char | 0.5681 | 78.87%% | Failed Baseline |

The Deep Learning model **BiLSTM** achieved the highest Macro-F1 of **0.9503** on the test set, demonstrating the value of sequential text processing over simple bag-of-words counting.

Η weighted δοκιμή ολοκληρώθηκε επιτυχώς τεχνικά, αλλά δεν κέρδισε το baseline: αύξησε το recall της κλάσης 0, όμως μείωσε precision, class-0 F1, συνολικό macro-F1 και accuracy. Θα επαληθεύσω τις τελικές τιμές και το καλύτερο validation checkpoint για να κλείσουμε το πείραμα με σαφές τελικό συμπέρασμα.

Θα εξετάσω πρώτα τις δύο εκτελέσεις, τις τελικές μετρικές και τις ρυθμίσεις τους· έπειτα θα συγκρίνω τις καμπύλες εκπαίδευσης ώστε να ξεχωρίσουμε πραγματική βελτίωση από πιθανό overfitting.

Η πρώτη εικόνα είναι σαφής: μόνο η νεότερη εκτέλεση κατέγραψε αποτελέσματα, ενώ η παλαιότερη ολοκληρώθηκε χωρίς καμία μετρική μοντέλου. Θα ελέγξω τώρα αν πρόκειται για ουσιαστικά διαφορετικό πείραμα ή για δοκιμαστική/κενή εκτέλεση, και πόσο σταθερή ήταν η εκπαίδευση του επιτυχημένου run.

Η σύγκριση δείχνει ότι τα configs είναι απολύτως ίδια. Το κενό run διήρκεσε ~8,4 λεπτά αλλά είχε 0% GPU utilization, ενώ το επιτυχημένο ολοκληρώθηκε σε ~2,1 λεπτά με ~47% μέση χρήση GPU—ένδειξη ότι το πρώτο μάλλον σταμάτησε πριν ξεκινήσει η εκπαίδευση. Θα επιβεβαιώσω την αιτία από τα logs και θα ανακτήσω τις ανά-epoch τιμές.

## Συμπέρασμα

Το έργο περιέχει **2 runs με ίδιο όνομα και ακριβώς ίδιο configuration**, αλλά ουσιαστικά υπάρχει **μόνο ένα έγκυρο πείραμα**. Το πρώτο run δεν ξεκίνησε εκπαίδευση λόγω σφάλματος import, ενώ το δεύτερο εκπαίδευσε επιτυχώς ένα BiLSTM και πέτυχε περίπου **96,3% accuracy** και **95,0% macro-F1** στο test set.

### 1. Επιτυχημένο run — `2aj9tzfv`

**Μοντέλο και δεδομένα**

- BiLSTM, 1 layer, hidden dimension 128
- Bidirectional: ναι
- Dropout: 0,3
- Pooling: τελευταίο hidden state
- Learning rate: 0,001
- Batch size: 128
- Weight decay: 0,0001
- Seed: 42
- Train/dev/test: **118.990 / 14.872 / 13.278** δείγματα
- Vocabulary: **15.138 tokens**

**Αποτελέσματα test**

| Μετρική | Τιμή |
|---|---:|
| Accuracy | **96,29%** |
| Macro-F1 | **95,03%** |

Η καλύτερη επίδοση validation εμφανίστηκε στο **3ο epoch**:

- dev accuracy: **96,48%**
- dev macro-F1: **94,93%**

Στη συνέχεια η training loss συνέχισε να μειώνεται, από **0,299 σε 0,076** συνολικά, αλλά οι validation μετρικές υποχώρησαν ελαφρά. Μετά από δύο epochs χωρίς βελτίωση ενεργοποιήθηκε σωστά το **early stopping** στο 5ο epoch. Αυτό δείχνει **ήπιο overfitting μετά το epoch 3**, όχι όμως σοβαρή αστάθεια ή έκρηξη του loss.

Η εκπαίδευση ολοκληρώθηκε σε περίπου **124 δευτερόλεπτα**, με μέση χρήση GPU περίπου **47%**.

### 2. Μη έγκυρο run — `aghjflvu`

Παρότι εμφανίζεται ως `finished`, δεν παρήγαγε μετρικές. Το log δείχνει:

```text
ModuleNotFoundError: No module named 'reviewnlp'
```

Άρα το preprocessing δεν ξεκίνησε λόγω λάθους στο Python package/module path. Το run παρέμεινε ενεργό περίπου **501 δευτερόλεπτα**, αλλά η GPU είχε **0% utilization**. Η κατάσταση `finished` είναι παραπλανητική· πιθανότατα το εξωτερικό command απέτυχε, αλλά το wrapper script τερμάτισε με exit code 0.
Αυτό είναι βελτίωση σε σχέση με τα προηγούμενα test αποτελέσματα και πλέον έχουμε και ανάλυση ανά κλάση. Θα επαληθεύσω ποιο config/seed χρησιμοποίησε το `mlluhlia` και αν οι νέες μετρικές αποθηκεύτηκαν στο W&B summary πριν συγκρίνω τις εκτελέσεις.

Η επαλήθευση δείχνει ότι το `mlluhlia` είναι seed 100 και όντως είναι το καλύτερο από τα τρία μοναδικά seeds. Υπάρχουν όμως διπλότυπα runs για seeds 50 και 100, οπότε θα υπολογίσω τη σταθερότητα μόνο πάνω στα τρία ανεξάρτητα αποτελέσματα.

## Το seed 100 είναι το καλύτερο αποτέλεσμα μέχρι τώρα

Το run `mlluhlia` ολοκληρώθηκε σωστά και πέτυχε:

- **Accuracy:** 96,42%
- **Macro-F1:** 95,21%
- **Weighted-F1:** 96,43%
- **Σωστές προβλέψεις:** 12.802/13.278
- **Λάθη:** 476 — error rate 3,58%

### Ανάλυση ανά κλάση

| Κλάση | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0 | 3.278 | 91,95% | 93,69% | 92,81% |
| 1 | 10.000 | 97,92% | 97,31% | 97,61% |

Το dataset έχει αναλογία περίπου **3,05:1** υπέρ της κλάσης 1. Η κλάση 0 παραμένει η δυσκολότερη:

- 207 δείγματα της κλάσης 0 ταξινομήθηκαν ως 1 — ποσοστό λάθους **6,31%**
- 269 δείγματα της κλάσης 1 ταξινομήθηκαν ως 0 — ποσοστό λάθους **2,69%**

Άρα το ποσοστό λάθους στην κλάση 0 είναι περίπου **2,3 φορές μεγαλύτερο**, παρότι συνολικά η επίδοση είναι πολύ καλή.

## Σταθερότητα μεταξύ seeds

Αφαιρώντας τα διπλότυπα runs, υπάρχουν τρία ανεξάρτητα αποτελέσματα:

| Seed | Test accuracy | Test macro-F1 |
|---:|---:|---:|
| 42 | 96,29% | 95,03% |
| 50 | 96,23% | 94,97% |
| **100** | **96,42%** | **95,21%** |

Στα τρία seeds:

- Μέση accuracy: **96,31% ± 0,09 ποσοστιαίες μονάδες**
- Μέσο macro-F1: **95,07% ± 0,12 ποσοστιαίες μονάδες**

Η πολύ μικρή διακύμανση δείχνει ότι το αποτέλεσμα είναι **σταθερό ως προς το seed**. Η βελτίωση του seed 100 είναι πραγματική αριθμητικά, αλλά μικρή—δεν αποτελεί διαφορετικό μοντέλο ή ουσιαστικό άλμα.

Σημείωση: τα `macro_precision`, `macro_recall`, `weighted_f1`, `per_class` και `confusion_matrix` εμφανίστηκαν στην έξοδο, αλλά **δεν αποθηκεύτηκαν στο W&B summary**. Στα επόμενα runs αξίζει να καταγραφούν ώστε να παραμένουν συγκρίσιμα στο workspace.

Η πιο χρήσιμη επόμενη δοκιμή είναι class-weighted loss ή threshold tuning πάνω στο **dev set**, με κύριο objective το macro-F1/κλάση-0 F1· το test set πρέπει να χρησιμοποιηθεί μόνο μετά την επιλογή του threshold.

## Τι σημαίνουν συνολικά τα αποτελέσματα

Το BiLSTM έχει ισχυρό αποτέλεσμα, αλλά δεν μπορούμε ακόμη να εκτιμήσουμε αξιόπιστα τη γενίκευση ή να πούμε ότι είναι η καλύτερη επιλογή, επειδή υπάρχει:

- μόνο **ένα επιτυχημένο seed**,
- κανένα baseline για σύγκριση,
- καμία μετρική ανά κλάση ή confusion matrix,
- καμία δοκιμή άλλων hyperparameters.

Η πιο χρήσιμη επόμενη κίνηση είναι να επαναληφθεί το ίδιο configuration με **3–5 διαφορετικά seeds** και να καταγραφούν mean ± standard deviation για test macro-F1. Έπειτα αξίζει σύγκριση του `pooling="last"` με mean/max pooling και ενός απλού baseline. Επίσης, το training wrapper πρέπει να προωθεί non-zero exit codes ώστε αποτυχημένα commands να σημειώνονται ως `failed` αντί για `finished`.

## Τελικό συμπέρασμα: κρατάμε το **unweighted BiLSTM**

Το weighted cross-entropy **δεν βελτίωσε** τον βασικό στόχο. Σε αυστηρή σύγκριση με ίδιο seed (`100`), ίδιο split και ίδια hyperparameters:

| Seed 100 | Best dev macro-F1 | Test accuracy | Test macro-F1 | Test errors |
|---|---:|---:|---:|---:|
| Unweighted baseline | **94.81%** | **96.42%** | **95.21%** | **476** |
| Weighted CE | 94.52% | 96.14% | 94.92% | 513 |
| Weighted − baseline | −0.29 pp | −0.28 pp | −0.29 pp | +37 |

### Τι συνέβη στην κλάση 0

Η στάθμιση πέτυχε τον άμεσο στόχο της, δηλαδή μεγαλύτερο minority recall, αλλά με υπερβολικό κόστος σε precision:

| Class 0 | Unweighted | Weighted | Μεταβολή |
|---|---:|---:|---:|
| Precision | **91.95%** | 89.51% | −2.44 pp |
| Recall | 93.69% | **95.55%** | +1.86 pp |
| F1 | **92.81%** | 92.43% | −0.38 pp |

Η confusion matrix άλλαξε από:

```text
Unweighted: [[3071, 207],
             [ 269, 9731]]
```

σε:

```text
Weighted:   [[3132, 146],
             [ 367, 9633]]
```

Δηλαδή διορθώθηκαν 61 λάθη της κλάσης 0, αλλά δημιουργήθηκαν 98 επιπλέον λάθη στην κλάση 1: καθαρά **37 περισσότερα λάθη**.

## Αποτέλεσμα των τριών baseline seeds

Για το unweighted μοντέλο, τα τρία μοναδικά seeds δίνουν:

- **Test accuracy:** `96.31% ± 0.09 pp`
- **Test macro-F1:** `95.07% ± 0.13 pp`
- Καλύτερο run: seed `100`, accuracy `96.42%`, macro-F1 `95.21%`

Η μικρή διακύμανση δείχνει ότι το αποτέλεσμα είναι αρκετά σταθερό. Δεν συνιστώ να ξοδέψεις χρόνο σε δύο ακόμη weighted seeds, επειδή το weighted run έχασε ήδη τόσο στο validation όσο και στο test με ελεγχόμενη σύγκριση ίδιου seed.


### Report-ready κείμενο

> Across three random seeds, the unweighted BiLSTM achieved a mean test accuracy of 96.31% ± 0.09 percentage points and a mean macro-F1 of 95.07% ± 0.13. The best run, using seed 100, reached 96.42% accuracy and 95.21% macro-F1. Balanced class-weighted cross-entropy increased minority-class recall from 93.69% to 95.55%, but reduced its precision from 91.95% to 89.51%. Consequently, minority-class F1 decreased from 92.81% to 92.43%, while overall test macro-F1 decreased from 95.21% to 94.92%. We therefore retained the unweighted cross-entropy objective as the final configuration.

Η διπλή εκτύπωση της γραμμής `vocab=...` είναι απλώς επειδή παρέμειναν δύο ίδια `print()` μέσα στη συνάρτηση· είναι ακίνδυνο και μπορείς να διαγράψεις το ένα.
