# Rilevamento di Volti e Riconoscimento delle Emozioni

[cite_start]Questo progetto implementa un sistema di visione artificiale in grado di rilevare volti umani all'interno di immagini statiche e di riconoscerne le espressioni emotive[cite: 4, 5]. [cite_start]Il sistema combina un approccio classico per il rilevamento dei volti con una rete neurale convoluzionale (CNN) pre-addestrata per la classificazione delle emozioni[cite: 4, 6, 9].

## 🎯 Obiettivo del Progetto

[cite_start]Lo scopo principale è sviluppare un sistema completo di analisi facciale che integri due compiti fondamentali[cite: 4]:
1.  [cite_start]**Rilevamento dei Volti**: Identificare e isolare le regioni facciali in un'immagine[cite: 5, 7].
2.  [cite_start]**Riconoscimento delle Emozioni**: Classificare l'espressione emotiva di ogni volto rilevato[cite: 5, 8].

[cite_start]Il sistema è progettato per analizzare immagini statiche, fornendo in output l'immagine originale con i volti cerchiati e annotati con l'emozione predetta[cite: 13, 15].

## ⚙️ Architettura del Sistema

Il progetto è suddiviso in due moduli principali che lavorano in sequenza.

### 1. Rilevamento dei Volti (C++ e OpenCV)

[cite_start]Il primo componente utilizza l'**algoritmo di Viola-Jones**, un metodo classico noto per la sua efficienza e accuratezza[cite: 6]. [cite_start]Questa parte è implementata in **C++** sfruttando la libreria **OpenCV** e il suo framework di classificatori a cascata per localizzare i volti all'interno di un'immagine[cite: 7].

### 2. Riconoscimento delle Emozioni (Python e CNN)

[cite_start]Una volta che un volto viene rilevato, la sua regione viene passata al secondo modulo[cite: 8]. [cite_start]Questo componente impiega una **Rete Neurale Convoluzionale (CNN)** pre-addestrata sul dataset FER-2013[cite: 9]. [cite_start]Il modulo, implementato in **Python**, classifica l'espressione facciale in una delle sette categorie predefinite[cite: 10]:
* [cite_start]Arrabbiato [cite: 10]
* [cite_start]Disgustato [cite: 10]
* [cite_start]Spaventato [cite: 10]
* [cite_start]Felice [cite: 10]
* [cite_start]Triste [cite: 10]
* [cite_start]Sorpreso [cite: 10]
* [cite_start]Neutrale [cite: 10]

## 📊 Dataset Utilizzati

### Dataset di Addestramento (FER-2013)

[cite_start]La CNN per il riconoscimento delle emozioni è stata addestrata sul **dataset FER-2013**[cite: 9, 42]. [cite_start]Questo dataset è composto da 35.887 immagini in scala di grigi di dimensioni 48x48 pixel, ciascuna etichettata con una delle sette emozioni[cite: 42].

### Dataset di Test

[cite_start]Per la valutazione complessiva del sistema, è stato utilizzato un dataset di test separato contenente 46 immagini[cite: 52]. [cite_start]Ogni immagine è annotata con i riquadri di delimitazione (bounding box) per tutti i volti visibili e le etichette emotive corrispondenti[cite: 53, 54].

## 📈 Metriche di Performance

[cite_start]La valutazione del sistema è stata condotta a più livelli per misurare l'efficacia di ciascun componente e del flusso di lavoro completo[cite: 26].

* **Valutazione del Rilevamento dei Volti**:
    * [cite_start]**Intersection over Union (IoU)**: Per misurare l'accuratezza dei riquadri di delimitazione predetti rispetto a quelli reali[cite: 28].
    * [cite_start]**Precision e Recall**: Per valutare il trade-off tra rilevamenti corretti, mancati e falsi, basandosi su una soglia IoU (comunemente 0.5)[cite: 29].

* **Valutazione del Riconoscimento delle Emozioni**:
    * [cite_start]**Accuracy di Classificazione**: Per misurare la proporzione di emozioni predette correttamente su tutti i volti rilevati[cite: 31].
    * [cite_start]**Matrice di Confusione**: Per analizzare nel dettaglio le performance della CNN su ciascuna categoria di emozione[cite: 32].

* **Valutazione a Livello di Sistema**:
    * [cite_start]La metrica principale è la **percentuale di volti rilevati con emozioni classificate correttamente**, per valutare le prestazioni end-to-end del sistema[cite: 34].

## 💡 Applicazioni Potenziali

[cite_start]Questo sistema può essere applicato in diversi domini[cite: 21]:
* [cite_start]**Interazione Uomo-Macchina (HCI)**: Creare sistemi più empatici e reattivi in grado di interpretare gli stati emotivi degli utenti[cite: 22].
* [cite_start]**Analisi Comportamentale**: Fornire insight in contesti come l'istruzione, il retail e la sicurezza, monitorando le risposte emotive delle persone[cite: 23].
* [cite_start]**Gestione di Fotografie**: Organizzare e taggare automaticamente le immagini in base alle espressioni facciali, semplificando la ricerca in grandi archivi fotografici[cite: 24].

## ⚠️ Limitazioni

Il sistema presenta alcune limitazioni importanti:
* [cite_start]Funziona **esclusivamente su immagini statiche** e non supporta l'elaborazione in tempo reale[cite: 17].
* [cite_start]Il riconoscimento delle emozioni dipende interamente dal successo del rilevamento dei volti[cite: 18]. [cite_start]Se l'algoritmo di Viola-Jones non rileva un volto, questo **non verrà analizzato** per l'emozione[cite: 19].

## 🚀 Come Eseguire il Progetto

*Per istruzioni dettagliate sull'installazione delle dipendenze e sull'esecuzione del codice, fare riferimento alla documentazione specifica all'interno del repository.*

**Prerequisiti (esempio):**
* C++ Compiler (GCC, Clang, etc.)
* OpenCV
* Python 3.x
* TensorFlow/Keras
* NumPy

**Esecuzione (esempio):**
```bash
# Compila ed esegui il modulo di rilevamento volti
g++ -o face_detection main.cpp `pkg-config --cflags --libs opencv4`
./face_detection input.jpg

# Esegui lo script di riconoscimento delle emozioni
python emotion_recognition.py --image output_faces/face_1.jpg
