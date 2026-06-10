# End-to-End Benchmark Pipeline
## One-Click Nutzung
Master-Config Vorlage erzeugen:
```powershell
python run_pipeline.py --init-master-config
```

Dann `eval_data/master_config.xlsx` ausfuellen und Pipeline starten:
```powershell
python run_pipeline.py
```

Optional einzelne Werte per CLI ueberschreiben (CLI > Master config):
```powershell
python run_pipeline.py --num-runs 8 --models openai/gpt-4o-mini
```

Direkt ohne Master-Config:
```powershell
python run_pipeline.py --input-file ./eval_data/questions.json --models google/gemini-2.0-flash-exp:free
```

Optional:
```powershell
python run_pipeline.py `
  --input-file ./eval_data/questions.json `
  --models google/gemini-2.0-flash-exp:free openai/gpt-4o-mini `
  --num-runs 5
```

## Master-Config Struktur (`master_config.xlsx`)
- Sheet `settings`: Spalten `key`, `value`
  - Beispiele: `num_runs`, `seed`, `temperature`, `default_max_words`, `judge_model`, `runs_root`, `auto_reindex`, `catalog_excel`, `auto_convert_questions`
  - Prompt-Pfade: `prompt_zero_shot_path`, `prompt_advanced_path`, `prompt_rag_path`, `judge_prompt_path`
- Sheet `models`: Spalten `model`, `enabled`, `temperature`
  - `enabled=true` aktiviert Modell
  - `temperature` kann pro Modell gesetzt werden
- Sheet `methods`: Spalten `method`, `enabled`
  - Aktiviert/deaktiviert `Zero-Shot`, `Advanced-Prompt`, `RAG`

## Hinweise
- Jede Frage soll `atomic_facts` enthalten (eine Spalte oder `atomic_fact_1..n`).
- Die Auswertung nutzt keine binaeren/extraktiven Regeln mehr.
- Bei `auto_reindex=true` wird die Chroma-DB nur neu gebaut, wenn sich Dateien in `processed_txt/rag_kb` geaendert haben.
- Bei `auto_convert_questions=true` wird der Fragenkatalog aus `catalog_excel` vor jedem Lauf nach `input_file` exportiert.
- Prompt-Vorlagen liegen in `prompts/*.txt` und werden ueber die Pfade in `settings` gesteuert.
- API-Schluessel muss in `.env` als `OPENROUTER_API_KEY` vorhanden sein.
- `enable_visualizations` steuert die Figure-Erzeugung explizit; bei `false` bleiben Figure-Artefakte deaktiviert.
- Die primaere Inferenz erfolgt ausschliesslich auf Nicht-Fangfragen auf Task-Ebene (`Model + ID`) via gepaartem Wilcoxon Signed-Rank Test.
- Die Wahl des Wilcoxon-Tests ist fuer kleine Samples, ordinale Likert-Scores und fehlende Normalverteilungsannahmen gedacht.
- Kosten und Laufzeit bleiben deskriptiv; sie werden nicht inferenzstatistisch getestet.
- Multiple Testing wird per Benjamini-Hochberg Korrektur pro Modell ueber alle Methodenvergleiche gegen die Baseline behandelt.
- Confidence wird nur aus expliziten Fact-Markern `#0`, `#1`, `#2`, `#3` in den Modellantworten geparst; dabei gilt `#0 = sehr sicher` und `#3 = stark unsicher/potenziell falsch`.
- Fuer die Auswertung wird daraus eine normierte Sicherheit berechnet: `confidence_norm = 1 - (confidence_mean / 3)`.
- Die primaere Qualitaetsmetrik ist der referenzzentrierte `Atomic_Coverage_Score` (`Support_Rate` bleibt als kompatibler Alias erhalten).





benchmark.py ist jetzt der Benchmark-only Einstieg, run_pipeline.py der End-to-End-Runner.

Nur Benchmark:

python benchmark.py --auto-convert --catalog-excel ./eval_data/Fragenkatalog_MA.xlsx
Mit Master-Config:

python benchmark.py --master-config ./eval_data/master_config.xlsx
End-to-End Benchmark + Evaluation + Analyse:

python run_pipeline.py
Falls du erst die Config-Vorlage neu erzeugen willst:

python benchmark.py --init-master-config
Wichtig:

.env muss OPENROUTER_API_KEY enthalten.
Der Fragenkatalog bleibt eval_data/Fragenkatalog_MA.xlsx, daraus wird bei --auto-convert nach eval_data/questions.json geschrieben.
Wenn du in VS Code startest, kannst du auch die Task Benchmark (One Click) verwenden.
