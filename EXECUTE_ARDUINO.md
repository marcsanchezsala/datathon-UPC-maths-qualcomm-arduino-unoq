# Guia d'Execució i Verificació a l'Arduino UNO Q (IP: 192.168.1.85)

Aquest document detalla pas a pas com connectar-se, verificar els serveis d'inferència, validar l'índex vectorial, executar proves de fum (*smoke tests*), mesurar la precisió amb `eval.py` i generar prediccions amb `predict.py` a la placa física **Arduino UNO Q (4 GB RAM)** amb processador **Qualcomm QRB2210**.

---

## 1. Connexió a la Placa (SSH i USB Fallback)

La placa es troba connectada a la xarxa local amb la IP assignada **`192.168.1.85`**.

### 1.1 Accés per SSH (Wi-Fi / LAN)
Des del terminal del teu ordinador (portàtil a la mateixa xarxa Wi-Fi):

```bash
ssh arduino@192.168.1.85
```
- **Usuari:** `arduino`
- **Contrasenya:** `datathon2026` (o `opensource` segons la imatge del sistema)

*Nota:* Si és el primer cop que et connectes, accepta l'empremta de clau RSA/ECDSA escrivint `yes`.

### 1.2 Accés alternatiu via USB (ADB Fallback)
En cas que la xarxa Wi-Fi no estigui disponible o caigui durant la presentació:
1. Connecta la placa directament al portàtil mitjançant el cable USB-C.
2. Comprova la connexió ADB:
   ```bash
   adb devices
   ```
3. Obre una sessió de terminal directe:
   ```bash
   adb shell
   # Un cop a dins, canvia a l'usuari arduino si cal:
   su - arduino
   ```

---

## 2. Preparació de l'Entorn a la Placa

Un cop dins la sessió SSH a la placa:

```bash
# Navega al directori del projecte (utilitza ~/copilot-prediction per a l'execució aïllada de producció)
cd ~/copilot-prediction
# (o ~/datathon-UPC-maths-qualcomm-arduino-unoq)

# Activa l'entorn virtual de Python
source .venv/bin/activate

# Comprova la versió de Python i les eines
python3 --version
```

---

## 3. Verificació dels Serveis d'Inferència (Ports 8081 i 8082)

El sistema utilitza dos servidors d'inferència locals basats en `llama.cpp` orquestrats com a serveis `systemd`:
- **Port 8081 (`math-llm`):** Servidor del Model de Llenguatge (`Qwen2.5-3B-Instruct` o `Qwen2.5-Math-1.5B`).
- **Port 8082 (`math-embed`):** Servidor d'Incrustacions Vectorials (`bge-small-en-v1.5`, vectors de 384 dimensions).

### 3.1 Comprovar l'estat dels serveis amb systemd

```bash
sudo systemctl status math-llm
sudo systemctl status math-embed
```
Els dos serveis han d'aparèixer amb l'estat **`active (running)`** en color verd.

### 3.2 Health Check directe mitjançant HTTP / curl

Comprova que les APIs responen correctament en local:

```bash
# Health check del servidor LLM (Port 8081)
curl -sf http://127.0.0.1:8081/health && echo "LLM server: OK (Port 8081)"

# Health check del servidor d'Embeddings (Port 8082)
curl -sf http://127.0.0.1:8082/health && echo "Embed server: OK (Port 8082)"
```

### 3.3 Test ràpid d'embeddings amb Python

Verifica que el model d'embeddings calcula distàncies semàntiques vàlides:

```bash
python3 - <<'EOF'
import time
from copilot.embed import embed
t0 = time.perf_counter()
v1, v2, v3 = embed([
    "A triangle has three interior angles.",
    "Every triangle consists of three edges.",
    "Chocolate cake tastes sweet."
])
sim_geom = sum(a * b for a, b in zip(v1, v2))
sim_diff = sum(a * b for a, b in zip(v1, v3))
print(f"Dimensions: {len(v1)} | Temps: {time.perf_counter() - t0:.2f}s")
print(f"Similitud semàntica geomètrica: {sim_geom:.3f} | No relacionada: {sim_diff:.3f}")
assert sim_geom > sim_diff + 0.2, "Error en les incrustacions"
print("✓ Servidor d'embeddings validat correctament!")
EOF
```

---

## 4. Verificació de la Base de Dades Vectorial (`data/index.db`)

El sistema compta amb un índex SQLite d'alta velocitat que emmagatzema els **2.932 problemes** del conjunt d'entrenament M3Kang per al Tier 0 RAG.

### 4.1 Comprovació del nombre de registres amb SQLite

```bash
sqlite3 data/index.db "SELECT COUNT(*) AS total_problemes FROM problems;"
```
**Sortida esperada:**
```text
2932
```

### 4.2 Inspecció del contingut i esquema

```bash
sqlite3 data/index.db "SELECT id, language, level, label FROM problems LIMIT 5;"
```

---

## 5. Execució de Proves de Fum (*Smoke Test*)

El test de fum valida tot el flux d'extrem a extrem sobre els 5 problemes de mostra (`data/sample_problems.jsonl`):

### 5.1 Executar el script de test automàtic

```bash
./board/05_smoke_test.sh
```

O manualment mitjançant `predict.py`:

```bash
# Executa la predicció sobre la mostra
python3 predict.py --input data/sample_problems.jsonl --output results/smoke.csv

# Visualitza els resultats generats
cat results/smoke.csv
```

**Sortida típica esperada a `results/smoke.csv`:**
```csv
id,language,answer,seconds,status
sample_001,eng,B,0.3,ok
sample_002,eng,A,0.4,ok
sample_003,cat,C,0.3,ok
sample_004,cat,D,0.2,ok
sample_005,eng,E,0.3,ok
```

Totes les respostes han d'estar compreses estrictament entre `A` i `E` i el temps per problema ha de ser molt inferior als 60 segons de límit màxim (normalment $< 1.0\text{ s}$ en Tier 1 o $< 15.0\text{ s}$ en CoT).

---

## 6. Avaluació de Rendiment (`eval.py`)

El script `eval.py` reprodueix el protocol oficial de competició, calculant la precisió global, per idioma, per nivell i mesurant la latència exacta.

### 6.1 Avaluació ràpida (Mostra de 50 problemes de validació)

```bash
python3 eval.py --limit 50 --name board_val50
```
- Mesura els primers 50 problemes de `data/val.jsonl`.
- Mostra taules detallades de precisió per idioma (`cat` / `eng`) i per nivell educatiu (0 a 7).
- Registra la mètrica obtinguda a `results/ablations.csv`.

### 6.2 Avaluació oficial completa (562 problemes)

```bash
python3 eval.py --limit 0 --name board_full_official
```
- Executa la totalitat del conjunt de validació oficial.
- La precisió esperada és del **28.8% global (162/562)** i superior al **30% en problemes amb opcions**.
- 0 timeouts (> 60s) garantits.

---

## 7. Generació de Prediccions Oficials (`predict.py`)

Per generar el fitxer CSV definitiu d'un conjunt de problemes:

```bash
python3 predict.py \
    --input data/val.jsonl \
    --output results/predictions_board.csv \
    --index data/index.db
```

Paràmetres clau:
- `--input`: Fitxer JSONL d'entrada (amb enunciat, llengua i opcions).
- `--output`: Fitxer CSV generat amb columnes `id,language,answer,seconds,status`.
- `--rag 0`: Ús del mode per defecte (Tier 0 RAG determinista actiu automàticament per a duplicats, sense sobrecàrrega de context a Tier 1).

---

## 8. Interacció Física amb la Matriu de LEDs 8x13

L'Arduino UNO Q disposa d'una arquitectura de doble nucli:
1. **Linux (SoC Qualcomm QRB2210):** Executa la intel·ligència artificial i el copilot en Python.
2. **Microcontrolador (STM32 MCU):** Controla el maquinari físic i la matriu de LEDs de $8\times 13$.

La comunicació entre ambdós dominis es realitza mitjançant l'**Arduino Router Bridge**:
- Quan el copilot determina la resposta (`A`, `B`, `C`, `D` o `E`), crida a la funció RPC:
  ```python
  from arduino_app_lab import Bridge
  Bridge.call("draw_letter", answer)
  ```
- El microcontrolador rep la lletra i il·lumina el glif corresponent de $5\times 7$ píxels centrat a la matriu de $8\times 13$.
- Durant la inferència d'un raonament complex (Micro-CoT), la matriu pot mostrar una animació de "pensant" (poliació suau dels píxels centrals).

---

## 9. Guia de Resolució de Problemes (*Troubleshooting*)

### 9.1 Un dels servidors no respon o ha caigut
Si `curl` al port 8081 o 8082 retorna *Connection refused*:
```bash
# Reiniciar el servei corresponent
sudo systemctl restart math-llm
sudo systemctl restart math-embed

# Comprovar els logs en temps real per veure l'error
journalctl -u math-llm -f -n 50
```

### 9.2 Monitorització de Memòria RAM i Processador
La placa compta amb 4 GB de RAM LPDDR4 compartida. Per verificar l'estat dels recursos:
```bash
free -h
```
- L'ús típic de RAM és de $\approx 2.4\text{ GB}$ a $2.8\text{ GB}$ amb els dos models carregats a memòria.
- Si la memòria lliure és inferior a 250 MB, atura altres aplicacions o buida la memòria cau:
  ```bash
  sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
  ```

### 9.3 Conflictes de Ports o Processos Residuals
Si el servei no pot obrir el port 8081 o 8082 perquè ja està ocupat:
```bash
sudo lsof -i :8081
sudo lsof -i :8082

# Aturar qualsevol instància manual de llama-server
sudo killall -9 llama-server
sudo systemctl restart math-llm
sudo systemctl restart math-embed
```

### 9.4 Control de Temperatura del SoC Qualcomm
Si la placa s'escalfa durant avaluacions massives continuades:
```bash
cat /sys/class/thermal/thermal_zone0/temp
```
(Divideix el valor per 1000 per obtenir els graus Celsius).
Valors per sota de 75°C són normals i garanteixen que el processador no fa *thermal throttling*.
