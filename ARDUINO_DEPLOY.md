# Desplegament de M3Kang Teaching Copilot a l'Arduino UNO Q (4 GB)

Aquesta guia detalla el procés complet pas a pas per desplegar i executar el **M3Kang Teaching Copilot** a la placa **Arduino UNO Q (4 GB RAM)**, funcionant 100% offline sense connexió a internet ni dependència del núvol.

---

## 1. Arquitectura de Hardware i Sistema

L'Arduino UNO Q combina dos entorns de computació en una única placa:
- **SoC Qualcomm QRB2210**: Processador quad-core ARM Cortex-A53 a 2.0 GHz amb 4 GB de memòria RAM LPDDR4 i emmagatzematge eMMC. Executa Linux Debian (Debian GNU/Linux 12 bookworm). Aquí s'executen els servidors d'inferència de `llama.cpp` i el servidor web del Teaching Copilot.
- **Microcontrolador STM32**: Gestiona el control de baix nivell en temps real i la matriu de LEDs 8x13 integrada per a senyalització hàptica/visual i telemetria d'estat de l'alumne.
- **Comunicació Inter-SoC**: Connectats mitjançant el bridge intern d'Arduino (RPC serial intern).

```
+--------------------------------------------------------------------------+
|                           Arduino UNO Q (4 GB)                           |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  | Qualcomm QRB2210 (Debian Linux, 4 GB RAM)                          |  |
|  |                                                                    |  |
|  |  +----------------------+  +------------------------+              |  |
|  |  | math-embed.service   |  | math-llm.service       |              |  |
|  |  | (bge-small-en-v1.5)  |  | (Qwen3.5-0.8B Q8_0)    |              |  |
|  |  | port: 8082           |  | port: 8081             |              |  |
|  |  +----------^-----------+  +-----------^------------+              |  |
|  |             | HTTP (127.0.0.1)         | HTTP (127.0.0.1)          |  |
|  |  +----------+--------------------------+------------+              |  |
|  |  | M3Kang Teaching Copilot (explorer/server.py:8000)|              |  |
|  |  | - Socratic Domain Tutor                          |              |  |
|  |  | - Trap / Distractor Detector                    |              |  |
|  |  | - Spaced Repetition Engine                       |              |  |
|  |  | - Local Vector RAG (SQLite: data/index.db)       |              |  |
|  |  +--------------------------+-----------------------+              |  |
|  +-----------------------------|--------------------------------------+  |
|                                |                                         |
|  +-----------------------------v--------------------------------------+  |
|  | Microcontrolador STM32                                             |  |
|  | - Matriu LED 8x13 per a feedback visual/estat                      |  |
|  +--------------------------------------------------------------------+  |
+--------------------------------------------------------------------------+
```

---

## 2. Preparació i Transferència del Codi a la Placa

Connecta't a l'Arduino UNO Q mitjançant SSH o shell ADB:

```bash
# Opció A: Connexió per ADB shell (USB)
adb shell

# Opció B: Connexió per SSH (IP de la placa)
ssh arduino@<ip-arduino>
```

Assegura't de tenir clonat el repositori a la placa i posiciona't a la branca `teaching`:

```bash
cd ~/datathon-teaching
git fetch origin
git checkout teaching
git pull origin teaching
```

Si treballes en local i transfereixes els arxius modificats via ADB o SCP:
```bash
# Des del teu ordinador de desenvolupament:
adb push copilot /home/arduino/datathon-teaching/
adb push explorer /home/arduino/datathon-teaching/
adb push build_index.py /home/arduino/datathon-teaching/
```

---

## 3. Verificació dels Serveis d'Inferència Locals

El Teaching Copilot requereix dos serveis `systemd` que allotgen els models en memòria mitjançant `llama-server`:
1. `math-llm.service` (Port 8081): **Qwen3.5 0.8B Q8_0** (generador de raonament Micro-CoT, pistes socràtiques i anàlisi de trampes).
2. `math-embed.service` (Port 8082): **bge-small-en-v1.5 Q8_0** (generació d'embeddings per a la cerca semàntica RAG).

Comprova l'estat dels serveis a la placa:

```bash
sudo systemctl status math-llm.service
sudo systemctl status math-embed.service
```

Tots dos han de mostrar `active (running)`. Si no estan encesos:

```bash
sudo systemctl restart math-llm.service math-embed.service
```

Pots verificar que responen correctament amb una crida local:

```bash
# Comprovació del model de llenguatge:
curl -s http://127.0.0.1:8081/health

# Comprovació del model d'embeddings:
curl -s http://127.0.0.1:8082/health
```

---

## 4. Construcció de l'Índex Vectorial RAG

Genera la base de dades vectorial SQLite local (`data/index.db`) a partir dels problemes d'entrenament. L'script processa els enunciats a través del servei `math-embed` (port 8082):

```bash
python3 build_index.py --input data/train.jsonl --db data/index.db --only-with-options
```

> **Nota**: `build_index.py` accepta tant `--input data/train.jsonl` com `--input data/train_full.jsonl`. Utilitza codificació UTF-8 nativa i crea directoris automàticament. L'índex resultant ocupa menys de 2 MB i es carrega instantàniament en RAM.

---

## 5. Execució del Servidor Teaching Copilot

Inicia el servidor web interactiu amb suport per a la càrrega de problemes de validació i l'índex vectorial:

```bash
python3 explorer/server.py --input data/val.jsonl --index data/index.db --host 0.0.0.0 --port 8000
```

El servidor s'inicia i queda escoltant al port `8000`.

---

## 6. Connexió i Accés des del Portàtil

Per presentar o provar el sistema des del teu ordinador portàtil:

### Mètode A: Connexió Directa per USB via ADB (Recomanat per a la Demo Oficial)
Aquest és el mètode més fiable i robust perquè **no requereix Wi-Fi**, és immune a saturacions de xarxa en esdeveniments públics i garanteix zero latència de xarxa:

1. Connecta el cable USB-C entre el teu portàtil i l'Arduino UNO Q.
2. Executa al terminal del teu portàtil:
   ```bash
   adb forward tcp:8000 tcp:8000
   ```
3. Obre el navegador al portàtil a:
   ```
   http://127.0.0.1:8000
   ```

### Mètode B: Connexió per Xarxa Local (Wi-Fi / Ethernet)
Si tant la placa com el portàtil estan a la mateixa xarxa:
1. Obtén la IP de la placa (`ip a show wlan0` o `hostname -I`).
2. Obre al navegador:
   ```
   http://<ip-arduino>:8000
   ```

---

## 7. Monitoratge de Recursos i Límits Edge

L'Arduino UNO Q disposa de 4 GB de memòria RAM compartida (entre CPU i GPU/NPU Adreno). La nostra arquitectura ha estat estrictament dissenyada per operar amb un marge segur d'estabilitat:

### Consum de Memòria RAM:
Pots monitorar la RAM en temps real amb:
```bash
watch -n 1 free -m
```

Perfil d'ús típic en execució completa:
- **Linux OS + Serveis base**: ~400 - 500 MB
- **math-llm (Qwen3.5 0.8B Q8_0 amb context 4096)**: ~1,150 MB
- **math-embed (bge-small-en-v1.5 Q8_0 amb context 512)**: ~180 MB
- **Teaching Server + Cache d'estat**: ~120 MB
- **Consum Total de RAM**: **~2.2 - 2.5 GB**
- **RAM Lliure Disponible**: **> 1.5 GB** (evitant qualsevol risc de saturació o OOM Killer).

### Latències i Límits de Temps (< 60s per norma oficial):
- Totes les consultes d'inferència pedagògica tenen un `timeout=35` segons com a mecanisme de contenció estricta.
- Gràcies a l'enfocament de prompts de domini concisos i generacions acotades (Micro-CoT de 20-30 tokens per pas socràtic o anàlisi de parany):
  - **Cerca d'embedding RAG**: ~30 - 60 ms.
  - **Pista Socràtica (1 pas)**: **1.2 - 3.8 segons**.
  - **Anàlisi de Parany / Distractor**: **1.5 - 4.1 segons**.
  - Totes les respostes es troben entre un **93% i 97% per sota del límit màxim de 60 segons**.

---

## 8. Verificació de Funcionament (Checklist Final abans del Jurat)

1. [ ] Els serveis `math-llm` i `math-embed` estan en estat `active (running)`.
2. [ ] `free -m` mostra almenys 1.4 GB lliures de memòria disponible.
3. [ ] `data/index.db` s'ha creat correctament amb `build_index.py`.
4. [ ] `adb forward tcp:8000 tcp:8000` està configurat al portàtil.
5. [ ] El navegador carrega la interfície web i respon a un pas socràtic en menys de 4 segons.
