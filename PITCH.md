# Guió Oficial del Pitch: M3Kang Teaching Copilot (3 Minuts)

Aquest document conté el guió complet, sincronització temporal (timings) i accions en directe per a la presentació davant el jurat de la Datathon UPC & Qualcomm.

- **Temps total**: 3:00 minuts (180 segons)
- **Dispositiu**: Arduino UNO Q (4 GB RAM, Qualcomm QRB2210)
- **Connexió**: 100% Offline via cable USB (`adb forward tcp:8000 tcp:8000`)
- **Estructura oficial**: Basada en la secció 14.5 del reglament de la datathon.

---

## Cronograma d'un Cop d'Ull (Timeline)

| Bloc | Temps | Durada | Tema | Acció en Directe |
|---|---|---|---|---|
| **1** | 0:00 - 0:20 | 20 s | **El Problema** | Mostrar la placa desconnectada de la xarxa |
| **2** | 0:20 - 0:50 | 30 s | **La Solució i Arquitectura Edge** | Ensenyar l'arquitectura local (Qwen 0.8B + BGE) |
| **3** | 0:50 - 1:50 | 60 s | **La Demo en Viu (El Plat Fort)** | 1. Socràtic 2. Anàlisi Paranys 3. Spaced Repetition |
| **4** | 1:50 - 2:30 | 40 s | **Les Xifres i Mètriques Edge** | Mostrar `free -m` (2.3 GB) i latències (1-4s) |
| **5** | 2:30 - 3:00 | 30 s | **Aprenentatges i Futur** | Lliçons apreses i matriu LED 8x13 |

---

## Guió Pas a Pas amb Notes per a l'Orador

### Bloc 1: El Problema (0:00 - 0:20 | 20 segons)
- **Objectiu**: Connectar immediatament amb el jurat; plantejar el dilema real de l'aprenentatge matemàtic a les aules.
- **Acció**: L'orador subjecta la placa Arduino UNO Q amb una mà. El portàtil té el Wi-Fi desactivat.

> **Guió verbal**:
> *"Bon dia. Els problemes de les proves Cangur (M3Kang) no mesuren memorització: mesuren pensament crític. Però a l'aula, quan un alumne es bloqueja, els assistents d'IA convencionals cometen dos pecats greus: primer, li donen la solució directa arruïnant el procés pedagògic; segon, depenen de connexions al núvol cares i prohibides en molts exàmens.*
> *La nostra pregunta va ser: pot un dispositiu que cap al palmell de la mà convertir-se en un tutor matemàtic socràtic, 100% autònom i sense enviar ni un sol bit a internet?"*

---

### Bloc 2: La Solució i Arquitectura Edge (0:20 - 0:50 | 30 segons)
- **Objectiu**: Explicar com encaixa la tecnologia a l'Arduino UNO Q.
- **Acció**: Diapositiva 2 en pantalla mostrant l'esquema de memòria i els dos models.

> **Guió verbal**:
> *"La resposta és el **M3Kang Teaching Copilot**, executant-se completament dins l'Arduino UNO Q amb SoC Qualcomm QRB2210.*
> *Hem dissenyat una arquitectura dual optimitzada amb llama.cpp:*
> *D'una banda, **Qwen 3.5 de 0.8 bilions de paràmetres** quantitzat a Q8, encarregat del raonament socràtic pas a pas i Micro-CoT.*
> *De l'altra, **bge-small** per a cerca vectorial semàntica mitjançant RAG sobre una base de dades SQLite local.*
> *Tot el sistema corre en 4 GB de RAM física, garantint privacitat absoluta i disponibilitat a qualsevol escola rural o aula sense Wi-Fi."*

---

### Bloc 3: La Demo en Viu (0:50 - 1:50 | 60 segons) ★ EL PLAT FORT ★
- **Objectiu**: Demostrar les tres capacitats clau funcionant en temps real des de la placa.
- **Acció**: Pantalla compartida mostrant `http://127.0.0.1:8000`.

> **Guió verbal**:
> *"Vegem-ho en funcionament real a la placa a través del túnel USB:*
> 
> **1. Tutor Socràtic adaptat per Dominis (20s)**:
> *Seleccionem un problema complex de geometria. En lloc d'escopir la resposta, el nostre copilot detecta el domini matemàtic, recupera problemes similars amb RAG i fa una pregunta guia:* 'Quina relació hi ha entre l'angle inscrit i el central?' *L'alumne pensa i respon; el tutor l'acompanya pas a pas.*
> 
> **2. Detector de Trampes i Anàlisi de Distractors (20s)**:
> *Si l'alumne tria l'opció B (que és incorrecta), no li diem només 't'has equivocat'. El model analitza el distractor: desmunta el parany típic d'haver oblidat dividir per dos o haver confós perímetre amb àrea, ajudant a entendre el perquè de l'error.*
> 
> **3. Entrenament Adaptatiu i Spaced Repetition (20s)**:
> *El sistema no és estàtic: manté una cua de repetició espaiada probabilística. Si l'alumne falla, el concepte es reintrodueix estratègicament; si encerta, el motor el promociona al següent nivell de dificultat Cangur."*

---

### Bloc 4: Les Xifres i Mètriques Edge (1:50 - 2:30 | 40 segons)
- **Objectiu**: Convèncer el jurat amb dades tècniques sòlides sobre eficiència i robustesa a l'edge.
- **Acció**: Mostrar breument terminal amb `free -m` o la diapositiva de dades tècniques.

> **Guió verbal**:
> *"Comportament i mètriques a l'edge:*
> - **Temps de resposta**: *La regla oficial de la datathon fixava un límit màxim de 60 segons. El nostre Teaching Copilot respon en **només 1.2 a 3.8 segons** per intervenció socràtica. Això ho aconseguim generant Micro-CoT concisos de 20 a 30 tokens enfocats a la interacció.*
> - **Consum de Memòria**: *El límit físic de la placa són 4 GB. Els dos servidors i l'aplicació ocupen conjuntament **2.3 GB de RAM**, deixant més d'1.5 GB lliures de marge d'estabilitat.*
> - **Robustesa**: *Amb un mecanisme de fallback determinista i límits de 35s per crida, el copilot no es penja mai."*

---

### Bloc 5: Aprenentatges i Visió de Futur (2:30 - 3:00 | 30 segons)
- **Objectiu**: Concloure amb maduresa tècnica i visió de producte pedagògic real.
- **Acció**: Mostrar la diapositiva final i aixecar la placa assenyalant la matriu LED.

> **Guió verbal**:
> *"Què hem après? Que un model petit de 0.8B no necessita inventar teories: amb prompts de domini concrets i RAG local, supera en utilitat pedagògica a models gegants que simplement donen la solució.*
> *I el futur? Aprofitar el microcontrolador STM32 de l'Arduino connectant la matriu de LEDs 8x13 per a semàfors visuals d'ajuda i gamificació directa a la taula de l'estudiant.*
> *M3Kang Teaching Copilot porta el millor tutor personalitzat allà on fa falta: directament a la placa, a la vora, i en mans de l'estudiant. Moltes gràcies!"*

---

## Pla de Contingència (Pla B d'Emergència)

- **Si la xarxa no funciona**: No cal Wi-Fi. La connexió s'efectua sempre via USB amb `adb forward tcp:8000 tcp:8000`.
- **Si el portàtil té problemes de pantalla**: Tingueu llesta la presentació `PITCH_TEACHING_COPILOT.pptx` i el vídeo gravat de la demo al disc dur.
- **Si el servei es reinicia**: El timeout està protegit a 35 segons i els serveis `systemd` es reinicien automàticament en segons (`Restart=on-failure`).
