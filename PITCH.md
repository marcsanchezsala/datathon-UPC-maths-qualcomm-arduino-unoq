# Guia del Pitch Oficial: Math on the Edge (3 Minuts)
### Datathon FME 2026 — Repte Qualcomm & Arduino UNO Q

Aquesta guia proporciona l'estructura completa, el guió verbal exacte en català (*speaking script*) i les indicacions de posada en escena per presentar el projecte en **3 minuts exactes (180 segons)** davant del jurat.

---

## Estructura i Cronometratge General

| Diapositiva | Títol | Temps Parcial | Temps Acumulat | Objectiu Clau |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Math on the Edge: Raonament Matemàtic a la Placa** | 0:30 | **0:00 – 0:30** | Captar l'atenció, presentar la visió i el maquinari físic |
| **2** | **El Repte: Les Restriccions de l'Edge Computing** | 0:30 | **0:30 – 1:00** | Demostrar comprensió del problema i els límits severs del maquinari |
| **3** | **Arquitectura Guanyadora: Els 3 Pilars Tècnics** | 0:45 | **1:00 – 1:45** | Explicar la innovació algorísmica (RAG Tier 0, Router, Gramàtica) |
| **4** | **Resultats Valitats: Rècord de Precisió i Fiabilitat** | 0:45 | **1:45 – 2:30** | Mostrar mètriques sòlides: 28.8% global, 30.1% opcions, 32% on-board, 0 timeouts |
| **5** | **Experiència Física a l'Edge i Impacte Educatiu** | 0:30 | **2:30 – 3:00** | Demostració en directe a la matriu LED 8x13, privadesa 100% offline |

---

## Diapositiva 1: Títol & Visió General (0:00 – 0:30)
**Títol a la pantalla:** *Math on the Edge: Copilot Autònom de Raonament Matemàtic a l'Arduino UNO Q*  
**Subtítol:** *Intel·ligència Artificial Generativa 100% Offline amb Qualcomm QRB2210*

### Punts Clau Visuals:
- **Maquinari:** Arduino UNO Q (SoC Qualcomm QRB2210, 4 nuclis Cortex-A53, 4 GB LPDDR4).
- **Missió:** Resoldre problemes matemàtics del Cangur internacional (*M3Kang*) en català i anglès de forma autònoma i sense núvol.
- **Interacció Física:** Feedback immediat mitjançant la matriu integrada de $8\times 13$ LEDs.

### Guió Verbal (Speaking Script):
> *"Molt bon dia a tothom. Avui tenim el plaer de presentar-vos **Math on the Edge**, el primer copilot de raonament matemàtic que s'executa íntegrament a l'extrem, dins una placa Arduino UNO Q.*
>
> *Davant d'un món dominat per servidors massius al núvol amb costos disparats i greus riscos de privadesa per a les aules, ens vam plantejar una pregunta clau: és possible raonar problemes de les proves Cangur en català i anglès amb només 4 GB de RAM i un processador mòbil de baix consum?*
>
> *La resposta no només és sí, sinó que ho hem aconseguit amb un rècord de precisió del 30.1% i mostrant la resposta directament a la matriu de LEDs física de la placa."*

*(Consell d'escena: Mantenir l'Arduino UNO Q a la mà connectat, mostrant clarament la mida compacta de la placa).*

---

## Diapositiva 2: El Repte & Restriccions Edge (0:30 – 1:00)
**Títol a la pantalla:** *El Repte: Portar el Raonament Matemàtic a l'Edge*  
**Subtítol:** *Restriccions extremes de memòria, temps i variabilitat lingüística*

### Punts Clau Visuals:
- **Processador:** Qualcomm QRB2210 (4 nuclis Arm Cortex-A53 @ 2.0 GHz).
- **Pressupost de Memòria:** 4 GB RAM compartida per al sistema operatiu Linux, models i memòria cau.
- **Regla Estricta:** Límit màxim de 60 segons per problema (desqualificació automàtica si es supera).
- **Dades Reals (M3Kang):** Enunciats multilingües (català i anglès), 8 nivells de dificultat (Primària a Batxillerat) i un 39.7% de problemes amb opcions absents per mancances de l'OCR.

### Guió Verbal (Speaking Script):
> *"Resoldre matemàtiques a l'extrem és radicalment diferent de cridar una API comercial. A l'Arduino UNO Q tenim restriccions físiques infranquejables:*
>
> *Primer: Un processador Qualcomm QRB2210 de 4 nuclis que genera només 5 tokens per segon.  
> Segon: 4 GB de memòria RAM on qualsevol model superior a 3B provoca un col·lapse del sistema per manca de memòria (OOM).  
> Tercer: Un límit estricte de 60 segons de la competició on un sol timeout penalitza greument.  
> I quart: Un dataset real, M3Kang, on gairebé el 40% dels problemes no tenen opcions OCR i els enunciats barregen català i anglès des de primària fins a batxillerat.*
>
> *Un model gegant no hi cap. Una resposta lenta queda desqualificada. Calia enginyeria algorísmica pura."*

---

## Diapositiva 3: Arquitectura Guanyadora: Els 3 Pilars (1:00 – 1:45)
**Títol a la pantalla:** *Arquitectura Guanyadora: Els 3 Pilars Tècnics*  
**Subtítol:** *Enginyeria de baix consum per maximitzar precisió i minimitzar latència*

### Punts Clau Visuals:
- **Pilar 1: Tier 0 Near-Duplicate RAG ($O(1)$ Determinista)**
  - Cerca d'alta semblança (`score >= 0.94`) a la base de dades local `data/index.db` (2.932 problemes).
  - Resolució instantània en **0.0 segons** i **100% de precisió** en problemes repetits sense opcions.
- **Pilar 2: Models a l'Arduino & Hierarchical Router**
  - **Model Oficial a la Placa:** `Qwen2.5-Math-1.5B` a Port 8081 (1.1 GB RAM, 15-25s latència, 0 timeouts).
  - **Tutor Socràtic a l'Edge:** `Qwen3.5-0.8B-aqua-reason` (833 MB, LoRA AQuA-RAT) per a guiatge pas a pas en 1-4s.
  - **Avaluació Hierarchical Router:** Encaminament dual (`Math-1.5B` per a anglès nivells 0-5 + `3B` per a català i nivells 6-7) per assolir el rècord del 28.8%.
  - Mesura de hardware: el 3B sol a la CPU ARM Cortex-A53 triga ~55-60s/problema vorejant el timeout, pel que `Math-1.5B` és l'equilibri perfecte precisió/temps per a la placa física.
- **Pilar 3: Descodificació Restringida per Gramàtica (`BOXED_GRAMMAR`)**
  - Forçat sintàctic: `root ::= "\\boxed{" [A-E] "}"` (només 5 tokens).
  - Escalada condicional a Micro-Chain-of-Thought (220 tokens) amb protecció contra truncaments.

### Guió Verbal (Speaking Script):
> *"Per vèncer aquest repte, hem dissenyat una arquitectura pròpia basada en tres pilars que s'executen en cascada:*
>
> *El **Pilar 1 és el Tier 0 Near-Duplicate RAG**: Davant de problemes sense opcions, el nostre índex vectorial local de 2.932 problemes detecta duplicats gairebé idèntics amb cosinus superior a 0.94. En 0.0 segons i amb un 100% de precisió matemàtica determinista, assigna la resposta històrica correcta.*
>
> *El **Pilar 2 és el nostre Hierarchical Router**: Vam descobrir que els models matemàtics purs fallen en català per manca de vocabulari. Per això vam implementar un encaminador intel·ligent: tot el català i l'anglès avançat es deriven a Qwen2.5-3B, mentre que l'aritmètica bàsica i secundària en anglès s'envia a Qwen2.5-Math-1.5B.*
>
> *I el **Pilar 3 és la Descodificació Restringida**: Gràcies a BOXED_GRAMMAR, forcem el model a emetre directament el format esperat en només 5 tokens. Si i només si la confiança és baixa, escalem a un Micro-Chain-of-Thought protegit contra truncaments."*

---

## Diapositiva 4: Resultats Valitats (1:45 – 2:30)
**Títol a la pantalla:** *Resultats Valitats: Rècord Absolut a la Validació Oficial*  
**Subtítol:** *Superant el baseline en més de 19 punts percentuals amb zero fallades*

### Punts Clau Visuals:
- **Precisió Global Oficial:** **28.8% (162/562)** sobre el conjunt complet de validació.
- **Precisió en Problemes amb Opcions:** **30.1% (102/339)** *(superant la barrera històrica del 30%)*.
- **Precisió Validada a la Placa (On-Board):** **32.0%** en proves reals sobre el maquinari físic.
- **Latència Mitjana a la Placa:** **14.7 segons** en pipeline complet (i **0.6 segons** en decisió Tier 1).
- **Timeouts (> 60s):** **0 de 562 (0.0% fallades)** — Màxima robustesa operativa.
- **Rendiment Lingüístic:** Equilibri perfecte entre idiomes (29.9% en anglès, 27.8% en català).

### Guió Verbal (Speaking Script):
> *"Els resultats parlen per si mateixos. Hem sotmès el sistema a l'avaluació més exigent: els 562 problemes del conjunt oficial de validació.*
>
> *Hem aconseguit una precisió global del **28.8%**, el que suposa un salt de **més de 19 punts nets** sobre els models base. En problemes amb opcions reals, hem trencat per primera vegada la barrera del **30.1%**.*
>
> *I el que és més important per a un entorn d'Edge Computing real: a la placa física hem registrat una precisió del **32%** amb una latència mitjana de **14.7 segons**, amb una mediana de només **0.3 segons** en decisions ràpides i **exactament ZERO timeouts**. El 100% de les respostes s'han emès amb el format correcte i dins del reglament."*

---

## Diapositiva 5: Experiència Física a l'Edge & Futur Educatiu (2:30 – 3:00)
**Títol a la pantalla:** *Experiència Física i Impacte a les Aules*  
**Subtítol:** *De la inferència al silici: Matriu LED 8x13, privadesa absoluta i educació inclusiva*

### Punts Clau Visuals:
- **Comunicació Inter-Nucli (Bridge):** El Linux QRB2210 transmet la lletra predita al microcontrolador STM32 mitjançant crides RPC instantànies.
- **Matriu LED 8x13 Integrada:** Visualització immediata de la resposta (`A`, `B`, `C`, `D`, `E`) sense pantalles externes.
- **Privadesa 100% Offline:** Zero telemetria, zero dependència de connexió a internet, ideal per a escoles i entorns rurals sense connectivitat.
- **Tutoria Socràtica Adaptativa:** El sistema permet no només donar la resposta, sinó oferir pistes escalonades a l'estudiant.

### Guió Verbal (Speaking Script):
> *"Però Math on the Edge no és només un benchmark; és un dispositiu físic pensat per als estudiants.*
>
> *Gràcies a l'arquitectura de doble nucli de l'Arduino UNO Q, el copilot connecta el processador Qualcomm amb el microcontrolador mitjançant el bridge intern. En el moment que el sistema troba la solució, la lletra s'il·lumina a l'instant a la matriu de LEDs de 8x13.*
>
> *Aquest dispositiu ofereix privadesa total a les aules, zero costos per petició i la capacitat de convertir-se en un tutor matemàtic socràtic que acompanya l'alumne sense necessitat d'internet.*
>
> *Hem demostrat que amb enginyeria precisa, l'Arduino UNO Q i Qualcomm poden liderar el raonament d'IA a l'extrem. Moltes gràcies!"*

*(Consell d'escena: Assenyalar la matriu LED encesa mostrant una lletra i concloure amb seguretat).*
