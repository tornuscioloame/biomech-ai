import flet as ft
import mediapipe as mp
import math
import threading
import tkinter as tk
from tkinter import filedialog
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from groq import Groq
import socket
import os
import segno
import json
import shutil
from datetime import datetime
from pyngrok import ngrok

# ============================================================
#  CONFIGURAZIONE — INSERISCI LE TUE CHIAVI QUI
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")

PORT = 8550

# Cartella dove salvare i progressi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_DIR = os.path.join(BASE_DIR, "progressi")
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "storico.json")
os.makedirs(PROGRESS_DIR, exist_ok=True)

# --- CONFIGURAZIONE MEDIAPIPE ---
try:
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1)
    detector = vision.PoseLandmarker.create_from_options(options)
except Exception as e:
    print(f"ATTENZIONE: Modello MediaPipe non trovato! Errore: {e}")
    detector = None


# ============================================================
#  GESTIONE STORICO PROGRESSI
# ============================================================
def carica_storico() -> list:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def salva_storico(storico: list):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(storico, f, ensure_ascii=False, indent=2)


def salva_sessione(profile_data: dict, ratios: dict, analisi_foto: str, scheda: str, foto_paths: dict):
    storico = carica_storico()
    data_oggi = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Copia le foto nella cartella progressi
    cartella_sessione = os.path.join(PROGRESS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(cartella_sessione, exist_ok=True)

    foto_salvate = {}
    for tipo, path in foto_paths.items():
        if path:
            ext = os.path.splitext(path)[1]
            dest = os.path.join(cartella_sessione, f"{tipo}{ext}")
            shutil.copy2(path, dest)
            foto_salvate[tipo] = dest

    sessione = {
        "data": data_oggi,
        "profilo": {
            "eta": profile_data.get("eta"),
            "peso": profile_data.get("peso"),
            "altezza": profile_data.get("altezza"),
            "sesso": profile_data.get("sesso"),
            "obiettivo": profile_data.get("obiettivo"),
        },
        "ratios": ratios,
        "analisi_foto": analisi_foto,
        "scheda": scheda,
        "foto": foto_salvate,
    }

    storico.append(sessione)
    salva_storico(storico)
    return sessione


# ============================================================
#  ANALISI BIOMECCANICA
# ============================================================
class BioMechEngine:
    def __init__(self):
        self.ratios = {}

    def analyze_pose(self, image_path):
        if detector is None:
            return None
        try:
            mp_image = mp.Image.create_from_file(image_path)
        except Exception:
            return None

        result = detector.detect(mp_image)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]

        torso = math.dist([lm[11].x, lm[11].y], [lm[23].x, lm[23].y])
        femur = math.dist([lm[23].x, lm[23].y], [lm[25].x, lm[25].y])
        tibia = math.dist([lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
        humerus = math.dist([lm[11].x, lm[11].y], [lm[13].x, lm[13].y])
        avamb = math.dist([lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
        spalle = math.dist([lm[11].x, lm[11].y], [lm[12].x, lm[12].y])
        fianchi = math.dist([lm[23].x, lm[23].y], [lm[24].x, lm[24].y])

        if torso > 0:
            self.ratios = {
                "femur_torso": round(femur / torso, 3),
                "tibia_femur": round(tibia / femur, 3) if femur > 0 else 0,
                "humerus_torso": round(humerus / torso, 3),
                "avamb_humerus": round(avamb / humerus, 3) if humerus > 0 else 0,
                "spalle_fianchi": round(spalle / fianchi, 3) if fianchi > 0 else 0,
            }
            return self.ratios
        return None

    def interpreta_ratios(self):
        r = self.ratios
        note = []
        ft_val = r.get("femur_torso", 0)
        if ft_val > 0.65:
            note.append(f"Femore LUNGO (ratio {ft_val}): leva sfavorevole nel back squat.")
        elif ft_val > 0.55:
            note.append(f"Femore nella media (ratio {ft_val}): buona versatilità.")
        else:
            note.append(f"Femore CORTO (ratio {ft_val}): leva eccellente per squat.")
        ht_val = r.get("humerus_torso", 0)
        if ht_val > 0.48:
            note.append(f"Omero LUNGO (ratio {ht_val}): preferire manubri nella panca.")
        else:
            note.append(f"Omero corto/medio (ratio {ht_val}): ottimo per panca bilanciere.")
        sf_val = r.get("spalle_fianchi", 0)
        if sf_val > 1.4:
            note.append(f"Spalle LARGHE (ratio {sf_val}): struttura a V naturale.")
        else:
            note.append(f"Proporzionalità spalle-fianchi standard (ratio {sf_val}).")
        av_val = r.get("avamb_humerus", 0)
        if av_val > 0.85:
            note.append(f"Avambracci LUNGHI (ratio {av_val}): trazioni più efficaci.")
        return "\n".join(note)


# ============================================================
#  CHIAMATE GROQ API
# ============================================================
def analizza_foto_con_groq(profile_data: dict, foto_disponibili: list, ratios_text: str, storico: list) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    n_sessioni = len(storico)
    confronto = ""
    if n_sessioni > 0:
        ultima = storico[-1]
        confronto = f"""
SESSIONE PRECEDENTE ({ultima['data']}):
- Peso: {ultima['profilo'].get('peso')} kg
- Ratios biomeccanici precedenti: {json.dumps(ultima.get('ratios', {}), ensure_ascii=False)}
- Analisi precedente: {ultima.get('analisi_foto', 'Non disponibile')[:300]}...
"""

    prompt = f"""Sei un coach esperto di bodybuilding e composizione corporea, con conoscenze di:
    - Ipertrofia muscolare e periodizzazione (Eric Helms, Brad Schoenfeld)
    - Analisi visiva della composizione corporea
    - Identificazione di squilibri muscolari e punti carenti
    - Distribuzione del grasso corporeo e sua influenza sull'allenamento

    L'atleta ha caricato {len(foto_disponibili)} foto ({', '.join(foto_disponibili)}) per il check-in settimanale.
    L'obiettivo principale è: {profile_data.get('obiettivo')}

    PROFILO:
    - Sesso: {profile_data.get('sesso')}, Età: {profile_data.get('eta')} anni
    - Peso attuale: {profile_data.get('peso')} kg
    - Settimana numero: {n_sessioni + 1}

    DATI BIOMECCANICI (MediaPipe):
    {ratios_text}

    {f"CONFRONTO CON SETTIMANA PRECEDENTE:{confronto}" if confronto else "Prima sessione — stabilisci la baseline."}

    Scrivi un'analisi FOCALIZZATA SULLA CRESCITA MUSCOLARE seguendo questo formato:

    ═══ COMPOSIZIONE CORPOREA ═══
    [Stima visiva della composizione: massa muscolare apparente, percentuale grasso stimata]
    [Distribuzione del grasso: dove si concentra (addome, fianchi, petto, schiena)]
    [Come questo influenza la strategia di allenamento e nutrizione]

    ═══ ANALISI DEI GRUPPI MUSCOLARI ═══
    [Vai gruppo per gruppo: petto, schiena, spalle, braccia, gambe, addome]
    [Per ognuno: sviluppo attuale, punti di forza, lacune evidenti]
    [Squilibri muscolari visibili (es: spalle anteriori vs posteriori, quadricipiti vs femorali)]

    ═══ PUNTI DI FORZA MUSCOLARI ═══
    [Gruppi muscolari già sviluppati bene — cosa sta funzionando]

    ═══ PUNTI CARENTI — PRIORITÀ DI SVILUPPO ═══
    [Gruppi muscolari sotto-sviluppati che frenano l'estetica e la performance]
    [Ordine di priorità: cosa allenare di più nelle prossime settimane]

    ═══ DISTRIBUZIONE DEL GRASSO ═══
    [Zone con accumulo di grasso più evidente]
    [Impatto sull'estetica e consigli per ridurlo mantenendo la massa muscolare]

    ═══ CONFRONTO CON SETTIMANA PRECEDENTE ═══
    {("[Progressi visibili: cambiamenti nella massa muscolare o nella composizione rispetto alla settimana scorsa]" if confronto else "[Prima sessione: questa è la tua baseline di partenza]")}

    ═══ PRIORITÀ PER LA PROSSIMA SETTIMANA ═══
    [Top 3 cose su cui concentrarsi: esercizi specifici per i punti carenti, aggiustamenti nutrizionali]

    Sii diretto, onesto e costruttivo. L'atleta vuole feedback reali per migliorare, non complimenti generici."""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )
    return completion.choices[0].message.content


def genera_scheda_con_groq(profile_data: dict, bio_note: str, storico: list) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    bmi = profile_data["peso"] / ((profile_data["altezza"] / 100) ** 2)
    bmi_cat = (
        "Sottopeso" if bmi < 18.5 else
        "Normopeso" if bmi < 25 else
        "Sovrappeso" if bmi < 30 else
        "Obesità"
    )

    n_sessioni = len(storico)
    storico_text = ""
    if n_sessioni > 0:
        storico_text = "STORICO ALLENAMENTI:\n"
        for i, s in enumerate(storico[-3:], 1):  # ultime 3 sessioni
            storico_text += f"""
Settimana {n_sessioni - len(storico[-3:]) + i} ({s['data']}):
- Peso: {s['profilo'].get('peso')} kg
- Scheda seguita: {s.get('scheda', '')[:200]}...
"""

    system_prompt = """Sei un personal trainer esperto NSCA-CSCS con conoscenza di:
- Biomeccanica, fisiologia dell'esercizio, periodizzazione
- Progressione dei carichi e principio di sovraccarico progressivo
- Prevenzione degli infortuni
- Adattamento dell'allenamento in base ai progressi reali

La sicurezza e la progressione graduale sono priorità assolute."""

    user_prompt = f"""Crea la scheda di allenamento per questa settimana basandoti sui progressi reali dell'atleta.

PROFILO:
- Età: {profile_data['eta']} anni | Sesso: {profile_data['sesso']}
- Peso: {profile_data['peso']} kg | Altezza: {profile_data['altezza']} cm | BMI: {bmi:.1f} ({bmi_cat})
- Livello: {profile_data['livello']}
- Obiettivo: {profile_data['obiettivo']}
- Giorni: {profile_data['giorni']} giorni/settimana
- Infortuni: {profile_data['infortunio']}
- Settimana numero: {n_sessioni + 1}

ANALISI BIOMECCANICA:
{bio_note if bio_note else "Non disponibile."}

{storico_text if storico_text else "Prima settimana — inizia con carichi moderati per stabilire la baseline."}

ISTRUZIONI:
{"Questa è la settimana " + str(n_sessioni + 1) + ". Aumenta progressivamente volume o intensità rispetto alle settimane precedenti seguendo il principio di sovraccarico progressivo." if n_sessioni > 0 else "Prima settimana: usa carichi conservativi per imparare i movimenti e valutare i punti di partenza."}

Segui ESATTAMENTE questo formato:

═══ PANORAMICA SETTIMANA {n_sessioni + 1} ═══
[Split e motivazione]
[Parametri: volume, intensità, recuperi]
[Progressioni rispetto alla settimana precedente se applicabile]
[Note nutrizionali]

═══ GIORNO X – [Nome] ═══
RISCALDAMENTO (10 min):
• [esercizio] – [durata]

ESERCIZI:
• [Esercizio] – [Serie x Rip] – RIR [X] – Recupero [X min] – Carico suggerito: [X kg o % 1RM]
  → [Note tecniche e motivazione biomeccanica]

NOTE SESSIONE: [avvisi]

[Ripeti per ogni giorno]

═══ PROGRESSIONE ═══
[Come aumentare i carichi questa settimana]
[Quando fare deload]

═══ AVVISI ═══
[Controindicazioni e sicurezza]"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=4000,
    )
    return completion.choices[0].message.content


# ============================================================
#  GENERA QR CODE
# ============================================================
def genera_qr(url: str, nome_file: str):
    qr = segno.make(url)
    qr.save(nome_file, scale=10)


# ============================================================
#  INTERFACCIA
# ============================================================
def main(page: ft.Page):
    page.title = "BioMech AI Coach"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 560
    page.window.height = 880
    page.scroll = ft.ScrollMode.ADAPTIVE
    page.bgcolor = "#0d0d0d"

    engine = BioMechEngine()
    storico = carica_storico()

    CYAN = ft.Colors.CYAN_ACCENT
    GREY9 = "#1a1a1a"
    GREY8 = "#181818"
    BORDER = ft.Colors.CYAN_900

    profile_data = {}
    bio_note_ref = [""]
    ratios_ref = [{}]
    foto_paths = {"frontale": None, "sinistra": None, "destra": None, "schiena": None}

    def titolo(t):
        return ft.Text(t, size=12, weight="bold", color=ft.Colors.GREY_500)

    def chip_settimane():
        n = len(storico)
        if n == 0:
            return ft.Text("Prima sessione — nessun dato precedente", size=11, color=ft.Colors.GREY_600, italic=True)
        return ft.Text(f"📈 {n} settiman{'a' if n == 1 else 'e'} di allenamento salvate", size=11, color=CYAN)

    # ============================================================
    #  STEP 1 – FORM
    # ============================================================
    def build_step1():
        page.controls.clear()

        f_eta = ft.TextField(label="Età (anni)", keyboard_type=ft.KeyboardType.NUMBER,
                             bgcolor=GREY9, border_color=BORDER, color="white", width=160)
        f_peso = ft.TextField(label="Peso (kg)", keyboard_type=ft.KeyboardType.NUMBER,
                              bgcolor=GREY9, border_color=BORDER, color="white", width=160)
        f_altezza = ft.TextField(label="Altezza (cm)", keyboard_type=ft.KeyboardType.NUMBER,
                                 bgcolor=GREY9, border_color=BORDER, color="white", width=160)
        f_infort = ft.TextField(
            label="Infortuni / limitazioni — vuoto se nessuno",
            bgcolor=GREY9, border_color=BORDER, color="white", min_lines=2, max_lines=3
        )
        f_note = ft.TextField(
            label="Altre note (lavoro sedentario, sport extra...)",
            bgcolor=GREY9, border_color=BORDER, color="white", min_lines=2, max_lines=3
        )

        # Pre-compila dal profilo precedente se disponibile
        if storico:
            ultimo = storico[-1]["profilo"]
            f_altezza.value = str(ultimo.get("altezza", ""))

        sesso_dd = ft.Dropdown(label="Sesso biologico",
                               options=[ft.dropdown.Option("Maschio"), ft.dropdown.Option("Femmina")],
                               bgcolor=GREY9, color="white")
        giorni_dd = ft.Dropdown(label="Giorni/settimana",
                                options=[ft.dropdown.Option(str(i)) for i in range(1, 7)],
                                bgcolor=GREY9, color="white")
        obiettivo_dd = ft.Dropdown(label="Obiettivo principale",
                                   options=[
                                       ft.dropdown.Option("Massa muscolare (ipertrofia)"),
                                       ft.dropdown.Option("Dimagrimento (perdita di grasso)"),
                                       ft.dropdown.Option("Ricomposizione corporea (asciugarsi)"),
                                       ft.dropdown.Option("Forza massimale"),
                                       ft.dropdown.Option("Resistenza muscolare"),
                                       ft.dropdown.Option("Benessere generale e salute"),
                                       ft.dropdown.Option("Recupero post-infortunio"),
                                   ], bgcolor=GREY9, color="white")
        livello_dd = ft.Dropdown(label="Livello di esperienza",
                                 options=[
                                     ft.dropdown.Option("Principiante (< 1 anno)"),
                                     ft.dropdown.Option("Intermedio (1-3 anni)"),
                                     ft.dropdown.Option("Avanzato (> 3 anni)"),
                                 ], bgcolor=GREY9, color="white")

        # Pre-compila dal profilo precedente
        if storico:
            ultimo = storico[-1]["profilo"]
            sesso_dd.value = ultimo.get("sesso", "")
            obiettivo_dd.value = ultimo.get("obiettivo", "")

        errore = ft.Text("", color=ft.Colors.RED_ACCENT, size=12)

        def vai_step2(e):
            try:
                eta = int(f_eta.value or "0")
                peso = float((f_peso.value or "0").replace(",", "."))
                altezza = float((f_altezza.value or "0").replace(",", "."))
                if eta <= 0 or peso <= 0 or altezza <= 0:
                    errore.value = "Inserisci valori validi per età, peso e altezza."
                    page.update();
                    return
                if not sesso_dd.value or not obiettivo_dd.value or not giorni_dd.value or not livello_dd.value:
                    errore.value = "Compila tutti i campi obbligatori."
                    page.update();
                    return
                profile_data.update({
                    "eta": eta, "peso": peso, "altezza": altezza,
                    "sesso": sesso_dd.value, "giorni": int(giorni_dd.value),
                    "obiettivo": obiettivo_dd.value, "livello": livello_dd.value,
                    "infortunio": f_infort.value or "Nessuno",
                    "note_extra": f_note.value or "Nessuna",
                })
                build_step2()
            except ValueError:
                errore.value = "Inserisci solo numeri validi."
                page.update()

        page.add(ft.Container(
            content=ft.Column([
                ft.Column([
                    ft.Text("BIOMECH AI", size=30, weight="bold", color=CYAN),
                    ft.Text("Coach Biomeccanico — Powered by LLaMA 3.3 70B via Groq",
                            italic=True, color=ft.Colors.GREY_600, size=11),
                    chip_settimane(),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=15, color=BORDER),
                titolo("DATI FISICI"),
                ft.Row([f_eta, f_peso, f_altezza], wrap=True),
                sesso_dd,
                ft.Divider(height=8, color="transparent"),
                titolo("ALLENAMENTO"),
                livello_dd, giorni_dd, obiettivo_dd,
                ft.Divider(height=8, color="transparent"),
                titolo("INFORTUNI E NOTE"),
                f_infort, f_note,
                ft.Divider(height=12, color="transparent"),
                errore,
                ft.Row([
                    ft.FilledButton("AVANTI →", on_click=vai_step2,
                                    style=ft.ButtonStyle(bgcolor=CYAN, color="#000000")),
                    ft.OutlinedButton("📊 STORICO", on_click=lambda _: build_storico(),
                                      style=ft.ButtonStyle(side=ft.BorderSide(1, BORDER), color=ft.Colors.GREY_400)),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
            ], spacing=10),
            padding=20,
        ))

    # ============================================================
    #  STEP 2 – 4 FOTO
    # ============================================================
    def build_step2():
        page.controls.clear()

        # Reset foto
        for k in foto_paths:
            foto_paths[k] = None

        tipi_foto = [
            ("frontale", "FRONTALE", "In piedi, braccia lungo i fianchi, guarda avanti"),
            ("sinistra", "LATO SINISTRO", "Di fianco sinistro, braccia lungo i fianchi"),
            ("destra", "LATO DESTRO", "Di fianco destro, braccia lungo i fianchi"),
            ("schiena", "SCHIENA", "Di spalle, braccia lungo i fianchi"),
        ]

        previews = {}
        stati = {}
        checkmarks = {}

        for tipo, _, _ in tipi_foto:
            previews[tipo] = ft.Image(src="", width=110, height=150, visible=False, border_radius=8)
            stati[tipo] = ft.Text("Non caricata", size=10, color=ft.Colors.GREY_600)
            checkmarks[tipo] = ft.Text("", size=14)

        errore = ft.Text("", color=ft.Colors.RED_ACCENT, size=11)
        ring = ft.ProgressRing(visible=False, color=CYAN, width=30, height=30)
        stato_gen = ft.Text("", color=ft.Colors.GREY_400, size=12, text_align=ft.TextAlign.CENTER)
        btn_genera = ft.FilledButton(
            "ANALIZZA E GENERA SCHEDA →",
            visible=False,
            on_click=lambda _: build_step3(),
            style=ft.ButtonStyle(bgcolor=CYAN, color="#000000")
        )

        def aggiorna_btn():
            caricate = sum(1 for v in foto_paths.values() if v)
            btn_genera.visible = caricate >= 1  # basta anche 1 foto
            if caricate == 4:
                stato_gen.value = "✅ Tutte e 4 le foto caricate! Ottimo per un'analisi completa."
            elif caricate > 0:
                stato_gen.value = f"📷 {caricate}/4 foto caricate. Puoi procedere o aggiungerne altre."
            page.update()

        def scegli_foto(tipo):
            def _handler(e):
                def _pick():
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    path = filedialog.askopenfilename(
                        title=f"Scegli foto {tipo}",
                        filetypes=[("Immagini", "*.jpg *.jpeg *.png *.bmp *.webp")]
                    )
                    root.destroy()
                    if not path:
                        return

                    foto_paths[tipo] = path
                    previews[tipo].src = path
                    previews[tipo].visible = True
                    stati[tipo].value = "✅ Caricata"
                    stati[tipo].color = ft.Colors.GREEN_400
                    checkmarks[tipo].value = "✅"

                    # Analisi pose sulla foto frontale
                    if tipo == "frontale":
                        ring.visible = True
                        stato_gen.value = "Analisi biomeccanica foto frontale..."
                        page.update()
                        ratios = engine.analyze_pose(path)
                        ring.visible = False
                        if ratios:
                            ratios_ref[0] = ratios
                            bio_note_ref[0] = engine.interpreta_ratios()
                    aggiorna_btn()

                threading.Thread(target=_pick, daemon=True).start()

            return _handler

        # Costruisci griglia 2x2 foto
        card_list = []
        for tipo, label, istruzione in tipi_foto:
            card_list.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(label, size=11, weight="bold", color=CYAN),
                        ft.Text(istruzione, size=9, color=ft.Colors.GREY_500),
                        previews[tipo],
                        stati[tipo],
                        ft.ElevatedButton(
                            "📷 Carica",
                            on_click=scegli_foto(tipo),
                            style=ft.ButtonStyle(bgcolor=GREY9, color=CYAN),
                            height=32,
                        ),
                    ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=10, bgcolor=GREY8, border_radius=10,
                    border=ft.border.all(1, BORDER), width=230,
                )
            )

        page.add(ft.Container(
            content=ft.Column([
                ft.Text("📸 CHECK-IN SETTIMANALE", size=20, weight="bold", color=CYAN),
                ft.Text(
                    f"Settimana {len(storico) + 1} • {profile_data.get('sesso')} • "
                    f"{profile_data.get('peso')}kg • {profile_data.get('obiettivo')}",
                    size=11, color=ft.Colors.GREY_500
                ),
                ft.Divider(height=10, color=BORDER),
                ft.Container(
                    content=ft.Text(
                        "Carica le foto in posizione anatomica standard: corpo intero visibile, "
                        "sfondo neutro, buona illuminazione. Almeno la frontale è necessaria.",
                        size=11, color=ft.Colors.GREY_400, text_align=ft.TextAlign.CENTER
                    ),
                    padding=10, bgcolor=GREY9, border_radius=8,
                    border=ft.border.all(1, BORDER)
                ),
                ft.Row(card_list[:2], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                ft.Row(card_list[2:], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                ft.Row([ring], alignment=ft.MainAxisAlignment.CENTER),
                stato_gen,
                errore,
                ft.Row([btn_genera], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(height=10, color="transparent"),
                ft.Row([
                    ft.TextButton("← Indietro", on_click=lambda _: build_step1(),
                                  style=ft.ButtonStyle(color=ft.Colors.GREY_600)),
                    ft.TextButton("Salta foto (scheda senza analisi visiva)",
                                  on_click=lambda _: build_step3(),
                                  style=ft.ButtonStyle(color=ft.Colors.GREY_700)),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
            ], spacing=12, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20,
        ))

    # ============================================================
    #  STEP 3 – ANALISI + GENERAZIONE SCHEDA
    # ============================================================
    def build_step3():
        page.controls.clear()

        ring = ft.ProgressRing(color=CYAN, width=50, height=50)
        stato = ft.Text(
            "Analisi in corso...\nL'AI sta esaminando le tue foto e generando la scheda.",
            color=ft.Colors.GREY_400, size=13, text_align=ft.TextAlign.CENTER
        )
        errore = ft.Text("", color=ft.Colors.RED_ACCENT, size=12)
        step_text = ft.Text("", color=CYAN, size=12, text_align=ft.TextAlign.CENTER)

        page.add(ft.Container(
            content=ft.Column([
                ft.Text("🤖 ELABORAZIONE IN CORSO", size=20, weight="bold", color=CYAN),
                ft.Divider(height=15, color=BORDER),
                ft.Row([ring], alignment=ft.MainAxisAlignment.CENTER),
                stato, step_text, errore,
            ], spacing=15, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20,
        ))

        def _genera():
            try:
                # Step 1: analisi foto
                foto_caricate = [k for k, v in foto_paths.items() if v]
                step_text.value = "📊 Step 1/2: Analisi posturale e progressi..."
                page.update()

                analisi = analizza_foto_con_groq(
                    profile_data,
                    foto_caricate,
                    bio_note_ref[0],
                    storico
                )

                # Step 2: genera scheda
                step_text.value = "🏋️ Step 2/2: Generazione scheda personalizzata..."
                page.update()

                scheda = genera_scheda_con_groq(profile_data, bio_note_ref[0], storico)

                # Salva sessione
                salva_sessione(profile_data, ratios_ref[0], analisi, scheda, foto_paths)
                storico.append({})  # aggiorna conteggio locale
                storico.pop()
                storico.extend(carica_storico()[-1:])

                # Mostra risultati
                page.controls.clear()
                page.add(ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text("✅ ANALISI COMPLETATA", size=18, weight="bold", color=CYAN, expand=True),
                            ft.IconButton(icon=ft.Icons.REFRESH, tooltip="Ricomincia",
                                          on_click=lambda _: build_step1(), icon_color=ft.Colors.GREY_500)
                        ]),
                        ft.Text(
                            f"Settimana {len(carica_storico())} • {profile_data.get('sesso')} • "
                            f"{profile_data.get('peso')}kg • {profile_data.get('obiettivo')}",
                            size=11, color=ft.Colors.GREY_500
                        ),
                        ft.Divider(height=10, color=BORDER),

                        # Analisi foto
                        ft.Text("📸 ANALISI POSTURALE", size=14, weight="bold", color=CYAN),
                        ft.Container(
                            content=ft.Text(analisi, size=12, color="white", selectable=True),
                            padding=15, bgcolor=GREY9, border_radius=10,
                            border=ft.border.all(1, BORDER)
                        ),

                        ft.Divider(height=10, color=BORDER),

                        # Scheda
                        ft.Text("🏋️ SCHEDA DELLA SETTIMANA", size=14, weight="bold", color=CYAN),
                        ft.Container(
                            content=ft.Text(scheda, size=12, color="white", selectable=True),
                            padding=15, bgcolor=GREY8, border_radius=10,
                            border=ft.border.all(1, BORDER)
                        ),

                        ft.Divider(height=15, color="transparent"),
                        ft.Row([
                            ft.FilledButton("🔄 Nuova settimana",
                                            on_click=lambda _: build_step1(),
                                            style=ft.ButtonStyle(bgcolor=GREY9, color=ft.Colors.GREY_300)),
                            ft.FilledButton("📊 Vedi storico",
                                            on_click=lambda _: build_storico(),
                                            style=ft.ButtonStyle(bgcolor=CYAN, color="#000000")),
                        ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
                    ], spacing=12),
                    padding=20,
                ))
                page.update()

            except Exception as ex:
                ring.visible = False
                errore.value = f"❌ Errore: {str(ex)}"
                page.update()

        threading.Thread(target=_genera, daemon=True).start()

    # ============================================================
    #  STORICO PROGRESSI
    # ============================================================
    def build_storico():
        page.controls.clear()
        dati = carica_storico()

        sezioni = ft.Column(spacing=15)
        sezioni.controls.append(ft.Row([
            ft.Text("📊 STORICO PROGRESSI", size=20, weight="bold", color=CYAN, expand=True),
            ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: build_step1(),
                          icon_color=ft.Colors.GREY_500)
        ]))

        if not dati:
            sezioni.controls.append(
                ft.Text("Nessuna sessione salvata ancora.", color=ft.Colors.GREY_500, size=13)
            )
        else:
            # Grafico peso nel tempo
            pesi = [(s["data"][:10], s["profilo"].get("peso", 0)) for s in dati]
            sezioni.controls.append(ft.Container(
                content=ft.Column([
                    ft.Text("⚖️ ANDAMENTO PESO", size=13, weight="bold", color=CYAN),
                    *[ft.Row([
                        ft.Text(data, size=11, color=ft.Colors.GREY_500, width=100),
                        ft.Container(
                            width=max(10, int((peso / max(p for _, p in pesi)) * 200)),
                            height=18, bgcolor=CYAN, border_radius=4
                        ),
                        ft.Text(f"{peso} kg", size=11, color="white"),
                    ]) for data, peso in pesi],
                ], spacing=5),
                padding=15, bgcolor=GREY9, border_radius=10,
                border=ft.border.all(1, BORDER)
            ))

            # Lista sessioni
            for i, s in enumerate(reversed(dati), 1):
                sezioni.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Text(f"Settimana {len(dati) - i + 1} — {s['data']}",
                                size=13, weight="bold", color=CYAN),
                        ft.Text(f"Peso: {s['profilo'].get('peso')} kg | Obiettivo: {s['profilo'].get('obiettivo')}",
                                size=11, color=ft.Colors.GREY_400),
                        ft.Text(
                            s.get("analisi_foto", "")[:250] + "..." if s.get("analisi_foto") else "",
                            size=11, color=ft.Colors.GREY_500, italic=True
                        ),
                    ], spacing=4),
                    padding=12, bgcolor=GREY8, border_radius=10,
                    border=ft.border.all(1, BORDER)
                ))

        page.add(ft.Container(content=sezioni, padding=20))

    build_step1()


# ============================================================
#  AVVIO
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8550))
    ft.run(main, view=ft.AppView.WEB_BROWSER, host="0.0.0.0", port=port)