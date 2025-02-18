# A22 (Cigna-Anlage-H) Optimizer
#
# initial 20250122 TG
#         20250216
#
# erstellt mit teilweiser Verwendung von ChatGPT

# pip install pandas plotly convertdate
# eventuell setuptool downgraden:
# pip install setuptools~=70.0

# pandas data frames werden hier gut erklärt:
# https://www.ionos.de/digitalguide/websites/web-entwicklung/python-pandas-dataframe/

import pandas as pd
import csv
import time
from datetime import datetime
from convertdate import julian,gregorian
import argparse
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

ROT="\033[31m"
NORM="\033[0m"

"""
\033[32m  Grün  
\033[33m  Gelb  
\033[34m  Blau  
\033[35m  Magenta  
\033[36m  Cyan  
\033[37m  Weiß  
"""

# Funktion zur Umwandlung des Julianischen Tages in ein Gregorianisches Datum
def j2g(jd):
    # Umwandlung des Julianischen Tages in Jahr, Monat, Tag (Gregorianisches Datum)
    year, month, day = gregorian.from_jd(jd)
    return pd.to_datetime(f"{year}-{month:02d}-{day:02d}").date() # we do not use the time component

# Funktion zur Umwandlung des Julianischen Tages in ein Gregorianisches Datum YYYYMMDD
def j2YYYYMMDD(jd):
    # Umwandlung des Julianischen Tages in Jahr, Monat, Tag (Gregorianisches Datum)
    year, month, day = gregorian.from_jd(jd)
    return pd.to_datetime(f"{year}{month:02d}{day:02d}").date() # we do not use the time component

def T(jd):
   d=j2g(jd)
   return f"{d.year:0>4}{d.month:0>2}{d.day:0>2}"

def t(jd):
   global offset
   d=j2g(jd+offset)
   return f"{d.year:0>4}{d.month:0>2}{d.day:0>2}"

# Betrag (limitiert)
def b(betrag):
   if betrag < 0.009:
      return "(         - ⊽)"
   else:
      return "("+f"{betrag:10,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")+" ⊽)"

# Betrag (unlimitiert, roh)
def B(betrag):
   if betrag < 0.009:
      return "(         - €)"
   else:
      return "("+f"{betrag:10,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")+" €)"

def tb(jd):
   global offset
   d=j2g(jd+offset)
   return f"{d.year:0>4}{d.month:0>2}{d.day:0>2} {b(Z[jd])}"

def is_leap_year(year):
    """Überprüft, ob ein Jahr ein Schaltjahr ist."""
    return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))

class Counter:
    # Klassenvariable
    global_count = 0

    def __call__(self):
        """Wird aufgerufen, wenn die Klasse wie eine Funktion aufgerufen wird."""
        Counter.global_count += 1
        return f"[{Counter.global_count:02d}] "

# Kommandozeilenparameter parsen
def parse_arguments():
    parser = argparse.ArgumentParser(description=
        "CIGNA-Anlage-H-Optimizer\n \nSimulationsdaten und eigene Gehaltsdaten als CSV-Dateien einlesen, optimieren und graphisch darstellen.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=45))
    parser.add_argument('Simulationsergebnisse', type=str, help="CSV-Datei der antragsberechtigten Beträge aus dem Simulator")

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('Durchschnittsgehälter', nargs="?", type=str, help="CSV-Datei der durchschnittlichen Monatsgehälter (optional, bei Fehlen wird ein Standardwert angenommen)")
    group1.add_argument('--averagesalary', type=float, default=5000.00 , help="optionaler Wert, der als durchschnittliches Gehalt angenommen wird (default: 5000,00)")

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--long', action='store_true', help="Liste von Tagesdaten bei Änderungen; mit ASCII-Plot")
    group2.add_argument('--full', action='store_true', help="Liste von Tagesdaten bei Änderungen, Liste für jeden Tag; beide mit ASCII-Plots")

    parser.add_argument('--logformat', action='store_true', help="Ausgabeformat, das für eine Log-Datei optimiert ist")
    parser.add_argument('--no3', action='store_true', help="ohne 3-Jahres-Optimierung (für schnelle Tests)")
    parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    return parser.parse_args()

# Funktion zur automatischen Erkennung des Separators
def detect_separator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(1024)  # Liest die ersten 1024 Zeichen
        delimiter = csv.Sniffer().sniff(sample).delimiter
    return delimiter

# Einlesen der Z-Werte aus der dritten Spalte der Datei daten.csv (Header überspringen und " €" entfernen)
def load_Z_values(file_name='daten.csv'):
    # Einlesen der CSV-Datei mit Pandas
    df = pd.read_csv(file_name,
        delimiter=detect_separator(file_name),
        skiprows=1,
        header=None,
        names=['Datum','JD','Betrag','Kommentar']
    )

    # Entfernen des " €"-Zeichens aus der dritten Spalte (Index 2) und Umwandeln in Floats
    df['Betrag'] = df['Betrag'].str.replace(" €", "").str.replace(",",".").astype(float)
    return df

# Benutzerdefinierte Funktion zum Bereinigen des Währungszeichens und Umwandeln in einen float-Wert
def parse_amount(amount):
    # Entfernen von Währungszeichen und Leerzeichen und Umwandeln in float
    return float(amount.replace(' €', '').replace(' ', '').replace(',', '.'))

def date_to_julian(date_str):
    # Das Datum im Format YYYYMMDD in ein datetime-Objekt umwandeln
    date = datetime.strptime(str(date_str), '%Y%m%d')

    # Umwandlung der year, month und day in das Julianische Datum
    jd = int(julian.to_jd(date.year, date.month, date.day) - 12.5) # Korrektur für Julian Dates
    return jd

# Einlesen der Gehalt-Werte aus der Datei gehalt.csv und optionale " €" entfernen
def load_gehalt_values(file_name='gehalt.csv'):
    # CSV-Datei einlesen
    df = pd.read_csv(file_name, header=None, names=['Datum', 'gehalt'])

    # Das Datum als Julian Date speichern und die Beträge mit der Funktion 'parse_amount' umwandeln
    df['gehalt'] = df['gehalt'].apply(parse_amount)
    df['JD'] = df['Datum'].apply(date_to_julian)
    df['formatted_gehalt']=df['gehalt'].apply(B)
    return df

# Einlesen der Gehalt-Werte aus der Datei gehalt.csv und optionale " €" entfernen
def load_constant_gehalt_values(salary, startJD):

    df = pd.DataFrame({'JD': [startJD], 'gehalt': [str(salary)]})
  
    # Das Datum als Julian Date speichern und die Beträge mit der Funktion 'parse_amount' umwandeln
    df['gehalt'] = df['gehalt'].apply(parse_amount)
    df['Datum'] = df['JD'].apply(j2YYYYMMDD)
    df['formatted_gehalt']=df['gehalt'].apply(B)
    return df

class Progressbar:
    def __init__(self, endprogress):
        self.lastprogress = 0
        self.endprogress = endprogress
        self.delta = int(endprogress/20)
        
    def __call__(self, progress):
        if args.logformat:
            return
        if ( progress >= self.lastprogress + self.delta ):
            self.lastprogress = progress
            ratio = round( 100.0 * progress/self.endprogress )
            for i in range(int( ratio / 2)):
                sys.stdout.write(f"\rFortschritt: [{(i+1) * '#':<50}] {ratio} %")
                sys.stdout.flush()
            if ratio > 99:
                print() # end the progressbar line

# adjust for 365/366 search windows
# wenn der aktuelle Beantragungszeitraum einen Schalttag umfasst,
# dann darf das nachfolgende Suchintervall erst nach 366 Tagen beginnen

def leapspan(jd):
    if not LEAPDAY:
        return YEAR
    if jd in range(LEAPDAY-YEAR,LEAPDAY+1):
        return LEAPYEAR
    else:
        return YEAR

# Erstellen der Listen valider Tupel T2 und Triplets T3
def generatorT2():
    for i in range(MAXDAYS):
        for j in range(i + leapspan(i), MAXDAYS):
            yield (i, j)

def generatorT3():
    for i in range(MAXDAYS):
        for j in range( i + leapspan(i), MAXDAYS):
            for k in range(j + leapspan(j), MAXDAYS):
                yield (i, j, k)  # Triplet erstellen

def bestimmeEinenOptimalenBeantragungszeitraum():
    jd = merged_df['cut'].idxmax()
    return Z[jd], jd

def bestimmeZweiOptimaleBeantragungszeiträume():
    # Initialisierung der maximalen Summe und maximalen Tupel-Index'
    max_sum = float('-inf')
    max_tupel = None
    tupel_sums = []
    countT2 = 0
    progressT2 = Progressbar(599753)
    
    # Berechnung der Summen und Bestimmung der maximalen Summe
    for tupel in generatorT2():
        progressT2(countT2)
        i, j = tupel
        countT2 += 1
        sumT2 = Z[i] + Z[j]

        if sumT2 > max_sum or (sumT2 == max_sum and tupel < max_tupel):
            max_sum = sumT2
            max_tupel = tupel
            tupel_sums.append((sumT2, tupel))
    
    # Sortieren der Tupel: Zuerst nach Summe (absteigend), dann nach Indizes (aufsteigend)
    tupel_sums.sort(key=lambda x: (-x[0], x[1]))  # -x[0] für absteigende Sortierung der Summe
    return countT2, tupel_sums[0]

def bestimmeDreiOptimaleBeantragungszeiträume():
    # Initialisierung der maximalen Summe und des maximalen Tripel-Index'
    max_sum = float('-inf')
    max_triplet = None
    triplet_sums = []
    countT3 = 0
    progressT3 = Progressbar(64925721)
            
    # Berechnung der Summen und Bestimmung der maximalen Summe
    for triplet in generatorT3():
        progressT3(countT3)
        i, j, k = triplet
        countT3 += 1
        sumT3 = Z[i] + Z[j] + Z[k]

        if sumT3 > max_sum or (sumT3 == max_sum and triplet < max_triplet):
            max_sum = sumT3
            max_triplet = triplet
            triplet_sums.append((sumT3, triplet))
    
    # Sortieren der Tripel: Zuerst nach Summe (absteigend), dann nach Indizes (aufsteigend)
    triplet_sums.sort(key=lambda x: (-x[0], x[1]))  # -x[0] für absteigende Sortierung der Summe

    # Das erste Tripel ist das mit der größten Summe und den kleineren Indizes
    return countT3, triplet_sums[0]
   
def iprint(outputline):
    ii = Counter()
    print()
    print(ii()+outputline)

def plot3(A, Amin, Amax, B, Bmin, Bmax, C, Cmin, Cmax):
    
    # Skaliere die Werte auf eine Breite von maxcolumn Zeichen
    maxcolumn = 80
    
    def scale_value(value, min_val, max_val):
        # Berechne die Position auf einer Skala von 0 bis maxcolumn
        return int((value - min_val) / (max_val - min_val) * (maxcolumn - 1))  # -1 wegen Skalenstart bei 0
        
    # Skaliere A, B, C separat
    pos_A = scale_value(A, Amin, Amax)
    pos_B = scale_value(B, Bmin, Bmax)
    pos_C = scale_value(C, Cmin, Cmax)

    # Erstelle eine Liste von Leerzeichen für die Skala
    plot = [' '] * maxcolumn

    # Erstelle eine Liste mit den skalierten Positionen und Symbolen
    scaled_values = [(pos_A, '|'), (pos_B, '#'), (pos_C, '⊽')]

    # Sortiere die Symbole gemäß ihrer Druckposition
    scaled_values.sort()  # Sortiere nach der Position (kleinster Wert -> größte Position)
    
    # Setze die Symbole an die richtigen Positionen
    for pos, plotsymbol in scaled_values:
        plot[pos] = plotsymbol

    # Gebe die Darstellung als eine einzelne Zeile zurück
    return ''.join(plot)

            
def main():

    global args
    args = parse_arguments()

    global ROT
    global NORM
    
    if args.logformat:
        # lösche die Escapesequencen, damit sie nicht im Logfile auftauchen
        ROT=""
        NORM=""
    
    global start_time
    start_time = time.time()

    # Automatische Erkennung des CSV-Separators
    separator = detect_separator(args.Simulationsergebnisse)

    global Z
    # Z-Werte aus der dritten Spalte einlesen (und " €" entfernen)
    df = load_Z_values(args.Simulationsergebnisse)

    startJD = df['JD'].min()
    endJD = df['JD'].max()
    
    global offset
    offset = startJD
    
    print("Cigna-Anlage-H-Optimizer (früher Artikel 22, a22)")
    
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    iprint(f"Datum des Optimizer-Laufs: {today}")

    iprint(f"Die Berechnung und Ausgabewerte berücksichtigen die Eigenbeteiligung auf der Basis der durchschnittlichen Monatsgehälter in der Datei '{args.Durchschnittsgehälter}'.")
    print("Beträge, die die Eigenbeteiligung berücksichtigen, sind mit '⊽' gekennzeichnet.")
    print("'Beantragungszeitraum' meint den freiwählbaren Ein-Jahres-Zeitraum (365 Tage), falls ein Schalttag beeinhaltet ist: 366 Tage.")

    iprint(f"Die eingelesenen Cigna-Daten umfassen den Zeitraum {T(startJD)} → {T(endJD)}, das sind {endJD-startJD+1} Tage.")
    iprint(f"Hinweis: Index[0] = Offset Julian date {startJD} ({t(0)})")

    # Salary values

    if args.Durchschnittsgehälter and os.path.exists(args.Durchschnittsgehälter):
        df_gehalt_sparse = load_gehalt_values(args.Durchschnittsgehälter)
    else:
        iprint("Keine Gehaltsdatei benannt bzw. Gehaltsdatei nicht gefunden.")
        iprint(f"Als durchschnittliches Gehalt werden {B(args.averagesalary)} angenommen.")
        df_gehalt_sparse = load_constant_gehalt_values(args.averagesalary, startJD)
   
    iprint("Die angenommenen Durchschnittsgehälter für die Bestimmung der Schwelle und der Eigenbeteiligung (20 %) mit ihren Änderungsdaten sind:")
    print(df_gehalt_sparse[['Datum','formatted_gehalt']].to_string(index=False))
    
    # Erstelle eine Serie für die gesamten JD-Werte im gewünschten Bereich
    jd_range = df['JD']

    # Ein leeres Dictionary für die expandierten gehalt-Werte
    expanded_gehalt = {jd: None for jd in jd_range}

    # 1. Auffüllen des gehalt-Feldes vor dem ersten JD
    first_gehalt_value = df_gehalt_sparse['gehalt'].iloc[0]  # Der erste gehalt-Wert

    for jd in range(startJD, df_gehalt_sparse['JD'].iloc[0]):
        expanded_gehalt[jd] = first_gehalt_value  # Vor dem ersten JD den ersten gehalt-Wert setzen

    # 2. Übertrage die gehalt-Werte für die Intervallbereiche zwischen den sparsamen Daten
    for i in range(len(df_gehalt_sparse) - 1):
        start_jd = df_gehalt_sparse['JD'].iloc[i]
        end_jd = df_gehalt_sparse['JD'].iloc[i + 1]
        gehalt_value = df_gehalt_sparse['gehalt'].iloc[i]
    
        # Fülle alle JD-Werte zwischen start_jd und end_jd-1 mit dem entsprechenden gehalt-Wert
        for jd in range(start_jd, end_jd):
            expanded_gehalt[jd] = gehalt_value

    # 3. Der letzte gehalt-Wert wird für die verbleibenden JD-Werte bis zum Enddatum gesetzt
    last_start_jd = df_gehalt_sparse['JD'].iloc[-1]
    last_gehalt_value = df_gehalt_sparse['gehalt'].iloc[-1]

    for jd in range(last_start_jd, endJD+1):
        expanded_gehalt[jd] = last_gehalt_value
    
    # Umwandlung des expanded_gehalt-Dictionary in ein DataFrame
    df_gehalt = pd.DataFrame(list(expanded_gehalt.items()), columns=['JD', 'gehalt'])
    
    # Apply 20 % of gehalt value to "Betrag"   
    # Merge der beiden DataFrames auf JD
    global merged_df
    merged_df = pd.merge(df, df_gehalt, on='JD')

    # Modifikation der Betrag-Werte gemäß der Bedingung
    merged_df['cut'] = merged_df.apply(
        lambda row: row['Betrag'] - 0.2 * row['gehalt'] if row['Betrag'] >= 0.2 * row['gehalt'] else 0.0,
        axis=1
    )

    merged_df["antragsberechtigt"] = merged_df["Betrag"].apply(B)
    merged_df["Gehalt"] = merged_df["gehalt"].apply(B)
    merged_df["gehalt20"] = merged_df["gehalt"].apply(lambda gehalt: 0.2 * gehalt)
    merged_df["Cut"] = merged_df["cut"].apply(b)
    merged_df["jd"] = merged_df["JD"].apply(lambda JD: JD - offset)
   
    # Überschreibe Datensatz mit dem "cut"-Ergebnis
    global RAW
    RAW = df['Betrag'].values
    Z = merged_df['cut'].values
    
    global LEAPDAY
    LEAPDAY = False

    iprint("Zum Kalenderjahr synchrone Beantragungszeiträume ab dem ältesten 02.01. (ältester möglicher Beantragungszeitraum):")
    # Durchlaufe die Julian Dates und prüfe auf den Schalttag (29.02.) und Jahresbeginne (02.01.)
    for jd in jd_range:
    
        year, month, day = gregorian.from_jd(jd)
        if month == 2 and day == 29:
            iprint(f"Hinweis: Schalttag für {year} (Schaltjahr) gefunden: {T(jd)} Index[{jd-offset}]")
            LEAPDAY = jd-offset
            
        if month == 1 and day == 2:
            print(f"{T(jd)} {B(RAW[jd-offset])} {b(Z[jd-offset])} [{jd-offset}]")

    # Ein Jahr ist Schaltjahr, meistens
    # LEAPDAY=1153 # 29.02.2024 jd 1153
    # 02.01.2021 (jd 0) - 31.12.2024 (jd 1459) Beispiel

    global YEAR,MAXDAYS,LEAPYEAR
    YEAR=365
    if not LEAPDAY:
        LEAPYEAR=365
        MAXDAYS=YEAR-1+YEAR+YEAR+YEAR
    else:
        LEAPYEAR=366
        MAXDAYS=YEAR-1+YEAR+YEAR+LEAPYEAR

    iprint("Globales Maximum eines (einzigen) ältesten Beantragungszeitraums im gesamten Zeitfenster:")
    max_Sum1, opt_Mono = bestimmeEinenOptimalenBeantragungszeitraum()
    print(f"{tb(opt_Mono)} => {ROT}{b(max_Sum1)}{NORM} [{opt_Mono}]")

    iprint(f"Zwei älteste Beantragungszeiträume, deren Summe maximal ist:")
    countT2, (max_Sum2, opt_Tupel) = bestimmeZweiOptimaleBeantragungszeiträume()
    print(f"{tb(opt_Tupel[0])} + {tb(opt_Tupel[1])} => {ROT}{b(max_Sum2)}{NORM} Tupel{opt_Tupel}")
    iprint(f"Hinweis: es wurden {countT2:,}".replace(',', '.') + " Kombinationen von zwei Abrechnungszeiträumen getestet.")

    if not args.no3:
        iprint(f"Drei älteste Beantragungszeiträume, deren Summe maximal ist:")
        countT3, (max_Sum3, opt_Triplet) = bestimmeDreiOptimaleBeantragungszeiträume()
        print(f"{tb(opt_Triplet[0])} + {tb(opt_Triplet[1])} + {tb(opt_Triplet[2])} " +
            f"=> {ROT}{b(max_Sum3)}{NORM} Triplet[{opt_Triplet[0]}, {opt_Triplet[1]}, {opt_Triplet[2]}]")
        iprint(f"Hinweis: es wurden {countT3:,}".replace(',', '.') + " Kombinationen von drei Abrechnungszeiträumen getestet.")

    else:
        iprint(f"Hinweis: Die Optimierung von drei Abrechnungszeiträumen wurde wunschgemäß übersprungen.")
        
    iprint(f"Total running time: {(time.time()-start_time):.2f} Sekunden.")
    
    minScale = min( merged_df['Betrag'].min(), merged_df['gehalt20'].min(), merged_df['cut'].min() )
    maxScale = max( merged_df['Betrag'].max(), merged_df['gehalt20'].max(), merged_df['cut'].min() )
    
    # Berechne die neue Spalte 'asciiplot' und füge sie hinzu
    merged_df['asciiplot'] = merged_df.apply(lambda row:
        plot3(
            row['Betrag'], minScale, maxScale,
            row['gehalt20'], minScale, maxScale,
            row['cut'], minScale, maxScale
        ),
        axis=1
    )

    # Erstellen einer neuen Spalte, die den Wert des nächsten Index enthält
    merged_df['next_value'] = merged_df['Cut'].shift(-1)

    # Filtere, so dass nur Zeilen angezeigt werden, wenn sich der Wert geändert hat
    df_compact = merged_df[merged_df['Cut'] != merged_df['next_value']]

    if args.long or args.full:
        iprint(f"Kontrollausgabe der Eingabedaten '{args.Simulationsergebnisse}' und '{args.Durchschnittsgehälter}' (differenziell: liste Daten nur bei Änderung)")
        print(df_compact[['Datum', 'antragsberechtigt', 'Gehalt', 'Cut', 'asciiplot']].to_string(index=False))

    if args.full:
        iprint(f"Kontrollausgabe der Eingabedaten '{args.Simulationsergebnisse}' und '{args.Durchschnittsgehälter}' (liste die Daten jeden Tages)")
        # print(merged_df[['Datum', 'JD', 'jd', 'antragsberechtigt', 'Gehalt', 'Cut', 'asciiplot']].to_string(index=False))
        print(merged_df[['Datum', 'antragsberechtigt', 'Gehalt', 'Cut', 'asciiplot']].to_string(index=False))
    
    # Bestimme, ob das Jahr des Datums in einem Gregorianischen Schaltjahr liegt
    # df['Schaltjahr'] = df['Datum'].apply(lambda x: is_leap_year(x.year))

    # Konvertiere den Julianischen Tag in ein Gregorianisches Datum
    merged_df['Datum'] = merged_df['JD'].apply(j2g)

    # Neue Spalten für farbliche Trennung
    merged_df["Betrag_yellow"] = merged_df["Betrag"].where(merged_df["Betrag"] < merged_df["gehalt20"])
    merged_df["Betrag_red"] = merged_df["Betrag"].where(merged_df["Betrag"] > merged_df["gehalt20"])
    merged_df["cut_white"] = merged_df["cut"].where(merged_df["cut"] < 0.009)
    merged_df["cut_cyan"] = merged_df["cut"].where(merged_df["cut"] > 0.001)

    # Umwandlung ins Long-Format für Plotly
    long_df = merged_df.melt(id_vars='Datum', value_vars=['Betrag_yellow', 'Betrag_red', 'gehalt20', 'cut_cyan', 'cut_white'], 
        var_name='Kategorie', value_name='Wert')

    # Farbzuordnung
    color_map = {
        "Betrag_yellow": "lightyellow",
        "Betrag_red": "lightsalmon",
        "gehalt20": "lightblue",
        "cut_cyan": "cyan",
        "cut_white": "grey"
    }
    
    if args.no3:
        title_of_plot = "Anlage H (vormals A22) Optimizer<br>1 und 2 Abrechnungszeiträume"
    else:
        title_of_plot = "Anlage H (vormals A22) Optimizer<br>1, 2 und 3 Abrechnungszeiträume"
    
    # Plotly Diagramm
    fig = px.line(long_df,
              x='Datum', 
              y='Wert', 
              color='Kategorie',
              color_discrete_map=color_map,
              title= title_of_plot,
              category_orders={"Kategorie": ["antragsberechtigt", "gehalt20", "cut_cyan", "cut_white"]},
              labels={'Datum': 'Startdatum', 'Wert': '(€)', 'Kategorie': 'Kurven'},
              template='plotly')

    def plotLine( jd, offset, linename, text, color, width ):
        datum_range = []
        for i in range(jd+offset, jd+offset+leapspan(jd) ):
            datum_range.append(j2g(i))
            
        betrag = Z[jd]
        fig.add_trace(go.Scatter(
            x = datum_range,   # Mehrere Punkte zwischen Datum1 und Datum2
            y = [betrag]* len(datum_range),
            mode = 'lines',  
            name = linename,
            line = dict(color = color, width = width ),
            hoverinfo = "x+y",
            hovertemplate = text+"<br>Datum: %{x}<br>Wert: %{y}€"  
        ))

    plotLine( opt_Mono, offset, "line1", "Zeitraum", "black", 7)

    try:
        if opt_Tupel:
            plotLine( opt_Tupel[0], offset, "line2", "1. Zeitraum", "lime", 5)
            plotLine( opt_Tupel[1], offset, "line2", "2. Zeitraum", "lime", 5)
    except:
        pass

    try:
        if opt_Triplet:
            plotLine( opt_Triplet[0], offset, "line3", "1. Zeitraum", "blue", 3)
            plotLine( opt_Triplet[1], offset, "line3", "2. Zeitraum", "blue", 3)
            plotLine( opt_Triplet[2], offset, "line3", "3. Zeitraum", "blue", 3)
    except:
        pass
                
    # Texte anpassen
    newnames = { "Betrag_yellow": "antragsberechtigt (< 20 % Gehalt)",
        "Betrag_red": "antragsberechtigt (≥ 20 % Gehalt)",
        "gehalt20": "20 % Gehalt",
        "cut_cyan": "Auszahlungsbetrag<br>unter Berücksichtigung<br>des Gehalts",
        "cut_white": "Keine Auszahlung,<br>da unterhalb des Eigenbeteiligungsmaximums",
        "line1": "Ein optimaler Abrechnungszeitraum<br>(globales Maximum)<br>" + b(max_Sum1) + "<br>" + tb(opt_Mono),
        "line2": "Zwei optimale Abrechnungszeiträume<br>Σ2 " + b(max_Sum2) + "<br>" + tb(opt_Tupel[0]) + " " + tb(opt_Tupel[1]) }
        
    if not args.no3:
        newnames['line3'] = "Drei optimale Abrechnungszeiträume<br>Σ3 " + b(max_Sum3) + "<br>" + tb(opt_Triplet[0]) + " " + tb(opt_Triplet[1]) + " " + tb(opt_Triplet[2])

    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
        legendgroup = newnames[t.name],
        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name]))
    )
                                      
    # Zeige das Diagramm an
    fig.update_layout(
        xaxis_title="Startdatum des 365/366-Tage-Intervals",
        yaxis_title="antragsberechtigt, 20 % Gehalt, Auszahlungsbetrag (€)",
        xaxis_tickformat="%Y-%m-%d"  # Formatierung des Datums auf der x-Achse
    )

    # Zeige das Diagramm auch interaktiv an
    fig.show()


if __name__ == "__main__":
    main()

