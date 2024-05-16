# OBD-JuicIR: OBD Daten zur Widerstands- und SoH-Schätzung


Dieses Projekt nutzt OBD-Daten, um den Widerstand und den State of Health (SoH) von Fahrzeugbatterien zu schätzen. Es besteht aus zwei Hauptskripten: `ir.py` und `soh.py`.

## Dateien

### ir.py
Dieses Skript ist für die Berechnung des Innenwiderstands (IR) der Batterie zuständig. Es verwendet OBD-Daten zur Durchführung der Berechnungen und gibt den geschätzten Innenwiderstand aus.

python .\ir.py --data_path="test_data.csv" --threshold=0.99 --cells=96

Hilfe: python .\ir.py --help


### soh.py
Dieses Skript berechnet den State of Health (SoH) der Batterie basierend auf den OBD-Daten. Der SoH gibt Auskunft über die verbleibende Kapazität der Batterie im Vergleich zu ihrem Neuzustand.

## Installation

Um dieses Projekt zu verwenden, müssen Sie Python installiert haben. 


## Support und Fragen
Im Akkudoktor Forum bitte -> forum.akkudoktor.net

Autor: Dr. Andreas Schmitz
