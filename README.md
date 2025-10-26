# Prosty klasyfikator Irysów (Scikit-learn)

To jest projekt rekrutacyjny.

## Opis

Skrypt `main.py` wczytuje zbiór danych "Iris", dzieli go na zbiór treningowy i testowy, a następnie buduje i ocenia prosty model uczenia maszynowego (K-Najbliższych Sąsiadów).

## Wyniki

Model osiągnął **100% skuteczności (accuracy)** na wydzielonym zbiorze testowym. Interpretacja i propozycje ulepszeń znajdują się w komentarzach na końcu skryptu `main.py`.

## Jak uruchomić

1.  Sklonuj repozytorium.
2.  Stwórz i aktywuj środowisko wirtualne (np. `python -m venv .venv`).
3.  Zainstaluj zależności: `pip install -r requirements.txt`
4.  Uruchom skrypt: `python main.py`