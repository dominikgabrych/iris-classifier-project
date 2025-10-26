from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Wczytanie i przygotowanie danych ---

iris = load_iris()
X = iris.data  # Cechy (pomiary kwiatów)
y = iris.target # Cel (gatunki jako liczby 0, 1, 2)

# Printy do podejrzenia danych
print("--- 1. Podgląd Danych ---")
print(f"Nazwy cech: {iris.feature_names}")
print(f"Nazwy gatunków: {iris.target_names}")
print(f"Pierwsze 5 próbek (X): \n{X[:5]}")
print(f"Pierwsze 5 etykiet (y): {y[:5]}")


# --- 2. Podział danych na zbiór treningowy i testowy ---

# Dzielimy dane: 70% na trening, 30% na test
# random_state=42 zapewnia, że podział jest zawsze taki sam (powtarzalność)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n--- 2. Podział Danych ---")
print(f"Całkowita liczba próbek: {len(X)}")
print(f"Liczba próbek treningowych: {len(X_train)}")
print(f"Liczba próbek testowych: {len(X_test)}")


# --- 3. Trenowanie modelu ---

print("\n--- 3. Trenowanie Modelu ---")
# Wybieramy model K-Najbliższych Sąsiadów (z 3 sąsiadami)
model_knn = KNeighborsClassifier(n_neighbors=3)

# "Uczymy" model na danych treningowych
model_knn.fit(X_train, y_train)
print(f"Model {type(model_knn).__name__} został wytrenowany.")


# --- 4. Ewaluacja i Wyniki ---

print("\n--- 4. Ewaluacja Modelu ---")
# Model "zgaduje" gatunki dla danych testowych (których nie widział)
y_pred = model_knn.predict(X_test)

# Porównanie przewidywań z prawdą
print(f"Przewidziane przez model: {y_pred}")
print(f"Prawdziwe gatunki:     {y_test}")

# Porównujemy przewidywania (y_pred) z prawdą (y_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nSkuteczność (Accuracy) modelu: {acc * 100:.2f}%")


# --- 5. Interpretacja i Wnioski ---

"""
Krótka interpretacja wyników i propozycje ulepszeń
---------------------------------------------------

Interpretacja Wyniku (100%):
Model osiągnął 100% skuteczności na naszym zbiorze testowym. Oznacza to,
że poprawnie zidentyfikował gatunek dla każdego z 45 kwiatków, których
wcześniej nie widział.

Jest to świetny wynik, ale warto pamiętać, że zbiór "Iris" jest bardzo
"czysty" i prosty. Wynik 100% mógł być też częściowo efektem "szczęśliwego"
podziału danych (ustawionego przez random_state=42).

Propozycje Ulepszeń:
Mimo 100%, zawsze można coś sprawdzić, aby upewnić się, że model jest stabilny:

1. Sprawdzić inne 'k': Użyliśmy 'k=3'. Warto sprawdzić, czy inne wartości
   'n_neighbors' (np. 5, 7) nie dałyby równie dobrych wyników.

2. Wypróbować inny model: Użycie dla porównania innego algorytmu,
   np. Drzewa Decyzyjnego (DecisionTreeClassifier), aby zobaczyć,
   jak on sobie poradzi.

3. Użyć walidacji krzyżowej: Zamiast pojedynczego podziału 70/30,
   użycie walidacji krzyżowej dałoby bardziej obiektywną i uśrednioną
   ocenę skuteczności, niezależną od jednego 'random_state'.
"""