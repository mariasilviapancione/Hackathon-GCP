import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Carica i dati
df = pd.read_csv("../data/mushrooms.csv")

#  Cambia i nomi delle colonne (cap-shape → cap_shape)
df.columns = [col.replace("-", "_") for col in df.columns]

# Encoding
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Salva encoders
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

# Split
selected_features = ["odor", "bruises", "cap_shape", "gill_color", "ring_type"]
X = df[selected_features]
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensemble
ensemble = VotingClassifier([
    ('mnb', MultinomialNB()),
    ('svc', SVC()),
    ('rf', RandomForestClassifier())
])

# GridSearch
clf = GridSearchCV(
    ensemble,
    {
        'mnb__alpha': [0.1, 1, 2],
        'svc__C': [0.1, 1, 10],
        'svc__class_weight': ['balanced'],
        'rf__n_estimators': [10, 100],
        'rf__criterion': ['gini', 'entropy']
    },
    cv=5,
    scoring='f1_macro'
)

clf.fit(X_train, y_train)

# Salva modello
pickle.dump(clf, open("model.pkl", "wb"))

