import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dat = yf.Ticker("GS")

# Get 2 years of daily data
data = dat.history(period="5y")

def did_it_go_up_the_next_day(data):

    # add a columns to flag if the stock went up the next day
    data["Next_Day_Up"] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # remove the last row since theres no next day data
    data = data[:-1]

def create_dataset(data):
    did_it_go_up_the_next_day(data)

    # daily returns
    data['Return'] = data['Close'].pct_change()

    # 5 day and 20 day moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # volatility 5 day rolling standard of returns
    data['Volatility'] = data['Return'].rolling(window=5).std()

    # volume change 
    data['Volume_Change'] = data['Volume'].pct_change()

    # drop rows with NaN from rolling calculations
    data = data.dropna().copy()
    

    print(f"\nData shape after features: {data.shape}")
    print(f"\nFeature columns: {data.columns.tolist()}")
    print("\nTarget distribution:")
    print(data['Next_Day_Up'].value_counts())
    print(f"\nPercentage of up days: {data['Next_Day_Up'].mean():.1%}")



create_dataset(data)


# select features for the model
feature_cols = ['Return', 'MA5', 'MA20', 'Volatility', 'Volume_Change']
X = data[feature_cols]
y = data['Next_Day_Up']

# split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# train a Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)


print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.1%}")
print(f"Baseline (always predict up): {y_test.mean():.1%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

