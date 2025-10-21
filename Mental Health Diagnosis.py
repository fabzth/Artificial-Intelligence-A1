"""
Mental Health Diagnosis
Author: Fabiola Zeth Patero
Student ID: 2023-191

Note: AI assistance was used as a tool to enhance productivity and code quality, similar to how developers use debuggers and other programming aids.
All fundamental concepts, research design, and final implementation decisions were made by Fabiola Patero.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('/Users/fabiolazeth/PycharmProjects/PythonProject/mental_disorders_dataset.csv')
df.columns = df.columns.str.strip()

print("Dataset Info:")
print(f"Total patients: {len(df)}")
print("Diagnoses:", df['Expert Diagnose'].value_counts().to_dict())


def preprocess_data(df):
    df_clean = df.copy()

    # Convert binary columns
    binary_cols = ['Mood Swing', 'Suicidal thoughts', 'Aggressive Response']
    for col in binary_cols:
        df_clean[col] = (df_clean[col] == 'YES').astype(int)

    # Convert frequency to scores
    freq_map = {'Seldom': 1, 'Sometimes': 2, 'Usually': 3, 'Most-Often': 4}
    freq_cols = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']
    for col in freq_cols:
        df_clean[col] = df_clean[col].map(freq_map)

    # Convert score columns
    def get_score(val):
        if 'From' in str(val):
            return int(str(val).split(' ')[0])
        return 5

    df_clean['Sexual Activity'] = df_clean['Sexual Activity'].apply(get_score)
    df_clean['Optimisim'] = df_clean['Optimisim'].apply(get_score)

    return df_clean


df_clean = preprocess_data(df)


# System 1: Forward Chaining (Symptom-Based)
def forward_chaining_predict(row):
    """Simple forward chaining: symptoms -> conclusions"""

    # Count symptoms for each condition
    bipolar1_score = 0
    bipolar2_score = 0
    depression_score = 0
    normal_score = 0

    # Bipolar Type-1 rules (mania focused)
    if row['Euphoric'] >= 3:  # Often euphoric
        bipolar1_score += 2
    if row['Sexual Activity'] >= 6:  # High sexual activity
        bipolar1_score += 1
    if row['Mood Swing'] == 1:  # Has mood swings
        bipolar1_score += 1

    # Bipolar Type-2 rules (depression + hypomania)
    if row['Sadness'] >= 3 and row['Euphoric'] >= 2:
        bipolar2_score += 2
    if row['Mood Swing'] == 1:
        bipolar2_score += 1
    if row['Sleep dissorder'] >= 3:
        bipolar2_score += 1

    # Depression rules
    if row['Sadness'] >= 3:  # Often sad
        depression_score += 2
    if row['Exhausted'] >= 3:  # Often exhausted
        depression_score += 1
    if row['Suicidal thoughts'] == 1:  # Suicidal thoughts
        depression_score += 1
    if row['Optimisim'] <= 4:  # Low optimism
        depression_score += 1

    # Normal rules
    if row['Sadness'] <= 2:  # Seldom sad
        normal_score += 2
    if row['Exhausted'] <= 2:  # Seldom exhausted
        normal_score += 1
    if row['Optimisim'] >= 6:  # High optimism
        normal_score += 1
    if row['Mood Swing'] == 0:  # No mood swings
        normal_score += 1

    # Find highest score
    scores = {
        'Bipolar Type-1': bipolar1_score,
        'Bipolar Type-2': bipolar2_score,
        'Depression': depression_score,
        'Normal': normal_score
    }

    return max(scores, key=scores.get)


# System 2: Backward Chaining (Diagnostic Accuracy)
def backward_chaining_predict(row):
    """Simple backward chaining: test each hypothesis"""

    # Test Bipolar Type-1 hypothesis
    manic_symptoms = 0
    if row['Euphoric'] >= 3: manic_symptoms += 1
    if row['Sexual Activity'] >= 6: manic_symptoms += 1
    if row['Mood Swing'] == 1: manic_symptoms += 1

    bipolar1_confident = manic_symptoms >= 2

    # Test Bipolar Type-2 hypothesis
    hypomanic_symptoms = 0
    depressive_symptoms = 0

    if row['Euphoric'] >= 2: hypomanic_symptoms += 1
    if row['Mood Swing'] == 1: hypomanic_symptoms += 1

    if row['Sadness'] >= 3: depressive_symptoms += 1
    if row['Exhausted'] >= 3: depressive_symptoms += 1
    if row['Optimisim'] <= 4: depressive_symptoms += 1

    bipolar2_confident = (hypomanic_symptoms >= 1 and depressive_symptoms >= 2)

    # Test Depression hypothesis
    if bipolar1_confident or bipolar2_confident:
        depression_confident = False  # Rule out depression if bipolar features present
    else:
        depression_symptoms = 0
        if row['Sadness'] >= 3: depression_symptoms += 1
        if row['Exhausted'] >= 3: depression_symptoms += 1
        if row['Suicidal thoughts'] == 1: depression_symptoms += 1
        if row['Optimisim'] <= 4: depression_symptoms += 1

        depression_confident = depression_symptoms >= 3

    # Test Normal hypothesis
    if not (bipolar1_confident or bipolar2_confident or depression_confident):
        normal_symptoms = 0
        if row['Sadness'] <= 2: normal_symptoms += 1
        if row['Exhausted'] <= 2: normal_symptoms += 1
        if row['Optimisim'] >= 6: normal_symptoms += 1
        if row['Mood Swing'] == 0: normal_symptoms += 1

        normal_confident = normal_symptoms >= 2
    else:
        normal_confident = False

    # Return diagnosis
    if bipolar1_confident:
        return 'Bipolar Type-1'
    elif bipolar2_confident:
        return 'Bipolar Type-2'
    elif depression_confident:
        return 'Depression'
    elif normal_confident:
        return 'Normal'
    else:
        return 'Normal'  # Default


# Evaluate both systems
def evaluate_system(true_labels, pred_labels, system_name):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    print(f"\n{system_name} Results:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")

    return accuracy, precision, recall, f1


# Get predictions
true_diagnoses = df_clean['Expert Diagnose']
forward_predictions = df_clean.apply(forward_chaining_predict, axis=1)
backward_predictions = df_clean.apply(backward_chaining_predict, axis=1)

# Evaluate
print("\n" + "=" * 50)
forward_acc, forward_prec, forward_rec, forward_f1 = evaluate_system(
    true_diagnoses, forward_predictions, "FORWARD CHAINING SYSTEM"
)

print("\n" + "=" * 50)
backward_acc, backward_prec, backward_rec, backward_f1 = evaluate_system(
    true_diagnoses, backward_predictions, "BACKWARD CHAINING SYSTEM"
)

# Compare systems
print("\n" + "=" * 50)
print("COMPARISON:")
print("=" * 50)
print(f"{'Metric':<10} {'Forward':<8} {'Backward':<8} {'Winner':<8}")
print(f"{'-' * 35}")
print(
    f"{'Accuracy':<10} {forward_acc:.3f}    {backward_acc:.3f}    {'Forward' if forward_acc > backward_acc else 'Backward'}")
print(
    f"{'Precision':<10} {forward_prec:.3f}    {backward_prec:.3f}    {'Forward' if forward_prec > backward_prec else 'Backward'}")
print(
    f"{'Recall':<10} {forward_rec:.3f}    {backward_rec:.3f}    {'Forward' if forward_rec > backward_rec else 'Backward'}")
print(
    f"{'F1-Score':<10} {forward_f1:.3f}    {backward_f1:.3f}    {'Forward' if forward_f1 > backward_f1 else 'Backward'}")

# Show some examples
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS:")
print("=" * 50)
print(f"{'Patient':<10} {'Actual':<15} {'Forward':<15} {'Backward':<15}")
print(f"{'-' * 55}")

for i in range(5):
    actual = true_diagnoses.iloc[i]
    forward = forward_predictions.iloc[i]
    backward = backward_predictions.iloc[i]

    print(f"{i + 1:<10} {actual:<15} {forward:<15} {backward:<15}")
