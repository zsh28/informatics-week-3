import pandas as pd

def calculate_metrics(data, actual_col, predicted_col, annotator_a_col, annotator_b_col, mapping_a, mapping_b):
    # Convert string labels to numeric values
    label_mapping = {mapping_a: 1, mapping_b: 0}
    data[actual_col] = data[actual_col].map(label_mapping)
    data[predicted_col] = data[predicted_col].map(label_mapping)

    # Calculating the precision
    TP = sum((data[predicted_col] == 1) & (data[actual_col] == 1))
    FP = sum((data[predicted_col] == 1) & (data[actual_col] == 0))
    FN = sum((data[predicted_col] == 0) & (data[actual_col] == 1))
    TN = sum((data[predicted_col] == 0) & (data[actual_col] == 0))

    # Handling division by zero errors and calculating the metrics
    if TP + FP == 0:
        P = 0
    else:
        P = TP / (TP + FP)

    if TP + FN == 0:
        R = 0
    else:
        R = TP / (TP + FN)

    if P + R == 0:
        F = 0
    else:
        F = 2 * (P * R) / (P + R)

    if TP + TN + FP + FN == 0:
        ACC = 0
    else:
        ACC = (TP + TN) / (TP + TN + FP + FN)

    # Calculating Cohen's Kappa
    agreement = sum(data[annotator_a_col] == data[annotator_b_col])
    P0 = agreement / len(data)
    categories = sorted(data[annotator_a_col].unique())
    Pe = sum((sum(data[annotator_a_col] == cat) / len(data)) * (sum(data[annotator_b_col] == cat) / len(data)) for cat in categories)
    if 1 - Pe == 0:
        K = 0
    else:
        K = (P0 - Pe) / (1 - Pe)

    # Print the results and round them to 2 decimal places
    print("Precision: ", round(P, 2))
    print("Recall: ", round(R, 2))
    print("F-measure: ", round(F, 2))
    print("Accuracy: ", round(ACC, 2))
    print("Cohen's Kappa: ", round(K, 2))

# Main function
if __name__ == "__main__":
    # Ask for the name of the file
    filename = input("Enter the name of the file (must end in .csv): ")
    # Read the CSV file
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print("File not found.")
        exit()

    # Input column names and class mappings
    actual_col = input("Enter the name of the actual column (press enter if name is Actual): ")
    if actual_col == "":
        actual_col = "Actual"
    predicted_col = input("Enter the name of the predicted column (press enter if name is Predicted): ")
    if predicted_col == "":
        predicted_col = "Predicted"
    annotator_a_col = input("Enter the name of the annotator A column (press enter if name is Annotator.A):")
    if annotator_a_col == "":
        annotator_a_col = "Annotator.A"
    annotator_b_col = input("Enter the name of the annotator B column (press enter if name is Annotator.B):")
    if annotator_b_col == "":
        annotator_b_col = "Annotator.B"
    mapping_a = input("Enter the Positive Class: ")
    if mapping_a == "":
        print("Positive class not provided.")
        exit()
    mapping_b = input("Enter the Negative Class: ")
    if mapping_b == "":
        print("Negative class not provided.")
        exit()

    # Call the function
    calculate_metrics(data, actual_col, predicted_col, annotator_a_col, annotator_b_col, mapping_a, mapping_b)