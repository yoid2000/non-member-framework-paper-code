from diffTools import makeModel, getColType, prepDataframes, getPrecisionFromBestGuess
import joblib
import pandas as pd
import numpy as np
import json
import pprint
import os

pp = pprint.PrettyPrinter(indent=4)

def sample_rows(df, num_rows=500):
    # Randomly sample rows and create a new dataframe
    sampled_df = df.sample(n=num_rows)
    # Remove the sampled rows from the original dataframe
    df = df.drop(sampled_df.index)
    return df, sampled_df

def replicate_rows(df, frac = 0.1):
    # Calculate the number of rows to replicate
    num_rows = int(len(df) * frac)
    # Randomly select rows
    replicate_rows = df.sample(n=num_rows)
    # Append the replicated rows to the dataframe
    df = pd.concat([df, replicate_rows], ignore_index=True)
    return df

def csv_to_dataframe(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Return the DataFrame
    return df

def print_dataframe_columns(df):
    # Loop through each column
    for column in df.columns:
        # Print the column description
        print("-----")
        print(df[column].describe())

if __name__ == "__main__":
    doTpot = False
    numVictims = 500
    filePath = 'BankChurnersNoId_ctgan.json'
    with open(filePath, 'r') as f:
        testData = json.load(f)
    '''
        dfAnon is the synthetic data generated from dfOrig
        dfTest is additional data not used for the synthetic data
    '''
    dfOrig = pd.DataFrame(testData['originalTable'], columns=testData['colNames'])
    dfAnon = pd.DataFrame(testData['anonTable'], columns=testData['colNames'])
    dfTest = pd.DataFrame(testData['testTable'], columns=testData['colNames'])

    dfOrig, dfTest, dfAnon = prepDataframes(dfOrig, dfTest, dfAnon)

    print(f"Got {dfOrig.shape[0]} original rows")
    print(f"Got {dfAnon.shape[0]} synthetic rows")
    print(f"Got {dfTest.shape[0]} test rows")
    print(list(dfOrig.columns))

    ''' The following runs the tests for the case where the victims are
        completely distinct from the original data
    '''
    print_dataframe_columns(dfOrig)
    print('===============================================')
    print('===============================================')
    results = []
    for target in dfOrig.columns:
        targetType = getColType(dfOrig, target)
        if targetType != 'cat':
            continue
        print(f"Use target {target} with {dfOrig[target].nunique()} distinct values")
        # Here, we are using the original rows to measure the baseline
        # using ML models (this is meant to give a high-quality baseline)
        print("\n----  DIFFERENTIAL FRAMEWORK  ----")
        fileBaseName = filePath + target
        savedModelName = fileBaseName + '.tpot.joblib'
        savedModelPath = os.path.join('models', savedModelName)
        if False and os.path.exists(savedModelPath):
            print("Using auto tpot")
            model = joblib.load(savedModelPath)
        else:
            print("Using non-auto")
            model = makeModel(filePath, target, dfOrig, numVictims=numVictims)
        ''' I screwed up a little, because if the tpot model was stored, then
            I've lost the train and test sets. Fortunately, the training set
            came only from dfOrig, and test entries in dfTest are distinct from
            those, so I can use dfTest as my new test set.
        '''
        X_test = dfTest.drop(target, axis=1)
        y_test = dfTest[target]
        y_pred = model.predict(X_test)
        # probabilities for each class
        probs = model.predict_proba(X_test)
        print(f"probs is type {type(probs)}")
        classLabels = model.classes_
        print("classLabels:")
        print(classLabels)
        precLowerBound = getPrecisionFromBestGuess(y_test)
        print(f"precLowerBound is {precLowerBound}")
        # classLabels contains the distinct values in the target column
        # probs contains the probability that each row in X_test predicts
        # each distinct value, where the column index in classLabels matches
        # that in probs

        # maxProbs has the highest probability in each row of probs
        maxProbs = np.amax(probs, axis=1)
        minMaxProb = np.amin(maxProbs, axis=0)
        maxMaxProb = np.amax(maxProbs, axis=0)
        print(f"maxProbs ranges from {minMaxProb} to {maxMaxProb}")

        ''' I want to step through different thresholds to produce
            different precision/recall values
        '''
        thresh = minMaxProb
        numTests = maxProbs.shape[0]
        bestPrecSoFar = 0
        while True:
            if thresh >= maxMaxProb:
                thresh = maxMaxProb
            doPredicts = (maxProbs >= thresh).astype(int)
            print(f"                     thresh is {thresh}")
            numPredicts = np.sum(doPredicts)
            recall = numPredicts / numTests
            print(f"Recall is {recall} ({numPredicts} out of {numTests})")
            # I only want to make predictions when doPredicts is 1
            numCorrect = 0
            for i in range(numTests):
                if doPredicts[i] == 0:
                    continue
                if y_test.values[i] == y_pred[i]:
                    numCorrect += 1
            prec = numCorrect / numPredicts
            if prec < precLowerBound:
                print(f"Set precision {prec} to lower bound {precLowerBound}")
                prec = precLowerBound
            if prec < bestPrecSoFar:
                print(f"Set precision {prec} to best so far {bestPrecSoFar}")
                prec = bestPrecSoFar
            bestPrecSoFar = max(bestPrecSoFar, prec)
            print(f"Precision is {prec}")
            results.append({'target':target,
                            'recall':recall,
                            'numPredicts':int(numPredicts),
                            'prec':prec})
            if numPredicts <= 20:
                break
            if prec >= 1.0:
                break
            if maxMaxProb == thresh:
                break
            thresh += (maxMaxProb - thresh) / 3
    pp.pprint(results)
    with open('resultsRecall.json', 'w') as f:
        json.dump(results, f, indent=4)
    


'''
# Choose a class and set a higher decision threshold for it
class_index = 0  # index of the class you want to adjust precision for
threshold = 0.7  # set higher decision threshold

# Apply decision threshold
predictions = np.argmax(probabilities, axis=1)  # default predictions
high_precision_preds = (probabilities[:, class_index] > threshold).astype(int)

# Replace predictions for the chosen class with high precision predictions
predictions[predictions == class_index] = high_precision_preds
'''