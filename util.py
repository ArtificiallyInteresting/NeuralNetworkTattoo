import pandas as pd
import numpy as np

def lettersToNumbers(df):
    # newDf = pd.DataFrame(columns=['x_train', 'y_train'])
    x_train = pd.DataFrame()
    y_train = pd.DataFrame()
    for index, row in df.iterrows():
        # print(row)
        x_train = x_train.append(pd.Series(oneWordToNumbers(row["Input"]), name="x_train"))
        y_train = y_train.append(pd.Series(oneWordToNumbers(row["Output"]), name="y_train"))
        # newDf = newDf.append(pd.DataFrame([row["Input"], row["Output"]], columns=newDf.columns))
    # newDf = pd.DataFrame({"x_train": x_train, "y_train": y_train}, index=np.arange(0,len(x_train)))
    # print(newDf)
    return x_train, y_train

def numbersToLetters(df):
    outputs = []
    for item in df:
        outputs.append(numbersToOneWord(item))
    return outputs

def numbersToOneWord(numbers):
    output = ""
    for number in numbers:
        output += chr(int(ord(' ') + number + 0.5))
    return output

def oneWordToNumbers(word):
    numbers = []
    for letter in word:
        numbers.append(ord(letter) - ord(' ')) #A lot of junk between space and letters. Maybe use "A" instead and give space a special number
    while len(numbers) < 8:
        numbers.append(0)
    if len(numbers) > 8:
        numbers = numbers[:8]
    return numbers

def finalStats(predictions, y_train):
    hits = 0
    for prediction, actual in zip(predictions, y_train.values):
        predictionWord = numbersToOneWord(prediction)
        actualWord = numbersToOneWord(actual)
        print("Actual: {}, Predicted:{}".format(actualWord, predictionWord))
        if predictionWord == actualWord:
            hits += 1
    print("Correct: {}, Total: {}".format(hits, len(predictions)))
