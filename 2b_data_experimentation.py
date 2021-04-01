import pandas as pd
import matplotlib.pyplot as plt 


train =  pd.read_csv("C:/BIG DATA/q2b/train.csv")

count=0
pos_count=0
neg_count=0
questionLengthDict = dict()
posQuestionLengthDict = dict()
negQuestionLengthDict = dict()

for index in range(len(train['Id'])):

    q1 = str(train['Question1'][index])
    q2 = str(train['Question2'][index])

    q1l = len(q1)
    q2l = len(q2)
    if q1l in questionLengthDict:
        questionLengthDict[q1l] = questionLengthDict[q1l] + 1
    else:
        questionLengthDict[q1l] = 1
    
    if q2l in questionLengthDict:
        questionLengthDict[q2l] = questionLengthDict[q2l] + 1
    else:
        questionLengthDict[q2l] = 1
    
    if bool(train['IsDuplicate'][index]):
        pos_count+=1
        if q1l in posQuestionLengthDict:
            posQuestionLengthDict[q1l] = posQuestionLengthDict[q1l] + 1
        else:
            posQuestionLengthDict[q1l] = 1

        if q2l in posQuestionLengthDict:
            posQuestionLengthDict[q2l] = posQuestionLengthDict[q2l] + 1
        else:
            posQuestionLengthDict[q2l] = 1
    else:
        neg_count+=1
        if q1l in negQuestionLengthDict:
            negQuestionLengthDict[q1l] = negQuestionLengthDict[q1l] + 1
        else:
            negQuestionLengthDict[q1l] = 1

        if q2l in negQuestionLengthDict:
            negQuestionLengthDict[q2l] = negQuestionLengthDict[q2l] + 1
        else:
            negQuestionLengthDict[q2l] = 1
        
    count+=1

plt.bar(list(questionLengthDict.keys()), list(questionLengthDict.values()))
plt.xlim(0,250)
plt.ylabel('Number')
plt.xlabel('length')
plt.title('Question length distribution')
plt.show()
print(count)

def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    
fig, ax = plt.subplots()
question_pairs= ['Duplicates', 'No Duplicates']
number = [pos_count , neg_count]
rect = ax.bar(question_pairs,number)
ax.set_ylabel('Number')
ax.set_title('Training class imbalance' , fontstyle='italic')
autolabel(rect)
plt.show()















