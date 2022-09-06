## _Group name (Group2)_
Deep Learning เป็นการเรียนรู้เชิงลึกที่เลียนแบบการทำงานของโครงข่ายประสาทของมนุษย์ โดยมีการทำงานซ้อนกันหลายๆ ชั้นหรือเรียกว่า Layer จากข้อมูลตัวอย่างเพื่อหา Pattern ของข้อมูล 
โดยการศึกษาในครั้งนี้มีจุดมุ่งหมายเพื่อเปรียบเทียบประสิทธิภาพของ **`Traditional Machine Learning (ML)`** และ **`Multilayer Perceptron (MLP)`** รวมถึงการปรับ Hyperparameter เพื่อความสมบูรณ์ของโมเดล ทางทีมคาดหวังว่าจะได้ประสบการณ์ในเรื่องนี้ และเป็นประโยชน์ต่อผู้ที่กำลังสนใจในเรียนนี้ 

## _Key Highlight_
- XX
- XX
- XX

## 1. Introduction
การศึกษาในครั้งนี้ใช้ข้อมูลเรื่อง Stroke Prediction โดยข้อมูลจะมีลักษณะเป็น **`Binary classification`** คือการทำนายว่าผู้ป่วยคนนั้นมีโอกาสเป็นผู้ป่วยเป็นโรคหลอดเลือดสมองหรือไม่ (1: หากผู้ป่วยเป็นโรคหลอดเลือดสมอง, 0: หากผู้ป่วยไม่เป็นโรคหลอดเลือดสมอง) ซึ่งพิจารณาจากรายละเอียดข้อมูลของผู้ป่วยเช่น อายุ, เพศ, โรคความดันโลหิตสูง, โรคหัวใจ และค่าเฉลี่ยน้ำตาลในเลือด เป็นต้น

## 2. Data

### Data source
**Dataset descrption:**: Stroke Prediction Dataset (Ref: [Stroke Prediction Dataset | Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset))))<br>
**Total Patient:** 5,110 <br>
**Total Features:** 11 <br>
**Fields:**
1) id: รหัสประจำตัวผู้ป่วย 
2) gender: "Male", "Female" หรือ "Other"  
3) age: อายุของผู้ป่วย  
4) hypertension: 0 หากผู้ป่วยไม่เป็นโรคความดันโลหิตสูง, 1 หากผู้ป่วยไม่เป็นโรคความดันโลหิตสูง  
5) heart_disease: 0 หากผู้ป่วยไม่เป็นโรคหัวใจ, 1 หากผู้ป่วยเป็นโรคหัวใจ
6) ever_married: "No" หรือ "Yes"  
7) work_type: "children", "Govt_jov", "Never_worked", "Private" หรือ "Self-employed"  
8) Residence_type: "Rural" หรือ "Urban"  
9) avg_glucose_level: ค่าเฉลี่ยน้ำตาลในเลือด  
10) bmi: ดัชนีมวลกาย
11) smoking_status: "formerly smoked", "never smoked", "smokes" หรือ "Unknown"*  
*Note: "Unknown" ในคอลัมน์ smoking_status หมายถึงไม่มีข้อมูลสำหรับผู้ป่วยรายนี้* <br>

### EDA
- XX
- XX
- XX

### Data preparation
การเตรียมข้อมูลก่อน train model เราทำการ drop ค่า outliner ออก หลังจากนั้นจึงจัดการข้อมูล Binary category และ Multicategory โดยใช้ **`One-Hot encoding`** เพื่อเปลี่ยนข้อมูลที่เก็บในลักษณะ categorical ให้อยู่ในรูป Binary values เนื่องจากการทำ Machine leaning นั้น ต้องการข้อมูลในรูปแบบตัวเลขเพื่อใช้ในการ train และ predict โดยแปลงค่าในคอลัมน์ gender, ever_married, work_type, residence_type และ smoking_status เพื่อให้อยู่ในรูปแบบดังกล่าว <br>

เนื่องจากข้อมูลของเรามีความ imbalance เราจึงเลือกใช้ **`SMOTE`** (synthetic minority over-sampling technique) ซึ่งเป็นเทคนิคที่ใช้ในการแก้ปัญหาการจำแนกข้อมูลที่ไม่สมดุลและทำการ normalize ค่าด้วย StandardScaler <br>

และจากข้อมูลทั้งหมด 5,110 มีการแบ่ง Data splitting (train/val/test) ดังนี้<br>
ML: Train 80% และ Test 20%<br>
MLP: Train 80%, validation 20% ของ Train set และ Test 20%<br>


![image](https://user-images.githubusercontent.com/101736826/187707794-38780d34-8cc0-4fd0-95de-48e3eda8c46f.png)

## 3. Network architecture
Total params: 179,201<br>
Trainable params: 179,201<br>
Non-trainable params: 0<br>
<table>
  <tr>
    <th>Layer (type)</th>
    <th>Output Shape</th>
    <th>Number of Parameter</th>
    <th>Activation function</th>
  </tr>
  <tr>
    <td>hidden1 (Dense)</td>
    <td>(None, 512)</td>
    <td>4,608</td>
    <td>sigmoid</td>
  </tr>
  <tr>
    <td>hidden1 (Dense)</td>
    <td>(None, 512)</td>
    <td>4608</td>
    <td>sigmoid</td>
  </tr>
  <tr>
    <td>hidden2 (Dense)</td>
    <td>(None, 256)</td>
    <td>131,328</td>
    <td>sigmoid</td>
  </tr>
  <tr>
    <td>hidden3 (Dense)</td>
    <td>(None, 128)</td>
    <td>32,896</td>
    <td>sigmoid</td>
  </tr>
  <tr>
    <td>hidden4 (Dense)</td>
    <td>(None, 64)</td>
    <td>8,256</td>
    <td>sigmoid</td>
  </tr>
  <tr>
    <td>hidden5 (Dense)</td>
    <td>(None, 32)</td>
    <td>2,080</td>
    <td>sigmoid</td>
  </tr>
  <tr>
    <td>output (Dense)</td>
    <td>(None, 1)</td>
    <td>33</td>
    <td>sigmoid</td>
  </tr>
</table>

## 4. Training
### 4.1 Traditional Machine Learning (ML)
เราใช้ Scikit-learn ซึ่งเป็น library ใน Python ในการเทรนโมเดลแบบ Traditional Machine Learning ซึ่งประกอบไปด้วย **`RidgeClassifier`**, **`LinearSVC`**, **`SVC`**, **`LogisticRegression`**, **`KNeighborsClassifier`** และ **`RandomForestClassifier`** โดยในแต่ละ model มีการ tuning ดังนี้

- RidgeClassifier     : RidgeClassifier(alpha = 0.0001, solver = 'lsqr')
- LinearSVC           : LinearSVC(C= 0.001,multi_class = 'crammer_singer',penalty = 'l2',loss='hinge')
- SVC                 : SVC(C= 1.5, gamma = 'scale', kernel = 'rbf', random_state = 0)
- LogisticRegression  : LogisticRegression(C= 1, max_iter= 100,solver = 'sag', penalty = 'l2')
- KNeighborsClassifier: KNeighborsClassifier(n_neighbors=5, leaf_size = 10, weights='distance')
- RandomForestClassifier: RandomForestClassifier(criterion= 'log_loss',  max_depth = 3, max_features = 'log2', n_estimators= 400)

### 4.2 Multilayer Perceptron (MLP)
ในการเทรนโมเดล  Multilayer Perceptron (MLP) เราใช้วิธี trial-and-error เพื่อหาโมเดลที่ดีที่สุดเพื่อพยากรณ์การเป็นโรคหลอดเลือดสมองสำหรับ dataset ข้างต้น โดยเราทำการเทรนโมเดลด้วย initial random weights ที่แตกต่างกัน 5 รอบ

## 5. Results
### 5.1 Traditional Machine Learning (ML)

<table>
  <tr>
    <th>Classification algorithm</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>RidgeClassifier</td>
    <td>0.737769</td>
    <td>0.722222</td>
    <td>0.133562</td>
    <td>0.225434</td>
  </tr>
  <tr>
    <td>LinearSVC</td>
    <td>0.723092</td>
    <td>0.777778</td>
    <td>0.134185</td>
    <td>0.228883</td>
  </tr>
  <tr>
    <td>SVC</td>
    <td>0.752446</td>
    <td>0.722222</td>
    <td>0.140794</td>
    <td>0.235650</td>
  </tr>
  <tr>
    <td>LogisticRegression</td>
    <td>0.758317</td>
    <td>0.703704</td>
    <td>0.141264</td>
    <td>0.235294</td>
  </tr>
  <tr>
    <td>KNeighborsClassifier</td>
    <td>0.826810</td>
    <td>0.407407</td>
    <td>0.131737</td>
    <td>0.199095</td>
  </tr>
  <tr>
    <td>RandomForestClassifier</td>
    <td>0.718200</td>
    <td>0.759259</td>
    <td>0.129747</td>
    <td>0.221622</td>
  </tr>
</table>


### 5.2 Multilayer Perceptron (MLP)
Mean±SD of Accuracy = ( , )<br>
<table>
  <tr>
    <th>Round</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>2</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>3</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>4</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  </table>

## 6. Experiment result and discussion
- สำหรับการ train model หนึ่งในสิ่งสำคัญคือการเลือกใช้ฟีเจอร์เพื่อไม่ให้ model มีความ overfit มากเกินไป ดังนั้น เราจึงเริ่มจากการดูค่า correlation ของตัวแปรต่างๆ ต่อการเป็นโรคหลอดเลือดสมอง (stroke) ซึ่งหาก correlation มีค่ามาก หมายถึงมีความสัมพันธ์ต่อการเป็น stroke มาก เช่น อายุ การเป็นโรคหัวใจ เป็นต้น
- การ normalization เราใช้ StandardScaler เนื่องจากข้อมูลในแต่ละ Features มีการแจกแจงปกติอยู่แล้ว
- ข้อมูลในเรื่อง Stroke Prediction มีความ imbalance เราจึงเลือกใช้ SMOTE (synthetic minority over-sampling technique)  จัดการกับข้อมูลในชุดนี้ก่อนจะนำไปใช้สร้าง Model จริง<br>
ในข้อมูล Train Set มีเพียง 195 instances เท่านั้นที่เป็น calss1 (หากผู้ป่วยเป็นโรคหลอดเลือดสมอง) แต่ในขณะที่ class0 (หากผู้ป่วยไม่เป็นโรคหลอดเลือดสมอง) มีถึง 3,892 instances<br>
จากเหตุผลด้านบนทำให้ Model ไม่สามารถหา Pattern ที่แน่นอนของ calss1 ได้ดีนัก ส่งผลให้ค่า Accuracy, Precision, Recall, F1 ของ calss1 น้อยตามไปด้วย
- Traditional Machine Learning (ML) ไม่จำเป็นต้องมีการ tuning hyperpsrsmeter มากนัก รวมถึงใช้เวลาและทรัพยากรในการเทรนค่อนข้างน้อยกว่า Multilayer Perceptron (MLP)


## 7. Conclusion

## 8. Recommendation

## _End Credit_
<table>
  <tr>
    <td>6410422005</td>
    <td>Metpiya Learakkakorn</td>
    <td>Prepare datase, Data cleaning, EDA, Traditional Machine Learning (ML), Multilayer Perceptron (MLP)</td>
  </tr>
  <tr>
    <td>6410422015</td>
    <td>Khodchapan Vitheethum</td>
    <td>Prepare datase, Data cleaning, EDA, Traditional Machine Learning (ML), Multilayer Perceptron (MLP)</td>
  </tr>
  <tr>
    <td>6410422017</td>
    <td>Peerat Pookpanich</td>
    <td>Prepare datase, Data cleaning, EDA, Traditional Machine Learning (ML), Multilayer Perceptron (MLP)</td>
  </tr>
  <tr>
    <td>6410422031</td>
    <td>Anyamanee Pornpanvattana</td>
    <td>Prepare datase, Data cleaning, EDA, Traditional Machine Learning (ML), Multilayer Perceptron (MLP)</td>
  </tr>
</table>
