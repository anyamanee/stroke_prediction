##  <font color="#8450B2">_3PM (Group2)_</font> <br>
Deep Learning เป็นการเรียนรู้เชิงลึกที่เลียนแบบการทำงานของโครงข่ายประสาทของมนุษย์ โดยมีการทำงานซ้อนกันหลายๆ ชั้นหรือเรียกว่า Layer จากข้อมูลตัวอย่างเพื่อหา Pattern ของข้อมูล 
โดยการศึกษาในครั้งนี้มีจุดมุ่งหมายเพื่อเปรียบเทียบประสิทธิภาพของ **`Traditional Machine Learning (ML)`** และ **`Multilayer Perceptron (MLP)`** รวมถึงการปรับ Hyperparameter เพื่อความสมบูรณ์ของโมเดล ทางทีมคาดหวังว่าจะได้ประสบการณ์ในเรื่องนี้ และเป็นประโยชน์ต่อผู้ที่กำลังสนใจในเรียนนี้ 

## _Key Highlight_
- ผลจากการเทรนโมเดลพบว่า RandomForestClassifier ซึ่งเป็น traditional ML ให้ค่า Accuracy สูงที่สุดอยู่ที่ 0.882583 ในขณะที่ MLP ให้ค่า Accuracy สูงที่สุดอยู่ที่ 0.XXX เท่านั้น
- ReLU M
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
- จากการสำรวจ dataset พบว่าปัจจัยที่ส่งผลต่อการเป็นโรคหลอดเลือดสมองโดยดูจากค่า correlation มากที่สุด 5 อันดับแรก ได้แก่
  1. อายุ โดยมีค่า correlation ที่ 0.245281
  2. การเป็นโรคหัวใจ โดยมีค่า correlation ที่ 0.134905
  3. ระดับน้ำตาลในเลือด โดยมีค่า correlation ที่ 0.131991
  4. การเป็นโรคความดันโลหิตสูง โดยมีค่า correlation ที่ 0.127891
  5. สถานะการแต่งงาน โดยมีค่า correlation ที่ 0.108299
- โดยพบว่ามีผู้ที่ป่วยด้วยโรคหลอดเลือดสมองมักเป็นผู้สูงอายุ โดยจะเริ่มพบผู้ป่วยที่เป็นโรคนี้มากขึ้นตั้งแต่ช่วงอายุ 40 ปีขึ้นไป และการกระจายตัวของผู้ป่วยที่เป็นโรคหลอดเลือดสมอง อยู่ในช่วงอายุ มากกว่า 60 ปีขึ้นไป
- จำนวนคนที่เป็นโรคหัวใจคิดเป็น 5.4% (498 คน) จากจำนวนคนทั้งหมด โดย 0.9% (66 คน) เป็นคนที่มีอาการโรคหลอดเลือดสมองร่วมกับอาการโรคหัวใจ
- ปริมาณนำตาลในเลือดขอคนที่เป็นและไม่เป็นโรคหลอดเลือดสมองมีปริมาณใกล้เคียงกัน อย่างไรก็ตาม ในคนที่คนที่เป็นหลอดเือดสมองมีการกระจายตัวของระดับน้ำตาลในเลือดมากกว่าคนที่ไม่เป็น โดยมีค่าเฉลี่ยอยู่ในช่วงประมาณ 55 ถึง 130 ซึ่งเป็นการกระจายตัวจะมีลักษณะเป็น right skewed
- จำนวนคนที่เป็นโรคความดันโลหิตสูงคิดเป็น 9.7% (276 คน) จากจำนวนคนทั้งหมด โดย 1.3% (47 คน) เป็นคนที่มีอาการโรคหลอดเลือดสมองร่วมกับอาการความดันโลหิตสูง
- จำนวนคนที่แต่งงานแล้วคิดเป็น 65.6% (3,353 คน) จากจำนวนคนทั้งหมด โดย 4.3% (220 คน) เป็นคนที่มีอาการโรคหลอดเลือดสมอง

### Data preparation
การเตรียมข้อมูลก่อน train model เราทำการ drop ค่า outliner ออก หลังจากนั้นจึงจัดการข้อมูล Binary category และ Multicategory โดยใช้ **`One-Hot encoding`** เพื่อเปลี่ยนข้อมูลที่เก็บในลักษณะ categorical ให้อยู่ในรูป Binary values เนื่องจากการทำ Machine leaning นั้น ต้องการข้อมูลในรูปแบบตัวเลขเพื่อใช้ในการ train และ predict โดยแปลงค่าในคอลัมน์ gender, ever_married, work_type, residence_type และ smoking_status เพื่อให้อยู่ในรูปแบบดังกล่าว <br>

เนื่องจากข้อมูลของเรามีความ imbalance เราจึงเลือกใช้ **`SMOTE`** (synthetic minority over-sampling technique) ซึ่งเป็นเทคนิคที่ใช้ในการแก้ปัญหาการจำแนกข้อมูลที่ไม่สมดุลและทำการ normalize ค่าด้วย StandardScaler <br>

และจากข้อมูลทั้งหมด 5,110 มีการแบ่ง Data splitting (train/val/test) ดังนี้<br>
ML: Train 80% และ Test 20%<br>
MLP: Train 80%, validation 20% ของ Train set และ Test 20%<br>


![image](https://user-images.githubusercontent.com/101736826/187707794-38780d34-8cc0-4fd0-95de-48e3eda8c46f.png)

## 3. Network architecture
Total params: 7,354<br>
Trainable params: 7,354<br>
Non-trainable params: 0<br>
|Layer (type)|Output Shape|Number of Parameter|Activation function|
|------------|------------|-------------------|-------------------|
|hidden1 (Dense)|	(None, 89)|	801|	tanh|
|hidden2 (Dense)|	(None, 72)|	6,408|	tanh|
|dropout_1 (Dropout)| (None, 72)|	0|	-|
|output (Dense)|	(None, 1)|	73|	sigmoid|

## 4. Training
### 4.1 Traditional Machine Learning (ML)
เราใช้ Scikit-learn ซึ่งเป็น library ใน Python ในการเทรนโมเดลแบบ Traditional Machine Learning ซึ่งประกอบไปด้วย **`RidgeClassifier`**, **`LinearSVC`**, **`SVC`**, **`LogisticRegression`**, **`KNeighborsClassifier`** , **`Xgboost`** และ **`RandomForestClassifier`** <br>
จากนั้นเราใช้ **`K-Fold Cross Validation`** จำนวน 5 รอบในแต่ละโมเดลเพื่อหาค่าเฉลี่ยของ accuracy และเลือกโมเดลที่เหมาะสมกับชุดข้อมูล และนำโมเดลที่ได้ไปปรับหาหาค่า **`Hyperparameter`** โดยใช้ **`GridSearchCV`** เพื่อหาค่าที่เหมาะสมกับโมเดลนั้นๆ <br>
โดยในแต่ละ model มีการ tuning ดังนี้

- RidgeClassifier     : RidgeClassifier(alpha = 0.0001, solver = 'lsqr')
- LinearSVC           : LinearSVC(C= 0.001,multi_class = 'crammer_singer',penalty = 'l2',loss='hinge')
- SVC                 : SVC(C= 1.5, gamma = 'scale', kernel = 'rbf', random_state = 0)
- LogisticRegression  : LogisticRegression(C= 1, max_iter= 100,solver = 'sag', penalty = 'l2')
- KNeighborsClassifier: KNeighborsClassifier(n_neighbors=5, leaf_size = 10, weights='distance')
- Xgboost             : XGBClassifier(eval_metric= 'error', learning_rate=0.1)
- RandomForestClassifier: RandomForestClassifier(criterion= 'gini', random_state = 0, n_estimators= 100)

### 4.2 Multilayer Perceptron (MLP)
ในการเทรนโมเดล Multilayer Perceptron (MLP) เราเลือกใช้ The Keras ecosystem (KerasTuner) โดยจะ trial-and-error ในการปรับหา Hyperparamet เพื่อหาโมเดลที่ดีที่สุดเพื่อพยากรณ์การเป็นโรคหลอดเลือดสมองสำหรับ dataset ข้างต้น<br>
โดยรายละเอียดของข้อมูลในการ trial-and-error มีดังนี้

|Hyperparameter	|List of value	|Best value for dataset|
|---------------|---------------|----------------------|
|Number of layer|[1,2,3]       	|3|
|Number of node	|[21, 22, ..., 100]	|89, 72|
|Learning rate	|[0.01, 0.001, 0.0001, 0.00001]	|ExponentialDecay <br>(itial_learning_rate=1e-2, <br> decay_steps=10000, <br> decay_rate=0.9)|
|Activation	    |[relu, tanh, sigmoid]	|tanh, sigmoid|
|Optimizer      |Adam          | Adam|


## 5. Results
โดยผลลัพธ์ในตารางข้อ 5.1 และ 5.2 เป็นผลลัพธ์ที่ได้จาก test set ทั้ง 2 ตาราง
### 5.1 Traditional Machine Learning (ML)

|Classification algorithm|	Accuracy|	Precision|	Recall	|F1|
|------------------------|----------|----------|----------|--|
|RandomForest	|0.877691|	0.222222|	0.126316|	0.161074|
|KNN|	0.826810	|0.407407|	0.131737|	0.199095|
|XGBClassifier	|0.763209|	0.574074|	0.124000|	0.203947|
|LogisticRegression	|0.758317|	0.703704|	0.141264|	0.235294|
|SVC|	0.752446	|0.722222|	0.140794|	0.235650|
|RidgeClassifier	|0.737769|	0.722222|	0.133562|	0.225434|
|LinearSVC	|0.723092|	0.777778|	0.134185|	0.228883|


### 5.2 Multilayer Perceptron (MLP)
**`Mean±SD of Accuracy = ( , )`**<br>
โดยค่าเฉลี่ยของ Accuracy ด้านบนนั้น มากจากการคำนวณค่าเฉลี่ยของ Accuracy ในการเทรนโมเดลลด้วย initial random weights ที่แตกต่างกัน 5 รอบ

|Round|	Accuracy|	Precision|	Recall|	F1|
|-----|---------|----------|--------|---|
|1    |	0	      |   0      |	0     |	0 |
|2	  |0        |	0        |	0     |	0 |
|3	  |0        |	0        |	0     |	0 |
|4	  |0	      |0         |	0	    |0  |
|5	  |0        |	0	       |0	      |0  |

MEAN: XXX
SD: XXX

โดย ทำการเทรนโดยใช้ GPU 1 ตัว, GPU No. 0: Name = /physical_device:GPU:0 <br>
เวลาที่ใช้ในการเทรนโดยประมาณ XX.XX นาที
และบน CPU ใช้เวลาในการเทรนโดยประมาณ 6m 54.2s

#### กราฟการแสดงผลการเทรนโมเดลด้วย data train set vs. data test set
<PIC>
จากกราฟสังเกตได้ว่า ค่า accuracy ของ validation มีการขยับดีขึ้นเมื่อเทียบกับ train เมื่อมีการเทรนโมเดล XXX epoch ในขณะเดียวกัน loss ของ validation ลดลงมาอยู่ที่ประมาณ XXX แสดงให้เห็นว่าโมเดล goodfit

#### Evaluation Metric
  <PIC>


## 6. Experiment result and discussion
- สำหรับการ train model หนึ่งในสิ่งสำคัญคือการเลือกใช้ฟีเจอร์เพื่อไม่ให้ model มีความ overfit มากเกินไป ดังนั้น เราจึงเริ่มจากการดูค่า correlation ของตัวแปรต่างๆ ต่อการเป็นโรคหลอดเลือดสมอง (stroke) ซึ่งหาก correlation มีค่ามาก หมายถึงมีความสัมพันธ์ต่อการเป็น stroke มาก เช่น อายุ การเป็นโรคหัวใจ เป็นต้น

- การ normalization เราใช้ StandardScaler เนื่องจากข้อมูลในแต่ละ Features มีการแจกแจงปกติอยู่แล้ว

- ข้อมูลในเรื่อง Stroke Prediction มีความ imbalance เราจึงเลือกใช้ SMOTE (synthetic minority over-sampling technique)  จัดการกับข้อมูลในชุดนี้ก่อนจะนำไปใช้สร้าง Model จริง<br>
ในข้อมูล Train Set มีเพียง 195 instances เท่านั้นที่เป็น calss1 (หากผู้ป่วยเป็นโรคหลอดเลือดสมอง) แต่ในขณะที่ class0 (หากผู้ป่วยไม่เป็นโรคหลอดเลือดสมอง) มีถึง 3,892 instances<br>
จากเหตุผลด้านบนทำให้ Model ไม่สามารถหา Pattern ที่แน่นอนของ calss1 ได้ดีนัก ส่งผลให้ค่า Accuracy, Precision, Recall, F1 ของ calss1 น้อยตามไปด้วย

- Traditional Machine Learning (ML) ไม่จำเป็นต้องมีการ tuning hyperpsrsmeter มากนัก รวมถึงใช้เวลาและทรัพยากรในการเทรนค่อนข้างน้อยกว่า Multilayer Perceptron (MLP) 


## 7. Conclusion & Discussion
- การเทรนโมเดล MLP จำเป็นต้องทำการ tuning hyperparameter และใช้ทรัพยากรในการเทรนโมเดลมาก อย่างไรก็ตาม เนื่องจาก dataset นี้มีข้อมูลที่ไม่มากนัก ดังนั้น การใช้ traditional ML ซึ่งสามารถ tuning hyperparameter และใช้ทรัพยากรน้อยกว่าจึงเหมาะกับ dataset ชุดนี้มากกว่า
- ผลจากการเทรนโมเดลพบว่า RandomForestClassifier ซึ่งเป็น traditional ML ให้ค่า Accuracy สูงที่สุดอยู่ที่ 0.882583 ในขณะที่ MLP ให้ค่า Accuracy สูงที่สุดอยู่ที่ 0.XXX เท่านั้น
- สิ่งที่ไม่ตรงตามสมมติฐาน
    
|Performance<br> Measures|	MLP|	RandomForest | KNN | XGBClassifier | LogisticRegression |
|------------------------|-----|---------------|-----|---------------|--------------------|
|1. Accuracy            |0| | | | |
|2. Recall              |0| | | | |
|3. Precision           |0| | | | |
|4. F1 Score            | | | | | |

## 8. Recommendation
- Dataset ที่เหมาะกับการ predict ค่าด้วย MLP ควรเป็น Dataset ที่มีขนาดใหญ่ (เท่าไหร่คือใหญ่?) ทั้งนี้เพื่อให้มีข้อมูลจำนวนมากพอในการเทรนและเรียนรู้เพื่อนำไปพยากรณ์ผลลัพท์ที่แม่นยำมากยิ่งขึ้น (???)
- สำหรับ Dataset ขนาดเล็ก การใช้ traditional ML ในการพยากรณ์ข้อมูลนั้นเหมาะสมกว่าการใช้ MLP เพราะใช้ทรัพยากรในการ train model น้อยกว่า
- การปรับจำนวน layer ของ MLP ควรมีค่าไม่เกินจำนวน hyperparameter ที่เราจะทำการปรับ

## 9. References
- เทคนิคการเขียนโค้ดด้วยวิธี automl ของ The Keras ecosystem: <br>
    **Authors:** Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin <br>
    **Last modified:** 2021/10/27 <br>
    **Link:** https://keras.io/guides/keras_tuner/getting_started/ <br>
    
- เทคนิคการปรับโมเดล: <br>
    **Authors:** SIDDHESH SAWANT <br>
    **Last modified:** 2021/03/26 <br>
    **Link:** https://www.kaggle.com/code/siddheshera/stroke-eda-smote-9-models-90-accuracy/comments <br>

    **Authors:** OHM-Songpol, nuijth, robinoud, MartRideratGamaGama <br>
    **Last modified:** 2022/02/07 <br>
    **Link:** https://github.com/robinoud/BADS7604_HW3_Deep-Learning
 
 


## _End Credit_

<table>
  <tr>
    <td>25%</td>
    <td>6410422005</td>
    <td>Metpiya Learakkakorn</td>
    <td>Prepare datase, Data cleaning, EDA, ML, MLP, Report, Conclusion</td>
  </tr>
  <tr>
    <td>25%</td>
    <td>6410422015</td>
    <td>Khodchapan Vitheethum</td>
    <td>Prepare datase, Data cleaning, EDA, ML, MLP, Report, Conclusion</td>
  </tr>
  <tr>
    <td>25%</td>
    <td>6410422017</td>
    <td>Peerat Pookpanich</td>
    <td>Prepare datase, Data cleaning, EDA, ML, MLP, Report, Conclusion</td>
  </tr>
  <tr>
    <td>25%</td>
    <td>6410422031</td>
    <td>Anyamanee Pornpanvattana</td>
    <td>Prepare datase, Data cleaning, EDA, ML, MLP, Report, Conclusion </td>
  </tr>
</table>

