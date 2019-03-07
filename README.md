# kaggle-泰坦尼克生还率预测
[项目飞机票](https://www.kaggle.com/c/titanic):airplane:

``` python
import pandas as pd
import numpy as np
import matplotlib as plot
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sn
sn.set_style('darkgrid')
from pyecharts import Bar
from pyecharts import Pie
from pyecharts import Boxplot
import statsmodels.api as sm
```
1.了解数据
![one](06a4a4f147936679a02db1d5eed1c9f.png)
`df.info()`
![one](/image//info.png)
`df.describe()`
![one](/image//describe.png)

### 对原始数据基础描述：
- 描述一
  - Age(年龄)存在小部分缺失值
  - Cabin(客舱)存在大量缺失值
  - Embarked(登船口)存在不明显的缺失值（这事可以先放放）
- 描述二
  - 生还率0.38
  - 平均年龄 29.70
  - 票价均数32.20，中位数14.45（偏位）
  - 最低票价为零（可以作为后期建模的一个特征）

以下分别对上面提到的字段进行下简单的查看
``` python
one_sur_male = len(df.query('Pclass == 1').query('Sex == "male"').query('Survived == 1'))/len(df.query('Pclass == 1').query('Sex == "male"'))
one_sur_male = round(one_sur_male,2)
one_sur_male
one_sur_female = len(df.query('Pclass == 1').query('Sex == "female"').query('Survived == 1'))/len(df.query('Pclass == 1').query('Sex == "female"'))
one_sur_female = round(one_sur_female,2)
one_sur_female
two_sur_male = len(df.query('Pclass == 2').query('Sex == "male"').query('Survived == 1'))/len(df.query('Pclass == 2').query('Sex == "male"'))
two_sur_male = round(two_sur_male,2)
two_sur_male
two_sur_female = len(df.query('Pclass == 2').query('Sex == "female"').query('Survived == 1'))/len(df.query('Pclass == 2').query('Sex == "female"'))
two_sur_female = round(two_sur_female,2)
two_sur_female
three_sur_male = len(df.query('Pclass == 3').query('Sex == "male"').query('Survived == 1'))/len(df.query('Pclass == 3').query('Sex == "male"'))
three_sur_male = round(three_sur_male,2)
three_sur_male
three_sur_female = len(df.query('Pclass == 3').query('Sex == "female"').query('Survived == 1'))/len(df.query('Pclass == 3').query('Sex == "female"'))
three_sur_female = round(three_sur_female,2)
three_sur_female
index_list = ['一等舱','二等舱','三等舱']
male_list = [one_sur_male,two_sur_male,three_sur_male]
female_list = [one_sur_female,two_sur_female,three_sur_female]
bar = Bar('各仓位男女生还率对比')
bar.add('女性',index_list,female_list,mark_line=['average'],mark_point=['max','min'])
bar.add('男性',index_list,male_list,mark_line=['average'],mark_point=['max','min'])
bar
```
![one](male_and_female.png)
<br>**就图像来看，这次的海难能充分的体现出Ladies first，男女生还率按舱级递减，大致get到了当时的情形，灾难面前人们保护的最珍贵的女人和孩子，部分有社会地位的人还是享受到了救援的"优先权"**<br>
<br>**看下各舱位年龄段的分布**<br>
```python
plt.figure(figsize=(12,7))
df.query('Pclass == 1').Age.plot(kind='hist',color='red',histtype='step',label='一等舱')
df.query('Pclass == 2').Age.plot(kind='hist',color='blue',histtype='step',label='二等舱')
df.query('Pclass == 3').Age.plot(kind='hist',color='black',histtype='step',label='三等舱')
plt.title('舱位等级年龄分布')
plt.xlabel('Age')
plt.ylabel('amount')
plt.legend()
plt.show()
```
![one](/image//cabin.png)

<br>**emm...，可以看到三等舱的旅客年龄更加的集中于15 ~ 37 区间段，二等舱大致与三等舱主体分布相同，不过稍稍向x轴右边移动了些貌似为 14 ~ 42，一等舱较为独特的是看起来没有较为突出的年龄区间，看到32 ~ 40部分稍稍凸起且两侧对称，估计平均年龄在36左右，这很符合我对《泰坦尼克号》的印象，年轻人不仅镜头多而且实际上也是主体，另外，随着舱位等级的增加，平均年龄也会随之增加，也很好的体现出经济能力在某种程度上受年龄的影响**:relaxed:<br>
<br>**那么年龄与生还率有没有关系呢？要知道现在初步试探的一切不仅是要我们对数据有个清晰的认识，同样也是方便我们挑出一些影响最终生还率的特征方便我们训练模型，废话不多说，继续上代码**<br>
```python
plt.figure(figsize=(14,8))
survived = df[df.Age.notnull()].query('Survived == 1').Age.plot(kind='hist',color='red',bins=70,label='幸存')
un_survived = df[df.Age.notnull()].query('Survived == 0').Age.plot(kind='hist',color='blue',alpha=0.5,bins=70,label='遇难')
plt.title('幸存与遇难对比图')
plt.xlabel('Age')
plt.ylabel('amount')
plt.legend()
plt.show()
```
![one](/image//survived.png)
<br>**貌似有迹可循哦！直观的来看中间大部分红色区域被蓝色区域给覆盖掉了，说明这部分死亡率是大于存活率的，不过0 ~ 15之间的红色区域异常的显眼，这部分为0~15岁的儿童，回溯之前的各舱位男女生还比率来讲，这种情形在意料之中。**<br>
<br>**看过了性别、年龄这两个特征，那么剩下可供我们初步查看的有`SibSp`(兄弟姐妹/配偶)、`Parch`(父母/孩子)、`Fare`(票价)、`Cabin`(房间号)、`Embarked`(登船口)，那么我继续做出各个特征与生还率之间的关系**<br>
<br>**这里呢我把`SibSp`、`Parch`相加到了一起生成一列新的特征`family`，下面我就以`family`为自变量，生还率为因变量来展示，但是为了更好的做出解释，我另加了一个不同的family人数与数量关系图**<br>
```python
df['family'] = df['SibSp'] + df['Parch']
kind = sorted(list(df.family.value_counts().index))
proportion_list = []
for i in kind:
    proportion = len(df.query('family == {}'.format(i)).query('Survived == 1')) / len(df.query('family == {}'.format(i)))
    proportion_list.append(proportion)
plt.figure(figsize=(12,7))
sn.barplot(x=kind,y=proportion_list)
plt.title('family与生还率')
plt.xlabel('family')
plt.ylabel('生还率');
```
![one](/image//family_rate.png)
<br>**横轴是family的所有个数类型，这里我没有把乘客自身加到里面，那么从得出的图像来讲family为7、10的生还率均为零，family等于3的生还率最高，比较有趣的是当family分别为4、5、7、8的时候我以为从4之后随着family人数增加生还率也会随之降低，但是family等于6的时候生还率出现的明显的增高。这是为什么呢？各位来看下图**<br>
<br>照例先上代码<br>
```python
family_type = df.family.value_counts().index
family_amount = df.family.value_counts().values
plt.figure(figsize=(12,7))
sn.barplot(x=family_type,y=family_amount)
plt.title('family与数量')
plt.xlabel('family')
plt.ylabel('数量');
```
![one](/image//family_amount.png)
<br>**该图展示了各个种类family的数量，可以看到当family等于0占据了相当大的一部分，family 3 ~ 10在总体的比重相当的低，所以目标群体越小越不能排除偶然性，既然这样，我就把family等于6的乘客单独拿出来看看**<br>
<br>
`df.query('family == 6').query('Survived == 1')`
<br>
![one](/image//family_6.png)
<br>**短短的四条数据蕴含着很多我们之前没有挖掘的信息，首先`Name`特征下，三个那个...英国人的命名我不太清楚，所以我去简单查了下，英语姓名的一般结构为：教名+自取名+姓，具体就不继续考究了，总之三个教名`Asplund`一模一样，再来看看`SibSp`(兄弟姐妹/配偶)、`Parch`(父母/孩子)两个特征，数量上面让我又觉得好像有某种联系，再来看下`Fare`(票价)，给我强烈的感觉认为她们是一家人，并且可能买的是团体票，关于票价的分布状态后面我会放出来。另外她们三个的票号也是一样的`Ticket`！**<br>
<br>**那么现在明了了，这些是一个特殊的群体，她们面临着不利因素，家庭群体大，由之前的各类family生还率能推测，结果的生还率没有family为0、1、2、3的大，舱位等级三等舱，但同时她们有掌握着有利因素，`Sex`特征下female为主体、`Age`特征下有着3岁和5岁的baby，最终的结果影响了family为6时的生还率#family为6的乘客一共12人**<br>
<br>**接着是`Fare`(票价)一个我认为有作用的但实际在分类器中却没起到什么作用的特征，不过图片还是要放出来滴，各位见仁见智啊**<br>
```python
df_first = df.query('Pclass == 1')
plt.figure(figsize=(15,8))
df_first.query('Survived == 1').query('200 > Fare ').Fare.plot(kind='hist',color='red',bins=150,label='幸存')
df_first.query('Survived == 0').query('200 > Fare ').Fare.plot(kind='hist',color='blue',alpha=0.5,bins=150,label='遇难')
plt.title('票价与生存情况(一等舱)')
plt.xlabel('Fare(票价)')
plt.legend()
plt.ylabel('人数');
```
![one](/image//fare_1.png)
```python
df_two = df.query('Pclass == 2')
plt.figure(figsize=(15,8))
df_two.query('Survived == 1').query('200 > Fare ').Fare.plot(kind='hist',color='red',bins=150,label='幸存')
df_two.query('Survived == 0').query('200 > Fare ').Fare.plot(kind='hist',color='blue',alpha=0.5,bins=150,label='遇难')
plt.title('票价与生存情况(二等舱)')
plt.xlabel('Fare(票价)')
plt.legend()
plt.ylabel('人数');
```
![one](/image//fare_2.png)
```python
df_three = df.query('Pclass == 3')
plt.figure(figsize=(15,8))
df_three.query('Survived == 1').query('200 > Fare ').Fare.plot(kind='hist',color='red',bins=150,label='幸存')
df_three.query('Survived == 0').query('200 > Fare ').Fare.plot(kind='hist',color='blue',alpha=0.5,bins=150,label='遇难')
plt.title('票价与生存情况(三等舱)')
plt.xlabel('Fare(票价)')
plt.legend()
plt.ylabel('人数');
```
![one](/image//fare_3.png)








## 二更。。。<br>
虽然之前提到对头等舱男性的猜测得到了一个自我满足式的验证，可这不代表正确，有时候数据真的很会说谎，那么怎么验证更有说服力呢？恐怕要就数据和我们研究的具体内容下定论了，刚刚的头等舱男士年龄事实上是否真的集中在中年，那么我们可以出门左拐去看看历史文献或记录片吧啦吧啦。。。<br><br>
废话不多说，各位看官咱接着往下走:smiley_cat:<br><br>
虽说啊，这个年龄段分布大体上是看过了，可是对于年龄上的探索我可还不满足，谁叫它缺失值太多了，叫人寝食难安夜不能寐，这么大比例的缺失值丢掉很可能直接影响结果的准确性，所以得想方设法找个由头给它们填充上，**so** 我决定接着和大家絮叨几张图<br>
这个是生还者年龄段分布，最起码也要看看哪个年龄段获救占总获救的比重更大不是<br>
`df.query('Survived == 1').Age.plot(kind='hist',title='生还者年龄分布',figsize=(8,5));`
`sn.set_style(style='whitegrid')`<br>
![picture](survived_age.png)<br>
这样两块明显区域貌似清晰了多，0 ~ 10，18 ~ 40.虽说事件臭名昭著，但从结果数据来看，妇女儿童优先不是说说而已了（**妇女优先**请看开始的**各舱位男女生还率对比**）
最后是各舱位年龄分布，头等舱的图在最开始的位置，这里只放二等舱与三等舱的。<br>
![picture](pclass_2.png)
![picture](pclass_3.png)<br>
看样子舱位等级越高平均年龄也越大，所以嘛，每个大佬也都是从年轻屌丝过来滴，稳扎稳打与君共勉。<br>
到这为止，年龄上的探索告一段落吧，不过在这里可以告诉大家，在这次的年龄缺失值的填充上并没有使用到上述内容所带来的信息。**求都妈得！各位看官先别激动**:fire:，上述的信息可谓是对我第二次提交预测集起到了莫大的作用，这个暂时还急不得，容在下卖个关子小抖个机灵.<br>
那~老少爷们们,咱们在往下遛遛？<br>
## 前面絮叨了太多，下面开始正题。<br>
目前的数据集长这样，仅仅是拟合了下年龄对缺失的年龄进行的填充<br>
![picture](3-1.png)<br>
下面我需要对其它的特征采取点措施了，这么多的分类变量最先让我想到的就是因子化了。<br>
通过把类别返回成`0`或`1`，放入我们之后的分类器中。其中`1`表`是`,`0`表`否`。<br>
让我们再回到数据中看看，能够进行特征因子化的变量有`Pclass`,`Sex`,`Cabin`,`Embarked`。<br>
```python
df[['first','second','third']] = pd.get_dummies(df.Pclass)
df[['female','male']] = pd.get_dummies(df.Sex)
df[['no','yes']] = pd.get_dummies(df.Cabin)
df[['is_c','is_q','is_s']] = pd.get_dummies(df.Embarked)
```
查看下结果<br>

![picture](3-2.png)<br>
哼哼~很不错，所有给出的的分类变量都让我们因子化了。下面再考虑要做点什么呢？<br>
我的目光再次盯到`Age`上，直觉告诉我我们之间还会有故事。<br>
那就是标准化啦，我们所挑选准备放到分类器中的特征都是我们刚刚因子化得出的`0`或`1`。<br>
倘若我把`Age`一并放入会怎样？先看下我们之前所做的因子化的变量，所有经过因子化的变量的方差都在`0~1`之间，我们看下`Age`这一列的值。<br>
![picture](3-4.png)<br>
普遍都为两位数，如果加入了未处理的`Age`会怎样呢？该变量的方差为170，我们了解回归拟合原理是用到了最小二乘法的，可想而知单个变量170的方差会对一堆`0~1`方差变量的拟合造成怎样的影响！<br>
