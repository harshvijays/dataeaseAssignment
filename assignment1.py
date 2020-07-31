from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import sum 
from pyspark.sql import SQLContext
import numpy as np
from scipy import stats

conf=SparkConf()
sc=SparkContext(conf=conf)
sqlContext = SQLContext(sc)


dfcsv=sqlContext.read.format('csv').options(header='true').load("nonConfidential.csv")
dfparq=sqlContext.read.format('parquet').options(header='true').load("confidential.snappy.parquet")
mergedfile=dfcsv.union(dfparq)


#print("##########################################################printing answer 1 : ")
virginiadf= mergedfile.filter((mergedfile.State == "Virginia") | (mergedfile.State == "VA"))
virginiadf_group=virginiadf.groupBy("LEEDSystemVersionDisplayName").count()

ans1count=virginiadf_group.count()
ans1=sqlContext.createDataFrame([{"No of Leed Projects in Virginia":ans1count}]) 
ans1.coalesce(1).write.option("header","true").mode("overwrite").csv("/tmp/output/output_1")


#print("#################################################################printing anwser 2 : ")
ownertype=virginiadf.groupBy("OwnerTypes").count()
ans2count=ownertype.count()
ans2=sqlContext.createDataFrame([{"No of Leed Projects by owner type":ans2count}]) 
ans2.coalesce(1).write.option("header","true").mode("overwrite").csv("/tmp/output/output_2")

#print("#################################################################printing answer 3 : ")
certified_proj=virginiadf.filter(virginiadf.IsCertified == "Yes")
ans3=certified_proj.select(sum("GrossSqFoot") )
ans3.coalesce(1).write.option("header","true").mode("overwrite").csv("/tmp/output/output_3")



#print("######################################################################printing answer 4 : ")
zipcode_groupby=virginiadf.groupBy("Zipcode").count()
zipcodesort=zipcode_groupby.orderBy("count",ascending=False)
ans4=zipcodesort.limit(2)
ans4.coalesce(1).write.option("header","true").mode("overwrite").csv("/tmp/output/output_4")



#print("######################################################################printing answer 5 : ")
virginia_leed_nc=virginiadf.filter(virginiadf.LEEDSystemVersionDisplayName=="LEED-NC 2.2")
list_virginia=virginia_leed_nc.select("PointsAchieved")

californiadf= mergedfile.filter((mergedfile.State == "California") | (mergedfile.State == "CA"))
california_leed_nc=californiadf.filter(californiadf.LEEDSystemVersionDisplayName=="LEED-NC 2.2")
ca=np.array(california_leed_nc.select("PointsAchieved").collect())
v=np.array(virginia_leed_nc.select("PointsAchieved").collect())

ca=ca[ca !=None]
ca=ca[ca !=0]
v=v[v !=None]
v=v[v!=0]
ca=ca.astype(np.int)
v=v.astype(np.int)

if np.var(v) == np.var(ca):
	equal_var=True
else:
	equal_var=False
t,p=stats.ttest_ind(ca,v)
if float(p)>0.05:
	ans5=sqlContext.createDataFrame([{"t value":float(t),"p value":float(p),"Significant Difference ":"NO"}]) 
	
else :
	ans5=sqlContext.createDataFrame([{"t value":float(t),"p value":float(p),"Significant Difference ":"YES"}]) 
ans5.coalesce(1).write.option("header","true").mode("overwrite").csv("/tmp/output/output_5")	


