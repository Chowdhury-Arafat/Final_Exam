library(RSQLite)
library(dplyr)
library(lubridate)

data = read.csv("Healthcare Dataset.csv") %>% mutate(
  Date.of.Admission = strptime(Date.of.Admission,format = "%m/%d/%Y") %>% as.Date(.,format = "%Y-%m-%d") %>% as.character(),
  Discharge.Date = strptime(Discharge.Date,format = "%m/%d/%Y") %>% as.Date(.,format = "%Y-%m-%d") %>% as.character()
)
head(data)
set.seed(31)

con = dbConnect(SQLite(),":memory:")
copy_to(con,data,name = "Data")

#1.	Selecting Patient Records
dbGetQuery(con,"SELECT * FROM Data WHERE \"Medical.Condition\" = 'Diabetes'")

#2.	Filtering by Date
dbGetQuery(con,"SELECT patient, \"Discharge.Date\" FROM Data WHERE date(\"Discharge.Date\") > date(\"2024-01-01\")")

#3.	Counting Cases by Condition
dbGetQuery(con,"SELECT COUNT(*) AS n_Hypertension FROM Data WHERE \"Medical.Condition\" = 'Hypertension'")

#4.	Grouping and Aggregation
dbGetQuery(con,"Select Hospital, COUNT(patient) AS n_Patients FROM Data GROUP BY Hospital")

#5.	Summing Financial Data
dbGetQuery(con,"Select Hospital, SUM(\"Billing.Amount\") AS Billing_Amount FROM Data GROUP BY Hospital")

#6.	Joining Tables
dbGetQuery(con, "
  SELECT Data.patient, Data.\"Admission.Type\", Data1.\"Date.of.Admission\"
  FROM Data
  LEFT JOIN 
  (SELECT patient, \"Date.of.Admission\" FROM Data) AS Data1
  ON Data.patient = Data1.patient
")

# 7.	Filtering with Multiple Conditions
dbGetQuery(con,"SELECT * FROM Data WHERE \"Medical.Condition\" = 'Pneumonia' AND Age > 65")

#8.	Sorting Results
dbGetQuery(con,"Select Hospital, COUNT(patient) AS n_Patients FROM Data 
           GROUP BY Hospital
             ORDER BY n_Patients DESC")

#9.	Using Aggregate Functions
dbGetQuery(con, "
  SELECT AVG(JULIANDAY(\"Discharge.Date\") - JULIANDAY(\"Date.of.Admission\")) AS avg_length_of_stay
  FROM Data
")

#10.	Identifying Missing Data
dbGetQuery(con,"SELECT patient FROM Data WHERE \"Blood.Type\" = 'A+'")

