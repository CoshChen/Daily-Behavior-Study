# Daily-Behavior-Study
# Data Preprocessing
There are three raw SPSS datasets: *data_1_Baseline* (baseline information of study 1), *data_1_Daily* (11-day daily behavior of study 1), *data_2_Final* (baseline information and 11-day daily behavior of study 2). The goal the data step is to combine datasets of study 1 and study 2 to form a larger dataset that is ready for the model fitting.  

## Part 1: In SPSS
* **Name Change**

  Find the intersection of measurements in study 1 and study 2. Align variable names in study 1 along with study 2. Eg: DASS_1 -> DASS1, DNCIFVB_2 -> DHQ5.
  
  Obtain: *data_1_Baseline_nameChange*, *data_1_Daily_nameChange*
  
* **Clean** (for *data_1_Baseline_nameChange* and *data_1_Daily_nameChange*)

  - Correct email typo such as "name@email,address,com", "name2email.address.com", "name@email.address.con" -> "name@email.address.com"
    The email will be used as participant identification.
  - Calculate the day (vale=1-11) of each record from the timestamp. Note if a participant filled up the survey after 12:00am everyday, the day range will be calculated as 2-12. In this case, the values are adjusted to 1-11.
  - Change labels of Age, Gender, and Ethnicity so that the mappings are the same as that in study 2.
  
* **Subset**

  Remove variables in study 1 and study 2 that are not in the intersection.
  
  Obtain: *data_1_Baseline_cleaned_subset*, *data_1_Daily_cleaned_subset*, *data_2_Final_subset*
  
* **Merge**
  - Merge (Add Variables) *data_1_Baseline_cleaned_subset* and *data_1_Daily_cleaned_subset* by email address to obtain *data_1_Base_daily_merged*. Then create 'ID' column and delete 'Email' column in this merged table.
  - Merge (Add Cases) *data_2_Final_subset* and *data_1_Base_Daily_merged*.
  - Replace all system missing values by 999.
  
  Obtain: *Final_study_1_study_2_Merged*
  
## Part 2: Python Scripts
