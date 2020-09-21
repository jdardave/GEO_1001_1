# GEO_1001_1
Assignment 1-Ioannis Dardavesis

This is a README file to describe how to run geo1001_hw01

That file has all the code inside and there are no additional python files to run.
By running the python file, all the wanted results appear. There are 18 figures
in this code, which appear all together, when the program has finished running.

In the beginning of the file, some basic libraries were imported, in order to
complete the wanted tasks.

The data files were converted to .csv and where imported in python with encoding
UTF-8. As a result a dataframe was created for every data file (df_A,df_B etc.).
The first 5 rows of each file were skipped, because they did not include numerical values.

#Section A1

Mean, variance annd standard deviation were calculated for each sensor/each dataframe. The
results are saved in an excel file (Sensors_Stats.csv). To help creating the wanted figures,
a column from each dataframe was picked, so that temperature,wind speed and wind direction
values can be chosen. From each dataframe, 5th column represents temperature values,(df_A[4]).
Number 4 in brackets shows, that the 5th column of the dataframe can be used, because python starts
counting the columns from 0. The other variables were picked the same way. Now that the needed
variables are calculated, the figures can be constructed.

#Section A2

For this section, no new variables were created and the figures (PMF,PDF,CDF,KDE) can be plotted based on the
existing information. Το calculate PMF a function was created, that takes one varriable (sample). It returns p
which is the result of the number of the sample value divided to the sample's length. Using this function for 
every type of data leads to the creation of the PMF figures.

#Section A3

Crosswind and WBGT data was gained the same way as described in section A1 for the temperature data.
Pearson and Spearman coefficients were calculated for each combination of sensors (10 combinations). 
Interpolations had to be made for variables with different length. The coefficient results are exported
in a csv file named "correlation.xslx". In order to plot the figures, 2 lists were created. One including all 
the Pearson and another with all the Spearman coefficients. 3 scatter plots were created, one for
each variable. Their x axis includes the sensor relationships (ex. AB,AC etc.) and y axis the values of their
correlation.

#Section A4

In order to calculate the confidence intervals for temperatures and wind speed a function named mean_confidence_interval
was created. It takes two variables, the data and the confidence level which is 0.95. The function uses the scipy.stats
library to calculate the 3 intervals. Then, to calculate the intervals for each variable, the function is called for each
data and for confidence level 0.95. The results are saved in a csv file named "confidence_intervals.csv". Regarding the t
testing and p values, the statistics library was used for each variable. The results are saved in a csv file named "T_test.csv".

#Bonus question

A function named average_temperature was created that takes data as its only variable. The function which is thorougly explained
in the essay, calculates the maximum and minimum temperatures in order to find which day is the hottest and coolest respectively,
during the days that the data was acquired.

