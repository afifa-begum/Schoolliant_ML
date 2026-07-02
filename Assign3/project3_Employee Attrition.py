import pandas as pd

df = pd.read_csv("employee_attrition.csv")

print("\nfirst 10 records")
print(df.head(10))

print("\naverage salary")
print(df["Salary"].mean())
print("\naverage age")
print(df["Age"].mean())
print("\naverage experience")
print(df["Experience"].mean())
print("\n")

print("\nemployees in each dept:")
print(df["Department"].value_counts())

attrition = df[df["Attrition"] == "Yes"]
highest_attrition = attrition["Department"].value_counts()
print("\nDepartment with Highest Attrition:")
print(highest_attrition)

print("\nEmployees with Salary > 50000:")
print(df[df["Salary"] > 50000])

print("\nEmployees with Experience > 5 years:")
print(df[df["Experience"] > 5])

sorted_df = df.sort_values(by = "Salary", ascending = False)

print("\nemp sorted by salary:")
print(sorted_df)

grouped=df.groupby("Department")[["Salary", "Performance"]].mean()
print("\naverage salary and performance by department")
print(grouped)

def risk_level(row):
    if row["Attrition"]=="Yes":
        return "High Risk"
    elif row ["Performance"]<60:
        return "Medium Risk"
    else:
        return "Low Risk"
    
df["Risk_Level"] = df.apply(risk_level, axis=1)
print("\ndataset with risk_level:") 
print(df.head())   

df.to_csv("employee_attrition_cleaned.csv", index=False)
print("\ncleaned dataset")
print("File Name: employee_attrition_cleaned.csv")