import sys
import json
import pandas as pd
import os

input_json_path = "../frontend/data/class_summary.json"
students_json_path = "../frontend/data/students.json"
output_excel_path = "../frontend/public/reports/class_summary.xlsx"

with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def extract_roll_number(file_name):
    return os.path.splitext(file_name)[0]

for result in data["individual_results"]:
    result["roll_number"] = extract_roll_number(result["file_name"])

students_data = {}
for result in data["individual_results"]:
    roll_number = result["roll_number"]
    if roll_number not in students_data:
        students_data[roll_number] = {
            "Roll Number": roll_number,
            "Total Marks": 0,
            "Tips": set(),
            "Questions": {}
        }
    students_data[roll_number]["Total Marks"] += result["marks_awarded"]
    students_data[roll_number]["Tips"].add(result["weak_topics"])
    students_data[roll_number]["Questions"][result["question_number"]] = result["marks_awarded"]

excel_data = []
for roll_number, student in students_data.items():
    row = {
        "Roll Number": roll_number,
        "Total Marks": student["Total Marks"],
        "Tips": ", ".join(student["Tips"])
    }
    for q_num, marks in student["Questions"].items():
        row[f"Q{q_num}"] = marks
    excel_data.append(row)

df = pd.DataFrame(excel_data)
df.to_excel(output_excel_path, index=False)

def update_students_json(roll_number, new_weak_topics, students_json_path):
    if os.path.exists(students_json_path):
        with open(students_json_path, "r", encoding="utf-8") as f:
            students_data = json.load(f)
    else:
        students_data = []

    for student in students_data:
        if student["rollNumber"] == roll_number:
            if student["weakTopics"] == "-" or not student["weakTopics"]:
                student["weakTopics"] = new_weak_topics
            else:
                student["weakTopics"] += f", {new_weak_topics}"
            break

    with open(students_json_path, "w", encoding="utf-8") as f:
        json.dump(students_data, f, indent=4, ensure_ascii=False)

for result in data["individual_results"]:
    update_students_json(result["roll_number"], result["weak_topics"], students_json_path)
