import sqlite3
import json
from datetime import datetime

import pandas as pd

import requests
from tqdm import tqdm

import pdb


def get_token():
    base_url = "https://dora.dentall.dev/totoro8083/api/"
    user_name = "totorouser"
    pwd = "37GqK1M2QeZ8b4L3"

    url = base_url + "authenticate"
    response = requests.post(url, json={"username": user_name, "password": pwd})

    return response.json()["id_token"]


def get_patient_data(patient_id, base_url, token):
    url = base_url + "crm/" + patient_id
    response = requests.get(url, headers={"Authorization": "Bearer " + token})
    return response.json()


def server_data_to_db(DB_PATH="./data/clinic_data.db", n_patient=20000):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    token = get_token()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            gender TEXT,
            birthday TEXT,
            first_visit_date TEXT,
            discount_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS patient_status (
            patient_id TEXT,
            smoking BOOLEAN,
            smoking_count TEXT,
            pregnancy BOOLEAN,
            medications BOOLEAN,
            blood_sugar_pre REAL,
            blood_sugar_post REAL,
            eligible_scaling BOOLEAN,
            eligible_fluoride BOOLEAN,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );

        CREATE TABLE IF NOT EXISTS dental_treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            date TEXT,
            treatment_item TEXT,
            tooth_position TEXT,
            doctor TEXT,
            money REAL,
            doctor_comment TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            date TEXT,
            time_slot TEXT,
            status TEXT,
            doctor TEXT,
            content TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );

        CREATE TABLE IF NOT EXISTS expenses (
            patient_id TEXT PRIMARY KEY,
            accumulated_total REAL,
            billing_projects_amount_due REAL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
    """
    )

    def get(d, *keys):
        for k in keys:
            d = d.get(k, {})
        return d if d != {} else None

    for i in tqdm(range(1, n_patient)):
        patient = get_patient_data(str(i))

        if len(patient["Patient"]) == 0:
            continue

        if len(patient["PastTreatments"]["DentalTreatments"]) > 0:
            treatment_patient_cnt += 1

        if (treatment_patient_cnt + 1) % 100 == 0:
            print("Current TPC: {}".format(treatment_patient_cnt + 1))

        try:

            pid = patient.get("PatientId")

            info = patient.get("Patient", {}).get("PatientInfo", {})
            status = patient.get("Patient", {}).get("PatientStatus", {})

            cur.execute(
                """INSERT OR IGNORE INTO patients VALUES (?, ?, ?, ?, ?)""",
                (
                    pid,
                    info.get("Gender"),
                    info.get("Birthday"),
                    info.get("FirstVisitDate"),
                    info.get("DiscountReason"),
                ),
            )

            cur.execute(
                """INSERT OR IGNORE INTO patient_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pid,
                    status.get("Smoking", {}).get("Status"),
                    status.get("Smoking", {}).get("DailyCount"),
                    status.get("Pregnancy", {}).get("Status"),
                    status.get("Medications", {}).get("Status"),
                    status.get("BloodSugar", {}).get("PreMeal"),
                    status.get("BloodSugar", {}).get("PostMeal"),
                    status.get("EligibleForScaling"),
                    status.get("EligibleForFluorideTreatment"),
                ),
            )

            for t in patient.get("PastTreatments", {}).get("DentalTreatments", []):
                cur.execute(
                    """INSERT INTO dental_treatments VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid,
                        t.get("Date"),
                        t.get("TreatmentItem"),
                        t.get("ToothPosition"),
                        t.get("Doctor"),
                        t.get("Money"),
                        t.get("DoctorComment"),
                    ),
                )

            for a in patient.get("Appointments", {}).get("AppointmentRecords", []):
                cur.execute(
                    """INSERT INTO appointments VALUES (NULL, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid,
                        a.get("Date"),
                        a.get("TimeSlot"),
                        a.get("Status"),
                        a.get("Doctor"),
                        a.get("Content", ""),
                    ),
                )

            exp = patient.get("Expenses", {})
            cur.execute(
                """INSERT OR IGNORE INTO expenses VALUES (?, ?, ?)""",
                (pid, exp.get("AccumulatedTotal"), exp.get("BillingProjectsAmountDue")),
            )

        except Exception as e:
            print("An error occurred:", e)
            continue

    conn.commit()
    conn.close()
