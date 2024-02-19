# Python Libraries
from fpdf import FPDF
from datetime import datetime, timedelta
import os
import sqlite3
import pandas as pd
import sys


# Local Libraries
from charts_generations.pn_manual_and_status import pie_manual_pn_status
from charts_generations.face_detection_and_verification import pie_face_detection_verification
WIDTH = 210
HEIGHT = 297


def header_report(pdf, txt):
    # Unicode is not yet supported in the py3k version; use windows-1252 standard font
    pdf.set_font('Arial', 'B', 24,)  
    pdf.cell(32)
    pdf.cell(130, 12, f'{txt}', 0, 1, 'C')

def header_title(pdf, txt):
    # Unicode is not yet supported in the py3k version; use windows-1252 standard font
    pdf.set_font('Arial', 'B', 14,)  
    pdf.cell(32)
    pdf.cell(145, 10, f'{txt}', 0, 1, 'C')

def write_title(pdf, txt, x):
    # Unicode is not yet supported in the py3k version; use windows-1252 standard font
    left = 20
    pdf.set_font('Arial', 'B', 18,)  
    pdf.ln(75)
    pdf.set_x(x)
    pdf.write(5, f"{txt}")

def write_time(pdf, time_1, time_2, x):
    # Unicode is not yet supported in the py3k version; use windows-1252 standard font
    left = 26
    pdf.set_font('Arial', '', 11,)  
    pdf.ln(10)
    pdf.set_x(x)
    pdf.write(5, f"Avg Time = {time_1} sec                                                                        Avg Time = {time_2} sec")

def write_heading(pdf, txt, adjust, x):
    left = 20
    pdf.set_font('Arial', 'B', 15,)  
    pdf.ln(adjust)
    pdf.cell(20)
    pdf.cell(x, 12, f'{txt}', 0, 1, 'C')

def database_report_table():

    try:
        conn = sqlite3.connect('Database/immigration_DB.db')
        cur = conn.cursor()
    except:
        print("Database connection failed")
        sys.exit(1)

    query = "SELECT * FROM 'Report'"
    df = pd.read_sql_query(query, conn)
    # print(df)
    
    total_entitries_in_data = df['total_time'].count()
    # print(total_entitries_in_data)
    total_entitries_total_time_sum = df['total_time'].sum()
    average_time = total_entitries_total_time_sum/total_entitries_in_data

    # Pie_Chart for Manual Passport Numbers
    total_entries_for_passprt_extraction = df['pn_manual'].count()
    manual_entries_of_passport = (df['pn_manual'] == 1).sum()
    pie_manual_pn_status(total_entries_for_passprt_extraction, manual_entries_of_passport, "Auto", "Manual", "Total Entries", "pn_manul_pie_chart")


    # # Pie_Chart for Status Passengers
    # total_entries_for_passprt_extraction = df['status'].count()
    # manual_entries_of_passport = (df['status'] == 0).sum()
    # pie_manual_pn_status(total_entries_for_passprt_extraction, manual_entries_of_passport, "Verified", "Under Investigation", "Total Passengers", "passenger_status_pie_chart")

    # Pie_Chart for Face Detection Passport Verification
    total_pv_faces = df['passport_face_detected'].count()
    undetected_faces = (df['passport_face_detected'] == 0).sum()
    pie_face_detection_verification(total_pv_faces, undetected_faces, "Detected", "Not detected", "pass_verification_face_detection")

    # Pie_Chart for Face Verification Passport Verification
    total_pv_faces = df['passport_verified'].count()
    undetected_faces = (df['passport_verified'] == 0).sum()
    pie_face_detection_verification(total_pv_faces, undetected_faces, "Verified", "Not verified",  "pass_verification_face_verification")

    # Pie_Chart for Face Detection Person Verification
    total_pv_faces = df['person_face_detected'].count()
    undetected_faces = (df['person_face_detected'] == 0).sum()
    pie_face_detection_verification(total_pv_faces, undetected_faces, "Detected", "Not detected", "person_verification_face_detection")

    # Pie_Chart for Face Verification Person Verification
    total_pv_faces = df['person_verified'].count()
    undetected_faces = (df['person_verified'] == 0).sum()
    pie_face_detection_verification(total_pv_faces, undetected_faces, "Verified", "Not verified",  "person_verification_face_verification")



    conn.commit()
    cur.close()
    conn.close()
    return total_entitries_in_data ,average_time
    
def average_time_text(pdf, txt_1, text_2):
    text_upper = f'Total Entries = {txt_1}'
    text_lower = f'Avg Time = {int(text_2)} sec'
    pdf.set_font('Arial', 'BU', 20,)  
    pdf.cell(48)
    pos_lower = pdf.y + 20
    pdf.cell(165, 70, text_upper, 0, 1, 'C')
    pdf.y = pos_lower
    pdf.x = pdf.l_margin
    pdf.cell(48)
    pdf.cell(165, 55, text_lower, 0, 0, 'C')




def generate_report(filename='report.pdf'): 
    total_entries, average_time = database_report_table()    


    pdf = FPDF()
    # pdf.set_margins(0,0,0)
    '''Page Header with Devider Line'''
    pdf.add_page()
    
    pdf.image("./resources/logo/COMSATS_logo.png", 10, 5, 25, 25)
    header_report(pdf, "REPORT")
    header_title(pdf, "Forensic Face Matching at Airport during Passport Control")
    average_time_text(pdf,total_entries, average_time)
    pdf.image("./resources/report/page_devider.jpg",WIDTH/2, 120,)
    pdf.image("./resources/report/horizontal_devider.png",0, 40,WIDTH)
    pdf.image("./resources/report/horizontal_devider.png",0, 120,WIDTH)
    
    '''Left Side of Page'''
    write_title(pdf,"Passport Verification                          Person Verification", 20)
    write_time(pdf, 6, 30, 35)

    pdf.image(name = "./charts_images/pn_manul_pie_chart.png",x = 0,y = 38, w=100, h = 75)
    write_heading(pdf, "Passport Extraction", -30, 42)
    # pdf.image(name = "./charts_images/passenger_status_pie_chart.png",x = 110,y = 38, w=100, h = 75)
    # write_heading(pdf, "Verification Status", -12, 252)
    

    
    pdf.image(name = "./charts_images/pass_verification_face_detection.png",x = 0,y = 140, w=100, h = 75)
    pdf.image("./resources/report/Face_Detection.png",140, 210, 35, 4)
    
    pdf.image(name = "./charts_images/person_verification_face_detection.png",x = 110,y = 140, w=100, h = 75)
    pdf.image("./resources/report/Face_Detection.png",32, 210, 35, 4)


    pdf.image(name = "./charts_images/pass_verification_face_verification.png",x = 0,y = 215, w=100, h = 75)
    pdf.image("./resources/report/Face_Verification.png",32, 285, 40, 4)


    pdf.image(name = "./charts_images/person_verification_face_verification.png",x = 110,y = 215, w=100, h = 75)
    pdf.image("./resources/report/Face_Verification.png",140, 285, 40, 4)

    pdf.output(name=filename)

    # files = os.listdir(os.path.join(os.getcwd(), "charts_images"))
    #
    # for file in files:
    #     if os.path.exists(os.path.join(os.getcwd(), "charts_images\\")+file):
    #         print(file)
    #         os.remove(os.path.join(os.getcwd(), "charts_images\\")+file)