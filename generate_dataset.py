import json
import os

# --- Data Definition (Manual Variations) ---

UNIVERSITY_FACTS = [
    (
        ["Where is KLE Tech located?", "Campus location?", "Give me the university address."],
        "[LOCATION]: KLE Tech is located in Hubballi (Hubli), Karnataka, India. Its main campus is in BVB Campus, Vidyanagar."
    ),
    (
        ["How to reach the campus?", "Nearest airport or railway station?", "Travel info for KLE Tech."],
        "[LOCATION]: You can reach the campus via Hubballi Airport (HBX) or Hubballi Junction Railway Station. It is situated in Vidyanagar."
    ),
    (
        ["Tell me about placement records.", "How are the placements at KLE Tech?", "Placement statistics and companies."],
        "[PLACEMENT]: KLE Tech has an exceptional placement record with 90%+ students placed yearly. Top companies include Google, Amazon, AWS, Bosch, and Microsoft."
    ),
    (
        ["What is the highest package?", "Google placement package?", "Highest domestic and international package."],
        "[PLACEMENT]: The highest international and domestic packages at KLE Tech often exceed INR 30-40 Lakhs per annum for top engineering roles."
    ),
    (
        ["University ranking info?", "NIRF ranking of KLE Tech?", "Is it a top engineering college?"],
        "[RANKING]: KLE Tech is consistently ranked among the top engineering colleges in Karnataka and features in the NIRF ranking bands."
    )
]

FEES = [
    (
        ["What is the KCET fee?", "Government quota fee for B.E.?", "CET quota charges."],
        "[FEE]: The KCET / Government Quota fee for B.E. is approximately INR 1,25,000 per annum."
    ),
    (
        ["COMEDK fee structure?", "Management quota fee for B.E.?", "How much is COMEDK entry?"],
        "[FEE]: The COMEDK quota fee for B.E. is approximately INR 2,25,000 per annum."
    )
]

HOLIDAYS = [
    ("Chandramana Ugadi", "19th March 2026", ["When is Ugadi?", "Ugadi holiday date?", "Is 19th March a holiday?"]),
    ("Compensatory Holiday", "20th March 2026", ["Is 20th March a holiday?", "When is the compensatory holiday?", "Next holiday after Ugadi?"]),
    ("Ramzan (Eid-ul-Fitr)", "21st March 2026", ["Ramzan date?", "Eid-ul-Fitr holiday?", "When is Eid?"]),
    ("Mahavir Jayanti", "31st March 2026", ["Mahavir Jayanti date?", "Is 31st March a holiday?", "When is Mahavir Jayanti?"]),
    ("Good Friday", "3rd April 2026", ["Good Friday date?", "Is Good Friday a holiday?"]),
    ("Ambedkar Jayanti", "14th April 2026", ["Ambedkar Jayanti date?", "When is Ambedkar Jayanti?"]),
    ("Basava Jayanti", "20th April 2026", ["Basava Jayanti date?", "Basava Jayanti holiday?"]),
    ("May Day", "1st May 2026", ["When is May Day?", "Is 1st May a holiday?"]),
    ("Bakrid", "28th May 2026", ["Bakrid date?", "Bakrid holiday?"])
]

# Monthly Holiday Summaries
MONTHS_HOLIDAYS = [
    (
        ["List all March holidays?", "Holidays in March 2026?", "What holidays are there in March?"],
        "[CALENDAR]: March 2026 holidays: Ugadi (19th March), Compensatory (20th March), Ramzan (21st March), Mahavir Jayanti (31st March)."
    ),
    (
        ["List all April holidays?", "Holidays in April 2026?", "What holidays are there in April?"],
        "[CALENDAR]: April 2026 holidays: Good Friday (3rd April), Ambedkar Jayanti (14th April), Basava Jayanti (20th April)."
    ),
    (
        ["List all May holidays?", "Holidays in May 2026?", "What holidays are there in May?"],
        "[CALENDAR]: May 2026 holidays: May Day (1st May), Bakrid (28th May)."
    )
]

# ---------------- 4th Semester Timetable ----------------
IV_TIMETABLE = [
    # Monday
    (["Monday timetable for 4th Sem Div A?", "IV A Monday schedule?", "IV A Monday classes?"],
     "[ACADEMIC]: Monday (IV A): CS-SAR (8-10, SEE104), LA-NSS (10:15-11:15, SEE104), IT-PC (11:15-12:15, SEE104), S&S-ART (1:30-2:30, SEE208)."),
    (["Monday timetable for 4th Sem Div B?", "IV B Monday schedule?", "IV B Monday classes?"],
     "[ACADEMIC]: Monday (IV B): LA-BMS (9-10, SEE114), S&S-RMB+SMG (10:15-12:15, SEE114), LIC-SBH+LRD+KSS (1:30-3:30, SEE108), ARM LAB (3:30-5:30, SEE109)."),
    (["Monday timetable for 4th Sem Div C?", "IV C Monday schedule?", "IV C Monday classes?"],
     "[ACADEMIC]: Monday (IV C): S&S-PSP+ART (8-10, SEE112), CS-RVH (10:15-11:15, SEE204), LA-BMS+SAH+SMG (1:30-3:30, SEE112)."),
    (["Monday timetable for 4th Sem Div D?", "IV D Monday schedule?", "IV D Monday classes?"],
     "[ACADEMIC]: Monday (IV D): CS-SA (8-10, SEE209), S&S-NSR (10:15-11:15, SEE209), ARM-NS (1:30-2:30, SEE104), LIC-JP+KSS+RVB (3:30-5:30, SEE108)."),
    (["Monday timetable for 4th Sem Div E?", "IV E Monday schedule?", "IV E Monday classes?"],
     "[ACADEMIC]: Monday (IV E): CS-SA (10:15-11:15, SEE110), DSA-HS+SBN (1:30-3:30, SEE114), CS LAB E2-SA+CSS (3:30-5:30, SEE206)."),
    (["Monday timetable for 4th Sem Div F?", "IV F Monday schedule?", "IV F Monday classes?"],
     "[ACADEMIC]: Monday (IV F): DSA-HS+SMK (8-10, SEE110), LIC-JP+CJ+LRD (10:15-12:15, SEE108), ARM-PSP (1:30-2:30, SEE109)."),

    # Tuesday
    (["Tuesday timetable for 4th Sem Div A?", "IV A Tuesday schedule?", "IV A Tuesday classes?"],
     "[ACADEMIC]: Tuesday (IV A): PSC LAB-RMB+SMG (10:15-12:15, SEE110), LA-NSS (1:30-2:30, SEE112), IT-PC (2:30-3:30, SEE112)."),
    (["Tuesday timetable for 4th Sem Div B?", "IV B Tuesday schedule?", "IV B Tuesday classes?"],
     "[ACADEMIC]: Tuesday (IV B): S&S-RMB (9-10, SEE104), ARM-SVK (10:15-12:15, SEE104), LA-BMS+SAH+SMG (1:30-3:30, SEE104), CS LAB B2-SAR+VKK (3:30-5:30, SEE206)."),
    (["Tuesday timetable for 4th Sem Div C?", "IV C Tuesday schedule?", "IV C Tuesday classes?"],
     "[ACADEMIC]: Tuesday (IV C): ARM-BRK (9-10, SEE109), LA-BMS (10:15-11:15, SEE114), S&S-PSP (11:15-12:15, SEE114), LIC-SBH+RVB+CJ (1:30-3:30, SEE108)."),
    (["Tuesday timetable for 4th Sem Div D?", "IV D Tuesday schedule?", "IV D Tuesday classes?"],
     "[ACADEMIC]: Tuesday (IV D): LA-PC (9-10, SEE112), LIC-JP+KSS+RVB (10:15-12:15, SEE108), DSA-SBN+HS (1:30-3:30, SEE110)."),
    (["Tuesday timetable for 4th Sem Div E?", "IV E Tuesday schedule?", "IV E Tuesday classes?"],
     "[ACADEMIC]: Tuesday (IV E): LIC-MC+SHH+GK (8-10, SEE108), DSA-HS+SBN (10:15-12:15, SEE111), CS LAB E1-SA+VKK (1:30-3:30, SEE206)."),
    (["Tuesday timetable for 4th Sem Div F?", "IV F Tuesday schedule?", "IV F Tuesday classes?"],
     "[ACADEMIC]: Tuesday (IV F): LA-AB (9-10, SEE114), CS-RVH (10:15-11:15, SEE204), S&S-NSR+PSP (1:30-3:30, SEE114), LIC-JP+CJ+LRD (3:30-5:30, SEE108)."),

    # Wednesday
    (["Wednesday timetable for 4th Sem Div A?", "IV A Wednesday schedule?", "IV A Wednesday classes?"],
     "[ACADEMIC]: Wednesday (IV A): CS-SAR (8-10, SEE114), ARM LAB-VKK+RA+NS (10:15-12:15, SEE109), S&S-ART (1:30-2:30, SEE109), ARM-VKK (2:30-3:30, SEE109)."),
    (["Wednesday timetable for 4th Sem Div B?", "IV B Wednesday schedule?", "IV B Wednesday classes?"],
     "[ACADEMIC]: Wednesday (IV B): LIC-SBH+LRD+KSS (10:15-12:15, SEE108), S&S-RMB (1:30-2:30, SEE104)."),
    (["Wednesday timetable for 4th Sem Div C?", "IV C Wednesday schedule?", "IV C Wednesday classes?"],
     "[ACADEMIC]: Wednesday (IV C): ARM-BRK (10:15-11:15, SEE112), LA-BMS (11:15-12:15, SEE112), S&S-PSP (1:30-2:30, SEE112), LIC-SBH+RVB+CJ (3:30-5:30, SEE108)."),
    (["Wednesday timetable for 4th Sem Div D?", "IV D Wednesday schedule?", "IV D Wednesday classes?"],
     "[ACADEMIC]: Wednesday (IV D): CS LAB D1-SA+CSS (8-10, SEE206), LA-PC+SAH+SMG (10:15-12:15, SEE114), ARM-NS (1:30-2:30, SEE208), CS LAB D2-SA+CSS (3:30-5:30, SEE206)."),
    (["Wednesday timetable for 4th Sem Div E?", "IV E Wednesday schedule?", "IV E Wednesday classes?"],
     "[ACADEMIC]: Wednesday (IV E): LA-PC (9-10, SEE112), CS-SA (10:15-11:15, SEE104), LIC-MC+SHH+GK (1:30-3:30, SEE108)."),
    (["Wednesday timetable for 4th Sem Div F?", "IV F Wednesday schedule?", "IV F Wednesday classes?"],
     "[ACADEMIC]: Wednesday (IV F): DSA-HS+SMK (8-10, SEE110), CS LAB F1-SAR+CSS (10:15-12:15, SEE206), LA-AB+SAH+SMG (1:30-3:30, SEE114)."),

    # Thursday
    (["Thursday timetable for 4th Sem Div A?", "IV A Thursday schedule?", "IV A Thursday classes?"],
     "[ACADEMIC]: Thursday (IV A): LIC-MC+RVB (8-10, SEE108), PSC LAB-RMB+SMG (10:15-12:15, SEE110), ARM-VKK (1:30-2:30, SEE112), LA-NSS+SAH+SMG (2:30-4:30, SEE112)."),
    (["Thursday timetable for 4th Sem Div B?", "IV B Thursday schedule?", "IV B Thursday classes?"],
     "[ACADEMIC]: Thursday (IV B): CS-SAR (8-10, SEE114), DSA-SBN+HS (10:15-12:15, SEE114), LA-BMS (1:30-3:30, SEE114)."),
    (["Thursday timetable for 4th Sem Div C?", "IV C Thursday schedule?", "IV C Thursday classes?"],
     "[ACADEMIC]: Thursday (IV C): CS-RVH (8-10, SEE204), LIC-SBH+RVB+CJ (10:15-12:15, SEE108), ARM LAB-BRK+RA+NS (1:30-3:30, SEE109), CS LAB C2-SA+VKK (3:30-5:30, SEE206)."),
    (["Thursday timetable for 4th Sem Div D?", "IV D Thursday schedule?", "IV D Thursday classes?"],
     "[ACADEMIC]: Thursday (IV D): CS-SA (10:15-11:15, SEE104), LA-PC (1:30-2:30, SEE104), S&S-NSR (2:30-4:30, SEE104)."),
    (["Thursday timetable for 4th Sem Div E?", "IV E Thursday schedule?", "IV E Thursday classes?"],
     "[ACADEMIC]: Thursday (IV E): LA-PC (9-10, SEE104), ARM-RA (10:15-11:15, SEE109), LIC-MC+SHH+GK (1:30-3:30, SEE108), S&S-UBP (3:30-5:30, SEE104)."),
    (["Thursday timetable for 4th Sem Div F?", "IV F Thursday schedule?", "IV F Thursday classes?"],
     "[ACADEMIC]: Thursday (IV F): S&S-NSR (8-10, SEE112), LA-AB (10:15-11:15, SEE112), ARM-PSP (11:15-12:15, SEE112), CS LAB F2-SAR+CSS (1:30-3:30, SEE206)."),

    # Friday
    (["Friday timetable for 4th Sem Div A?", "IV A Friday schedule?", "IV A Friday classes?"],
     "[ACADEMIC]: Friday (IV A): S&S-ART+PSP (8-10, SEE112), ARM-VKK (10:15-11:15, SEE112), LA-NSS (11:15-12:15, SEE112), LIC-MC+RVB (1:30-3:30, SEE108)."),
    (["Friday timetable for 4th Sem Div B?", "IV B Friday schedule?", "IV B Friday classes?"],
     "[ACADEMIC]: Friday (IV B): CS-SAR (8-10, SEE104), LIC-SBH+LRD+KSS (10:15-12:15, SEE108), DSA-SBN+HS (1:30-3:30, SEE110), CS LAB B1-SAR+VKK (3:30-5:30, SEE206)."),
    (["Friday timetable for 4th Sem Div C?", "IV C Friday schedule?", "IV C Friday classes?"],
     "[ACADEMIC]: Friday (IV C): DSA-SBN+MK (8-10, SEE110), LA-BMS (10:15-11:15, SEE104), ARM-BRK (11:15-12:15, SEE104)."),
    (["Friday timetable for 4th Sem Div D?", "IV D Friday schedule?", "IV D Friday classes?"],
     "[ACADEMIC]: Friday (IV D): LA-PC (9-10, SEE114), S&S-NSR+ART (10:15-12:15, SEE109), ARM LAB-NS+VKK+RA (1:30-3:30, SEE109), LIC-JP+KSS+RVB (3:30-5:30, SEE108)."),
    (["Friday timetable for 4th Sem Div E?", "IV E Friday schedule?", "IV E Friday classes?"],
     "[ACADEMIC]: Friday (IV E): ARM LAB-RA+NS+VKK (8-10, SEE109), S&S-UBP+PSP (10:15-12:15, SEE209), LA-PC+SAH+SMG (1:30-3:30, SEE104)."),
    (["Friday timetable for 4th Sem Div F?", "IV F Friday schedule?", "IV F Friday classes?"],
     "[ACADEMIC]: Friday (IV F): LIC-JP+CJ+LRD (8-10, SEE108), CS-RVH (10:15-11:15, SEE204), LA-AB (1:30-2:30, SEE112), S&S-NSR (2:30-3:30, SEE112)."),
]

# ---------------- 6th Semester Timetable ----------------
VI_TIMETABLE = [
    # Monday
    (["Monday timetable for 6th Sem Div A?", "VI A Monday schedule?", "VI A Monday classes?"],
     "[ACADEMIC]: Monday (VI A): PALR (8-10, SEE303), Elective (10:15-12:15), Minor Project (1:30-4:30)."),
    (["Monday timetable for 6th Sem Div B?", "VI B Monday schedule?", "VI B Monday classes?"],
     "[ACADEMIC]: Monday (VI B): EIS (8-10, CLH304), Elective (10:15-12:15), CCN-SKG (1:30-2:30, SEE302), Minor Project (2:30-5:30)."),
    (["Monday timetable for 6th Sem Div C?", "VI C Monday schedule?", "VI C Monday classes?"],
     "[ACADEMIC]: Monday (VI C): EIS (8-10, CLH304), Elective (10:15-12:15), PALR (1:30-2:30, SEE303), CCN LAB C2 (2:30-5:30, SEE111)."),
    (["Monday timetable for 6th Sem Div D?", "VI D Monday schedule?", "VI D Monday classes?"],
     "[ACADEMIC]: Monday (VI D): EIS (8-10, CLH304), Elective (10:15-12:15), Minor Project (1:30-5:30)."),
    (["Monday timetable for 6th Sem Div E?", "VI E Monday schedule?", "VI E Monday classes?"],
     "[ACADEMIC]: Monday (VI E): EIS (8-10, CLH304), Elective (10:15-12:15), AE LAB E1 (1:30-3:30, SEE305), CCN LAB E2 (3:30-5:30, SEE111)."),
    (["Monday timetable for 6th Sem Div F?", "VI F Monday schedule?", "VI F Monday classes?"],
     "[ACADEMIC]: Monday (VI F): EIS (8-10, CLH304), Elective (10:15-12:15), GEN AI-SP+SSC (1:30-2:30, SEE209)."),

    # Tuesday
    (["Tuesday timetable for 6th Sem Div A?", "VI A Tuesday schedule?", "VI A Tuesday classes?"],
     "[ACADEMIC]: Tuesday (VI A): CCN LAB-SKG+MA (8-10, SEE111), Electives (10:15-1:30), Minor Project (1:30-3:30)."),
    (["Tuesday timetable for 6th Sem Div B?", "VI B Tuesday schedule?", "VI B Tuesday classes?"],
     "[ACADEMIC]: Tuesday (VI B): Minor Project (8-10), Electives (10:15-1:30), GEN AI-RT+ART (1:30-3:30, SEE209)."),
    (["Tuesday timetable for 6th Sem Div C?", "VI C Tuesday schedule?", "VI C Tuesday classes?"],
     "[ACADEMIC]: Tuesday (VI C): Minor Project (8-10), Electives (10:15-1:30), CCN-MA (1:30-3:30, SEE302)."),
    (["Tuesday timetable for 6th Sem Div D?", "VI D Tuesday schedule?", "VI D Tuesday classes?"],
     "[ACADEMIC]: Tuesday (VI D): CCN-SHS (8-10, SEE302), Electives (10:15-1:30), AE LAB D1 & CCN LAB D2 (1:30-3:30)."),
    (["Tuesday timetable for 6th Sem Div E?", "VI E Tuesday schedule?", "VI E Tuesday classes?"],
     "[ACADEMIC]: Tuesday (VI E): Minor Project (8-10), Electives (10:15-1:30), PALR (1:30-3:30, SEE303)."),
    (["Tuesday timetable for 6th Sem Div F?", "VI F Tuesday schedule?", "VI F Tuesday classes?"],
     "[ACADEMIC]: Tuesday (VI F): AE LAB F1-CJ+SM (8-10, SEE305), Electives (10:15-1:30), GEN AI-SP+SSC (1:30-3:30, SEE109), PALR (3:30-5:30, SEE303)."),

    # Wednesday
    (["Wednesday timetable for 6th Sem Div A?", "VI A Wednesday schedule?", "VI A Wednesday classes?"],
     "[ACADEMIC]: Wednesday (VI A): AE LAB-KSS+SM (8-10, SEE305), CCN-SKG (10:15-12:15, SEE303), Minor Project (1:30-3:30)."),
    (["Wednesday timetable for 6th Sem Div B?", "VI B Wednesday schedule?", "VI B Wednesday classes?"],
     "[ACADEMIC]: Wednesday (VI B): GEN AI-RT+ART (8-10, SEE302), AE-RSJ (10:15-12:15, SEE302), PALR (1:30-2:30, SEE303)."),
    (["Wednesday timetable for 6th Sem Div C?", "VI C Wednesday schedule?", "VI C Wednesday classes?"],
     "[ACADEMIC]: Wednesday (VI C): AE LAB C2 & CCN LAB C1 (10:15-12:15), GEN AI-SSC+SVK (1:30-3:30, SEE209)."),
    (["Wednesday timetable for 6th Sem Div D?", "VI D Wednesday schedule?", "VI D Wednesday classes?"],
     "[ACADEMIC]: Wednesday (VI D): GEN AI-SSC+SVK (8-10, SEE209), CCN-SHS (10:15-12:15, SEE204), AE-BP (1:30-2:30, SEE110)."),
    (["Wednesday timetable for 6th Sem Div E?", "VI E Wednesday schedule?", "VI E Wednesday classes?"],
     "[ACADEMIC]: Wednesday (VI E): CCN-GK (8-10, SEE204), GEN AI-ART+RT (10:15-12:15, SEE110), AE-PCN (1:30-2:30, SEE302)."),
    (["Wednesday timetable for 6th Sem Div F?", "VI F Wednesday schedule?", "VI F Wednesday classes?"],
     "[ACADEMIC]: Wednesday (VI F): CCN LAB F2-CA+SHS (8-10, SEE111), AE-CJ (10:15-12:15, SEE208), AE LAB F2 & CCN LAB F1 (1:30-3:30)."),

    # Thursday
    (["Thursday timetable for 6th Sem Div A?", "VI A Thursday schedule?", "VI A Thursday classes?"],
     "[ACADEMIC]: Thursday (VI A): AE-KSS (8-10, SEE209), GEN AI-RT+SP (10:15-12:15, SEE209), PALR (1:30-3:30, SEE302)."),
    (["Thursday timetable for 6th Sem Div B?", "VI B Thursday schedule?", "VI B Thursday classes?"],
     "[ACADEMIC]: Thursday (VI B): CCN-SKG (10:15-12:15, SEE302), AE LAB B2 & CCN LAB B1 (1:30-3:30)."),
    (["Thursday timetable for 6th Sem Div C?", "VI C Thursday schedule?", "VI C Thursday classes?"],
     "[ACADEMIC]: Thursday (VI C): AE LAB C1-GHM+KMR (8-10, SEE305), CCN-MA (10:15-12:15, SEE303), GEN AI-SSC+SVK (1:30-3:30, SEE209)."),
    (["Thursday timetable for 6th Sem Div D?", "VI D Thursday schedule?", "VI D Thursday classes?"],
     "[ACADEMIC]: Thursday (VI D): PALR (8-10, SEE303), GEN AI-SSC+SVK (10:15-12:15, SEE208)."),
    (["Thursday timetable for 6th Sem Div E?", "VI E Thursday schedule?", "VI E Thursday classes?"],
     "[ACADEMIC]: Thursday (VI E): CCN-GK (8-10, SEE302), AE LAB E2 & CCN LAB E1 (10:15-12:15), AE-PCN (1:30-3:30, SEE303)."),
    (["Thursday timetable for 6th Sem Div F?", "VI F Thursday schedule?", "VI F Thursday classes?"],
     "[ACADEMIC]: Thursday (VI F): AE-CJ (9-10, SEE109), CCN-CA (10:15-11:15, SEE204), Minor Project (1:30-4:30)."),

    # Friday
    (["Friday timetable for 6th Sem Div A?", "VI A Friday schedule?", "VI A Friday classes?"],
     "[ACADEMIC]: Friday (VI A): GEN AI-RT+SP (8-10, SEE209), CCN-SKG (10:15-11:15, SEE114), AE-KSS (1:30-2:30, SEE302)."),
    (["Friday timetable for 6th Sem Div B?", "VI B Friday schedule?", "VI B Friday classes?"],
     "[ACADEMIC]: Friday (VI B): CCN-SKG (9-10, SEE302), AE-RSJ (10:15-12:15, SEE110), AE LAB B1 & CCN LAB B2 (1:30-4:30)."),
    (["Friday timetable for 6th Sem Div C?", "VI C Friday schedule?", "VI C Friday classes?"],
     "[ACADEMIC]: Friday (VI C): PALR (8-10, SEE303), AE-GHM (10:15-11:15, SEE210), Minor Project (1:30-3:30)."),
    (["Friday timetable for 6th Sem Div D?", "VI D Friday schedule?", "VI D Friday classes?"],
     "[ACADEMIC]: Friday (VI D): AE-BP (9-10, SEE204), AE LAB D2 & CCN LAB D1 (10:15-12:15), PALR (1:30-3:30, SEE303)."),
    (["Friday timetable for 6th Sem Div E?", "VI E Friday schedule?", "VI E Friday classes?"],
     "[ACADEMIC]: Friday (VI E): Minor Project (8-10), PALR (10:15-11:15, SEE303), GEN AI-ART+RT (1:30-3:30, SEE204)."),
    (["Friday timetable for 6th Sem Div F?", "VI F Friday schedule?", "VI F Friday classes?"],
     "[ACADEMIC]: Friday (VI F): Minor Project (8-10), PALR (10:15-11:15, SEE302), CCN-CA (1:30-2:30, SEE114)."),
]

# ---------------- Electives ----------------
ELECTIVES = [
    (["When is ADIC elective class?", "ADIC class schedule?", "ADIC timetable?"],
     "[ACADEMIC]: ADIC-RM: Monday and Tuesday (10:15-12:15) in SEE207 (Mon) and SEE305 (Tue)."),
    (["When is AL elective class?", "AL class schedule?", "AL timetable?"],
     "[ACADEMIC]: AL elective: Div A,B,C on Monday/Wednesday (SEE302/207); Div D,E,F on Monday/Tuesday/Thursday (SEE302/207)."),
]

# ---------------- ESA Exam Dates ----------------
ESA_DATES = [
    (["When are practical exams?", "ESA practical exam dates?", "When do practical exams start?"],
     "[ACADEMIC]: ESA Practical Exams are from 12th June to 16th June 2026 for all programs including B.E., B.Arch, BBA, BCA, B.Com, and B.Sc. FAD."),
    (["When are theory exams?", "ESA theory exam dates?", "When do theory exams start?"],
     "[ACADEMIC]: ESA Theory Exams begin from 19th June 2026 onwards for all programs including B.E., B.Arch, BBA, BCA, B.Com, and B.Sc. FAD."),
    (["What are end semester exam dates?", "Tell me ESA schedule.", "When are final exams this semester?"],
     "[ACADEMIC]: ESA Even Semester 2025-26: Practical Exams 12th-16th June 2026, Theory Exams from 19th June 2026. Applies to all programs."),
    (["ESA dates for B.E. students?", "When is end semester assessment for engineering?"],
     "[ACADEMIC]: For B.E. students (2nd, 4th, 6th Sem): Practical Exams 12th-16th June 2026, Theory Exams from 19th June 2026."),
    (["ESA dates for B.Arch students?", "When is end semester assessment for architecture?"],
     "[ACADEMIC]: For B.Arch students (2nd, 4th, 6th, 8th Sem): Practical Exams 12th-16th June 2026, Theory Exams from 19th June 2026."),
    (["ESA dates for BBA BCA B.Com students?", "When is end semester assessment for management students?"],
     "[ACADEMIC]: For 4th Sem BBA, BCA, B.Com students: Practical Exams 12th-16th June 2026, Theory Exams from 19th June 2026."),
]

# --- Final Assembly ---
final_pairs = []

# 1. Identity
final_pairs.append({"user": "Who are you?", "assistant": "[PERSONA]: I am the official Virtual Assistant for KLE Technological University (KLE Tech)."})
final_pairs.append({"user": "Tell me your role.", "assistant": "[PERSONA]: I am an AI assistant designed to provide accurate information about KLE Tech university."})

# 2. Add University Facts
for questions, answer in UNIVERSITY_FACTS:
    for q in questions:
        final_pairs.append({"user": q, "assistant": answer})

# 3. Add Fees
for questions, answer in FEES:
    for q in questions:
        final_pairs.append({"user": q, "assistant": answer})

# 4. Add Individual Holidays
for name, date, questions in HOLIDAYS:
    ans = f"[CALENDAR]: {name} is a university holiday on {date}."
    for q in questions:
        final_pairs.append({"user": q, "assistant": ans})

# 5. Add Monthly Holiday Summaries
for questions, answer in MONTH_HOLIDAYS if 'MONTH_HOLIDAYS' in locals() else MONTHS_HOLIDAYS:
    for q in questions:
        final_pairs.append({"user": q, "assistant": answer})

# 6. Add Full Semester Holiday List
final_pairs.append({"user": "List all holidays this semester.", "assistant": "[CALENDAR]: All holidays: Ugadi (19 Mar), Compensatory (20 Mar), Ramzan (21 Mar), Mahavir Jayanti (31 Mar), Good Friday (3 Apr), Ambedkar Jayanti (14 Apr), Basava Jayanti (20 Apr), May Day (1 May), Bakrid (28 May)."})
final_pairs.append({"user": "How many holidays are there this semester?", "assistant": "[CALENDAR]: There are 9 holidays this semester: Ugadi, Compensatory, Ramzan, Mahavir Jayanti, Good Friday, Ambedkar Jayanti, Basava Jayanti, May Day, and Bakrid."})
final_pairs.append({"user": "All college holidays 2026?", "assistant": "[CALENDAR]: All holidays: Ugadi (19 Mar), Compensatory (20 Mar), Ramzan (21 Mar), Mahavir Jayanti (31 Mar), Good Friday (3 Apr), Ambedkar Jayanti (14 Apr), Basava Jayanti (20 Apr), May Day (1 May), Bakrid (28 May)."})

# 7. Add all timetable + ESA to final_pairs
for questions, answer in IV_TIMETABLE + VI_TIMETABLE + ELECTIVES + ESA_DATES:
    for q in questions:
        final_pairs.append({"user": q, "assistant": answer})

# --- Write to Output ---
# No shuffle, No system prompt, format: {"user": q, "assistant": a}
with open('kle_tech_dataset.jsonl', 'w', encoding='utf-8') as f:
    for pair in final_pairs:
        f.write(json.dumps(pair) + '\n')

print(f"Dataset Refactor Complete! Generated {len(final_pairs)} samples.")

