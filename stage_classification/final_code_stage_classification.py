import os
import re
import pandas as pd
# Directories
input_directory = "/content/drive/MyDrive/aws-textract-output"
output_directory = "/content/drive/MyDrive/downstaged-output"
os.makedirs(output_directory, exist_ok=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Invasion Extraction ===
def extract_myometrial_invasion_percentage(text):
    if match := re.search(r"(?:invasion|myometrial invasion)[^\d]{0,10}(\d{1,2}(?:\.\d+)?)\s*%", text):
        return float(match.group(1)), "Direct"
    if match := re.search(r"(\d+(?:\.\d+)?)\s*mm.*?(?:into|in)\s+a\s+(\d+(?:\.\d+)?)\s*mm", text):
        depth_mm = float(match.group(1))
        thickness_mm = float(match.group(2))
        return round((depth_mm / thickness_mm) * 100, 1), "Calculated (mm)"
    if match := re.search(r"(\d+(?:\.\d+)?)\s*cm.*?(?:into|in)\s+a\s+(\d+(?:\.\d+)?)\s*cm", text):
        depth_cm = float(match.group(1))
        thickness_cm = float(match.group(2))
        return round((depth_cm / thickness_cm) * 100, 1), "Calculated (cm)"
    if re.search(r"(less\s+than\s+half|<\s*50%|inner\s+half|no\s+deep\s+invasion)", text):
        return 45.0, "Assumed (<50%)"
    return None, "Missing"

# === Upstaging Rules ===
def get_upstaging_patterns():
    return {
        "Lymph Node Metastasis": r"(lymph\s+node[s]?\s+(metastasis|positive)|(metastasis\s+(to|in).*lymph\s+node))",
        "Lymphovascular Invasion": r"(lymphovascular\s+space\s+invasion\s*:\s*present)",
        "High Tumor Grade (Grade III)": r"\b(grade\s*(3|iii|III)|high[-\s]*grade)\b",
        "Invasion of Serosa or Adnexa": r"(serosa\s+involvement|adnexal\s+involvement|extension\s+to\s+(serosa|adnexa))",
        "Metastasis to Other Organs": r"(metastasis\s+to\s+(lungs|liver|brain|bone|omentum|other\s+organs))",
        "Critical Lymphovascular Invasion": r"(lymphovascular\s+space\s+invasion.*(critical|extensive))",
        "FIGO Grade 3 with Myometrial Invasion": r"FIGO\s+grade\s*(3|III|iii)\s+.*?(myometrial\s+(invasion|involvement))",
        "Distant Metastasis (Stage IVB)": r"(stage\s+(IVB|4B|ivb).*(metastasis|lungs|liver|brain|bone))",
        "p53 Mutation or Abnormality with Aggressive Features": r"p53\s+(mutation|abnormality|alteration).*(high-grade|aggressive|metastatic)",
    }

# === Downstaging Rules ===
def get_downstaging_patterns():
    return {
        "Low Grade Tumor": r"(figo\s*grade\s*[i1]\s*/?\s*[ii2]?|grade\s*(i|ii|1|2)|figo\s*grade\s*(i|ii|1|2)|endometrioid\s*(carcinoma|type)?\s*figo\s*grade\s*(i|ii|1|2)|well[-\s]?differentiated|low[-\s]?grade)",

        "Tumor Confined to Uterus": r"(confined\s+to\s+(the\s+)?uterus|confined\s+to\s+corpus\s+uteri|within\s+(the\s+)?uterus|limited\s+to\s+(endometrium|corpus|uterus)|no\s+(extra[-\s]?uterine|serosal|parametrial)\s+(spread|extension|involvement)|spread\s+not\s+beyond\s+uterus|disease\s+(confined|limited)\s+to\s+(the\s+)?uterus)",

        "No Lymphovascular Invasion": r"(no\s+(lymphovascular|vascular|lymphatic)\s+invasion|no\s+evidence\s+of\s+(lymphovascular|vascular|lymphatic)\s+invasion|not\s+identified|lvsi\s+not\s+(seen|identified|detected)|absence\s+of\s+(lvsi|lymphovascular|vascular|lymphatic)\s+invasion|lvsi[:\s]+none|negative\s+for\s+lymphovascular\s+invasion|lvsi\s+absent)",

        "Minimal or Superficial Invasion": r"(superficial\s+invasion|minimal\s+invasion|limited\s+to\s+inner\s+myometrium|<\s*50%\s+invasion|less\s+than\s+(half|50%)\s+of\s+myometrium|inner\s+half|no\s+deep\s+myometrial\s+invasion|tumor\s+extends\s+to\s+(lower uterine segment|cervical canal)\s+only)",

        "POLE Mutation Mentioned": r"(pole\s+(mutation|mutated|ultramutated|positive|type|detected))",

        "Ovary Unilateral + No Capsule + <50% Invasion": r"(low[-\s]?grade.*?(ovary|ovaries).*?(unilateral|single|only\s+one).*?(no\s+(capsular|capsule)\s+(involvement|invasion)).*?(<\s*50%|less\s+than\s+half).*?(no\s+(distant|additional).*?metastasis))"
    }

# === Check LVSI Presence ===
def has_lvsi_present(text):
    return bool(re.search(r"(lvsi\s+(present|identified|extensive|positive|foci)|lymphovascular\s+invasion\s+(present|identified))", text))

# === Run Staging Classification with File Tracking ===
downstaged_files = []
likely_downstaged_files = []
ovary_exception_files = []
upstaged_files = []
no_change_files = []
clinical_review_files = []

def process_reports_with_tracking():
    counts = {"Downstaged": 0, "Likely Downstaged": 0, "Ovary Exception": 0, "Upstaged": 0, "No Change": 0}
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".txt"):
            with open(os.path.join(input_directory, file_name), "r", encoding="utf-8") as f:
                text = f.read()

            clean_text = preprocess_text(text)
            invasion_percent, method = extract_myometrial_invasion_percentage(text)
            output_lines = [f"📄 {file_name}", "-" * 60]
            upstaged = False
            downstaged = False

            for rule, pattern in get_upstaging_patterns().items():
                if re.search(pattern, clean_text):
                    output_lines.append(f"❌ Upstaged: {rule}")
                    upstaged = True

            matched = []
            if not upstaged:
                if invasion_percent is not None and invasion_percent < 50:
                    matched.append("Numeric Invasion < 50%")
                if not has_lvsi_present(text):
                    for rule, pattern in get_downstaging_patterns().items():
                        if re.search(pattern, clean_text, re.IGNORECASE):
                            matched.append(rule)

                required = {"Low Grade Tumor", "Tumor Confined to Uterus", "No Lymphovascular Invasion"}
                invasion_ok = "Numeric Invasion < 50%" in matched

                if required.issubset(set(matched)) and invasion_ok:
                    downstaged = True
                    output_lines.append("✅ Downstaged")
                    counts["Downstaged"] += 1
                    downstaged_files.append(file_name)
                elif "Ovary Unilateral + No Capsule + <50% Invasion" in matched:
                    downstaged = True
                    output_lines.append("✅ Downstaged (Ovary Exception)")
                    counts["Ovary Exception"] += 1
                    ovary_exception_files.append(file_name)
                elif len(required.intersection(set(matched))) >= 2 and invasion_ok:
                    output_lines.append("⚠️ Likely Downstaged")
                    counts["Likely Downstaged"] += 1
                    likely_downstaged_files.append(file_name)

            if upstaged:
                counts["Upstaged"] += 1
                upstaged_files.append(file_name)
                output_lines.append("📌 Final: Upstaged")
            elif not downstaged:
                counts["No Change"] += 1
                no_change_files.append(file_name)
                output_lines.append("📌 Final: No Change")

            with open(os.path.join(output_directory, f"{file_name}_staging_result.txt"), "w") as out_f:
                out_f.write("\n".join(output_lines))

    return counts

# === Run and Display Summary Table ===
summary = process_reports_with_tracking()

df = pd.DataFrame({
    "Downstaged": pd.Series(downstaged_files),
    "Likely Downstaged": pd.Series(likely_downstaged_files),
    "Ovary Exception": pd.Series(ovary_exception_files),
    "Upstaged": pd.Series(upstaged_files),
    "No Change": pd.Series(no_change_files)
})

from IPython.display import display
display(df)

print("\n📊 STAGING SUMMARY")
for category, count in summary.items():
    print(f"{category}: {count}")

def process_reports_with_dual_check():
    counts = {
        "Downstaged": 0,
        "Likely Downstaged": 0,
        "Ovary Exception": 0,
        "Upstaged": 0,
        "No Change": 0,
        "Clinical Review": 0
    }

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".txt"):
            with open(os.path.join(input_directory, file_name), "r", encoding="utf-8") as f:
                text = f.read()

            clean_text = preprocess_text(text)
            invasion_percent, method = extract_myometrial_invasion_percentage(text)
            output_lines = [f"📄 {file_name}", "-" * 60]
            is_upstaged = False
            is_downstaged = False
            matched_downstage = []

            # Check upstage
            for rule, pattern in get_upstaging_patterns().items():
                if re.search(pattern, clean_text):
                    output_lines.append(f"❌ Upstaged: {rule}")
                    is_upstaged = True

            # Check downstage
            if invasion_percent is not None and invasion_percent < 50:
                matched_downstage.append("Numeric Invasion < 50%")

            if not has_lvsi_present(text):
                for rule, pattern in get_downstaging_patterns().items():
                    if re.search(pattern, clean_text, re.IGNORECASE):
                        matched_downstage.append(rule)

            required = {"Low Grade Tumor", "Tumor Confined to Uterus", "No Lymphovascular Invasion"}
            invasion_ok = "Numeric Invasion < 50%" in matched_downstage

            if required.issubset(set(matched_downstage)) and invasion_ok:
                is_downstaged = True
                downstaged_files.append(file_name)
                counts["Downstaged"] += 1
                output_lines.append("✅ Downstaged")
            elif "Ovary Unilateral + No Capsule + <50% Invasion" in matched_downstage:
                is_downstaged = True
                ovary_exception_files.append(file_name)
                counts["Ovary Exception"] += 1
                output_lines.append("✅ Downstaged (Ovary Exception)")
            elif len(required.intersection(set(matched_downstage))) >= 2 and invasion_ok:
                is_downstaged = True
                likely_downstaged_files.append(file_name)
                counts["Likely Downstaged"] += 1
                output_lines.append("⚠️ Likely Downstaged")

            # Conflict: needs clinical review
            if is_upstaged and is_downstaged:
                clinical_review_files.append(file_name)
                counts["Clinical Review"] += 1
                output_lines.append("🩺 Flagged for Clinical Review")
            elif is_upstaged:
                upstaged_files.append(file_name)
                counts["Upstaged"] += 1
                output_lines.append("📌 Final: Upstaged")
            elif is_downstaged:
                pass  # Already counted in downstaging categories
            else:
                no_change_files.append(file_name)
                counts["No Change"] += 1
                output_lines.append("📌 Final: No Change")

            with open(os.path.join(output_directory, f"{file_name}_staging_result.txt"), "w") as out_f:
                out_f.write("\n".join(output_lines))

    return counts

# Run the enhanced processor
summary = process_reports_with_dual_check()

# Display all results in a DataFrame
df = pd.DataFrame({
    "Downstaged": pd.Series(downstaged_files),
    "Likely Downstaged": pd.Series(likely_downstaged_files),
    "Ovary Exception": pd.Series(ovary_exception_files),
    "Upstaged": pd.Series(upstaged_files),
    "No Change": pd.Series(no_change_files),
    "Clinical Review": pd.Series(clinical_review_files)
})
from IPython.display import display
display(df)
# Create DataFrames for each category
df_upstaged = pd.DataFrame({"Upstaged Files": upstage_only})
df_downstaged = pd.DataFrame({"Downstaged Files": downstage_only})
df_clinical_review = pd.DataFrame({"Clinical Review Files (Both)": both_staged})
df_no_change = pd.DataFrame({"No Change Files": no_change})

# Display each table in Colab
from IPython.display import display, Markdown

display(Markdown("### ✅ Upstaged Cases"))
display(df_upstaged)

display(Markdown("### ✅ Downstaged Cases"))
display(df_downstaged)

display(Markdown("### 🩺 Clinical Review Needed"))
display(df_clinical_review)

display(Markdown("### 📌 No Change"))
display(df_no_change)


