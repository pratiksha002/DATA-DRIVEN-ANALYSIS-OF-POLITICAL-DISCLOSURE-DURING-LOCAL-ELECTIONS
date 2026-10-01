"""
SEC Maharashtra (mahasecelec.in) candidate/affidavit scraper skeleton.

Built from the actual network capture of MasterSearch.aspx for BMC (Bruhan Mumbai).
This is a STARTING POINT, not a finished tool — ASP.NET UpdatePanel sites are fragile
and you will likely need to adjust headers/field names after testing against the live
site. Test on ONE ward first before looping over all of them.

Confirmed constants (from your capture):
    LocalBody          = 5      (MUNICIPAL CORPORATION)
    Division           = 6      (Konkan)
    District           = 518    (Mumbai Suburban)
    MunicipalCorp      = 429    (Bruhan Mumbai)
    ElectionProgramme  = 402    (BMC General Election, poll date 15-01-2026)
    Ward dropdown value = 205497 + ward_number   (e.g. ward 8 -> 205505, ward 9 -> 205506)
        (verify this offset holds for wards beyond what you've seen so far)
"""

import re
import time
import requests
from pathlib import Path

BASE_URL = "https://mahasecelec.in/MasterSearch.aspx"
OUT_DIR = Path("affidavits")
OUT_DIR.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
})

# Fixed selections for this project
LOCAL_BODY = "5"
DIVISION = "6"
DISTRICT = "518"
MUNICIPAL_CORP = "429"
ELECTION_PROGRAMME = "402"

# Wards to pull — edit this list to your final chosen set
# R/Central (Borivali): 8-17, R/South (Kandivali): 18-28
WARD_NUMBERS = list(range(8, 18)) + list(range(18, 29))


def extract_hidden_from_html(html):
    """Extract ASP.NET hidden fields from a normal (non-AJAX) page load."""
    fields = {}
    for name in ["__VIEWSTATE", "__VIEWSTATEGENERATOR", "__VIEWSTATEENCRYPTED", "__EVENTVALIDATION"]:
        m = re.search(rf'id="{name}"[^>]*value="([^"]*)"', html)
        fields[name] = m.group(1) if m else ""
    return fields


def extract_hidden_from_delta(text):
    """Extract ASP.NET hidden fields from an UpdatePanel AJAX 'delta' response."""
    fields = {}
    for name in ["__VIEWSTATE", "__VIEWSTATEGENERATOR", "__VIEWSTATEENCRYPTED", "__EVENTVALIDATION"]:
        m = re.search(rf'hiddenField\|{name}\|([^|]*)\|', text)
        fields[name] = m.group(1) if m else ""
    return fields


def initial_page_load():
    """GET the page fresh to obtain a valid starting VIEWSTATE/EVENTVALIDATION."""
    r = session.get(BASE_URL)
    r.raise_for_status()
    return extract_hidden_from_html(r.text)


def ajax_postback(event_target, extra_fields, hidden):
    """
    Fire an UpdatePanel async postback. `extra_fields` are the current values of
    ALL relevant form controls (dropdowns etc.) that must be resent on every postback —
    ASP.NET WebForms resends full form state each time, not just the changed field.
    """
    payload = {
        "ctl00$ScriptManager1": "ctl00$ContentPlaceHolder1$UpdatePanel1|" + event_target,
        "__EVENTTARGET": event_target,
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
        "__VIEWSTATE": hidden["__VIEWSTATE"],
        "__VIEWSTATEGENERATOR": hidden["__VIEWSTATEGENERATOR"],
        "__VIEWSTATEENCRYPTED": hidden["__VIEWSTATEENCRYPTED"],
        "__EVENTVALIDATION": hidden["__EVENTVALIDATION"],
        "__ASYNCPOST": "true",
    }
    payload.update(extra_fields)

    headers = {
        "X-MicrosoftAjax": "Delta=true",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": BASE_URL,
    }
    r = session.post(BASE_URL, data=payload, headers=headers)
    r.raise_for_status()
    return r.text


def get_ward_candidates(ward_number):
    """Full flow: fresh load -> select all filters -> select ward -> search -> parse table."""
    hidden = initial_page_load()

    common_fields = {
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLLocalBody": LOCAL_BODY,
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLDivision": DIVISION,
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLDistrict": DISTRICT,
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLMunicipalcorporation": MUNICIPAL_CORP,
        "ctl00$ContentPlaceHolder1$SearchControl1$ddlEP": ELECTION_PROGRAMME,
        "ctl00$ContentPlaceHolder1$SearchControl1$hdnElectionProgram": ELECTION_PROGRAMME,
        "ctl00$ContentPlaceHolder1$SearchControl1$hdnLocalBody": MUNICIPAL_CORP,
    }

    # Step through each dropdown's onchange postback so the server-side session
    # state advances correctly. This mirrors what your browser did automatically.
    for target in [
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLLocalBody",
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLDivision",
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLDistrict",
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLMunicipalcorporation",
    ]:
        text = ajax_postback(target, common_fields, hidden)
        hidden = extract_hidden_from_delta(text)
        time.sleep(0.5)  # be polite to a government server

    ward_value = str(205497 + ward_number)
    ward_fields = dict(common_fields)
    ward_fields["ctl00$ContentPlaceHolder1$SearchControl1$DDLWARDGP"] = ward_value

    text = ajax_postback(
        "ctl00$ContentPlaceHolder1$SearchControl1$DDLWARDGP", ward_fields, hidden
    )
    hidden = extract_hidden_from_delta(text)
    time.sleep(0.5)

    # Now trigger Search
    search_fields = dict(ward_fields)
    search_fields["ctl00$ContentPlaceHolder1$btnSearch"] = "Search"
    text = ajax_postback("ctl00$ContentPlaceHolder1$btnSearch", search_fields, hidden)
    hidden = extract_hidden_from_delta(text)

    # Parse candidate rows: reg no, name, whether affidavit is downloadable
    rows = re.findall(
        r'lblRegistrationNo_(\d+)">([^<]+)</span>.*?lblname_\1">([^<]+)</span>.*?'
        r'(lnkDownload_\1"[^>]*>(View Affidavit|NA))',
        text, re.DOTALL
    )
    candidates = []
    for idx, reg_no, name, link_html, status in rows:
        candidates.append({
            "ward": ward_number,
            "index": idx,
            "reg_no": reg_no.strip(),
            "name": name.strip(),
            "affidavit_available": status == "View Affidavit",
        })
    return candidates, hidden


if __name__ == "__main__":
    import csv

    all_candidates = []
    for ward in WARD_NUMBERS:
        print(f"Fetching ward {ward}...")
        try:
            candidates, _ = get_ward_candidates(ward)
            all_candidates.extend(candidates)
            print(f"  -> {len(candidates)} candidates, "
                  f"{sum(c['affidavit_available'] for c in candidates)} with affidavit")
        except Exception as e:
            print(f"  !! failed on ward {ward}: {e}")
        time.sleep(1)  # rate-limit between wards

    with open(OUT_DIR / "candidate_list.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ward", "index", "reg_no", "name", "affidavit_available"])
        writer.writeheader()
        writer.writerows(all_candidates)

    print(f"\nDone. {len(all_candidates)} total candidates written to "
          f"{OUT_DIR / 'candidate_list.csv'}")

    # NEXT STEP (not yet implemented here): downloading individual affidavit PDFs.
    # The "View Affidavit" link is ALSO a postback (lnkDownload), not a direct URL —
    # e.g. __doPostBack('ctl00$ContentPlaceHolder1$GVData$ctl02$lnkDownload','')
    # You'll need to fire that as another ajax_postback() using the row's specific
    # control ID (ctl02, ctl03... increments per row) and inspect whether the response
    # is raw PDF bytes or a redirect/new-window trigger. Test this manually on one
    # candidate first — print response headers' Content-Type to see what comes back.
