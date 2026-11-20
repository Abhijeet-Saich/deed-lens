"""Generate the fictional sample land records used by the demo, the tests and the evaluation.

    python samples/make_samples.py        (or: make samples)

Everything here is invented: Fairhaven County, the Town of Alder Falls, every person,
address, parcel and recording reference. Every page carries the banner
"FICTIONAL SAMPLE DOCUMENT — NOT A REAL RECORD" at the top and bottom.

The set is small but built to exercise each rule DeedLens learned on real records:

  Parcel A 12.04-3-7  (Lot 7) - five documents forming a chain of title:
    Fore Jud 598 12    1974 tax foreclosure judgment; several parcels in an exhibit table
    Tax Deed 612 118   1975 tax deed, small real price          (SCANNED: image only)
                       cites the judgment at Page 14, its exhibit page -> page-range link
    Deed 1043 77       1988 quitclaim, nominal $10              (SCANNED: image only)
                       has NO tax map number -> parcel must propagate along the chain
    Deed 2218 401      2004 warranty deed, nominal $10; real price + transfer tax on the
                       clerk's cover page (page 401; deed text starts on 402)
    Deed 3307 52       2019 warranty deed, nominal $1; cover page; address, acreage,
                       a prior owner's death; cites "Book 2218 ... Page 402" -> page-range link
  Parcel B 12.04-3-8  (Lot 8) - two look-alike deeds (1996, 2012) -> parcel filter
  Parcel C 12.04-3-9  (Lot 9) - one look-alike deed (2015) -> more crowding

Output is deterministic (reportlab invariant mode, fixed noise seeds), so re-running the
script reproduces the committed PDFs instead of creating git churn.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageFilter
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

rl_config.invariant = 1  # no timestamps or random ids in the PDF -> reproducible files

BANNER = "FICTIONAL SAMPLE DOCUMENT — NOT A REAL RECORD"
OUT_DIR = Path(__file__).resolve().parent
SCAN_DPI = 200

# A page is (font kind, blocks). Blocks are small tuples rendered by `flowables()`:
#   ("title", text) ("center", text) ("p", text) ("small", text) ("space", points)
#   ("sig", [names])  ("table", rows, col_widths_in_inches, has_header_row)
Block = tuple
Page = tuple[str, list[Block]]

FONTS = {  # kind: (regular, bold, size, leading)
    "typed": ("Courier", "Courier-Bold", 10, 13.2),  # typewriter look for the 1970s/80s documents
    "modern": ("Times-Roman", "Times-Bold", 10.5, 13.4),
    "form": ("Helvetica", "Helvetica-Bold", 9.5, 12.5),  # clerk cover pages
}


# ---------------------------------------------------------------- money helpers

def money(x: float) -> str:
    return f"${x:,.2f}"


def ny_transfer_tax(amount: float) -> float:
    """New York real estate transfer tax: $2 per $500 of consideration or fraction thereof."""
    return math.ceil(amount / 500) * 2.0


# ---------------------------------------------------------------- shared wording

CLERK_1975 = "Herbert J. Lowe"
CLERK_1988 = "Dolores K. Whitcombe"
CLERK_2004 = "Rosalind M. Achterberg"
CLERK_2012 = "Theodore N. Vance"

MAP_214 = ("the Map of Juniper Hollow Subdivision filed in the Fairhaven County Clerk's Office "
           "on May 3, 1962 as Map No. 214")

# Road frontage per lot, in feet. Every lot is 210.00 feet deep, so frontage fixes the area:
# Lot 7 -> 0.62 acres, Lot 8 -> 0.71 acres, Lot 9 -> 0.58 acres.
FRONTAGE = {7: 128.50, 8: 147.30, 9: 120.30}


def premises(lot: int, *, tax_map: str | None = None, acres: str | None = None,
             address: str | None = None) -> str:
    """Legal description. Neighbouring lots get near-identical wording on purpose."""
    f = FRONTAGE[lot]
    text = (f"ALL THAT TRACT OR PARCEL OF LAND, situate in the Town of Alder Falls, County of Fairhaven "
            f"and State of New York, known and designated as Lot No. {lot} on {MAP_214}, bounded and "
            f"described as follows: BEGINNING at an iron pin set in the westerly line of Juniper Hollow "
            f"Road at the division line between Lots {lot - 1} and {lot}; running thence westerly along "
            f"said division line 210.00 feet; thence northerly parallel with Juniper Hollow Road "
            f"{f:.2f} feet; thence easterly along the division line between Lots {lot} and {lot + 1}, "
            f"210.00 feet to the westerly line of Juniper Hollow Road; thence southerly along said road "
            f"line {f:.2f} feet to the point or place of beginning.")
    if acres:
        text += f" Containing {acres} acres of land, more or less."
    if address:
        text += f" Being known as {address}."
    if tax_map:
        text += f" {tax_map}"
    return text


def acknowledgment(date: str, names: str, notary: str) -> list[Block]:
    return [
        ("p", "STATE OF NEW YORK, COUNTY OF FAIRHAVEN, ss.:"),
        ("p", f"On the {date}, before me, the undersigned, a Notary Public in and for said State, "
              f"personally appeared {names}, personally known to me or proved to me on the basis of "
              "satisfactory evidence to be the individual(s) whose name(s) is (are) subscribed to the "
              "within instrument and acknowledged to me that they executed the same in their "
              "capacity(ies), and that by their signature(s) on the instrument, the individual(s) "
              "executed the instrument."),
        ("sig", [f"{notary}, Notary Public, State of New York"]),
    ]


def cover_page(*, book: int, page: int, num_pages: int, control: str, recorded: str,
               grantors: list[str], grantees: list[str], amount: float, recording_fee: float,
               tax_map: str, clerk: str) -> Page:
    """Clerk's recording cover page: the only place the real price and transfer tax appear."""
    tax = ny_transfer_tax(amount)
    return ("form", [
        ("title", "FAIRHAVEN COUNTY CLERK"),
        ("center", "County Office Building, 1 Courthouse Square, Alder Falls, New York"),
        ("space", 4),
        ("title", "RECORDING AND ENDORSEMENT COVER PAGE"),
        ("space", 8),
        ("table", [["Instrument Type:", "DEED"],
                   ["Book / Page:", f"{book} / {page}"],
                   ["Number of Pages:", str(num_pages)],
                   ["Control Number:", control],
                   ["Recorded Date / Time:", recorded],
                   ["Municipality:", "Town of Alder Falls"],
                   ["Tax Map No. (SBL):", tax_map]], [2.2, 4.3], False),
        ("space", 10),
        ("table", [["GRANTOR(S)", "GRANTEE(S)"], ["\n".join(grantors), "\n".join(grantees)]],
         [3.25, 3.25], True),
        ("space", 10),
        ("table", [["MTG/DEED AMOUNT:", money(amount)],
                   ["Consideration:", money(amount)],
                   ["Transfer Tax:", money(tax)],
                   ["Recording Fee:", money(recording_fee)],
                   ["Total Fees Paid:", money(tax + recording_fee)]], [2.2, 4.3], False),
        ("space", 12),
        ("p", "THIS PAGE IS PART OF THE INSTRUMENT. DO NOT DETACH. The County Clerk's endorsement on "
              "this page certifies the recording information above; it does not certify the validity "
              "of the instrument."),
        ("sig", [f"{clerk}, Fairhaven County Clerk"]),
    ])


def warranty_deed(*, date: str, grantors: str, grantors_plural: bool, grantees: str,
                  grantees_plural: bool, consideration: str, nominal: bool, description: str,
                  being: str, signers: list[str], ack_date: str, notary: str, return_to: str,
                  recording_note: str | None = None) -> list[Page]:
    """Two pages of modern warranty-deed text. Parcels A, B and C all share this wording."""
    first = "parties" if grantors_plural else "party"
    second = "parties" if grantees_plural else "party"
    verb, has, is_ = ("do", "have", "are") if grantors_plural else ("does", "has", "is")
    other = " and other good and valuable consideration" if nominal else ""
    page1 = ("modern", [
        ("title", "WARRANTY DEED WITH FULL COVENANTS"),
        ("center", "CONSULT YOUR LAWYER BEFORE SIGNING THIS INSTRUMENT."),
        ("space", 6),
        ("p", f"THIS INDENTURE, made the {date}, BETWEEN {grantors}, {first} of the first part, and "
              f"{grantees}, {second} of the second part,"),
        ("p", f"WITNESSETH, that the {first} of the first part, in consideration of {consideration}"
              f"{other} paid by the {second} of the second part, {verb} hereby grant and release unto "
              f"the {second} of the second part, their heirs and assigns forever,"),
        ("p", description),
        ("p", being),
    ])
    page2_blocks: list[Block] = [
        ("p", "TOGETHER with all right, title and interest, if any, of the "
              f"{first} of the first part in and to any streets and roads abutting the above described "
              "premises to the center lines thereof; TOGETHER with the appurtenances and all the estate "
              f"and rights of the {first} of the first part in and to said premises; TO HAVE AND TO "
              f"HOLD the premises herein granted unto the {second} of the second part, their heirs and "
              "assigns forever."),
        ("p", f"AND the {first} of the first part covenant(s) as follows: FIRST, that the {first} of "
              f"the first part {is_} seized of the said premises in fee simple, and {has} good right to "
              f"convey the same; SECOND, that the {second} of the second part shall quietly enjoy the "
              "said premises; THIRD, that the said premises are free from incumbrances; FOURTH, that "
              f"the {first} of the first part will execute or procure any further necessary assurance "
              f"of the title to said premises; FIFTH, that the {first} of the first part will forever "
              "warrant the title to said premises. The grantor(s), in compliance with Section 13 of "
              "the Lien Law, will receive the consideration for this conveyance as a trust fund to be "
              "applied first for the purpose of paying the cost of any improvement."),
        ("p", f"IN WITNESS WHEREOF, the {first} of the first part {has} duly executed this deed the day "
              "and year first above written."),
        ("sig", signers),
        *acknowledgment(ack_date, " and ".join(signers), notary),
        ("small", f"Record and return to: {return_to}"),
    ]
    if recording_note:
        page2_blocks.append(("small", recording_note))
    return [page1, ("modern", page2_blocks)]


# ---------------------------------------------------------------- the documents

@dataclass
class Sample:
    file: str
    book: int
    first_page: int
    pages: list[Page]
    scanned: bool = False
    seed: int = 0


def judgment_1974() -> Sample:
    exhibit = [
        ["Tax Map No.", "Reputed Owner", "Location", "Amount Due"],
        ["12.04-3-7", "Harlan Voss", "Lot 7, Juniper Hollow Road", "$412.37"],
        ["12.04-3-11", "Estate of Ida Brandt", "Lot 11, Juniper Hollow Road", "$198.05"],
        ["15.02-1-4", "Cyrus Mendel", "Quarry Lane", "$1,067.40"],
        ["15.02-1-9", "Unknown Owner", "Quarry Lane", "$76.12"],
        ["", "", "TOTAL", "$1,753.94"],
    ]
    return Sample("Fore Jud 598 12.pdf", 598, 12, [
        ("typed", [
            ("center", "STATE OF NEW YORK"),
            ("center", "COUNTY COURT : COUNTY OF FAIRHAVEN"),
            ("space", 8),
            ("p", "In the Matter of the Foreclosure of Tax Liens by Proceeding In Rem by the COUNTY OF "
                  "FAIRHAVEN, Petitioner, against the parcels of real property listed in Exhibit A."),
            ("center", "Index No. 1974-0381"),
            ("space", 6),
            ("title", "JUDGMENT OF FORECLOSURE"),
            ("space", 6),
            ("p", "The County of Fairhaven, by Gerald P. Hume, County Treasurer, having filed a petition "
                  "of foreclosure of tax liens on the parcels of real property listed in Exhibit A "
                  "annexed hereto, for taxes levied for the years 1970, 1971 and 1972 and remaining "
                  "unpaid; and notice of the proceeding having been published in the Alder Falls Courier "
                  "and mailed to the reputed owners as required by law; and the period for redemption "
                  "having expired on September 30, 1974 without any of the said parcels having been "
                  "redeemed; and no answer having been interposed;"),
            ("p", "NOW, on motion of Lorraine D. Abbot, Esq., County Attorney, it is"),
        ]),
        ("typed", [
            ("p", "ORDERED, ADJUDGED AND DECREED that the County of Fairhaven is awarded possession of "
                  "each parcel of real property listed in Exhibit A, and shall be seized of an estate in "
                  "fee simple absolute in each such parcel; and it is further"),
            ("p", "ORDERED, ADJUDGED AND DECREED that the reputed owners named in Exhibit A, and all "
                  "persons claiming through them, are forever barred and foreclosed of all right, title "
                  "and interest in each such parcel; and it is further"),
            ("p", "ORDERED that the amounts due listed in Exhibit A, totalling $1,753.94, are the unpaid "
                  "taxes, interest and charges on each parcel as of the date of this judgment."),
            ("space", 6),
            ("p", "Dated: October 30, 1974"),
            ("sig", ["ENTER: Hon. Miriam T. Calloway, County Judge"]),
            ("space", 10),
            ("small", "Recorded in the Fairhaven County Clerk's Office on November 18, 1974 at 2:15 P.M. "
                      f"in Book 598 of Deeds at Page 12. {CLERK_1975}, County Clerk."),
        ]),
        ("typed", [
            ("title", "EXHIBIT A"),
            ("center", "Parcels Subject to Judgment of Foreclosure, Index No. 1974-0381"),
            ("space", 8),
            ("table", exhibit, [1.2, 1.8, 2.3, 1.2], True),
        ]),
    ])


def tax_deed_1975() -> Sample:
    return Sample("Tax Deed 612 118.pdf", 612, 118, scanned=True, seed=1975, pages=[
        ("typed", [
            ("title", "TAX DEED"),
            ("space", 6),
            ("p", "THIS INDENTURE, made the 14th day of April, nineteen hundred seventy-five, BETWEEN "
                  "the COUNTY OF FAIRHAVEN, a municipal corporation of the State of New York, acting by "
                  "Gerald P. Hume, its County Treasurer, party of the first part, and WALTER PELLEGRINO "
                  "and EDITH PELLEGRINO, his wife, both residing at 3 Mill Street, Alder Falls, New York, "
                  "as tenants by the entirety, parties of the second part."),
            ("p", "WITNESSETH, that the party of the first part, in consideration of the sum of EIGHT "
                  "HUNDRED FIFTY AND 00/100 DOLLARS ($850.00), lawful money of the United States, paid by "
                  "the parties of the second part as the highest bidders at the public auction of "
                  "tax-foreclosed property held by the County of Fairhaven on March 8, 1975, does hereby "
                  "grant and release unto the parties of the second part, their heirs and assigns "
                  "forever,"),
            ("p", premises(7, tax_map="Designated on the tax map of the Town of Alder Falls as Section "
                                      "12.04, Block 3, Lot 7 (Tax Map No. 12.04-3-7).")),
        ]),
        ("typed", [
            # Page 14 is the judgment's exhibit page, not its first page (12): page-range link.
            ("p", "BEING the same premises acquired by the County of Fairhaven by judgment of "
                  "foreclosure of tax liens, Index No. 1974-0381, dated October 30, 1974 and recorded in "
                  "the Fairhaven County Clerk's Office on November 18, 1974 in Book 598 of Deeds at Page "
                  "14, the said premises having been assessed to Harlan Voss."),
            ("p", "This conveyance is made without covenants of any kind, the party of the first part "
                  "conveying only such title as it acquired by the said judgment."),
            ("p", "IN WITNESS WHEREOF, the party of the first part has caused this instrument to be "
                  "signed by its County Treasurer the day and year first above written."),
            ("sig", ["COUNTY OF FAIRHAVEN, By: Gerald P. Hume, County Treasurer"]),
            ("p", "STATE OF NEW YORK, COUNTY OF FAIRHAVEN, ss.: On the 14th day of April, 1975, before me "
                  "personally came Gerald P. Hume, to me known, who being by me duly sworn did depose and "
                  "say that he is the County Treasurer of the County of Fairhaven, the corporation "
                  "described in and which executed the foregoing instrument."),
            ("sig", ["Beatrice L. Sorensen, Notary Public"]),
            ("small", "Recorded April 22, 1975 at 11:05 A.M. in Book 612 of Deeds at Page 118. "
                      f"{CLERK_1975}, County Clerk."),
        ]),
    ])


def quitclaim_1988() -> Sample:
    return Sample("Deed 1043 77.pdf", 1043, 77, scanned=True, seed=1988, pages=[
        ("typed", [
            ("title", "QUITCLAIM DEED"),
            ("space", 6),
            ("p", "THIS INDENTURE, made the 2nd day of June, 1988, BETWEEN WALTER PELLEGRINO and EDITH "
                  "PELLEGRINO, his wife, both residing at 14 Juniper Hollow Road, Alder Falls, New York, "
                  "parties of the first part, and EDITH PELLEGRINO, residing at 14 Juniper Hollow Road, "
                  "Alder Falls, New York, and MARTA QUILL, residing at 88 Ferris Avenue, Harrow Point, "
                  "New York, as joint tenants with right of survivorship, parties of the second part."),
            ("p", "WITNESSETH, that the parties of the first part, in consideration of TEN AND 00/100 "
                  "DOLLARS ($10.00), lawful money of the United States, and other good and valuable "
                  "consideration paid by the parties of the second part, do hereby remise, release and "
                  "quitclaim unto the parties of the second part, their heirs and assigns forever,"),
            # No tax map number on purpose: this deed joins Parcel A only through the chain.
            ("p", premises(7)),
        ]),
        ("typed", [
            ("p", "BEING the same premises conveyed to the parties of the first part by the County of "
                  "Fairhaven by tax deed dated April 14, 1975 and recorded in the Fairhaven County "
                  "Clerk's Office on April 22, 1975 in Book 612 of Deeds at Page 118."),
            ("p", "TOGETHER with the appurtenances and all the estate and rights of the parties of the "
                  "first part in and to said premises. TO HAVE AND TO HOLD the premises herein granted "
                  "unto the parties of the second part, their heirs and assigns forever."),
            ("p", "IN WITNESS WHEREOF, the parties of the first part have hereunto set their hands and "
                  "seals the day and year first above written."),
            ("sig", ["Walter Pellegrino", "Edith Pellegrino"]),
            *acknowledgment("2nd day of June, 1988", "Walter Pellegrino and Edith Pellegrino",
                            "Harold V. Mertz"),
            ("small", "Recorded June 10, 1988 at 9:40 A.M. in Book 1043 of Deeds at Page 77. "
                      f"{CLERK_1988}, County Clerk."),
        ]),
    ])


def deed_2004() -> Sample:
    return Sample("Deed 2218 401.pdf", 2218, 401, pages=[
        cover_page(book=2218, page=401, num_pages=3, control="2004-018832",
                   recorded="08/27/2004 10:42 AM", grantors=["QUILL, MARTA"],
                   grantees=["OKAFOR, DESMOND", "OKAFOR, PRIYA"], amount=142_500, recording_fee=45,
                   tax_map="12.04-3-7", clerk=CLERK_2004),
        *warranty_deed(
            date="20th day of August, 2004",
            grantors="MARTA QUILL, residing at 88 Ferris Avenue, Harrow Point, New York, as surviving "
                     "joint tenant",
            grantors_plural=False,
            grantees="DESMOND OKAFOR and PRIYA OKAFOR, husband and wife, residing at 230 Harbor View "
                     "Drive, Elmsted, New York, as tenants by the entirety",
            grantees_plural=True,
            consideration="TEN AND 00/100 DOLLARS ($10.00)", nominal=True,
            description=premises(7, tax_map="Tax Map No. 12.04-3-7."),
            being="BEING the same premises conveyed to Edith Pellegrino and Marta Quill, as joint tenants "
                  "with right of survivorship, by Walter Pellegrino and Edith Pellegrino by quitclaim "
                  "deed dated June 2, 1988 and recorded June 10, 1988 in Book 1043 of Deeds at Page 77. "
                  "Edith Pellegrino died on January 11, 2002, whereupon title vested in Marta Quill as "
                  "surviving joint tenant.",
            signers=["Marta Quill"], ack_date="20th day of August, 2004", notary="Camille R. Duquesne",
            return_to="Hollis & Brandvold, LLP, 40 Market Street, Alder Falls, New York"),
    ])


def deed_2019() -> Sample:
    return Sample("Deed 3307 52.pdf", 3307, 52, pages=[
        cover_page(book=3307, page=52, num_pages=3, control="2019-011206",
                   recorded="05/22/2019 03:08 PM", grantors=["OKAFOR, DESMOND"],
                   grantees=["MARSH, LENA", "RENN, TOBIAS"], amount=236_000, recording_fee=165,
                   tax_map="12.04-3-7", clerk=CLERK_2012),
        *warranty_deed(
            date="15th day of May, 2019",
            grantors="DESMOND OKAFOR, residing at 14 Juniper Hollow Road, Alder Falls, New York",
            grantors_plural=False,
            grantees="LENA MARSH and TOBIAS RENN, both residing at 27 Wren Street, Elmsted, New York, "
                      "as joint tenants with right of survivorship",
            grantees_plural=True,
            consideration="ONE AND 00/100 DOLLAR ($1.00)", nominal=True,
            description=premises(7, acres="0.62", address="14 Juniper Hollow Road, Alder Falls, New York",
                                 tax_map="Tax Map No. 12.04-3-7."),
            # Cites Page 402: the deed text, one page after the cover page (401). Page-range link.
            being="BEING the same premises conveyed to Desmond Okafor and Priya Okafor, as tenants by the "
                  "entirety, by Marta Quill by deed dated August 20, 2004 and recorded August 27, 2004 in "
                  "Book 2218 of Deeds at Page 402. Priya Okafor died on February 6, 2017, whereupon title "
                  "vested in Desmond Okafor as surviving tenant by the entirety.",
            signers=["Desmond Okafor"], ack_date="15th day of May, 2019", notary="Anselm J. Whitlock",
            return_to="Pruett Title Agency, 12 Depot Square, Elmsted, New York"),
    ])


def deed_1996_b() -> Sample:
    price = 71_500
    return Sample("Deed 1580 233.pdf", 1580, 233, pages=warranty_deed(
        date="9th day of September, 1996",
        grantors="GORDON FAYE, residing at 16 Juniper Hollow Road, Alder Falls, New York",
        grantors_plural=False,
        grantees="RUTH ABERNATHY and SAMUEL ABERNATHY, wife and husband, residing at 52 Orchard Terrace, "
                  "Elmsted, New York, as tenants by the entirety",
        grantees_plural=True,
        consideration="SEVENTY-ONE THOUSAND FIVE HUNDRED AND 00/100 DOLLARS ($71,500.00)", nominal=False,
        description=premises(8, acres="0.71", address="16 Juniper Hollow Road, Alder Falls, New York",
                             tax_map="Tax Map No. 12.04-3-8."),
        being="BEING the same premises conveyed to Gordon Faye by deed dated July 1, 1981 and recorded "
              "July 8, 1981 in Book 905 of Deeds at Page 340.",
        signers=["Gordon Faye"], ack_date="9th day of September, 1996", notary="Harold V. Mertz",
        return_to="Ruth and Samuel Abernathy, 16 Juniper Hollow Road, Alder Falls, New York",
        recording_note=f"Recorded September 16, 1996 at 1:52 P.M. in Book 1580 of Deeds at Page 233. "
                       f"Real Estate Transfer Tax paid: {money(ny_transfer_tax(price))}. "
                       f"{CLERK_1988}, County Clerk.",
    ))


def deed_2012_b() -> Sample:
    return Sample("Deed 2874 610.pdf", 2874, 610, pages=[
        cover_page(book=2874, page=610, num_pages=3, control="2012-004517",
                   recorded="03/09/2012 01:17 PM", grantors=["ABERNATHY, RUTH", "ABERNATHY, SAMUEL"],
                   grantees=["OSTROWSKI, CALVIN"], amount=189_900, recording_fee=125,
                   tax_map="12.04-3-8", clerk=CLERK_2012),
        *warranty_deed(
            date="2nd day of March, 2012",
            grantors="RUTH ABERNATHY and SAMUEL ABERNATHY, wife and husband, both residing at 16 Juniper "
                     "Hollow Road, Alder Falls, New York",
            grantors_plural=True,
            grantees="CALVIN OSTROWSKI, residing at 9 Larkspur Court, Elmsted, New York",
            grantees_plural=False,
            consideration="TEN AND 00/100 DOLLARS ($10.00)", nominal=True,
            description=premises(8, acres="0.71", address="16 Juniper Hollow Road, Alder Falls, New York",
                                 tax_map="Tax Map No. 12.04-3-8."),
            being="BEING the same premises conveyed to Ruth Abernathy and Samuel Abernathy by Gordon Faye "
                  "by deed dated September 9, 1996 and recorded September 16, 1996 in Book 1580 of Deeds "
                  "at Page 233.",
            signers=["Ruth Abernathy", "Samuel Abernathy"], ack_date="2nd day of March, 2012",
            notary="Anselm J. Whitlock",
            return_to="Pruett Title Agency, 12 Depot Square, Elmsted, New York"),
    ])


def deed_2015_c() -> Sample:
    return Sample("Deed 3105 288.pdf", 3105, 288, pages=[
        cover_page(book=3105, page=288, num_pages=3, control="2015-007733",
                   recorded="10/14/2015 11:26 AM", grantors=["FERREIRA, NADIA"],
                   grantees=["BLACKWOOD, OWEN", "BLACKWOOD, JUN"], amount=205_000, recording_fee=145,
                   tax_map="12.04-3-9", clerk=CLERK_2012),
        *warranty_deed(
            date="6th day of October, 2015",
            grantors="NADIA FERREIRA, residing at 18 Juniper Hollow Road, Alder Falls, New York",
            grantors_plural=False,
            grantees="OWEN BLACKWOOD and JUN BLACKWOOD, a married couple, residing at 301 Canal Street, "
                     "Harrow Point, New York, as tenants by the entirety",
            grantees_plural=True,
            consideration="ONE AND 00/100 DOLLAR ($1.00)", nominal=True,
            description=premises(9, acres="0.58", address="18 Juniper Hollow Road, Alder Falls, New York",
                                 tax_map="Tax Map No. 12.04-3-9."),
            being="BEING the same premises conveyed to Nadia Ferreira by deed dated April 18, 2007 and "
                  "recorded April 25, 2007 in Book 2290 of Deeds at Page 15.",
            signers=["Nadia Ferreira"], ack_date="6th day of October, 2015", notary="Camille R. Duquesne",
            return_to="Hollis & Brandvold, LLP, 40 Market Street, Alder Falls, New York"),
    ])


def all_samples() -> list[Sample]:
    return [judgment_1974(), tax_deed_1975(), quitclaim_1988(), deed_2004(), deed_2019(),
            deed_1996_b(), deed_2012_b(), deed_2015_c()]


# ---------------------------------------------------------------- rendering

def _styles(kind: str) -> dict[str, ParagraphStyle]:
    regular, bold, size, leading = FONTS[kind]
    base = ParagraphStyle(kind, fontName=regular, fontSize=size, leading=leading,
                          alignment=TA_JUSTIFY, spaceAfter=leading * 0.55)
    return {
        "p": base,
        "title": ParagraphStyle(kind + "t", parent=base, fontName=bold, fontSize=size + 3,
                                leading=leading + 4, alignment=TA_CENTER),
        "center": ParagraphStyle(kind + "c", parent=base, alignment=TA_CENTER, spaceAfter=2),
        "small": ParagraphStyle(kind + "s", parent=base, fontSize=size - 1.5, leading=leading - 2),
        "sig": ParagraphStyle(kind + "g", parent=base, leftIndent=3.0 * inch, alignment=0, spaceAfter=0),
        "cell": ParagraphStyle(kind + "x", parent=base, alignment=0, spaceAfter=0),
        "head": ParagraphStyle(kind + "h", parent=base, fontName=bold, alignment=0, spaceAfter=0),
    }


def _para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def flowables(kind: str, blocks: list[Block]) -> list:
    st = _styles(kind)
    out: list = []
    for block in blocks:
        tag = block[0]
        if tag == "space":
            out.append(Spacer(1, block[1]))
        elif tag == "sig":
            for name in block[1]:
                out += [Spacer(1, 16), _para("_" * 34, st["sig"]), _para(name, st["sig"])]
            out.append(Spacer(1, 8))
        elif tag == "table":
            _, rows, widths, header = block
            cells = [[_para(c, st["head"] if header and r == 0 else st["cell"]) for c in row]
                     for r, row in enumerate(rows)]
            table = Table(cells, colWidths=[w * inch for w in widths])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            out.append(table)
        else:
            out.append(_para(block[1], st[tag]))
    return out


def build_pdf(sample: Sample) -> bytes:
    """Text-layer PDF for one sample. Raises if the text overflowed the planned page count."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=inch, rightMargin=inch,
                            topMargin=1.1 * inch, bottomMargin=0.9 * inch,
                            title=f"Fictional sample record: {sample.file}",
                            author="DeedLens sample generator", invariant=1)
    story: list = []
    for i, (kind, blocks) in enumerate(sample.pages):
        if i:
            story.append(PageBreak())
        story += flowables(kind, blocks)

    def decorate(canvas, _doc) -> None:
        width, height = letter
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#9b1c1c"))
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(width / 2, height - 0.45 * inch, BANNER)
        canvas.drawCentredString(width / 2, 0.42 * inch, BANNER)
        canvas.setFillColor(colors.black)
        canvas.setFont("Courier-Bold", 11)
        stamp = f"BK {sample.book} PG {sample.first_page + canvas.getPageNumber() - 1}"
        canvas.drawRightString(width - inch, height - 0.8 * inch, stamp)
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    data = buf.getvalue()
    pages = len(pdfium.PdfDocument(data))
    if pages != len(sample.pages):
        raise RuntimeError(f"{sample.file}: planned {len(sample.pages)} pages, got {pages}; shorten text")
    return data


def scan(pdf_bytes: bytes, seed: int) -> list[Image.Image]:
    """Make a text PDF look like an old photocopy: grey paper, slight tilt, noise, specks, blur."""
    rng = np.random.default_rng(seed)
    pdf = pdfium.PdfDocument(pdf_bytes)
    images = []
    for i in range(len(pdf)):
        img = pdf[i].render(scale=SCAN_DPI / 72).to_pil().convert("L")
        img = img.rotate(float(rng.uniform(-1.0, 1.0)), resample=Image.BICUBIC, fillcolor=255)
        arr = np.asarray(img, dtype=np.float32) * 0.88 + 22  # toner not pure black, paper not white
        arr += rng.normal(0, 9, arr.shape)
        specks = rng.random(arr.shape) < 0.0005
        arr[specks] = rng.uniform(0, 120, int(specks.sum()))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
        images.append(img.filter(ImageFilter.GaussianBlur(0.6)))
    pdf.close()
    return images


def write_sample(sample: Sample, out_dir: Path = OUT_DIR) -> Path:
    path = out_dir / sample.file
    data = build_pdf(sample)
    if sample.scanned:
        images = scan(data, sample.seed)
        fixed = time.strptime("2026-01-01", "%Y-%m-%d")  # Pillow stamps "now" otherwise
        images[0].save(path, "PDF", save_all=True, append_images=images[1:], resolution=SCAN_DPI,
                       quality=80, title=f"Fictional sample record: {sample.file}",
                       creationDate=fixed, modDate=fixed)
    else:
        path.write_bytes(data)
    return path


def main() -> None:
    for s in all_samples():
        path = write_sample(s)
        kind = "image only" if s.scanned else "text layer"
        print(f"{s.file:<22} {len(s.pages)} pages  {kind:<10}  {path.stat().st_size / 1024:>6.0f} KB")


if __name__ == "__main__":
    main()
