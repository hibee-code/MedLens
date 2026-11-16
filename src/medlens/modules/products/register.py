"""Normalising register (Greenbook) and pack text so the two can be compared: NAFDAC numbers,
strengths, dosage forms, active-ingredient names and company names.

The register writes multi-ingredient products as parallel lists separated by "; " ("Artemether;
Lumefantrine" / "20 mg; 120 mg"), salts in parentheses ("Amoxicillin (Amoxicillin Trihydrate)")
and concentrations per volume ("125 mg/5 mL"). Packs write the same facts loosely ("Amoxicillin
trihydrate 500mg", "Artemether/Lumefantrine 20/120 mg", "0.5 g"). Every function here is pure
and deterministic; comparisons happen on the normalised values.

Equal strengths: "500mg", "500 mg" and "0.5 g" (masses in mg); "125 mg/5 ml" and "25 mg/mL"
(concentrations per mL); "100 mcg/Metered-Dose" and "100 mcg/dose"; "1%" and "1 %".
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from medlens.core.identifiers import normalize_nafdac_number

__all__ = [
    "KNOWN_INGREDIENTS",
    "Ingredient",
    "Strength",
    "canonical_text",
    "dosage_form_family",
    "dosage_form_mentions",
    "ingredients_from_json",
    "normalize_company",
    "normalize_ingredient",
    "normalize_nafdac_number",
    "parse_ingredients",
    "parse_strength",
    "parse_strengths",
    "same_strength",
]


def _fold(text: str) -> str:
    """Lower case, accents removed (µ kept as u), whitespace collapsed."""
    plain = unicodedata.normalize("NFKD", text.replace("µ", "u").replace("μ", "u"))
    ascii_text = plain.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", ascii_text.lower()).strip()


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# --------------------------------------------------------------------------- strengths
_MASS_TO_MG = {
    "mg": Decimal(1),
    "g": Decimal(1000),
    "mcg": Decimal("0.001"),
    "ng": Decimal("0.000001"),
}
_VOLUME_TO_ML = {"ml": Decimal(1), "l": Decimal(1000)}
_UNIT_ALIASES = {
    "mg": "mg", "g": "g", "gm": "g", "mcg": "mcg", "ug": "mcg", "ng": "ng",
    "ml": "ml", "l": "l", "iu": "IU", "i.u.": "IU", "unit": "IU", "units": "IU",
    "%": "%", "%w/w": "%", "%w/v": "%", "%v/v": "%",
}  # fmt: skip
_VOLUME_WORDS = _words("ml l millilitre milliliter litre liter")
_DOSE_WORDS = _words("dose metered-dose actuation puff spray drop")
_NUMBER = r"(\d+(?:[.,]\d+)?)"
_UNIT = r"(mg|mcg|ug|ng|g|gm|ml|l|iu|i\.u\.|units?|%(?:\s*[wv]/[wv])?)"
_ONE = re.compile(rf"^{_NUMBER}\s*{_UNIT}$")
_PER = re.compile(rf"^{_NUMBER}\s*{_UNIT}\s*/\s*(?:{_NUMBER}\s*)?([a-z][a-z\-\s.]*)$")
_COMBINED = re.compile(rf"^{_NUMBER}(?:\s*{_UNIT})?(?:\s*/\s*{_NUMBER}(?:\s*{_UNIT})?)+$")
_PIECE = re.compile(rf"{_NUMBER}\s*{_UNIT}?")


@dataclass(frozen=True, slots=True)
class Strength:
    """An amount per unit of the product. ``unit``: mg (any mass), IU, % or mL; ``per_unit``: mL,
    dose or None (per tablet, capsule or whatever the dosage form counts). ``per_value`` is the
    volume or count the amount is in (5 for "125 mg/5 mL")."""

    value: Decimal
    unit: str
    per_value: Decimal | None = None
    per_unit: str | None = None

    def canonical(self) -> tuple[Decimal, str, str | None]:
        """Amount per one ``per_unit``: 125 mg/5 mL and 25 mg/mL give the same."""
        amount = self.value / self.per_value if self.per_value else self.value
        return amount.normalize(), self.unit, self.per_unit

    def as_json(self) -> dict[str, Any]:
        """{"strength_value": 125.0, "strength_unit": "mg/5 mL"}."""
        unit = self.unit
        if self.per_unit:
            count = "" if self.per_value in (None, Decimal(1)) else f"{_plain(self.per_value)} "
            unit = f"{unit}/{count}{self.per_unit}"
        return {"strength_value": float(self.value), "strength_unit": unit}


def _plain(value: Decimal | None) -> str:
    return format((value or Decimal(0)).normalize(), "f")


def _amount(number: str, unit: str) -> tuple[Decimal, str]:
    """(value in mg or mL, canonical unit); IU and % stay as they are."""
    canonical = _UNIT_ALIASES[unit.replace(" ", "")]
    value = Decimal(number.replace(",", "."))
    if canonical in _MASS_TO_MG:
        return value * _MASS_TO_MG[canonical], "mg"
    if canonical in _VOLUME_TO_ML:
        return value * _VOLUME_TO_ML[canonical], "mL"
    return value, canonical


def parse_strength(text: str) -> Strength | None:
    """One strength ("500 mg", "0.5g", "125 mg/5 mL", "10 IU/mL", "100 mcg/Metered-Dose",
    "1%"), or None when it is not one."""
    folded = _fold(text).replace(" per ", "/")
    try:
        if match := _ONE.match(folded):
            value, unit = _amount(match[1], match[2])
            return Strength(value, unit)
        if match := _PER.match(folded):
            value, unit = _amount(match[1], match[2])
            count = Decimal(match[3].replace(",", ".")) if match[3] else Decimal(1)
            per_word = match[4].strip(" .")
            if per_word in _VOLUME_WORDS:
                volume, _ = _amount(str(count), "l" if per_word.startswith("l") else "ml")
                return Strength(value, unit, volume, "mL")
            if per_word in _DOSE_WORDS:
                return Strength(value, unit, count, "dose")
    except (InvalidOperation, KeyError):
        return None
    return None


def parse_strengths(text: str) -> list[Strength] | None:
    """Every strength in a field: "80 mg; 480 mg", "20/120 mg", "20mg/120mg", "500 mg + 125 mg".
    None when a part does not parse (the caller decides whether that is an error)."""
    folded = _fold(text)
    if not folded:
        return []
    parts = [part.strip() for part in re.split(r";|\+|,\s(?=\d)| and ", folded) if part.strip()]
    found: list[Strength] = []
    for part in parts:
        single = parse_strength(part)
        if single is not None:
            found.append(single)
            continue
        combined = _combined(part)
        if combined is None:
            return None
        found += combined
    return found


def _combined(part: str) -> list[Strength] | None:
    """ "20/120 mg" or "20mg/120mg": amounts of several ingredients, the unit written once or on
    each. (A volume or a dose after the slash makes a concentration: parse_strength.)"""
    if not _COMBINED.match(part):
        return None
    pieces = _PIECE.findall(part)
    units = [unit for _, unit in pieces if unit]
    if not units:
        return None
    found = []
    for number, unit in pieces:
        value, canonical = _amount(number, unit or units[-1])
        if canonical == "mL":
            return None
        found.append(Strength(value, canonical))
    return found


def same_strength(a: Strength, b: Strength) -> bool:
    return a.canonical() == b.canonical()


# --------------------------------------------------------------------------- dosage forms
# Families a pack and its registration are compared by: a pack may say "tablets" for a
# film-coated tablet, or "syrup" for an oral suspension. Checked in this order.
_FORM_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("injection", ("injection", "injectable", "infusion", "intravenous", "intramuscular",
                   "ampoule", "vial")),
    ("tablet", ("tablet", "caplet", "tab")),
    ("capsule", ("capsule", "softgel", "soft gel", "cap")),
    ("drops", ("drops", "drop", "eye", "ophthalmic", "ear", "otic", "nasal")),
    ("oral liquid", ("syrup", "suspension", "oral solution", "elixir", "mixture", "linctus",
                     "oral liquid", "solution")),
    ("powder", ("powder", "granules", "sachet")),
    ("cream", ("cream",)),
    ("ointment", ("ointment",)),
    ("gel", ("gel",)),
    ("suppository", ("suppository", "pessary")),
    ("inhaler", ("inhaler", "aerosol", "metered-dose", "inhalation")),
    ("lotion", ("lotion",)),
)  # fmt: skip


def dosage_form_family(text: str) -> str | None:
    """ "Film-coated tablet" -> tablet, "Oral suspension" -> oral liquid, "Powder for injection"
    -> injection (injection is checked first). None when no family is recognised."""
    words = " " + re.sub(r"[^a-z\s-]", " ", _fold(text)) + " "
    for family, synonyms in _FORM_FAMILIES:
        for synonym in synonyms:
            if re.search(rf"(?<![a-z]){re.escape(synonym)}s?(?![a-z])", words):
                return family
    return None


def dosage_form_mentions(text: str) -> Counter[str]:
    """How often each family is named in free text (pack text names a form several times)."""
    words = " " + re.sub(r"[^a-z\s-]", " ", _fold(text)) + " "
    counts: Counter[str] = Counter()
    claimed: list[tuple[int, int]] = []  # "powder for injection" counts once, as injection
    for family, synonyms in _FORM_FAMILIES:
        for synonym in synonyms:
            for match in re.finditer(rf"(?<![a-z]){re.escape(synonym)}s?(?![a-z])", words):
                span = match.span()
                if any(start < span[1] and span[0] < end for start, end in claimed):
                    continue
                claimed.append(span)
                counts[family] += 1
    return counts


# --------------------------------------------------------------------------- ingredients
# Salt and hydrate words that follow the active moiety ("metformin hydrochloride").
_SALT_WORDS = _words(
    "hydrochloride hcl hydrobromide hbr dihydrochloride sodium disodium potassium calcium "
    "magnesium trihydrate dihydrate monohydrate hemihydrate sesquihydrate anhydrous phosphate "
    "diphosphate sulfate besilate besylate maleate mesylate mesilate citrate acetate tartrate "
    "bitartrate fumarate succinate axetil diethylamine dipropionate propionate bromide chloride "
    "nitrate lactate gluconate hyclate embonate pamoate valerate base"
)
# Metal ions: with a salt word they are the ingredient ("ferrous sulfate", "sodium citrate").
_CATIONS = _words("sodium potassium calcium magnesium aluminium ferrous ferric zinc iron lithium")
_WORD_VARIANTS = {  # spelling
    "sulphate": "sulfate", "sulphur": "sulfur", "aluminum": "aluminium",
    "amoxycillin": "amoxicillin", "cephalexin": "cefalexin", "cephradine": "cefradine",
    "frusemide": "furosemide", "lignocaine": "lidocaine", "oestradiol": "estradiol",
    "phenobarbitone": "phenobarbital", "chlorpheniramine": "chlorphenamine",
    "rifampin": "rifampicin", "glibenclamide": "glyburide",
}  # fmt: skip
_NAME_VARIANTS = {  # other names of the same moiety
    "acetaminophen": "paracetamol",
    "acetylsalicylic acid": "aspirin",
    "albuterol": "salbutamol",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "clavulanate": "clavulanic acid",
    "valproate": "valproic acid",
    "artenimol": "dihydroartemisinin",
    "vitamin c": "ascorbic acid",
    "vitamin b12": "cyanocobalamin",
    "co-trimoxazole": "sulfamethoxazole; trimethoprim",
    "cotrimoxazole": "sulfamethoxazole; trimethoprim",
}


def normalize_ingredient(text: str) -> str:
    """The active moiety: "Amoxicillin (Amoxicillin Trihydrate)" and "amoxicillin trihydrate"
    -> "amoxicillin"; "Diclofenac Potassium" -> "diclofenac"; "Potassium Clavulanate" ->
    "clavulanic acid"; "Acetaminophen" -> "paracetamol". Where the salt is the ingredient, it
    stays: "magnesium hydroxide", "ferrous sulphate" -> "ferrous sulfate", "sodium citrate"."""
    name = re.sub(r"\(.*?\)", " ", _fold(text))  # the register's salt note in parentheses
    name = re.sub(r"[^a-z0-9\s-]", " ", name)
    words = [_WORD_VARIANTS.get(word, word) for word in name.split()]
    if not words:
        return ""
    if " ".join(words) in _NAME_VARIANTS:
        return _NAME_VARIANTS[" ".join(words)]
    core = list(words)
    while len(core) > 1 and core[-1] in _SALT_WORDS:
        core.pop()  # trailing salt or hydrate words
    if len(core) == 1 and core[0] in _CATIONS:
        core = list(words)  # the salt is the ingredient
    elif len(core) == 2 and core[0] in _CATIONS and core[1] in _NAME_VARIANTS:
        core = core[1:]  # "potassium clavulanate" -> "clavulanate" (a drug, not "trisilicate")
    result = " ".join(core)
    return _NAME_VARIANTS.get(result, result)


_PHRASE_VARIANTS = sorted(
    (name for name in _NAME_VARIANTS if " " in name or "-" in name), key=len, reverse=True
)


def canonical_text(text: str) -> str:
    """Free text with ingredient names in their normalised form, word by word ("Amoxycillin
    trihydrate" -> "amoxicillin trihydrate", "Acetylsalicylic acid" -> "aspirin"), so a
    dictionary of normalised names can be searched in it."""
    folded = _fold(text)
    for phrase in _PHRASE_VARIANTS:
        folded = re.sub(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", _NAME_VARIANTS[phrase], folded)
    return " ".join(normalize_ingredient(word) or word for word in folded.split())


@dataclass(frozen=True, slots=True)
class Ingredient:
    name: str  # normalised (normalize_ingredient)
    strength: Strength | None

    def as_json(self) -> dict[str, Any]:
        if self.strength is None:
            return {"name": self.name, "strength_value": None, "strength_unit": None}
        return {"name": self.name, **self.strength.as_json()}


def ingredients_from_json(items: Iterable[dict[str, Any]]) -> tuple[Ingredient, ...]:
    """products.active_ingredients back as Ingredients (unparseable entries are skipped)."""
    found = []
    for item in items:
        name = str(item.get("name") or "")
        if not name:
            continue
        value, unit = item.get("strength_value"), item.get("strength_unit")
        strength = (
            parse_strength(f"{_plain(Decimal(str(value)))} {unit}") if value and unit else None
        )
        found.append(Ingredient(name, strength))
    return tuple(found)


def parse_ingredients(names: str, strengths: str) -> list[Ingredient]:
    """The register's parallel lists as ingredients. ValueError when the lists do not pair up or
    a strength does not parse: a wrong register row must not import silently."""
    normalised: list[str] = []
    for part in (piece.strip() for piece in names.split(";")):
        if part:
            normalised += [name.strip() for name in normalize_ingredient(part).split(";")]
    if not normalised:
        raise ValueError("no active ingredient")
    parsed = parse_strengths(strengths) if strengths.strip() else []
    if parsed is None:
        raise ValueError(f"strength {strengths!r} does not parse")
    if parsed and len(parsed) != len(normalised):
        raise ValueError(
            f"{len(normalised)} ingredient(s) but {len(parsed)} strength(s) "
            f"({names!r} / {strengths!r})"
        )
    return [
        Ingredient(name, parsed[index] if parsed else None) for index, name in enumerate(normalised)
    ]


# --------------------------------------------------------------------------- companies
_LEGAL = re.compile(
    r"\b(?:public limited company|private limited company|limited|ltd|plc|inc|incorporated|llc"
    r"|co|corp|corporation|gmbh|pvt|private|enterprises?|ventures?|international|intl)\b"
)


def normalize_company(text: str) -> str:
    """ "GlaxoSmithKline Consumer Nigeria PLC" and "Glaxosmithkline Consumer Nig. Ltd." ->
    "glaxosmithkline consumer nigeria": case, punctuation, "&", legal suffixes and "Nig"."""
    name = _fold(text).replace("&", " and ")
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\bnig\b", "nigeria", name)
    name = re.sub(r"\bpharmaceuticals?\b|\bpharma\b", "pharmaceutical", name)
    name = re.sub(r"\bindustries\b", "industry", name)
    name = _LEGAL.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


# --------------------------------------------------------------------------- dictionary
# Active ingredients the pack reader looks for (normalised names), besides those of the product
# being checked: medicines widely sold in Nigeria, the sample register's included. A pack naming
# an ingredient outside this list and the product's own reads as "unread", never as a mismatch.
KNOWN_INGREDIENTS: frozenset[str] = _words("""
    paracetamol ibuprofen diclofenac aspirin caffeine naproxen piroxicam meloxicam ketoprofen
    indometacin celecoxib tramadol codeine morphine pentazocine artemether lumefantrine
    artesunate amodiaquine dihydroartemisinin piperaquine sulfadoxine pyrimethamine chloroquine
    quinine mefloquine primaquine proguanil atovaquone amoxicillin ampicillin cloxacillin
    flucloxacillin benzylpenicillin phenoxymethylpenicillin cefuroxime ceftriaxone cefixime
    cefalexin cefradine cefpodoxime ciprofloxacin levofloxacin ofloxacin norfloxacin
    tinidazole metronidazole secnidazole azithromycin erythromycin clarithromycin doxycycline
    tetracycline gentamicin sulfamethoxazole trimethoprim nitrofurantoin fluconazole
    ketoconazole clotrimazole miconazole griseofulvin nystatin terbinafine albendazole
    mebendazole ivermectin praziquantel metformin glyburide glimepiride gliclazide insulin
    amlodipine nifedipine lisinopril enalapril captopril losartan valsartan
    hydrochlorothiazide furosemide methyldopa atenolol propranolol bisoprolol spironolactone
    amiloride atorvastatin simvastatin omeprazole esomeprazole lansoprazole pantoprazole
    ranitidine cimetidine famotidine loperamide metoclopramide salbutamol prednisolone
    dexamethasone hydrocortisone betamethasone cetirizine loratadine chlorphenamine
    promethazine diphenhydramine pseudoephedrine phenylephrine guaifenesin dextromethorphan
    bromhexine carbocisteine levonorgestrel ethinylestradiol misoprostol oxytocin diazepam
    tenofovir lamivudine efavirenz dolutegravir zidovudine nevirapine isoniazid rifampicin
    ethambutol pyrazinamide lidocaine cyanocobalamin""") | frozenset(
    {
        "clavulanic acid", "ascorbic acid", "folic acid", "mefenamic acid", "tranexamic acid",
        "aluminium hydroxide", "magnesium hydroxide", "magnesium trisilicate",
        "sodium bicarbonate", "sodium citrate", "sodium chloride", "potassium chloride",
        "ferrous sulfate", "ferrous fumarate", "zinc sulfate", "hyoscine butylbromide",
        "valproic acid",
    }
)  # fmt: skip
