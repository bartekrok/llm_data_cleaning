"""Seeded generator for the large-scale test dataset (large_tests/).

Produces >= 1000 input values PER SCENARIO (accept_without_renaming,
accept_with_renaming, decline, suggestion) plus a large mixed scenario,
each with an expanded scope and expected.jsonl predicate gold specs
compatible with evaluate.py (open answer space for suggest via
same_referent_as_input; closed one_of sets for acceptance).

Design principles (matching the referent-based decision rules in script.py):
- accept_without_renaming: input is byte-identical to a scope item.
- accept_with_renaming: input is a SURFACE VARIANT of a scope item
  (typo, casing, doubled/dropped character, noise characters, or a
  well-established alias/abbreviation). Same referent, different form.
- suggestion: input is a DISTINCT real entity of the same category as the
  scope, deliberately held out of the scope.
- decline: input is gibberish, a placeholder (N/A, null, ...), a free-text
  sentence, or a real entity from a DIFFERENT category than the scope.

Everything is derived from a fixed random seed, so the dataset is fully
reproducible: python generate_data.py
"""

import csv
import json
import os
import random
import shutil

SEED = 42
OUT_ROOT = "large_tests"


def _lines(block):
    return [ln.strip() for ln in block.strip().splitlines() if ln.strip()]


def _pool(block):
    """Parse 'Canonical|Alias1|Alias2' lines -> (canonicals, {canonical: [aliases]})."""
    canonicals, aliases = [], {}
    for ln in _lines(block):
        parts = [p.strip() for p in ln.split("|")]
        canonicals.append(parts[0])
        if len(parts) > 1:
            aliases[parts[0]] = parts[1:]
    assert len(set(c.lower() for c in canonicals)) == len(canonicals), "duplicate canonicals"
    return canonicals, aliases


# ---------------------------------------------------------------------------
# Entity pools
# ---------------------------------------------------------------------------

COUNTRIES, COUNTRY_ALIASES = _pool("""
Afghanistan
Albania
Algeria
Andorra
Angola
Antigua and Barbuda
Argentina
Armenia
Australia
Austria
Azerbaijan
Bahamas
Bahrain
Bangladesh
Barbados
Belarus
Belgium
Belize
Benin
Bhutan
Bolivia
Bosnia and Herzegovina
Botswana
Brazil
Brunei
Bulgaria
Burkina Faso
Burundi
Cambodia
Cameroon
Canada
Cape Verde
Central African Republic
Chad
Chile
China
Colombia
Comoros
Costa Rica
Croatia
Cuba
Cyprus
Czech Republic|Czechia
Democratic Republic of the Congo|DR Congo
Denmark
Djibouti
Dominica
Dominican Republic
East Timor
Ecuador
Egypt
El Salvador
Equatorial Guinea
Eritrea
Estonia
Eswatini
Ethiopia
Fiji
Finland
France
Gabon
Gambia
Georgia
Germany
Ghana
Greece
Grenada
Guatemala
Guinea
Guinea-Bissau
Guyana
Haiti
Honduras
Hungary
Iceland
India
Indonesia
Iran
Iraq
Ireland
Israel
Italy
Ivory Coast
Jamaica
Japan
Jordan
Kazakhstan
Kenya
Kiribati
Kuwait
Kyrgyzstan
Laos
Latvia
Lebanon
Lesotho
Liberia
Libya
Liechtenstein
Lithuania
Luxembourg
Madagascar
Malawi
Malaysia
Maldives
Mali
Malta
Marshall Islands
Mauritania
Mauritius
Mexico
Micronesia
Moldova
Monaco
Mongolia
Montenegro
Morocco
Mozambique
Myanmar|Burma
Namibia
Nauru
Nepal
Netherlands|Holland
New Zealand|NZ
Nicaragua
Niger
Nigeria
North Korea
North Macedonia
Norway
Oman
Pakistan
Palau
Panama
Papua New Guinea
Paraguay
Peru
Philippines
Poland
Portugal
Qatar
Republic of the Congo
Romania
Russia
Rwanda
Saint Kitts and Nevis
Saint Lucia
Saint Vincent and the Grenadines
Samoa
San Marino
Sao Tome and Principe
Saudi Arabia
Senegal
Serbia
Seychelles
Sierra Leone
Singapore
Slovakia
Slovenia
Solomon Islands
Somalia
South Africa
South Korea|Republic of Korea
South Sudan
Spain
Sri Lanka
Sudan
Suriname
Sweden
Switzerland
Syria
Taiwan
Tajikistan
Tanzania
Thailand
Togo
Tonga
Trinidad and Tobago
Tunisia
Turkey
Turkmenistan
Tuvalu
Uganda
Ukraine
United Arab Emirates|UAE
United Kingdom|UK
United States|USA
Uruguay
Uzbekistan
Vanuatu
Vatican City
Venezuela
Vietnam
Yemen
Zambia
Zimbabwe
""")

CITIES, CITY_ALIASES = _pool("""
Tokyo
Delhi
Shanghai
Sao Paulo
Mexico City
Cairo
Mumbai
Beijing
Dhaka
Osaka
New York City|NYC
Karachi
Buenos Aires
Chongqing
Istanbul
Kolkata
Manila
Lagos
Rio de Janeiro|Rio
Tianjin
Kinshasa
Guangzhou
Los Angeles|LA
Moscow
Shenzhen
Lahore
Bangalore
Paris
Bogota
Jakarta
Chennai
Lima
Bangkok
Seoul
Nagoya
Hyderabad
London
Tehran
Chicago
Chengdu
Nanjing
Wuhan
Ho Chi Minh City
Luanda
Ahmedabad
Kuala Lumpur
Hong Kong
Riyadh
Baghdad
Santiago
Surat
Madrid
Suzhou
Pune
Harbin
Houston
Dallas
Toronto
Dar es Salaam
Miami
Belo Horizonte
Philadelphia|Philly
Atlanta
Fukuoka
Khartoum
Barcelona
Johannesburg
Saint Petersburg|St. Petersburg
Qingdao
Dalian
Washington
Yangon
Alexandria
Jinan
Guadalajara
Boston
Phoenix
San Francisco|SF
Seattle
Denver
Detroit
Minneapolis
San Diego
Austin
San Antonio
San Jose
Columbus
Charlotte
Indianapolis
Jacksonville
Fort Worth
El Paso
Nashville
Memphis
Portland
Oklahoma City
Las Vegas|Vegas
Louisville
Baltimore
Milwaukee
Albuquerque
Tucson
Fresno
Sacramento
Kansas City
Omaha
Raleigh
Cleveland
Tampa
Pittsburgh
Cincinnati
Orlando
Vancouver
Montreal
Calgary
Ottawa
Edmonton
Quebec City
Winnipeg
Hamilton
Munich
Berlin
Hamburg
Cologne
Frankfurt
Stuttgart
Dusseldorf
Leipzig
Dortmund
Essen
Rome
Milan
Naples
Turin
Palermo
Genoa
Bologna
Florence
Venice
Verona
Amsterdam
Rotterdam
The Hague
Utrecht
Eindhoven
Brussels
Antwerp
Ghent
Vienna
Graz
Linz
Salzburg
Zurich
Geneva
Basel
Bern
Lausanne
Stockholm
Gothenburg
Malmo
Oslo
Bergen
Copenhagen
Aarhus
Helsinki
Espoo
Tampere
Warsaw
Krakow
Lodz
Wroclaw
Poznan
Gdansk
Szczecin
Katowice
Lublin
Bialystok
Prague
Brno
Ostrava
Budapest
Debrecen
Bucharest
Cluj-Napoca
Timisoara
Sofia
Plovdiv
Athens
Thessaloniki
Lisbon
Porto
Dublin
Cork
Edinburgh
Glasgow
Manchester
Birmingham
Liverpool
Leeds
Sheffield
Bristol
Newcastle
Nottingham
Cardiff
Belfast
Lyon
Marseille
Toulouse
Nice
Nantes
Strasbourg
Montpellier
Bordeaux
Lille
Rennes
Kyiv
Kharkiv
Odesa
Dnipro
Lviv
Minsk
Riga
Vilnius
Tallinn
Zagreb
Belgrade
Sarajevo
Skopje
Tirana
Ljubljana
Bratislava
Chisinau
Yerevan
Tbilisi
Baku
Almaty
Astana
Tashkent
Bishkek
Dushanbe
Ashgabat
Kabul
Islamabad
Colombo
Kathmandu
Thimphu
Dubai
Abu Dhabi
Doha
Kuwait City
Manama
Muscat
Amman
Beirut
Damascus
Jerusalem
Tel Aviv
Ankara
Izmir
Casablanca
Rabat
Tunis
Tripoli
Algiers
Accra
Abidjan
Dakar
Bamako
Nairobi
Addis Ababa
Kampala
Kigali
Lusaka
Harare
Gaborone
Windhoek
Maputo
Antananarivo
Cape Town
Durban
Pretoria
Auckland
Wellington
Christchurch
Sydney
Melbourne
Brisbane
Perth
Adelaide
Canberra
Hobart
Darwin
Honolulu
Anchorage
Havana
Kingston
Port-au-Prince
Santo Domingo
Panama City
San Salvador
Tegucigalpa
Managua
Caracas
Quito
Guayaquil
Montevideo
Asuncion
La Paz
Brasilia
Salvador
Fortaleza
Curitiba
Recife
Porto Alegre
Manaus
Medellin
Cali
Cartagena
""")

ANIMALS, ANIMAL_ALIASES = _pool("""
Aardvark
Alligator
Alpaca
Anteater
Antelope
Armadillo
Baboon
Badger
Bat
Bear
Beaver
Bison
Boar
Butterfly
Camel
Capybara
Caribou
Cassowary
Cat
Caterpillar
Chameleon
Cheetah
Chicken
Chimpanzee
Chinchilla
Chipmunk
Cobra
Cockatoo
Condor
Cougar
Cow
Coyote
Crab
Crane
Crocodile
Crow
Deer
Dingo
Dog
Dolphin
Donkey
Dove
Dragonfly
Duck
Eagle
Echidna
Eel
Elephant
Elk
Emu
Falcon
Ferret
Finch
Flamingo
Fox
Frog
Gazelle
Gecko
Gerbil
Gibbon
Giraffe
Goat
Goose
Gorilla
Grasshopper
Hamster
Hare
Hawk
Hedgehog
Heron
Hippopotamus
Hornet
Horse
Hummingbird
Hyena
Ibex
Iguana
Impala
Jackal
Jaguar
Jellyfish
Kangaroo
Kingfisher
Kiwi
Koala
Komodo Dragon
Kudu
Ladybug
Lemming
Lemur
Leopard
Lion
Lizard
Llama
Lobster
Lynx
Macaw
Magpie
Manatee
Mandrill
Meerkat
Mole
Mongoose
Monkey
Moose
Mosquito
Mouse
Mule
Narwhal
Newt
Nightingale
Ocelot
Octopus
Okapi
Opossum
Orangutan
Ostrich
Otter
Owl
Ox
Oyster
Panda
Pangolin
Panther
Parrot
Peacock
Pelican
Penguin
Pheasant
Pig
Pigeon
Platypus
Porcupine
Puffin
Puma
Quail
Rabbit
Raccoon
Rat
Rattlesnake
Raven
Reindeer
Rhinoceros
Rooster
Scorpion
Seahorse
Seal
Shark
Sheep
Skunk
Sloth
Snail
Sparrow
Squid
Squirrel
Starfish
Stork
Swan
Tapir
Tarantula
Tiger
Toad
Toucan
Turtle
Vulture
Wallaby
Walrus
Wasp
Weasel
Whale
Wolf
Wolverine
Wombat
Woodpecker
Yak
Zebra
""")

LANGUAGES, LANGUAGE_ALIASES = _pool("""
Python
Java
JavaScript|JS
TypeScript|TS
C
C++|CPP
C#|C Sharp
Go|Golang
Rust
Ruby
PHP
Swift
Kotlin
Scala
Perl
Haskell
Erlang
Elixir
Clojure
F#
OCaml
R
MATLAB
Julia
Fortran
COBOL
Ada
Common Lisp
Scheme
Prolog
Lua
Dart
Groovy
Objective-C
Visual Basic
Delphi
Pascal
Assembly
Bash
PowerShell
Zig
Nim
Crystal
Elm
PureScript
Solidity
Verilog
VHDL
ABAP
Apex
ActionScript
CoffeeScript
D
Eiffel
Forth
Hack
Racket
Raku
Smalltalk
Tcl
VBScript
APL
AWK
Ballerina
Chapel
Haxe
Idris
Mojo
Carbon
Gleam
Odin
V
Janet
Fantom
Gosu
J
Modula-2
Oberon
RPG
Scratch
Standard ML
X10
Q#
""")

COLORS, COLOR_ALIASES = _pool("""
Alice Blue
Antique White
Aqua
Aquamarine
Azure
Beige
Bisque
Black
Blanched Almond
Blue
Blue Violet
Brown
Burlywood
Cadet Blue
Chartreuse
Chocolate
Coral
Cornflower Blue
Cornsilk
Crimson
Cyan
Dark Blue
Dark Cyan
Dark Goldenrod
Dark Gray
Dark Green
Dark Khaki
Dark Magenta
Dark Olive Green
Dark Orange
Dark Orchid
Dark Red
Dark Salmon
Dark Sea Green
Dark Slate Blue
Dark Slate Gray
Dark Turquoise
Dark Violet
Deep Pink
Deep Sky Blue
Dim Gray
Dodger Blue
Firebrick
Floral White
Forest Green
Fuchsia
Gainsboro
Ghost White
Gold
Goldenrod
Gray
Green
Green Yellow
Honeydew
Hot Pink
Indian Red
Indigo
Ivory
Khaki
Lavender
Lavender Blush
Lawn Green
Lemon Chiffon
Light Blue
Light Coral
Light Cyan
Light Goldenrod
Light Gray
Light Green
Light Pink
Light Salmon
Light Sea Green
Light Sky Blue
Light Slate Gray
Light Steel Blue
Light Yellow
Lime Green
Linen
Magenta
Maroon
Medium Aquamarine
Medium Blue
Medium Orchid
Medium Purple
Medium Sea Green
Medium Slate Blue
Medium Spring Green
Medium Turquoise
Medium Violet Red
Midnight Blue
Mint Cream
Misty Rose
Moccasin
Navajo White
Navy
Old Lace
Olive
Olive Drab
Orange
Orange Red
Orchid
Pale Goldenrod
Pale Green
Pale Turquoise
Pale Violet Red
Papaya Whip
Peach Puff
Pink
Plum
Powder Blue
Purple
Rebecca Purple
Red
Rosy Brown
Royal Blue
Saddle Brown
Salmon
Sandy Brown
Sea Green
Seashell
Sienna
Silver
Sky Blue
Slate Blue
Slate Gray
Snow
Spring Green
Steel Blue
Tan
Teal
Thistle
Turquoise
Violet
Wheat
White
White Smoke
Yellow
Yellow Green
""")

ELEMENTS_RAW = """
Hydrogen|H
Helium|He
Lithium|Li
Beryllium|Be
Boron|B
Carbon|C
Nitrogen|N
Oxygen|O
Fluorine|F
Neon|Ne
Sodium|Na
Magnesium|Mg
Aluminum|Al
Silicon|Si
Phosphorus|P
Sulfur|S
Chlorine|Cl
Argon|Ar
Potassium|K
Calcium|Ca
Scandium|Sc
Titanium|Ti
Vanadium|V
Chromium|Cr
Manganese|Mn
Iron|Fe
Cobalt|Co
Nickel|Ni
Copper|Cu
Zinc|Zn
Gallium|Ga
Germanium|Ge
Arsenic|As
Selenium|Se
Bromine|Br
Krypton|Kr
Rubidium|Rb
Strontium|Sr
Yttrium|Y
Zirconium|Zr
Niobium|Nb
Molybdenum|Mo
Technetium|Tc
Ruthenium|Ru
Rhodium|Rh
Palladium|Pd
Silver|Ag
Cadmium|Cd
Indium|In
Tin|Sn
Antimony|Sb
Tellurium|Te
Iodine|I
Xenon|Xe
Caesium|Cs
Barium|Ba
Lanthanum|La
Cerium|Ce
Praseodymium|Pr
Neodymium|Nd
Promethium|Pm
Samarium|Sm
Europium|Eu
Gadolinium|Gd
Terbium|Tb
Dysprosium|Dy
Holmium|Ho
Erbium|Er
Thulium|Tm
Ytterbium|Yb
Lutetium|Lu
Hafnium|Hf
Tantalum|Ta
Tungsten|W
Rhenium|Re
Osmium|Os
Iridium|Ir
Platinum|Pt
Gold|Au
Mercury|Hg
Thallium|Tl
Lead|Pb
Bismuth|Bi
Polonium|Po
Astatine|At
Radon|Rn
Francium|Fr
Radium|Ra
Actinium|Ac
Thorium|Th
Protactinium|Pa
Uranium|U
Neptunium|Np
Plutonium|Pu
Americium|Am
Curium|Cm
Berkelium|Bk
Californium|Cf
Einsteinium|Es
Fermium|Fm
Mendelevium|Md
Nobelium|No
Lawrencium|Lr
Rutherfordium|Rf
Dubnium|Db
Seaborgium|Sg
Bohrium|Bh
Hassium|Hs
Meitnerium|Mt
Darmstadtium|Ds
Roentgenium|Rg
Copernicium|Cn
Nihonium|Nh
Flerovium|Fl
Moscovium|Mc
Livermorium|Lv
Tennessine|Ts
Oganesson|Og
"""
ELEMENTS, _ELEM_SYMBOLS = _pool(ELEMENTS_RAW)
# Only keep symbols with >= 2 characters as usable aliases (single letters are
# too ambiguous even for a human annotator).
ELEMENT_ALIASES = {c: syms for c, syms in _ELEM_SYMBOLS.items() if len(syms[0]) >= 2}

FRUITS_VEG, FRUIT_ALIASES = _pool("""
Apple
Apricot
Avocado
Banana
Blackberry
Blueberry
Cantaloupe
Cherry
Clementine
Coconut
Cranberry
Dragonfruit
Durian
Elderberry
Fig
Gooseberry
Grape
Grapefruit
Guava
Honeydew Melon
Jackfruit
Kumquat
Lemon
Lime
Lychee
Mango
Mandarin
Nectarine
Papaya
Passion Fruit
Peach
Pear
Persimmon
Pineapple
Pomegranate
Pomelo
Quince
Raspberry
Starfruit
Strawberry
Tangerine
Watermelon
Mulberry
Boysenberry
Currant
Artichoke
Arugula
Asparagus
Beetroot
Bell Pepper
Bok Choy
Broccoli
Brussels Sprouts
Cabbage
Carrot
Cauliflower
Celery
Chard
Chickpea
Corn
Cucumber
Eggplant
Endive
Fennel
Garlic
Ginger
Green Bean
Kale
Kohlrabi
Leek
Lentil
Lettuce
Mushroom
Okra
Onion
Parsnip
Pea
Potato
Pumpkin
Radicchio
Radish
Rhubarb
Rutabaga
Scallion
Shallot
Spinach
Squash
Sweet Potato
Tomato
Turnip
Watercress
Zucchini
Yam
Horseradish
Jalapeno
Habanero
Cayenne Pepper
Basil
Cilantro
Parsley
Dill
Oregano
Thyme
Rosemary
Sage
Mint
""")

DEPARTMENTS, DEPARTMENT_ALIASES = _pool("""
Human Resources|HR
Information Technology|IT
Research and Development|R&D
Quality Assurance|QA
Public Relations|PR
Operations|Ops
Marketing
Sales
Finance
Accounting
Legal
Procurement
Customer Support
Customer Success
Engineering
Product Management
Design
Data Analytics
Business Development|Biz Dev
Supply Chain
Logistics
Manufacturing
Facilities
Security
Compliance
Internal Audit
Treasury
Payroll
Recruiting
Learning and Development|L&D
Corporate Communications|Comms
Investor Relations
Strategy
Risk Management
Health and Safety
Administration
Field Services
Partnerships
Merchandising
Category Management
""")

CURRENCIES, CURRENCY_ALIASES = _pool("""
US Dollar|USD
Euro|EUR
Japanese Yen|JPY
British Pound|GBP
Swiss Franc|CHF
Canadian Dollar|CAD
Australian Dollar|AUD
New Zealand Dollar|NZD
Chinese Yuan|CNY
Indian Rupee|INR
Brazilian Real|BRL
Mexican Peso|MXN
Russian Ruble|RUB
South Korean Won|KRW
Singapore Dollar|SGD
Hong Kong Dollar|HKD
Norwegian Krone|NOK
Swedish Krona|SEK
Danish Krone|DKK
Polish Zloty|PLN
Czech Koruna|CZK
Hungarian Forint|HUF
Romanian Leu|RON
Bulgarian Lev|BGN
Turkish Lira|TRY
Israeli Shekel|ILS
Saudi Riyal|SAR
UAE Dirham|AED
Qatari Riyal|QAR
Kuwaiti Dinar|KWD
Egyptian Pound|EGP
South African Rand|ZAR
Nigerian Naira|NGN
Kenyan Shilling|KES
Moroccan Dirham|MAD
Thai Baht|THB
Vietnamese Dong|VND
Indonesian Rupiah|IDR
Malaysian Ringgit|MYR
Philippine Peso
Pakistani Rupee|PKR
Bangladeshi Taka|BDT
Sri Lankan Rupee|LKR
Chilean Peso|CLP
Colombian Peso|COP
Peruvian Sol|PEN
Argentine Peso|ARS
Uruguayan Peso|UYU
Ukrainian Hryvnia|UAH
Kazakhstani Tenge|KZT
Georgian Lari|GEL
Armenian Dram|AMD
Azerbaijani Manat|AZN
Icelandic Krona|ISK
Croatian Kuna|HRK
Serbian Dinar|RSD
Taiwanese Dollar|TWD
Jordanian Dinar|JOD
Bahraini Dinar|BHD
Omani Rial|OMR
""")

_LEVELS = ["Junior", "Mid-Level", "Senior", "Staff", "Principal", "Lead"]
_ENG_ROLES = [
    "Software Engineer", "Data Engineer", "QA Engineer", "DevOps Engineer",
    "Machine Learning Engineer", "Security Engineer", "Cloud Engineer",
    "Frontend Engineer", "Backend Engineer", "Mobile Engineer",
    "Platform Engineer", "Site Reliability Engineer",
]
_STANDALONE = _pool("""
Chief Executive Officer|CEO
Chief Technology Officer|CTO
Chief Financial Officer|CFO
Chief Operating Officer|COO
Chief Information Officer|CIO
Chief Marketing Officer|CMO
Product Manager
Project Manager
Program Manager
Business Analyst
Data Analyst
Data Scientist
UX Designer
UI Designer
Graphic Designer
Scrum Master
Solutions Architect
Enterprise Architect
Software Architect
Engineering Manager
Technical Writer
Database Administrator|DBA
System Administrator|SysAdmin
Network Engineer
Support Engineer
Release Manager
Product Owner
VP of Engineering
Director of Engineering
Head of Product
""")

JOB_TITLES = [f"{lvl} {role}" for lvl in _LEVELS for role in _ENG_ROLES] + _STANDALONE[0]
JOB_ALIASES = dict(_STANDALONE[1])
for lvl, abbr in [("Junior", "Jr."), ("Senior", "Sr.")]:
    for role in _ENG_ROLES:
        JOB_ALIASES[f"{lvl} {role}"] = [f"{abbr} {role}"]

DOMAINS = {
    "countries": (COUNTRIES, COUNTRY_ALIASES),
    "cities": (CITIES, CITY_ALIASES),
    "animals": (ANIMALS, ANIMAL_ALIASES),
    "programming_languages": (LANGUAGES, LANGUAGE_ALIASES),
    "colors": (COLORS, COLOR_ALIASES),
    "chemical_elements": (ELEMENTS, ELEMENT_ALIASES),
    "fruits_vegetables": (FRUITS_VEG, FRUIT_ALIASES),
    "departments": (DEPARTMENTS, DEPARTMENT_ALIASES),
    "currencies": (CURRENCIES, CURRENCY_ALIASES),
    "job_titles": (JOB_TITLES, JOB_ALIASES),
}

PLACEHOLDERS = [
    "N/A", "NA", "N.A.", "null", "None", "nil", "-", "--", "?", "??", "???",
    "Unknown", "TBD", "TBA", "missing", "undefined", "(empty)", "(blank)",
    "NaN", "#N/A", "#REF!", "no data", "not applicable", "xxx", "12345", "0",
    "true", "false", "...", "void",
]

_SENT_SUBJECTS = [
    "I", "My neighbor", "The intern", "Nobody", "The customer",
    "Our team", "The previous vendor", "Someone", "The manager", "A stranger",
]
_SENT_PREDICATES = [
    "accidentally typed this into the wrong field",
    "left this comment during testing",
    "wrote a long note about their weekend plans",
    "pasted the wrong clipboard content here",
    "described their favorite movie in this cell",
    "entered free text instead of a valid option",
]
SENTENCES = [f"{s} {p}" for s in _SENT_SUBJECTS for p in _SENT_PREDICATES]


# ---------------------------------------------------------------------------
# Variant generation (surface forms of the SAME referent)
# ---------------------------------------------------------------------------

NOISE_SUFFIXES = ["!@#", "##", "??", "!!", "***"]


def _typo_swap(s, rng):
    idxs = [i for i in range(len(s) - 1) if s[i].isalpha() and s[i + 1].isalpha()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _typo_drop(s, rng):
    idxs = [i for i in range(len(s)) if s[i].isalpha()]
    if len(idxs) < 4:
        return None
    i = rng.choice(idxs[1:])  # never drop the first letter
    return s[:i] + s[i + 1:]


def _typo_double(s, rng):
    idxs = [i for i in range(len(s)) if s[i].isalpha()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    return s[:i] + s[i] + s[i:]


def make_variants(canonical, aliases, rng, scope_lower, taken_lower, count=1):
    """Generate up to `count` distinct surface variants of `canonical`."""
    strategies = []
    for alias in aliases.get(canonical, []):
        strategies.append(("alias", alias))
    strategies += [("upper", None), ("lower", None), ("noise", None),
                   ("swap", None), ("drop", None), ("double", None)]
    rng.shuffle(strategies)

    out = []
    for kind, alias in strategies:
        if len(out) >= count:
            break
        if kind == "alias":
            candidate = alias
        elif kind == "upper":
            candidate = canonical.upper()
        elif kind == "lower":
            candidate = canonical.lower()
        elif kind == "noise":
            candidate = canonical + rng.choice(NOISE_SUFFIXES)
        elif kind == "swap":
            candidate = _typo_swap(canonical, rng)
        elif kind == "drop":
            candidate = _typo_drop(canonical, rng)
        else:
            candidate = _typo_double(canonical, rng)

        if not candidate or candidate == canonical:
            continue
        low = candidate.lower()
        # must not collide with any scope item or an already-used input
        if kind not in ("upper", "lower") and low in scope_lower:
            continue
        if low in taken_lower:
            continue
        taken_lower.add(low)
        out.append(candidate)
    return out


def make_gibberish(rng, n, taken_lower):
    consonants = "bcdfghjklmnpqrstvwxz"
    out = []
    while len(out) < n:
        s = "".join(rng.choice(consonants) for _ in range(rng.randint(6, 14)))
        if s not in taken_lower:
            taken_lower.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Folder writing
# ---------------------------------------------------------------------------

def write_case(path, scope, rows):
    """rows: list of (raw_value, state, value, alt_state, alt_value)."""
    os.makedirs(path, exist_ok=True)
    # dedupe scope (case-insensitive, order-preserving)
    seen_scope, scope_unique = set(), []
    for s in scope:
        if s.lower() not in seen_scope:
            seen_scope.add(s.lower())
            scope_unique.append(s)
    scope = scope_unique
    # per-folder gold keys must be unique (evaluate.py keys on lowercased raw)
    seen = set()
    unique_rows = []
    for r in rows:
        key = r[0].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(r)

    with open(os.path.join(path, "scope.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope_value"])
        for s in scope:
            w.writerow([s])
    with open(os.path.join(path, "input_data.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw_value"])
        for r in unique_rows:
            w.writerow([r[0]])
    with open(os.path.join(path, "expected.jsonl"), "w", encoding="utf-8") as f:
        for r in unique_rows:
            allowed = [_gold_entry(r[1], r[2])]
            if r[3]:  # optional alternative answer
                allowed.append(_gold_entry(r[3], r[4]))
            f.write(json.dumps({"raw_value": r[0], "allowed": allowed}) + "\n")
    return len(unique_rows)


def _gold_entry(state, value):
    """Predicate gold spec (see evaluate.py): closed set for acceptance,
    empty for decline, open answer space for suggest."""
    if state == "decline":
        return {"state": "decline", "value": {"mode": "empty"}}
    if state == "suggest":
        return {"state": "suggest", "value": {"mode": "same_referent_as_input"}}
    return {"state": "acceptance", "value": {"mode": "one_of", "items": [value]}}


def main():
    rng = random.Random(SEED)
    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    counts = {"accept_without_renaming": 0, "accept_with_renaming": 0,
              "suggestion": 0, "decline": 0, "mixed": 0}

    # --- 1. accept_without_renaming: input == scope item -------------------
    for name, (pool, _aliases) in DOMAINS.items():
        rows = [(item, "acceptance", item, "", "") for item in pool]
        counts["accept_without_renaming"] += write_case(
            os.path.join(OUT_ROOT, "accept_without_renaming", name), pool, rows)

    # --- 2. accept_with_renaming: surface variants of scope items ----------
    for name, (pool, aliases) in DOMAINS.items():
        scope_lower = {s.lower() for s in pool}
        taken = set()
        rows = []
        for item in pool:
            for v in make_variants(item, aliases, rng, scope_lower, taken, count=1):
                rows.append((v, "acceptance", item, "", ""))
        counts["accept_with_renaming"] += write_case(
            os.path.join(OUT_ROOT, "accept_with_renaming", name), pool, rows)

    # --- 3. suggestion: held-out entities of the same category -------------
    holdouts = {}  # domain -> (scope_subset, heldout_items)
    for name, (pool, _aliases) in DOMAINS.items():
        shuffled = pool[:]
        rng.shuffle(shuffled)
        n_scope = max(5, len(pool) // 6)
        scope_subset, heldout = shuffled[:n_scope], shuffled[n_scope:]
        holdouts[name] = (sorted(scope_subset), sorted(heldout))
        rows = [(item, "suggest", item, "", "") for item in holdouts[name][1]]
        counts["suggestion"] += write_case(
            os.path.join(OUT_ROOT, "suggestion", name), holdouts[name][0], rows)

    # --- 4. decline: garbage / placeholders / sentences / wrong category ---
    decline_pairs = [
        ("colors_scope", COLORS, ANIMALS),
        ("chemical_elements_scope", ELEMENTS, CITIES),
        ("departments_scope", DEPARTMENTS, FRUITS_VEG),
        ("programming_languages_scope", LANGUAGES, COUNTRIES),
        ("currencies_scope", CURRENCIES, JOB_TITLES),
    ]
    gib_taken = set()
    gibberish = make_gibberish(rng, 250, gib_taken)
    gib_chunks = [gibberish[i::5] for i in range(5)]
    ph_chunks = [PLACEHOLDERS[i::5] for i in range(5)]
    sent_chunks = [SENTENCES[i::5] for i in range(5)]

    for i, (name, scope, wrong_category) in enumerate(decline_pairs):
        scope_lower = {s.lower() for s in scope}
        inputs = [w for w in wrong_category if w.lower() not in scope_lower]
        inputs += gib_chunks[i] + ph_chunks[i] + sent_chunks[i]
        rows = [(v, "decline", "", "", "") for v in inputs]
        counts["decline"] += write_case(
            os.path.join(OUT_ROOT, "decline", name), scope, rows)

    # --- 5. mixed: all four outcomes against one combined scope ------------
    mixed_domains = ["colors", "programming_languages", "countries",
                     "animals", "chemical_elements"]
    mixed_scope, mixed_heldout = [], []
    for name in mixed_domains:
        sub, held = holdouts[name]
        mixed_scope += sub
        mixed_heldout += held
    scope_lower = {s.lower() for s in mixed_scope}
    taken = set()
    rows = []
    # exact accepts
    for item in mixed_scope:
        rows.append((item, "acceptance", item, "", ""))
        taken.add(item.lower())
    # renaming accepts (3 variants per scope item)
    alias_union = {}
    for name in mixed_domains:
        alias_union.update(DOMAINS[name][1])
    for item in mixed_scope:
        for v in make_variants(item, alias_union, rng, scope_lower, taken, count=3):
            rows.append((v, "acceptance", item, "", ""))
    # suggestions (held-out same-category entities)
    for item in rng.sample(mixed_heldout, 350):
        if item.lower() in taken:
            continue
        taken.add(item.lower())
        rows.append((item, "suggest", item, "", ""))
    # declines
    mixed_pools_lower = set()
    for name in mixed_domains:
        mixed_pools_lower |= {s.lower() for s in DOMAINS[name][0]}
    decline_inputs = make_gibberish(rng, 200, gib_taken)
    decline_inputs += PLACEHOLDERS + SENTENCES
    decline_inputs += [c for c in rng.sample(CITIES, 60)
                       if c.lower() not in mixed_pools_lower]
    for v in decline_inputs:
        if v.lower() in taken:
            continue
        taken.add(v.lower())
        rows.append((v, "decline", "", "", ""))
    rng.shuffle(rows)
    counts["mixed"] += write_case(os.path.join(OUT_ROOT, "mixed"), mixed_scope, rows)

    # --- report + guarantees ------------------------------------------------
    print(f"Seed: {SEED}")
    total = 0
    for scenario, n in counts.items():
        total += n
        flag = "OK " if n >= 1000 else "LOW"
        print(f"  [{flag}] {scenario:<28} {n:>5} inputs")
        assert n >= 1000, f"{scenario} has only {n} inputs (< 1000)"
    print(f"  Total: {total} inputs")


if __name__ == "__main__":
    main()
