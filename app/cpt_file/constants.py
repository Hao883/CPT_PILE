"""Copyright (c) 2022 VIKTOR B.V.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from viktor.parametrization import ViktorParametrization, Step, TextField, NumberField,OptionField, OptionListElement

Pile_Class = [
    OptionListElement(label="Bore piles", value="Bore"),
    OptionListElement(label="CFA piles", value="CFA"),
    OptionListElement(label="Screw piles", value="Screw"),
    OptionListElement(label="Closed-ended driven piles", value="Closed"),
    OptionListElement(label="Open-ended driven piles", value="Open"),
    OptionListElement(label="Driven H piles", value="H"),
    OptionListElement(label="Driven sheet pile walls", value="sheet"),
    OptionListElement(label="Micro piles", value="Micro"),
]
Pile_Bore_Category = [
    OptionListElement(label="No support", value="No support"),
    OptionListElement(label="With Slurry", value="With Slurry"),
    OptionListElement(label="Permanent casing", value="Permanent casing"),
    OptionListElement(label="Recoverable casing", value="Recoverable casing"),
    OptionListElement(label="Dry bored pile or slurry", value="Dry bored pile or slurry"),
    OptionListElement(label="With grooved sockets", value="With grooved sockets"),

]

Pile_CFA_Category = [
    OptionListElement(label="CFA piles", value="CFA piles")

]

Pile_Screw_Category = [
    OptionListElement(label="Screw cast-in-place pile", value="cast-in"),
    OptionListElement(label="Screw piles with casing", value="casing")
]

Pile_Closed_Driven_Category = [
    OptionListElement(label="Pre-cast or pre-stressed concrete-driven pile", value="pre"),
    OptionListElement(label="Coated driven steel pile (coating: concrete, mortar, grout)", value="coated"),
    OptionListElement(label="Driven cast-in-place pile", value="cast-in-place"),
    OptionListElement(label="Driven steel pile, closed ended", value="steel")
]

Pile_Open_Driven_Category = [
    OptionListElement(label="Driven steel pile, open ended", value="steel")
]

Pile_Driven_H_Category = [
    OptionListElement(label="Driven H pile", value="driven"),
    OptionListElement(label="Driven grouted H pile", value="grouted")
]

Pile_Driven_Sheet_Category = [
    OptionListElement(label="Driven sheet pile", value="sheet")
]

Pile_Micro_piles_Category = [
    OptionListElement(label="Micropile I (gravity pressure)", value="gravity"),
    OptionListElement(label="Micropile II low pressure)", value="low"),
    OptionListElement(label="Micropile II high pressure)", value="high"),
    OptionListElement(label="Micropile IV high pressure)", value="TAM")
]

DEFAULT_MIN_LAYER_THICKNESS = 200

ADDITIONAL_COLUMNS = ["corrected_depth", "fs"]

DEFAULT_ROBERTSON_TABLE = [
    {
        "name": "Robertson zone unknown",
        "ui_name": "Unknown material",
        "color": "255, 0, 0",
        "gamma_dry": 0,
        "gamma_wet": 0,
        "phi": 0,
    },
    {
        "name": "Robertson zone 1",
        "ui_name": "Soil, fine grain",
        "color": "200, 25, 0",
        "gamma_dry": 10,
        "gamma_wet": 10,
        "phi": 15,
    },
    {
        "name": "Robertson zone 2",
        "ui_name": "Peat, organic material",
        "color": "188, 104, 67",
        "gamma_dry": 12,
        "gamma_wet": 12,
        "phi": 15,
    },
    {
        "name": "Robertson zone 3",
        "ui_name": "Clay, slightly silty to silty",
        "color": "29, 118, 29",
        "gamma_dry": 15.5,
        "gamma_wet": 15.5,
        "phi": 17.5,
    },
    {
        "name": "Robertson zone 4",
        "ui_name": "Clay, silty to loamy",
        "color": "213, 252, 181",
        "gamma_dry": 18,
        "gamma_wet": 18,
        "phi": 22.5,
    },
    {
        "name": "Robertson zone 5",
        "ui_name": "Sand, silty to loamy",
        "color": "213, 252, 155",
        "gamma_dry": 18,
        "gamma_wet": 20,
        "phi": 25,
    },
    {
        "name": "Robertson zone 6",
        "ui_name": "Sand, slightly silty to silty",
        "color": "255, 225, 178",
        "gamma_dry": 18,
        "gamma_wet": 20,
        "phi": 27,
    },
    {
        "name": "Robertson zone 7",
        "ui_name": "Sand, gravelly",
        "color": "255, 183, 42",
        "gamma_dry": 17,
        "gamma_wet": 19,
        "phi": 32.5,
    },
    {
        "name": "Robertson zone 8",
        "ui_name": "Sand, solid to clayey",
        "color": "200, 190, 200",
        "gamma_dry": 18,
        "gamma_wet": 20,
        "phi": 32.5,
    },
    {
        "name": "Robertson zone 9",
        "ui_name": "Soil, very stiff, finegrained",
        "color": "186, 205, 224",
        "gamma_dry": 20,
        "gamma_wet": 22,
        "phi": 40,
    },
]


DEFAULT_CLASSIFICATION_TABLE = [
    {
        "phi": 30,
        "name": "Sand, clean, loose",
        "color": "255,255,153",
        "qc_max": None,
        "qc_min": None,
        "rf_max": 0.8,
        "rf_min": None,
        "gamma_dry": 17,
        "gamma_wet": 19,
        "qc_norm_max": 5,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 32.5,
        "name": "Sand, clean, moderately",
        "color": "255,255,102",
        "qc_max": None,
        "qc_min": None,
        "rf_max": 0.8,
        "rf_min": None,
        "gamma_dry": 18,
        "gamma_wet": 20,
        "qc_norm_max": 15,
        "qc_norm_min": 5,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 35,
        "name": "Sand, clean, firm",
        "color": "255,255,0",
        "qc_max": None,
        "qc_min": None,
        "rf_max": 0.8,
        "rf_min": None,
        "gamma_dry": 19,
        "gamma_wet": 21,
        "qc_norm_max": None,
        "qc_norm_min": 15,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 27,
        "name": "Sand, slightly silty",
        "color": "255,204,102",
        "qc_max": None,
        "qc_min": None,
        "rf_max": 1.5,
        "rf_min": 0.8,
        "gamma_dry": 18,
        "gamma_wet": 20,
        "qc_norm_max": None,
        "qc_norm_min": 0,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 25,
        "name": "Sand, very silty",
        "color": "255,204,0",
        "qc_max": None,
        "qc_min": None,
        "rf_max": 1.8,
        "rf_min": 1.5,
        "gamma_dry": 18,
        "gamma_wet": 20,
        "qc_norm_max": None,
        "qc_norm_min": 0,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 17.5,
        "name": "Clay, clean, soft",
        "color": "0,255,0",
        "qc_max": 0.75,
        "qc_min": None,
        "rf_max": 5,
        "rf_min": 3,
        "gamma_dry": 14,
        "gamma_wet": 14,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 17.5,
        "name": "Clay, clean, moderately",
        "color": "0,204,0",
        "qc_max": 1.5,
        "qc_min": 0.75,
        "rf_max": 5,
        "rf_min": 3,
        "gamma_dry": 17,
        "gamma_wet": 17,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 17.5,
        "name": "Clay, clean, firm",
        "color": "0,204,102",
        "qc_max": None,
        "qc_min": 1.5,
        "rf_max": 5,
        "rf_min": 3,
        "gamma_dry": 19,
        "gamma_wet": 19,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 22.5,
        "name": "Clay, slightly sandy",
        "color": "102,255,102",
        "qc_max": None,
        "qc_min": 0,
        "rf_max": 3,
        "rf_min": 1.8,
        "gamma_dry": 18,
        "gamma_wet": 18,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 27.5,
        "name": "Clay, very sandy",
        "color": "102,255,51",
        "qc_max": None,
        "qc_min": 0,
        "rf_max": 3,
        "rf_min": 1.8,
        "gamma_dry": 18,
        "gamma_wet": 18,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 15,
        "name": "Clay, organic, soft",
        "color": "102,153,0",
        "qc_max": 0.2,
        "qc_min": None,
        "rf_max": 7,
        "rf_min": 5,
        "gamma_dry": 13,
        "gamma_wet": 13,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 15,
        "name": "Clay, organic, moderately",
        "color": "0,153,0",
        "qc_max": None,
        "qc_min": 0.2,
        "rf_max": 7,
        "rf_min": 5,
        "gamma_dry": 15,
        "gamma_wet": 15,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 15,
        "name": "Peat, not preloaded, soft",
        "color": "204,153,0",
        "qc_max": 0.1,
        "qc_min": None,
        "rf_max": None,
        "rf_min": 7,
        "gamma_dry": 10,
        "gamma_wet": 10,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
    {
        "phi": 15,
        "name": "Peat, moderately preloaded, moderately",
        "color": "153,102,51",
        "qc_max": None,
        "qc_min": 0.1,
        "rf_max": None,
        "rf_min": 7,
        "gamma_dry": 12,
        "gamma_wet": 12,
        "qc_norm_max": None,
        "qc_norm_min": None,
        "max_cone_res_mpa": 0,
        "max_cone_res_type": "Standard",
    },
]


DEFAULT_SOIL_NAMES = [
    "Unknown material",
    "Soil, fine grain",
    "Soil, very stiff, finegrained",
    "Gravel, slightly silty, loose",
    "Gravel, slightly silty, moderately",
    "Gravel, slightly silty, firm",
    "Gravel, very silty, loose",
    "Gravel, very silty, moderately",
    "Gravel, very silty, firm",
    "Sand, clean, loose",
    "Sand, clean, moderately",
    "Sand, clean, firm",
    "Sand, slightly silty",
    "Sand, slightly silty, clayey",
    "Sand, very silty",
    "Sand, very silty, clayey",
    "Sand, silty to loamy",
    "Sand, slightly silty to silty",
    "Sand, gravelly",
    "Sand, solid to clayey",
    "Loam, slightly sandy, soft",
    "Loam, slightly sandy, moderately",
    "Loam, slightly sandy, firm",
    "Loam, very sandy",
    "Clay, clean, soft",
    "Clay, clean, moderately",
    "Clay, clean, firm",
    "Clay, slightly sandy",
    "Clay, slightly sandy, soft",
    "Clay, slightly sandy, moderately",
    "Clay, slightly sandy, firm",
    "Clay, very sandy",
    "Clay, organic, soft",
    "Clay, organic, moderately",
    "Clay, slightly silty to silty",
    "Clay, silty to loamy",
    "Peat, not preloaded, soft",
    "Peat, moderately preloaded, moderately",
    "Peat, organic material",
]


MAX_CONE_RESISTANCE_TYPE = ["Standard", "Manual"]
