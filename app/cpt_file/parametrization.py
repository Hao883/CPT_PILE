from viktor.errors import UserError
from viktor.parametrization import (
    And,
    BooleanField,
    DownloadButton,
    FileField,
    HiddenField,
    IsEqual,
    IsFalse,
    Lookup,
    NumberField,
    OptionField,
    OptionListElement,
    Parametrization,
    SetParamsButton,
    Step,
    TableInput,
    Text,
    TextField,
    LineBreak
)

from .constants import (
    DEFAULT_CLASSIFICATION_TABLE,
    DEFAULT_MIN_LAYER_THICKNESS,
    DEFAULT_ROBERTSON_TABLE,
    DEFAULT_SOIL_NAMES,
    MAX_CONE_RESISTANCE_TYPE,
    Pile_Class,
    Pile_Bore_Category,
    Pile_CFA_Category,
    Pile_Screw_Category,
    Pile_Closed_Driven_Category,
    Pile_Open_Driven_Category,
    Pile_Driven_H_Category,
    Pile_Driven_Sheet_Category,
    Pile_Micro_piles_Category
)

CLASSIFICATION_METHODS = [
    OptionListElement(label="Robertson Method (1990)", value="robertson"),
    # OptionListElement(label="Table Method", value="table"),
]


def validate_step_1(params, **kwargs):
    """Validates step 1."""
    if not params.measurement_data:
        raise UserError("Classify soil layout before proceeding.")


class CPTFileParametrization(Parametrization):
    """Defines the input fields in left-side of the web UI in the CPT_file entity (Editor)."""

    classification = Step("Upload and classification", on_next=validate_step_1)
    classification.text_01 = Text(
        """# Welcome to the CPT Pile Parametric design app!

With this app you will be able to (1) classify and interpret GEF-formatted CPT files by uploading and automatically 
classifying the soil profile; (2) conduct parametric Pile design by selecting desired pile parameters. 

## Step 1: Upload a GEF file

For the users who want to try out the app, but do not have a GEF file at hand, feel free to use the sample 
GEF file available.
    """
    )
    classification.gef_file = FileField(
        "Upload GEF file",
        file_types=[".gef"],
        visible=IsFalse(Lookup("classification.get_sample_gef_toggle")),
    )
    classification.get_sample_gef_toggle = BooleanField("Get sample GEF file", default=False, flex=15)
    classification.download_sample_gef = DownloadButton(
        "Download sample GEF file",
        "download_sample_gef_file",
        visible=Lookup("classification.get_sample_gef_toggle"),
        flex=15,
    )
    classification.text_02 = Text(
        """## Step 2: Select Classification Method
        
Select your preferred classification method.
        """
    )
    classification.method = OptionField(
        "Classification method",
        options=CLASSIFICATION_METHODS,
        default="robertson",
        autoselect_single_option=True,
        variant="radio-inline",
        description="Robertson method (1990) \n"
        "\n Robertson method (2016)",
    )
    # classification.change_table = BooleanField("Change classification table")
    classification.robertson = TableInput(
        "Robertson table",
        default=DEFAULT_ROBERTSON_TABLE,
        # visible=And(
        #     Lookup("classification.change_table"),
        #     IsEqual(Lookup("classification.method"), "robertson"),
        visible=False,
        )
    classification.robertson.name = TextField("Robertson Zone")
    classification.robertson.ui_name = OptionField("Soil", options=DEFAULT_SOIL_NAMES)
    classification.robertson.color = TextField("Color (R, G, B)")
    classification.robertson.gamma_dry = NumberField("γ dry [kN/m³]", num_decimals=1)
    classification.robertson.gamma_wet = NumberField("γ wet [kN/m³]", num_decimals=1)
    classification.robertson.phi = NumberField("Friction angle Phi [°]", num_decimals=1)

    classification.table = TableInput(
        "Classification table",
        default=DEFAULT_CLASSIFICATION_TABLE,
        # visible=And(
        #     Lookup("classification.change_table"),
        #     IsEqual(Lookup("classification.method"), "table"),
        visible=False,
        )
    classification.table.name = OptionField("Naam", options=DEFAULT_SOIL_NAMES)
    classification.table.color = TextField("Kleur (R, G, B)")
    classification.table.qc_min = NumberField("qc min [MPa]", num_decimals=2)
    classification.table.qc_max = NumberField("qc max [MPa]", num_decimals=2)
    classification.table.qc_norm_min = NumberField("qc norm; min [MPa]", num_decimals=1)
    classification.table.qc_norm_max = NumberField("qc norm; max [MPa]", num_decimals=1)
    classification.table.rf_min = NumberField("Rf min [%]", num_decimals=1)
    classification.table.rf_max = NumberField("Rf max [%]", num_decimals=1)
    classification.table.gamma_dry = NumberField("γ dry [kN/m³]", num_decimals=1)
    classification.table.gamma_wet = NumberField("γ wet [kN/m³]", num_decimals=1)
    classification.table.phi = NumberField("Friction angle Phi [°]", num_decimals=1)
    classification.table.max_cone_res_type = OptionField(
        "Maximum cone resistance type", options=MAX_CONE_RESISTANCE_TYPE
    )
    classification.table.max_cone_res_mpa = NumberField("Maximum cone resistance [MPa]", num_decimals=1)
    classification.text_03 = Text(
        """## Step 3: Classify the soil layout
        
Classify the uploaded GEF file by clicking the "Classify soil layout" button. Proceed then to the next step.
        """
    )
    classification.classify_soil_layout_button = SetParamsButton("Classify soil layout", "classify_soil_layout")

    cpt = Step("CPT interpretation", views=["visualize_cpt", "visualize_map"])
    cpt.text = Text(
        "Use the table below to change the interpreted soil layout by changing the positions of the layers, "
        "adding rows or changing the material type."
    )

    cpt.filter_thin_layers = SetParamsButton(
        "Filter Layer Thickness",
        method="filter_soil_layout_on_min_layer_thickness",
        flex=60,
        description="Filter the soil layout to remove layers that are " "thinner than the minimum layer thickness",
    )
    cpt.min_layer_thickness = NumberField(
        "Minimum Layer Thickness",
        suffix="mm",
        min=0,
        step=50,
        default=DEFAULT_MIN_LAYER_THICKNESS,
        flex=40,
    )

    cpt.reset_original_layers = SetParamsButton(
        "Reset to original Soil Layout",
        method="reset_soil_layout_user",
        flex=100,
        description="Reset the table to the original soil layout",
    )

    cpt.ground_water_level = NumberField("Phreatic level", name="ground_water_level", suffix="RL", flex=50)
    cpt.ground_level = NumberField("Ground level", name="ground_level", suffix="RL", flex=50)
    cpt.soil_layout = TableInput("Soil layout", name="soil_layout")
    cpt.soil_layout.name = OptionField("Material", options=DEFAULT_SOIL_NAMES)
    cpt.soil_layout.top_of_layer = NumberField("Top (m RL)", num_decimals=1)

    # hidden fields
    cpt.gef_headers = HiddenField("GEF Headers", name="headers")
    cpt.bottom_of_soil_layout_user = HiddenField("GEF Soil bottom", name="bottom_of_soil_layout_user")
    cpt.measurement_data = HiddenField("GEF Measurement data", name="measurement_data")
    cpt.soil_layout_original = HiddenField("Soil layout original", name="soil_layout_original")

    PILE = Step("Pile Capacity Design",views=["visualize_pile"])

    PILE.text_02 = Text(
        """## Select your pile geometry
        """
    )

    PILE.Diameter = NumberField(
        "Pile Diameter (m)",
        default=1,
        min=0.5,
        step=0.1,
        max=3,
        # flex=33,
        variant='slider'
    )

    # PILE.Length = NumberField(
    #     "Pile Length (m)",
    #     default=10,
    #     min=3,
    #     step=0.2,
    #     max=50,
    #     # flex=33,
    #     variant='slider'
    # )

    PILE.Load = NumberField(
        "Pile Load Required (kN)",
        default=5000,
        min=10,
        # step=,
        max=30000,
        flex=50,
        variant='slider'
    )
    PILE.lb = LineBreak()  # split the fields in 2 pairs
    PILE.text_0 = Text(
        """## Select your pile class

Select your preferred pile types 
        """
    )
    PILE.method = OptionField(
        "Pile Class",
        options=Pile_Class,
        default="Bore",
        autoselect_single_option=True,
        variant="radio-inline",
        description="Different pile classes.",
    )

    PILE.text_01 = Text(
        """## Select your pile category

Select your preferred pile category based on the selected pile class
        """
    )
    PILE.bore_category = OptionField(
        "Bored Pile category",
        options=Pile_Bore_Category,
        default="No support",
        autoselect_single_option=True,
        variant="radio-inline",
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "Bore")
    )

    PILE.CFA_category = OptionField(
        "CFA Pile category",
        options=Pile_CFA_Category,
        default="CFA piles",
        autoselect_single_option=True,
        variant="radio-inline",
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "CFA")
    )

    PILE.Screw_category = OptionField(
        "Screw Pile category",
        options=Pile_Screw_Category,
        variant="radio-inline",
        default="cast-in",
        autoselect_single_option=True,
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "Screw")
    )

    PILE.Closed = OptionField(
        "Pile Closed Driven Category",
        options=Pile_Closed_Driven_Category,
        variant="radio-inline",
        default="pre",
        autoselect_single_option=True,
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "Closed")
    )

    PILE.Open = OptionField(
        "Open-ended driven piles Category",
        options=Pile_Open_Driven_Category,
        variant="radio-inline",
        default="steel",
        autoselect_single_option=True,
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "Open")
    )

    PILE.H = OptionField(
        "Driven H piles",
        options=Pile_Driven_H_Category,
        variant="radio-inline",
        default="driven",
        autoselect_single_option=True,
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "H")
    )

    PILE.Sheet = OptionField(
        "Driven sheet pile walls",
        options=Pile_Driven_Sheet_Category,
        variant="radio-inline",
        default="sheet",
        autoselect_single_option=True,
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "sheet")
    )

    PILE.Micro = OptionField(
        "Micropiles",
        options=Pile_Micro_piles_Category,
        variant="radio-inline",
        default="gravity",
        autoselect_single_option=True,
        description="Different pile category.",
        # flex=33,
        visible=IsEqual(Lookup("PILE.method"), "Micro")
    )


