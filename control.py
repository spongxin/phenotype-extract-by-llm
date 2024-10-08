from rapidfuzz import process, fuzz, utils
from pydantic import BaseModel, Field
from typing import Union
from enum import Enum
import json
import os
import re


class ItemsControl:
    enzyme_list_path = os.path.join('reference', 'enzymes', 'enzymes.txt')
    antibiotics_list_path = os.path.join('reference', 'antibiotics', 'antibiotics.txt')
    cnsource_list_path = os.path.join('reference', 'cnsource', 'cnsource.txt')
    
    with open(enzyme_list_path, 'r', encoding='utf-8') as f:
        enzyme_list = f.read().splitlines()
    with open(antibiotics_list_path, 'r', encoding='utf-8') as f:
        antibiotics_list = f.read().splitlines()
    with open(cnsource_list_path, 'r', encoding='utf-8') as f:
        cnsource_list = f.read().splitlines()

    @staticmethod
    def control_enzyme_list(enzymes: list) -> list[tuple[str, float, int]]:
        """Control the enzyme list with the given enzyme list.
        :param enzymes: list of enzyme names
        :return: list of tuples containing the enzyme name and its candidate
        """
        return ItemsControl._control_items_by_list(enzymes, ItemsControl.enzyme_list)
    
    @staticmethod
    def control_antibiotics_list(antibiotics: list) -> list[tuple[str, float, int]]:
        """Control the antibiotics list with the given antibiotics list.
        :param antibiotics: list of antibiotics names
        :return: list of tuples containing the antibiotics name and its candidate
        """
        return ItemsControl._control_items_by_list(antibiotics, ItemsControl.antibiotics_list)

    @staticmethod
    def control_cnsource_list(cnsources: list) -> list[tuple[str, float, int]]:
        """Control the cnsource list with the given cnsource list.
        :param cnsources: list of cnsource names
        :return: list of tuples containing the cnsource name and its candidate
        """
        return ItemsControl._control_items_by_list(cnsources, ItemsControl.cnsource_list)

    @staticmethod
    def _control_items_by_list(items: list, control_list: list) -> list[tuple[str, float, int]]:
        """Control the items with the given control list.
        :param items: list of items
        :param control_list: list of control items
        :return: list of tuples containing the item and its candidate
        """
        candidates = [process.extractOne(item, control_list, scorer=fuzz.WRatio, processor=utils.default_process) for item in items]
        return candidates


def extract_json_data(query: str) -> Union[dict, str]:
    try:
        json_data = re.search(r'{.*}', query, re.DOTALL).group()
        return json.loads(json_data)
    except Exception as e:
        return f"The following error occurred when extracting JSON Data from your output:\n {e}"

class Evidence(BaseModel):
    """原文依据"""
    evidences: list[str] = Field(
        default_factory=list,
        description="The evidences that support only its parent module.",
    )

class pH(BaseModel):
    """pH值"""
    range: list[str] | None = Field(
        default_factory=list,
        description="A growth range phenotype related to the pH range that allows growth of a cell or organism.",
        examples=[["7.0-8.0", "8.5-9.5"], ],
    )
    optimum: str | None = Field(
        description="The optimum pH value (or range) to grow.",
        examples=["7.0", "8.0-9.0"],
    )
    evidences: Evidence

class SaltTolerance(BaseModel):
    """耐盐性"""
    range: list[str] | None = Field(
        default_factory=list,
        description="A growth range phenotype related to the salt concentration range that allows growth of a cell or organism. (usually NaCl)",
        examples=[["10 % NaCl", "1-6 % NaCl"], ],
    )
    optimum: str | None = Field(
        description="The optimum salt concentration value (or range) to grow.",
        examples=["0-6 % NaCl"],
    )
    evidences: Evidence

class Temperature(BaseModel):
    """生长温度"""
    range: list[str] | None = Field(
        default_factory=list,
        description="A growth phenotype where the trait in question is the range of temperatures where growth of the microbe can occur.",
        examples=[["37-42", "16"], ],
    )
    optimum: str | None = Field(
        description="The optimum temperature value (or range) to grow.",
        examples=["25", "31-37"],
    )
    evidences: Evidence

class Spore(BaseModel):
    """孢子形成"""
    spore_formation: Enum | None = Enum(
        "spore_formation",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="Ability of spore formation.",
        examples=["Positive", "Negative"],
    )
    spore_type: str | None = Field(
        description="The type of spore formation.",
        examples=["endospore"],
    )
    spore_description: str | None = Field(
        description="Detail description of the spore formation.",
        examples=["ellipsoidal spores centrally and paracentrally in unswollen sporangia."],
    )
    evidences: Evidence

class GuanineCytosine(BaseModel):
    """鸟嘌呤胞嘧啶含量"""
    gc_content: float | None = Field(
        description="Guanine-Cytosine content in mol%.",
        examples=[58.1, ],
    )
    gc_method: str | None = Field(
        description="Method used to quantify GC content.",
        examples=["HPLC"],
    )
    evidence: Evidence

class CNSourceItem(BaseModel):
    """碳氮源单项"""
    source_name: str = Field(
        description="C/N source contains organic compound or inorganic compound.",
        examples=["D-glucose", "NH4Cl"],
    )
    utilization: Enum = Enum(
        "utilization",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="Ability of utilizing the referring C/N source.",
        examples=["Positive", "Negative"],
    )
    evidence: Evidence

class AntibioticItem(BaseModel):
    """抗生素单项"""
    substance: str = Field(
        description="The name of the antibiotic.",
        examples=["isoniazid", "rifampicin"],
    )
    concentration: str | None = Field(
        description="The concentration of the antibiotic.",
        examples=["0.1 µg/ml", "10 µG"],
    )
    sensitivity: Enum = Enum(
        "sensitivity",
        {
            "sensitive": "Sensitive",
            "resistant": "Resistant",
        },
        description="The sensitivity of the microorganism to the referring antibiotic.",
        examples=["Sensitive", "Resistant"],
    )
    evidence: Evidence

class Flagellum(BaseModel):
    """鞭毛"""
    flagellum: Enum | None = Enum(
        "flagellum",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="Ability of flagellum formation or If the cell has flagellum.",
        examples=["Positive", "Negative"],
    )
    arrangement: str | None = Field(
        description="Flagellum arrangement of the microorganism.",
        examples=["peritrichous flagella"],
    )
    evidences: Evidence

class Hemolysis(BaseModel):
    """溶血"""
    hemolysis: Enum | None = Enum(
        "hemolysis",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="Ability of hemolysis, which is the lysis of red blood cells and the release of the cytoplasm into surrounding fluid.",
        examples=["Positive", "Negative"],
    )
    hemolysis_type: str | None = Field(
        description="The type of hemolysis.",
        examples=["alpha"],
    )
    evidences: Evidence

class EnzymologyItem(BaseModel):
    """酶学单项"""
    enzyme_name: str = Field(
        description="The name of enzyme, which are synthesized by the bacterium.",
        examples=["catalase", "urease"],
    )
    enzyme_activity: Enum = Enum(
        "enzyme_activity",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="The activity of the enzyme.",
        examples=["Positive", "Negative"],
    )
    evidence: Evidence

class Cell(BaseModel):
    """细胞信息"""
    shape: str | None = Field(
        description="A cell morphology phenotype where the trait in question is the shape of a cell.",
        examples=["rod-shaped", "coccus"],
    )
    length: str | None = Field(
        description="The length of the cell.",
        examples=["1.5-2.0 µm"],
    )
    width: str | None = Field(
        description="The width of the cell.",
        examples=["0.5-0.7 µm"],
    )
    diameter: str | None = Field(
        description="The diameter of the cell.",
        examples=["0.5-0.7 µm"],
    )

class Colony(BaseModel):
    """菌落信息"""
    color: str | None = Field(
        description="The color of the colony, including pigmentation color.",
        examples=["saffron yellow"],
    )
    shape: str | None = Field(
        description="A colony morphology phenotype dealing with the basic shape or form of a colony or colonies.",
        examples=["rod-shaped", "coccus"],
    )
    size: str | None = Field(
        description="A colony morphology phenotype dealing with the size of the colony.",
    )
    cultivation_medium: str | None = Field(
        description="The medium used to cultivate the colony.",
        examples=["Yeast Extract-malt Extract Agar"],
    )
    incubation_period: str | None = Field(
        description="Incubation period in context with the description of colony morphology.",
        examples=["24 h", "4 days"],
    )
    evidence: Evidence

class Sample(BaseModel):
    """样本信息"""
    isolation_source: str | None = Field(
        description="The source from which the sample was isolated.",
        examples=["soil", "human"],
    )
    date: str | None = Field(
        description="The date when the sample was isolated.",
        examples=["2008-01-23"],
    )
    name: str | None = Field(
        description="The name of the sample.",
        examples=["sample1"],
    )
    host_disease: str | None = Field(
        description="The disease of the host caused by the bacterium.",
        examples=["Hypersensitivity pneumonitis; Wound infection"],
    )
    host_body_site: str | None = Field(
        description="The body site of the host where the sample was isolated from.",
        examples=["skin", "gut"],
    )
    environment: str | None = Field(
        description="The environmental information when the sample was collected.",
        examples=["root:Environmental:Aquatic:Thermal springs:Hot (42-90C):Sediment"],
    )
    evidence: Evidence

class Geography(BaseModel):
    """地理信息"""
    continent: str | None = Field(
        description="The continent where the sample material/strain was collected.",
        examples=["Asia"],
    )
    country: str | None = Field(
        description="The country or region where the sample material/strain was collected.",
        examples=["China"],
    )
    geographic_location: str | None = Field(
        description="The geographical origin of the sample as defined by the country or sea name followed by specific region name.",
        examples=["China: Sichuan: Chengdu"],
    )
    coordinate_latitude: str | None = Field(
        description="The latitude coordinate or range of the geographical location.",
        examples=["30.67", "30.67-30.70"],
    )
    coordinate_longitude: str | None = Field(
        description="The longitude coordinate or range of the geographical location.",
        examples=["104.07", "104.07-104.10"],
    )

class Enrichment(BaseModel):
    """富集培养"""
    enrichment_medium: str | None = Field(
        description="The medium used to enrich the sample.",
        examples=["Basal Culture Medium"],
    )
    enrichment_temperature: str | None = Field(
        description="The temperature used to enrich the sample.",
        examples=["37 °C"],
    )
    enrichment_period: str | None = Field(
        description="The period used to enrich the sample.",
        examples=["24 h"],
    )
    procedure_origin: str | None = Field(
        description="The method how the organism was isolated and enriched.",
        examples=["dilution-plating method"],
    )
    date: str | None = Field(
        description="The date when the sample was enriched.",
        examples=["2008-01-23"],
    )
    evidence: Evidence

class Ecology(BaseModel):
    """生态信息"""
    sample: Sample
    geography: Geography
    enrichment: Enrichment
    evidence: Evidence

class Morphology(BaseModel):
    """形态特性"""
    cell: Cell
    colony: Colony
    multicelluar_ability: Enum | None = Enum(
        "multicelluar_ability",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="Ability of forming multicellular structures (aggregations).",
        examples=["Positive", "Negative"],
    )
    gram_stain: Enum | None = Enum(
        "gram_stain",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="A cell staining phenotype where microorganisms are grouped based on their ability to retain crystal violet stain when decolorized with an organic solvent such as ethanol.",
        examples=["Positive", "Negative"],
    )
    motility: Enum | None = Enum(
        "motility",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="The ability of the microbe to move independently, using metabolic energy. A locomotion phenotype where the trait in question is the self-propelled movement of a microbe from one location to another.",
        examples=["Positive", "Negative"],
    )
    evidence: Evidence
        

class Physiology(BaseModel):
    """生理特性"""
    pathogenicity: Enum | None = Enum(
        "pathogenicity",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="Ability of causing pathological conditions within the referring organism.",
        examples=["Positive", "Negative"],
    )
    spore: Spore
    nutrition_type: str | None = Field(
        description="The nutrition type of the microorgainsm.",
        examples=["autotroph"],
    )
    cn_source: list[CNSourceItem] | None = Field(
        default_factory=list,
        description="Carbon/Nitrogen source list. A nutrient utilization phenotype related to the ability of a microbe to utilize a nutrient as a source of carbon or nitrogen.",
    )
    growth_rate: str | None = Field(
        description="The growth rate of the microorganism.",
        examples=["rapidly growing"],
    )
    oxygen_requirement: str | None = Field(
        description="The bacterium respiration requires the use of oxygen or not.",
        examples=["aerobic"],
    )
    flagellum: Flagellum
    temperature: Temperature
    pH: pH
    salt_tolerance: SaltTolerance
    antibiotic: list[AntibioticItem] = Field(
        default_factory=list,
        description="A phenotype related to the sensitivity of a microbe to antibiotics.",
    )
    metabolite_production: list[str] = Field(
        default_factory=list,
        description="Name of metabolite that was collected from manually curation.",
        examples=[["dihydroxyacetone", ], ],
    )
    guanine_cytosine: GuanineCytosine
    hemolysis: Hemolysis
    evidence: Evidence

class Enzymology(BaseModel):
    """酶学特性"""
    enzyme: list[EnzymologyItem] = Field(
        default_factory=list,
        description="Enzymology is a microbial phenotype that affects an enzymatic activity.",
    )
    evidence: Evidence

class Biochemical(BaseModel):
    """生化特性"""
    indole_production: Enum | None = Enum(
        "indole_production",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="A biochemical test performed on bacterial species to determine the ability of the bacterium to convert tryptophan into indole.",
        examples=["Positive", "Negative"],
    )
    voges_proskauer_test: Enum | None = Enum(
        "voges_proskauer_test",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="A biochemical test performed on bacterial species to determine the ability of the bacterium to convert glucose to acetoin.",
        examples=["Positive", "Negative"],
    )
    nitrate_reduction: Enum | None = Enum(
        "nitrate_reduction",
        {
            "positive": "Positive",
            "negative": "Negative",
        },
        description="A biochemical test performed on bacterial species to determine the ability of the bacterium to reduce nitrate to nitrite.",
        examples=["Positive", "Negative"],
    )
    fatty_acids_profile: str | None = Field(
        description="The fatty acid compositions of the bacterium.",
        examples=["Iso-c15 : 0, Anteiso-c15 : 0, Iso-c15 : 0 3-oh, C17 : 1v6c,iso-c16 : 0 3-oh, Iso-c17 : 0 3-oh And Summed Feature 4(comprising Iso-c17 : 1 I And Anteiso-c17 : 1 B)"],
    )
    cell_wall_animo_acid: str | None = Field(
        description="The amino acid composition in the cell wall of the microoriganism.",
        examples=["meso-diaminopimelic acid, alanine, glutamic acid"],
    )
    whole_cell_sugar: str | None = Field(
        description="The sugars of the microorganism, which are analyzed after whole-cell hydrolysis.",
        examples=["arabinose, madurose"],
    )
    cell_wall_sugar: str | None = Field(
        description="The cell-wall sugar composition of the microorganism.",
        examples=["arabinose, madurose"],
    )
    murein_types: str | None = Field(
        description="Characterization of the different murein types, including amino acid sequence and amino sugar composition.",
        examples=["A4alpha L-Lys-L-Glu"],
    )
    evidence: Evidence

class Biosafety(BaseModel):
    """生物安全性"""
    biosafety_level: str | None = Field(
        description="The biosafety level of the microorganism.",
        examples=["BSL-1", "BSL-2"],
    )
    biosafety_comment: str | None = Field(
        description="Additional comments or detail description on the biosafety level.",
        examples=["Risk group (German classification)"],
    )
    evidence: Evidence

class DefaultPhenotype(BaseModel):
    """基础表型信息"""
    strain_designations: list[str] = Field(
        default_factory=list,
        description="Type strain's name according to the International Nomenclature of Microorganisms.",
        examples=[["YIT 11304T", "DSM 16506T", "NCIMB 13994T", "VKM B-2757T", "BCRC 17990T", "JCM 15121T"], ],
    )
    scientific_name: str | None = Field(
        description="The scientific name of the microorganism. genus name + species name, a blank between genus name and species name;",
        examples=["Zeaxanthinibacter enoshimensis"],
    )
    species: str | None = Field(
        description="The species name of the microorganism (Type Strain's).",
        examples=["enoshimensis"],
    )
    etymology: str | None = Field(
        description="The etymology of the microorganism's name (Type Strain's).",
        examples=["en.o.shi.men'sis. N.L. masc. adj. enoshimensis pertaining to Enoshima Island in Japan, where the type strain was isolated."],
    )
    type_strain: bool | None = Field(
        description="wheather the strain `strain_designation` is a type strain.",
        examples=[True, False],
    )
    morphology: Morphology
    physiology: Physiology
    enzymology: Enzymology
    biochemical: Biochemical
    biosafety: Biosafety
    evidence: Evidence

class PathogenicPhenotype(BaseModel):
    """致病表型信息"""
    strain_designations: list[str] = Field(
        default_factory=list,
        description="Type strain's name according to the International Nomenclature of Microorganisms.",
        examples=[["YIT 11304T", "DSM 16506T", "NCIMB 13994T", "VKM B-2757T", "BCRC 17990T", "JCM 15121T"], ],
    )
    scientific_name: str | None = Field(
        description="The scientific name of the microorganism. genus name + species name, a blank between genus name and species name;",
        examples=["Zeaxanthinibacter enoshimensis"],
    )
    species: str | None = Field(
        description="The species name of the microorganism (Type Strain's).",
        examples=["enoshimensis"],
    )
    type_strain: bool | None = Field(
        description="wheather the strain `strain_designation` is a type strain.",
        examples=[True, False],
    )
    # TODO: Add pathogenicity related phenotypes
    evidence: Evidence