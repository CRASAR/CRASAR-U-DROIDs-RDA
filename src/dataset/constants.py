FAILED_IMAGES = ["1002-Palm-Acers.4.geo.tif_2048_2048_tile_(0,17554)_2048x2048_rgb.png",
                 "1002-Palm-Acers.3.geo.tif_2048_2048_tile_(19504,9752)_2048x2048_rgb.png",
                 "1001-Summerlin-San-Carlos.geo.tif_2048_2048_tile_(7801,35108)_2048x2048_rgb.png",
                 "1001-Ft-Myers-Beach-Boone.geo.tif_2048_2048_tile_(37059,74118)_2048x2048_rgb.png",
                 "1001-Ft-Myers-Beach-Boone.geo.tif_2048_2048_tile_(52662,89721)_2048x2048_rgb.png",
                 "10142018-MexicoBeach.geo.tif_2048_2048_tile_(33158,50712)_2048x2048_rgb.png",
                 "10142018-MexicoBeach.geo.tif_2048_2048_tile_(42910,9752)_2048x2048_rgb.png",
                 "20210831-LA-DIV-01.geo.tif_2048_2048_tile_(40960,17554)_2048x2048_rgb.png",
                 "20210831-LA-DIV-01.geo.tif_2048_2048_tile_(68266,25356)_2048x2048_rgb.png",
                 "20211214-Mayfield.geo.tif_2048_2048_tile_(85820,46811)_2048x2048_rgb.png"]

DATASET_BASE_NAME = "CRASAR-U-DROIDs"
TRAIN_FOLDER_NAME = "train"
TEST_FOLDER_NAME = "test"
IMAGERY_FOLDER_NAME = "imagery"
UAS_FOLDER_NAME = "UAS"

ANNOTATIONS_FOLDER_NAME = "annotations"
BDA_FOLDER_NAME = "building_damage_assessment"
RDA_FOLDER_NAME = "road_damage_assessment"
BDA_ADJ_FOLDER_NAME = "building_alignment_adjustments"
RDA_ADJ_FOLDER_NAME = "road_alignment_adjustments"

LAT_LON_CRS = "EPSG:4326"

PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS = "Passable with Difficulty (Obstructions)"
PASSABLE_WITH_DIFFICULTY_FLOODING = "Passable with Difficulty (Flooding)"
PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION = "Passable with Difficulty (Road Condition)"
NOT_PASSABLE_OBSTRUCTIONS = "Not Passable (Obstructions)"
NOT_PASSABLE_FLOODING = "Not Passable (Flooding)"
NOT_PASSABLE_DESTRUCTION = "Not Passable (Destruction)"
NOT_ABLE_TO_DETERMINE = "Not Able To Determine"
PARTICULATE_PARTIAL = "Particulate Partial"
PARTICULATE_TOTAL = "Particulate Total"
ROAD_LINE = "Road Line"
BACKGROUND = "Background"
PARTIAL = "Partial"
TOTAL = "Total"

TOTAL_FLOODING = "Total Flooding"
TOTAL_OBSTRUCTION = "Total Obstruction"
TOTAL_DESTRUCTION = "Total Destruction"
PARTIAL_FLOODING = "Partial Flooding"
PARTIAL_OBSTRUCTION = "Partial Obstruction"
PARTIAL_DESTRUCTION = "Partial Road Condition"

ROAD_DEBUG_WIDTH = 1

RDA_LABELBOX_CLASSES = [PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS,
                        PASSABLE_WITH_DIFFICULTY_FLOODING,
                        PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION,
                        NOT_PASSABLE_OBSTRUCTIONS,
                        NOT_PASSABLE_FLOODING,
                        NOT_PASSABLE_DESTRUCTION,
                        NOT_ABLE_TO_DETERMINE,
                        PARTICULATE_PARTIAL,
                        PARTICULATE_TOTAL]
RDA_DATASET_CLASSES = [NOT_ABLE_TO_DETERMINE,
                       PARTICULATE_PARTIAL,
                       PARTICULATE_TOTAL,
                       TOTAL_FLOODING,
                       TOTAL_OBSTRUCTION,
                       TOTAL_DESTRUCTION,
                       PARTIAL_FLOODING,
                       PARTIAL_OBSTRUCTION,
                       PARTIAL_DESTRUCTION]
RDA_MODEL_CLASSES = RDA_DATASET_CLASSES + [ROAD_LINE]

RDA_COLOR_MAP = {
    PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS: [255, 0, 255],
    PASSABLE_WITH_DIFFICULTY_FLOODING: [0, 255, 255],
    PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION: [255, 255, 0],
    NOT_PASSABLE_OBSTRUCTIONS: [127, 0, 255],
    NOT_PASSABLE_FLOODING: [0,0,255],
    NOT_PASSABLE_DESTRUCTION: [255,0,0],
    NOT_ABLE_TO_DETERMINE: [0,0,0],
    PARTICULATE_TOTAL: [255, 75, 51],
    PARTICULATE_PARTIAL: [255, 102, 0],
    ROAD_LINE: [0,255,0],
}



NO_DAMAGE = "no damage"
MINOR_DAMAGE = "minor damage"
MAJOR_DAMAGE = "major damage"
DESTROYED = "destroyed"
UNCLASSIFIED = "un-classified"

BDA_DAMAGE_CLASSES = [NO_DAMAGE, MINOR_DAMAGE, MAJOR_DAMAGE, DESTROYED, UNCLASSIFIED]

LABEL_SCORE_PRIORITY_MAP = {
    NO_DAMAGE: 0,
    MINOR_DAMAGE: 1,
    MAJOR_DAMAGE: 2,
    DESTROYED: 3,
    UNCLASSIFIED: -1
}

BDA_CATEGORY_COLOR_MAP = {
    NO_DAMAGE: [0,255,0,255],
    MINOR_DAMAGE: [255,255,0,255],
    MAJOR_DAMAGE: [255,165,0,255],
    DESTROYED: [255,0,0,255],
    UNCLASSIFIED: [128,0,128,255]
}



BDA_IDX_TO_COLOR_MAP = [
    [0,0,0],
    [0,255,0],
    [255,255,0],
    [255,165,0],
    [255,0,0],
    [128,0,128]
]

ROAD_DEBUG_SIZE_MAP = {
    "motorway":370,
    "trunk":320,
    "primary":300,
    "secondary":250,
    "tertiary":150,
    "residential":120,
    "motorway_link":100,
    "trunk_link":120,
    "primary_link":100,
    "secondary_link":100,
    "tertiary_link":100,
    "living_street":100,
    "turning_loop":100,
    "turning_circle":100,
    "service": 100,
    "road": 100
}

ADJ_LABELBOX_PREFIX_ORTHO_TITLE = {
    "090302-Pecan-Grove-Levee.geo.tif": "090302-Pecan-Grove-Levee.geo.tif",
    "090401-DMS-Assessment-Westpark.geo.tif": "090401-DMS-Assessment-Westpark.geo.tif",
    "090402-DMS-Assessment-Sienna-Village.geo.tif": "090402-DMS-Assessment-Sienna-Village.geo.tif",
    "090403-Lancaster-Canyon-Gate.geo.tif": "090403-Lancaster-Canyon-Gate.geo.tif",
    "1001-Ft-Myers-Beach-Boone.geo.tif": "1001-Ft-Myers-Beach-Boone.geo.tif",
    "1001-Ft-Myers-Beach-DIRT.geo.tif": "1001-Ft-Myers-Beach-DIRT.geo.tif",
    "1001-Harlem-Heights.geo.tif": "1001-Harlem-Heights.geo.tif",
    "1001-Iona-Point.geo.tif": "1001-Iona-Point.geo.tif",
    "1001-Kennedy-Green-Mobile-Homes.geo.tif": "1001-Kennedy-Green-Mobile-Homes.geo.tif",
    "1001-McGregor-College-Pkwy-South.1.geo.tif": "1001-McGregor-College-Pkwy-South.1.geo.tif",
    "1001-McGregor-College-Pkwy-South.2.geo.tif": "1001-McGregor-College-Pkwy-South.2.geo.tif",
    "1001-McGregor-College-Pkwy-South.3.geo.tif": "1001-McGregor-College-Pkwy-South.3.geo.tif",
    "1001-Palmeto-Palms.geo.tif": "1001-Palmeto-Palms.geo.tif",
    "1001-San-Carlos-Island.geo.tif": "1001-San-Carlos-Island.geo.tif",
    "1001-Summerlin-San-Carlos.geo.tif": "1001-Summerlin-San-Carlos.geo.tif",
    "1002-Boca-Grande.1.geo.tif": "1002-Boca-Grande.1.geo.tif",
    "1002-Boca-Grande.2.geo.tif": "1002-Boca-Grande.2.geo.tif",
    "1002-Boca-Grande.3.geo.tif": "1002-Boca-Grande.3.geo.tif",
    "1002-Boca-Grande.4.geo.tif": "1002-Boca-Grande.4.geo.tif",
    "1002-Boca-Grande.5.geo.tif": "1002-Boca-Grande.5.geo.tif",
    "1002-Boca-Grande.6.geo.tif": "1002-Boca-Grande.6.geo.tif",
    "1002-Ft-Myers-Beach-LCSO.geo.tif": "1002-Ft-Myers-Beach-LCSO.geo.tif",
    "1002-Ft-Myers-Beach-TFD.geo.tif": "1002-Ft-Myers-Beach-TFD.geo.tif",
    "1002-Kelly-Road.geo.tif": "1002-Kelly-Road.geo.tif",
    "1002-Palm-Acres.1.geo.tif": "1002-Palm-Acers.1.geo.tif",
    "1002-Palm-Acers.2.geo.tif": "1002-Palm-Acers.2.geo.tif",
    "1002-Palm-Acers.3.geo.tif": "1002-Palm-Acers.3.geo.tif",
    "1002-Palm-Acers.4.geo.tif": "1002-Palm-Acers.4.geo.tif",
    "1002-Sanibel-Causeway-North.geo.tif": "1002-Sanibel-Causeway-North.geo.tif",
    "10132018-MexicoBeach.geo.tif": "10132018-MexicoBeach.geo.tif",
    "10142018-MexicoBeach.geo.tif": "10142018-MexicoBeach.geo.tif",
    "20230830-SteinhatcheeRiver.geo.tif": "20230830-SteinhatcheeRiver.geo.tif",
    "20230831-Jena-SteinhatcheeRiverSouth.geo.tif": "20230831-Jena-SteinhatcheeRiverSouth.geo.tif",
    "05-08-2020-MBF-01.geo.tif": "05-08-2020-MussettBayouFire-01.geo.tif",
    "05-08-2020-MBF-N98.geo.tif": "05-08-2020-MussettBayouFire-NorthOf98.geo.tif",
    "05-08-2020-MBF-S98-ALD.geo.tif": "05-08-2020-MussettBayouFire-SouthOf98-AnchorLakeDr.geo.tif",
    "05-08-2020-MBF-S98-DLn.geo.tif": "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif",
    "05-08-2020-MBF-S98-LPC.geo.tif": "05-08-2020-MussettBayouFire-SouthOf98-LakeParkCove.geo.tif",
    "0827-A-01.geo.tif": "0827-A-01.geo.tif",
    "0827-B-02.geo.tif": "0827-B-02.geo.tif",
    "2018-05-18-X4S-visible-CentralPark.geo.tif": "2018-05-18-X4S-visible-CentralPark.geo.tif",
    "2018-05-18-X5-visible-Geothermal.geo.tif": "2018-05-18-X5-visible-Geothermal.geo.tif",
    "2018-05-18-X5-visible-Kahukai.geo.tif": "2018-05-18-X5-visible-Kahukai.geo.tif",
    "20210703-Champlain-Towers -South.geo.tif": "20210703-Champlain-Towers -South.geo.tif",
    "20210831-LA-DIV-01.geo.tif": "20210831-LA-DIV-01.geo.tif",
    "20210901-Cocodrie-1.geo.tif": "20210901-Cocodrie-1.geo.tif",
    "20210901-Cocodrie-2.geo.tif": "20210901-Cocodrie-2.geo.tif",
    "20210901-Cocodrie-3.geo.tif": "20210901-Cocodrie-3.geo.tif",
    "20210902-LA-DIV-01.geo.tif": "20210902-LA-DIV-01.geo.tif",
    "20211213-Candle-Factory-AO.geo.tif": "20211213-Candle-Factory-AO.geo.tif",
    "20211214-Mayfield.geo.tif": "20211214-Mayfield.geo.tif",
    "20211215-Russelville-Middle.geo.tif": "20211215-Russelville-Middle.geo.tif",
    "1002-Ft-Myers-Beach.5.geo.tif":None,
    "1002-Ft-Myers-Beach.4.geo.tif":None,
    "1002-Ft-Myers-Beach.3.geo.tif":None,
    "1002-Ft-Myers-Beach.2.geo.tif":None,
    "1002-Ft-Myers-Beach.1.geo.tif":None
}

ORTHO_TITLE_TO_LABELBOX_DATASET = {
    "090302-Pecan-Grove-Levee.geo.tif": "090302-Pecan-Grove-Levee.geo.tif",
    "090401-DMS-Assessment-Westpark.geo.tif": "090401-DMS-Assessment-Westpark.geo.tif",
    "090402-DMS-Assessment-Sienna-Village.geo.tif": "090402-DMS-Assessment-Sienna-Village.geo.tif",
    "090403-Lancaster-Canyon-Gate.geo.tif": "090403-Lancaster-Canyon-Gate.geo.tif",
    "1001-Ft-Myers-Beach-Boone.geo.tif": "1001-Ft-Myers-Beach-Boone.geo.tif",
    "1001-Ft-Myers-Beach-DIRT.geo.tif": "1001-Ft-Myers-Beach-DIRT.geo.tif",
    "1001-Harlem-Heights.geo.tif": "1001-Harlem-Heights.geo.tif",
    "1001-Iona-Point.geo.tif": "1001-Iona-Point.geo.tif",
    "1001-Kennedy-Green-Mobile-Homes.geo.tif": "1001-Kennedy-Green-Mobile-Homes.geo.tif",
    "1001-McGregor-College-Pkwy-South.1.geo.tif": "1001-McGregor-College-Pkwy-South.1.geo.tif",
    "1001-McGregor-College-Pkwy-South.2.geo.tif": "1001-McGregor-College-Pkwy-South.2.geo.tif",
    "1001-McGregor-College-Pkwy-South.3.geo.tif": "1001-McGregor-College-Pkwy-South.3.geo.tif",
    "1001-Palmeto-Palms.geo.tif": "1001-Palmeto-Palms.geo.tif",
    "1001-San-Carlos-Island.geo.tif": "1001-San-Carlos-Island.geo.tif",
    "1001-Summerlin-San-Carlos.geo.tif": "1001-Summerlin-San-Carlos.geo.tif",
    "1002-Boca-Grande.1.geo.tif": "1002-Boca-Grande.1.geo.tif",
    "1002-Boca-Grande.2.geo.tif": "1002-Boca-Grande.2.geo.tif",
    "1002-Boca-Grande.3.geo.tif": "1002-Boca-Grande.3.geo.tif",
    "1002-Boca-Grande.4.geo.tif": "1002-Boca-Grande.4.geo.tif",
    "1002-Boca-Grande.5.geo.tif": "1002-Boca-Grande.5.geo.tif",
    "1002-Boca-Grande.6.geo.tif": "1002-Boca-Grande.6.geo.tif",
    "1002-Ft-Myers-Beach-LCSO.geo.tif": "1002-Ft-Myers-Beach-LCSO.geo.tif",
    "1002-Ft-Myers-Beach-TFD.geo.tif": "1002-Ft-Myers-Beach-TFD.geo.tif",
    "1002-Kelly-Road.geo.tif": "1002-Kelly-Road.geo.tif",
    "1002-Palm-Acers.1.geo.tif": "1002-Palm-Acres.1.geo.tif",
    "1002-Palm-Acers.2.geo.tif": "1002-Palm-Acers.2.geo.tif",
    "1002-Palm-Acers.3.geo.tif": "1002-Palm-Acers.3.geo.tif",
    "1002-Palm-Acers.4.geo.tif": "1002-Palm-Acers.4.geo.tif",
    "1002-Sanibel-Causeway-North.geo.tif": "1002-Sanibel-Causeway-North.geo.tif",
    "10132018-MexicoBeach.geo.tif": "10132018-MexicoBeach.geo.tif",
    "10142018-MexicoBeach.geo.tif": "10142018-MexicoBeach.geo.tif",
    "20230830-SteinhatcheeRiver.geo.tif": "20230830-SteinhatcheeRiver.geo.tif",
    "20230831-Jena-SteinhatcheeRiverSouth.geo.tif": "20230831-SteinhatcheeRiverSouth.geo.tif",
    "05-08-2020-MussettBayouFire-01.geo.tif": "05-08-2020-MussettBayouFire-01.geo.tif",
    "05-08-2020-MussettBayouFire-NorthOf98.geo.tif": "05-08-2020-MussettBayouFire-N98.geo.tif",
    "05-08-2020-MussettBayouFire-SouthOf98-AnchorLakeDr.geo.tif": "05082020MussBayFireS98AnchorLakeDr.geo.tif",
    "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif": "05082020MussBayFireS98DelbertLn.geo.tif",
    "05-08-2020-MussettBayouFire-SouthOf98-LakeParkCove.geo.tif": "05082020MussBayFireS98LakeParkCove.geo.tif",
    "0827-A-01.geo.tif": "0827-A-01.geo.tif",
    "0827-B-02.geo.tif": "0827-B-02.geo.tif",
    "2018-05-18-X4S-visible-CentralPark.geo.tif": "2018-05-18-X4S-visible-CentralPark.geo.tif",
    "2018-05-18-X5-visible-Geothermal.geo.tif": "2018-05-18-X5-visible-Geothermal.geo.tif",
    "2018-05-18-X5-visible-Kahukai.geo.tif": "2018-05-18-X5-visible-Kahukai.geo.tif",
    "20210703-Champlain-Towers -South.geo.tif": "20210703-Champlain-Towers -South.geo.tif",
    "20210831-LA-DIV-01.geo.tif": "20210831-LA-DIV-01.geo.tif",
    "20210901-Cocodrie-1.geo.tif": "20210901-Cocodrie-1.geo.tif",
    "20210901-Cocodrie-2.geo.tif": "20210901-Cocodrie-2.geo.tif",
    "20210901-Cocodrie-3.geo.tif": "20210901-Cocodrie-3.geo.tif",
    "20210902-LA-DIV-01.geo.tif": "20210902-LA-DIV-01.geo.tif",
    "20211213-Candle-Factory-AO.geo.tif": "20211213-Candle-Factory-AO.geo.tif",
    "20211214-Mayfield.geo.tif": "20211214-Mayfield.geo.tif",
    "20211215-Russelville-Middle.geo.tif": "20211215-Russelville-Middle.geo.tif"

}

LABELBOX_DATASET_TO_ORTHO_TITLE = {v:k for k, v in ORTHO_TITLE_TO_LABELBOX_DATASET.items()}

HURRICANE_IAN = "Hurricane Ian"
HURRICANE_IDA = "Hurricane Ida"
HURRICANE_IDALIA = "Hurricane Idalia"
HURRICANE_HARVEY = "Hurricane Harvey"
HURRICANE_MICHAEL = "Hurricane Michael"
HURRICANE_LAURA = "Hurricane Laura"
MAYFIELD_TORNADO = "Mayfield Tornado"
KILAUEA_VOLCANO = "Kilauea Volcano Eruption"
MUSSETT_BAYOU_FIRE = "Mussett Bayou Fire"
CHAMPLAIN_TOWERS_COLLAPSE = "Champlain Towers Collapse"


EVENTS = [HURRICANE_IAN, HURRICANE_IDALIA, HURRICANE_HARVEY, HURRICANE_MICHAEL]

EVENTS_ORTHO = {HURRICANE_IAN: ["1001-Ft-Myers-Beach-Boone.geo.tif", "1001-Ft-Myers-Beach-DIRT.geo.tif",
                                "1001-Harlem-Heights.geo.tif", "1001-Iona-Point.geo.tif", "1001-Kennedy-Green-Mobile-Homes.geo.tif",
                                "1001-McGregor-College-Pkwy-South.1.geo.tif", "1001-McGregor-College-Pkwy-South.2.geo.tif",
                                "1001-McGregor-College-Pkwy-South.3.geo.tif", "1001-Palmeto-Palms.geo.tif",
                                "1001-San-Carlos-Island.geo.tif", "1001-Summerlin-San-Carlos.geo.tif",
                                "1002-Boca-Grande.1.geo.tif", "1002-Boca-Grande.2.geo.tif",
                                "1002-Boca-Grande.3.geo.tif", "1002-Boca-Grande.4.geo.tif",
                                "1002-Boca-Grande.5.geo.tif", "1002-Boca-Grande.6.geo.tif",
                                "1002-Ft-Myers-Beach-LCSO.geo.tif", "1002-Ft-Myers-Beach-TFD.geo.tif",
                                "1002-Kelly-Road.geo.tif", "1002-Palm-Acers.1.geo.tif",
                                "1002-Palm-Acers.2.geo.tif", "1002-Palm-Acers.3.geo.tif",
                                "1002-Palm-Acers.4.geo.tif", "1002-Sanibel-Causeway-North.geo.tif"],
                HURRICANE_IDALIA: ["20230830-SteinhatcheeRiver.geo.tif", "20230831-Jena-SteinhatcheeRiverSouth.geo.tif"],
                HURRICANE_LAURA: ["0827-A-01.geo.tif", "0827-B-02.geo.tif"],
                HURRICANE_HARVEY: ["090302-Pecan-Grove-Levee.geo.tif", "090401-DMS-Assessment-Westpark.geo.tif",
                                   "090402-DMS-Assessment-Sienna-Village.geo.tif", "090403-Lancaster-Canyon-Gate.geo.tif"],
                HURRICANE_MICHAEL: ["10132018-MexicoBeach.geo.tif", "10142018-MexicoBeach.geo.tif"],
                HURRICANE_IDA: ["20210831-LA-DIV-01.geo.tif", "20210901-Cocodrie-1.geo.tif", "20210901-Cocodrie-2.geo.tif",
                                "20210901-Cocodrie-3.geo.tif", "20210902-LA-DIV-01.geo.tif"],
                MAYFIELD_TORNADO: ["20211213-Candle-Factory-AO.geo.tif", "20211214-Mayfield.geo.tif", "20211215-Russelville-Middle.geo.tif"],
                KILAUEA_VOLCANO: ["2018-05-18-X4S-visible-CentralPark.geo.tif", "2018-05-18-X5-visible-Geothermal.geo.tif",
                                  "2018-05-18-X5-visible-Kahukai.geo.tif"],
                CHAMPLAIN_TOWERS_COLLAPSE: ["20210703-Champlain-Towers -South.geo.tif"],
                MUSSETT_BAYOU_FIRE: ["05-08-2020-MussettBayouFire-01.geo.tif", "05-08-2020-MussettBayouFire-NorthOf98.geo.tif",
                                     "05-08-2020-MussettBayouFire-SouthOf98-AnchorLakeDr.geo.tif",
                                     "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif",
                                     "05-08-2020-MussettBayouFire-SouthOf98-LakeParkCove.geo.tif"]
                }

ORTHO_EVENT = {}
for event, orthos in EVENTS_ORTHO.items():
    for ortho in orthos:
        ORTHO_EVENT[ortho] = event

TRAIN_EVENTS = [HURRICANE_IAN, HURRICANE_LAURA, HURRICANE_HARVEY, HURRICANE_IDA, KILAUEA_VOLCANO, CHAMPLAIN_TOWERS_COLLAPSE]
TEST_EVENTS = [MAYFIELD_TORNADO, MUSSETT_BAYOU_FIRE, HURRICANE_MICHAEL, HURRICANE_IDALIA]

ORTHO_DATETIME = {
    "090302-Pecan-Grove-Levee.geo.tif": "09-03-2017",
    "090401-DMS-Assessment-Westpark.geo.tif": "09-04-2017",
    "090402-DMS-Assessment-Sienna-Village.geo.tif": "09-04-2017",
    "090403-Lancaster-Canyon-Gate.geo.tif": "09-04-2017",
    "1001-Ft-Myers-Beach-Boone.geo.tif": "10-01-2022",
    "1001-Ft-Myers-Beach-DIRT.geo.tif": "10-01-2022",
    "1001-Harlem-Heights.geo.tif": "10-01-2022",
    "1001-Iona-Point.geo.tif": "10-01-2022",
    "1001-Kennedy-Green-Mobile-Homes.geo.tif": "10-01-2022",
    "1001-McGregor-College-Pkwy-South.1.geo.tif": "10-01-2022",
    "1001-McGregor-College-Pkwy-South.2.geo.tif": "10-01-2022",
    "1001-McGregor-College-Pkwy-South.3.geo.tif": "10-01-2022",
    "1001-Palmeto-Palms.geo.tif": "10-01-2022",
    "1001-San-Carlos-Island.geo.tif": "10-01-2022",
    "1001-Summerlin-San-Carlos.geo.tif": "10-01-2022",
    "1002-Boca-Grande.1.geo.tif": "10-02-2022",
    "1002-Boca-Grande.2.geo.tif": "10-02-2022",
    "1002-Boca-Grande.3.geo.tif": "10-02-2022",
    "1002-Boca-Grande.4.geo.tif": "10-02-2022",
    "1002-Boca-Grande.5.geo.tif": "10-02-2022",
    "1002-Boca-Grande.6.geo.tif": "10-02-2022",
    "1002-Ft-Myers-Beach-LCSO.geo.tif": "10-02-2022",
    "1002-Ft-Myers-Beach-TFD.geo.tif": "10-02-2022",
    "1002-Kelly-Road.geo.tif": "10-02-2022",
    "1002-Palm-Acers.1.geo.tif": "10-02-2022",
    "1002-Palm-Acers.2.geo.tif": "10-02-2022",
    "1002-Palm-Acers.3.geo.tif": "10-02-2022",
    "1002-Palm-Acers.4.geo.tif": "10-02-2022",
    "1002-Sanibel-Causeway-North.geo.tif": "10-02-2022",
    "10132018-MexicoBeach.geo.tif": "10-13-2018",
    "10142018-MexicoBeach.geo.tif": "10-14-2018",
    "20230830-SteinhatcheeRiver.geo.tif": "08-30-2023",
    "20230831-Jena-SteinhatcheeRiverSouth.geo.tif": "08-31-2023",
    "05-08-2020-MussettBayouFire-01.geo.tif": "05-08-2020",  # TODO: Double Check Date
    "05-08-2020-MussettBayouFire-NorthOf98.geo.tif": "05-08-2020",  # TODO: Double Check Date
    "05-08-2020-MussettBayouFire-SouthOf98-AnchorLakeDr.geo.tif": "05-08-2020",  # TODO: Double Check Date
    "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif": "05-08-2020",  # TODO: Double Check Date
    "05-08-2020-MussettBayouFire-SouthOf98-LakeParkCove.geo.tif": "05-08-2020",  # TODO: Double Check Date
    "0827-A-01.geo.tif": "08-27-2020", # TODO: Double Check Date
    "0827-B-02.geo.tif": "08-27-2020",
    "2018-05-18-X4S-visible-CentralPark.geo.tif": "05-18-2018",  # TODO: Double Check Date
    "2018-05-18-X5-visible-Geothermal.geo.tif": "05-18-2018",  # TODO: Double Check Date
    "2018-05-18-X5-visible-Kahukai.geo.tif": "05-18-2018",  # TODO: Double Check Date
    "20210703-Champlain-Towers -South.geo.tif": "07-03-2021",  # TODO: Double Check Date
    "20210831-LA-DIV-01.geo.tif": "08-31-2021",
    "20210901-Cocodrie-1.geo.tif": "09-01-2021", # TODO: Double Check Date
    "20210901-Cocodrie-2.geo.tif": "09-01-2021",
    "20210901-Cocodrie-3.geo.tif": "09-01-2021",
    "20210902-LA-DIV-01.geo.tif": "09-02-2021",
    "20211213-Candle-Factory-AO.geo.tif": "12-13-2021", # TODO: Double Check Date
    "20211214-Mayfield.geo.tif": "12-14-2021", # TODO: Double Check Date
    "20211215-Russelville-Middle.geo.tif": "12-15-2021" # TODO: Double Check Date
}


ORTHO_GSD = {
    "090302-Pecan-Grove-Levee.geo.tif": 1.947011898,
    "090401-DMS-Assessment-Westpark.geo.tif": 1.9599834,
    "090402-DMS-Assessment-Sienna-Village.geo.tif": 2.11934952646896, # TODO: Need to Update
    "090403-Lancaster-Canyon-Gate.geo.tif": 3.651,
    "1001-Ft-Myers-Beach-Boone.geo.tif": 3.498758824,
    "1001-Ft-Myers-Beach-DIRT.geo.tif": 2.999266396,
    "1001-Harlem-Heights.geo.tif": 4.672015432,
    "1001-Iona-Point.geo.tif": 3.81,
    "1001-Kennedy-Green-Mobile-Homes.geo.tif": 3.458573726,
    "1001-McGregor-College-Pkwy-South.1.geo.tif": 3.838880264,
    "1001-McGregor-College-Pkwy-South.2.geo.tif": 1.997849149,
    "1001-McGregor-College-Pkwy-South.3.geo.tif": 4.620378947,
    "1001-Palmeto-Palms.geo.tif": 2.54,
    "1001-San-Carlos-Island.geo.tif": 4.092435469,
    "1001-Summerlin-San-Carlos.geo.tif": 4.059608099,
    "1002-Boca-Grande.1.geo.tif": 2.494279102,
    "1002-Boca-Grande.2.geo.tif": 3.797910926,
    "1002-Boca-Grande.3.geo.tif": 3.795145385,
    "1002-Boca-Grande.4.geo.tif": 2.474423818,
    "1002-Boca-Grande.5.geo.tif": 3.876947425,
    "1002-Boca-Grande.6.geo.tif": 3.911851623,
    "1002-Ft-Myers-Beach-LCSO.geo.tif": 3.348838574,
    "1002-Ft-Myers-Beach-TFD.geo.tif": 3.084663289,
    "1002-Kelly-Road.geo.tif": 4.09004026,
    "1002-Palm-Acers.1.geo.tif": 3.422865012,
    "1002-Palm-Acers.2.geo.tif": 4.54592715,
    "1002-Palm-Acers.3.geo.tif": 3.449713824,
    "1002-Palm-Acers.4.geo.tif": 2.574879524,
    "1002-Sanibel-Causeway-North.geo.tif": 3.302688085,
    "10132018-MexicoBeach.geo.tif": 1.964605832,
    "10142018-MexicoBeach.geo.tif": 1.946466023,
    "20230830-SteinhatcheeRiver.geo.tif": 12.7, # # TODO: Need to Update
    "20230831-Jena-SteinhatcheeRiverSouth.geo.tif": 12.7, # TODO: Need to Update
    "05-08-2020-MussettBayouFire-01.geo.tif": 3.215340129,
    "05-08-2020-MussettBayouFire-NorthOf98.geo.tif": 4.401283824,
    "05-08-2020-MussettBayouFire-SouthOf98-AnchorLakeDr.geo.tif": 2.019042353,
    "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif": 2.001503155,
    "05-08-2020-MussettBayouFire-SouthOf98-LakeParkCove.geo.tif": 2.042948968,
    "0827-A-01.geo.tif": 4.034476404,
    "0827-B-02.geo.tif": 3.85107563,
    "2018-05-18-X4S-visible-CentralPark.geo.tif": 7.829940969,
    "2018-05-18-X5-visible-Geothermal.geo.tif": 6.672295171,
    "2018-05-18-X5-visible-Kahukai.geo.tif": 6.532728592,
    "20210703-Champlain-Towers -South.geo.tif": 1.774649562,
    "20210831-LA-DIV-01.geo.tif": 2.498187547,
    "20210901-Cocodrie-1.geo.tif": 3.067821765,
    "20210901-Cocodrie-2.geo.tif": 3.179036317,
    "20210901-Cocodrie-3.geo.tif": 3.208421751,
    "20210902-LA-DIV-01.geo.tif": 3.148085978,
    "20211213-Candle-Factory-AO.geo.tif": 2.927101439,
    "20211214-Mayfield.geo.tif": 3.110499127,
    "20211215-Russelville-Middle.geo.tif": 2.197146592
}
