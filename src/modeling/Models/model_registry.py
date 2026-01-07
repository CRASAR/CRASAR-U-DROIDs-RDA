from modeling.Models.MaskedUNet.LitMaskedUNetModel import LitMaskedUNetModel, LitAlignedMaskedUNetModel
from modeling.Models.ZampieriEtAl2018.LitZampieriEtAl2018 import LitZampieriEtAl2018
from modeling.Models.Segmenter.SegmenterVit import SegmenterVit
from modeling.Models.UperNet.UperNetVit import UperNetVit
from modeling.Models.Baselines.RandomBaselineModel import RandomBaselineModel
from modeling.Models.PSPNet.PSPNetResNet import PSPNetResNet
from modeling.Models.DeepLabV3Plus.DeepLabV3PlusResNet import DeepLabV3PlusResNet
from modeling.Models.BaseModelBDA import BaseModelBDA
from modeling.Models.BaseModelRDA import BaseModelRDA
from modeling.Models.BaseModelBDAADJ import BaseModelBDAADJ

from modeling.utils.sample_presentation import (
    IndexSampleLocationPresentationStrategy,
    MostRecentlyObservedSampleLocationPresentationStrategy,
    WeightedSampleLocationPresentationStrategy
)

from modeling.utils.sample_location_generator import (
    RandomSampleLocationGenerationStrategy,
    CenteredBuildingSampleStrategy,
    GridSampleStrategy,
    BDASampleAnnotator,
    RDASampleAnnotator
)
from modeling.utils.mask_generation import (
    MaskingStrategyBDA,
    MaskingStrategyRDA
)

from modeling.utils.data_augmentations import (
    KeyPointConversionStrategyRDA,
    KeyPointConversionStrategyBDA
)

# TODO: Need to think about how this transfers for train and inference
# TODO: We may want to further extend this to different datasets
LOCATIONSTRATEGY2MODULEMAPPING = {
    "random": RandomSampleLocationGenerationStrategy,
    "centered": CenteredBuildingSampleStrategy,
    "grid": GridSampleStrategy
}

PRESENTATIONTRATEGY2MODULEMAPPING = {
    "indexed": IndexSampleLocationPresentationStrategy,
    "most_recently_observed": MostRecentlyObservedSampleLocationPresentationStrategy,
    "weighted": WeightedSampleLocationPresentationStrategy
}

MASKINGSTRATEGY2MODULEMAPPING = {
    "BDA": MaskingStrategyBDA,
    "BDAADJ": MaskingStrategyBDA,
    "RDA": MaskingStrategyRDA,
    "RDAADJ": MaskingStrategyRDA,
}

KEYPOINTSTRATEGY2MODULEMAPPING = {
    "BDA": KeyPointConversionStrategyBDA,
    "BDAADJ": KeyPointConversionStrategyBDA,
    "RDA": KeyPointConversionStrategyRDA,
    "RDAADJ": None,
}

SAMPLEANNOTATORTRATEGY2MODULEMPAPPING = {
    "BDA": BDASampleAnnotator,
    "BDAADJ": BDASampleAnnotator,
    "RDA": RDASampleAnnotator,
    "RDAADJ": None,
}

STR2TASKMODELCLASS = {
    "BDA": BaseModelBDA,
    "RDA": BaseModelRDA,
    "BDAADJ": BaseModelBDAADJ,
}

STR2MODELCLASS = {
    "AlignedMaskedUNet": LitAlignedMaskedUNetModel,
    "ZampieriEtAl2018": LitZampieriEtAl2018,
    "MaskedUNet": LitMaskedUNetModel,
    "SegmenterVit": SegmenterVit,
    "UperNetVit": UperNetVit,
    "RandomBaseline": RandomBaselineModel,
    "PSPNetResNet": PSPNetResNet,
    "DeepLabV3PlusResNet": DeepLabV3PlusResNet
}
